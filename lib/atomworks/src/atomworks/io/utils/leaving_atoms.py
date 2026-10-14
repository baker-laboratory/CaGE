"""Utilities for handling leaving atoms in atomic structures."""

import functools
import logging
import os
from collections import defaultdict
from typing import Literal

import biotite.structure as struc
import networkx as nx
import numpy as np

from atomworks.constants import (
    CCD_MIRROR_PATH,
    DEFAULT_VALENCE,
    DO_NOT_MATCH_CCD,
    HYDROGEN_LIKE_SYMBOLS,
)
from atomworks.io.utils.atom_array import (
    _find_bonded_hydrogens,
    _get_bond_neighbors,
    get_bond_degree_per_atom,
)
from atomworks.io.utils.ccd import _standard_ccd_only_cache, atom_array_from_ccd_code

logger = logging.getLogger(__name__)


def _find_connected_components_after_removal(graph: nx.Graph, node_to_remove: int) -> list[list[int]]:
    """Identifies connected components that would form after removing a node from a graph.

    Args:
        graph: The input graph.
        node_to_remove: The node to hypothetically remove.

    Returns:
        List of lists containing node indices in each new component.
    """
    # Get the neighbors before removal
    neighbors = set(graph.neighbors(node_to_remove))
    if not neighbors:
        return []

    # Create subgraph without the node
    subgraph = graph.subgraph(set(graph.nodes) - {node_to_remove})

    # Use BFS from any neighbor to find components
    components = []
    unvisited = neighbors.copy()

    while unvisited:
        start = unvisited.pop()
        if start not in subgraph:
            continue

        # Find all nodes reachable from this neighbor
        component = list(nx.bfs_tree(subgraph, start))
        components.append(component)
        unvisited -= set(component)

    return components


@_standard_ccd_only_cache(functools.cache)
def _get_chem_comp_leaving_atom_names_cached(
    ccd_code: str, ccd_mirror_path_str: str, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[str, ...]]:
    """Internal cached implementation. Do not call directly."""
    # Skip CCD lookup for codes that shouldn't be matched against CCD (e.g., UNL, water-like codes)
    if ccd_code in DO_NOT_MATCH_CCD:
        if mode == "warn":
            logger.debug(f"Skipping CCD lookup for `{ccd_code}` as it's in DO_NOT_MATCH_CCD")
        return {}

    try:
        chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path_str, coords=None)
    except (ValueError, AttributeError) as e:
        if mode == "warn":
            logger.warning(f"Failed to compute leaving groups for `{ccd_code}`: {e}")
        elif mode == "raise":
            raise ValueError(f"Failed to compute leaving groups for `{ccd_code}`: {e}") from e
        return {}

    if "is_leaving_atom" not in chem_comp.get_annotation_categories():
        if mode == "warn":
            logger.warning(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups, returning empty dictionary. "
                "Check if your CCD mirror is up to date."
            )
        elif mode == "raise":
            raise ValueError(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups. Check if your CCD mirror is up to date."
            )
        return {}

    # ... initialize output
    leaving_atom_names = defaultdict(list)

    # ... get relevant annotations
    is_leaving_atom = chem_comp.get_annotation("is_leaving_atom")
    atom_name = chem_comp.get_annotation("atom_name")
    element = chem_comp.get_annotation("element")

    # ... skip if no atoms are annotated as leaving atoms (majority of CCD entries)
    if not any(is_leaving_atom):
        return {}

    # ... compute the leaving groups based on the bond graph and annotation
    bond_graph = chem_comp.bonds.as_graph()
    for atom_idx in range(chem_comp.array_length()):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(bond_graph, atom_idx)

        # ... check if all atoms in the connected group are flagged as leaving atoms
        #     by the CCD entry
        for connected_group in connected_groups:
            heavy_atoms: list[int] = list(filter(lambda x: element[x] != "H", connected_group))
            is_leaving_group = (
                all(is_leaving_atom[heavy_atoms]) if len(heavy_atoms) > 0 else all(is_leaving_atom[connected_group])
            )

            if is_leaving_group:
                leaving_atom_names[atom_name[atom_idx]] += [atom_name[idx] for idx in connected_group]

    # ... turn leaving_atom_names into a dictionary of tuples
    leaving_atom_names = {k: tuple(v) for k, v in leaving_atom_names.items()}

    return leaving_atom_names


def get_chem_comp_leaving_atom_names(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[str, ...]]:
    """Compute leaving atom names for a given CCD entry.

    The returned dictionary maps the name of the atom to the names of the atoms that would
    become disconnected if the atom were removed.

    Example:
        >>> get_chem_comp_leaving_atom_names("ALA")
        {'N': ('H2',), 'C': ('OXT', 'HXT'), 'OXT': ('HXT',)}
    """
    # Normalize path and delegate to cached implementation (use empty string for None)
    cache_key = os.fspath(ccd_mirror_path) if ccd_mirror_path else ""
    return _get_chem_comp_leaving_atom_names_cached(ccd_code, cache_key, mode)


def get_leaving_atom_bond_type(ccd_code: str, atom_name: str) -> struc.BondType | None:
    """Return the bond type between ``atom_name`` and its CCD leaving atoms."""
    leaving_names = get_chem_comp_leaving_atom_names(ccd_code)
    if atom_name not in leaving_names:
        return None

    try:
        chem_comp = atom_array_from_ccd_code(ccd_code)
    except (ValueError, AttributeError):
        return None

    names = chem_comp.atom_name
    matches = np.where(names == atom_name)[0]
    if not len(matches):
        return None

    atom_idx = int(matches[0])
    leaving_set = set(leaving_names[atom_name])
    rows = chem_comp.bonds.as_array()
    rows = rows[(rows[:, 0] == atom_idx) | (rows[:, 1] == atom_idx)]
    for a1, a2, bt in rows:
        nb = a2 if a1 == atom_idx else a1
        if names[nb] in leaving_set and (bond_type := struc.BondType(int(bt))) != struc.BondType.SINGLE:
            return bond_type
    return None


def _check_bonds_in_direction(
    bonds_array: np.ndarray,
    leaving_atom_mask: np.ndarray,
    parent_atom_mask: np.ndarray,
    leaving_col: int,
    other_col: int,
    atom_array: struc.AtomArray,
) -> None:
    """Check that leaving atoms at ``leaving_col`` only bond to valid atoms at ``other_col``."""
    is_leaving = leaving_atom_mask[bonds_array[:, leaving_col]]
    bonds_with_leaving = bonds_array[is_leaving]

    if len(bonds_with_leaving) == 0:
        return

    other_atoms = bonds_with_leaving[:, other_col]
    # Valid bonds: to parent atoms OR to other leaving atoms
    is_other_valid = parent_atom_mask[other_atoms] | leaving_atom_mask[other_atoms]
    invalid_mask = ~is_other_valid

    if np.any(invalid_mask):
        # Find first invalid bond for error message
        invalid_bond = bonds_with_leaving[invalid_mask][0]
        leaving_idx = invalid_bond[leaving_col]
        other_idx = invalid_bond[other_col]
        raise ValueError(
            f"Leaving atom '{atom_array.atom_name[leaving_idx]}' in residue "
            f"{atom_array.res_name[leaving_idx]} (index {leaving_idx}) is bonded to "
            f"'{atom_array.atom_name[other_idx]}' (index {other_idx}), which is neither "
            f"a parent atom making inter-residue bonds nor another leaving atom. "
            f"Cannot safely remove."
        )


def _validate_leaving_atoms_have_no_unexpected_bonds(
    atom_array: struc.AtomArray,
    leaving_atom_mask: np.ndarray,
    atoms_with_inter_bonds: np.ndarray,
) -> None:
    """Validate that atoms marked for removal don't have unexpected bonds."""
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return

    bonds_array = atom_array.bonds.as_array()
    parent_atom_mask = np.zeros(len(atom_array), dtype=bool)
    parent_atom_mask[atoms_with_inter_bonds] = True

    # Check both directions
    _check_bonds_in_direction(bonds_array, leaving_atom_mask, parent_atom_mask, 0, 1, atom_array)
    _check_bonds_in_direction(bonds_array, leaving_atom_mask, parent_atom_mask, 1, 0, atom_array)


def _get_inter_residue_bonds(atom_array: struc.AtomArray) -> np.ndarray:
    """Return Nx3 array of inter-residue bonds ``[atom1, atom2, bond_type]``."""
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return np.empty((0, 3), dtype=int)

    bonds_array = atom_array.bonds.as_array()

    # Define residue identity
    group_by = ["chain_id", "res_id", "ins_code"]
    if "transformation_id" in atom_array.get_annotation_categories():
        group_by.append("transformation_id")

    # For each bond, check if atoms are in same residue
    bond_atom1_keys = [tuple(atom_array.get_annotation(field)[bonds_array[:, 0]]) for field in group_by]
    bond_atom2_keys = [tuple(atom_array.get_annotation(field)[bonds_array[:, 1]]) for field in group_by]

    # Stack keys into arrays for comparison
    atom1_residue_keys = np.column_stack(bond_atom1_keys)
    atom2_residue_keys = np.column_stack(bond_atom2_keys)

    # Find bonds where atoms are in different residues
    is_inter_bond = np.any(atom1_residue_keys != atom2_residue_keys, axis=1)

    return bonds_array[is_inter_bond]


def get_inter_residue_atom_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Get boolean mask indicating which atoms are involved in inter-residue bonds."""
    inter_bonds = _get_inter_residue_bonds(atom_array)

    if len(inter_bonds) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    # Get unique atoms involved in inter-residue bonds
    atoms_with_inter_bonds = np.unique(inter_bonds[:, :2])

    # Create atom mask
    atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
    atom_mask[atoms_with_inter_bonds] = True

    return atom_mask


def _maybe_fix_overvalent_carbon(atom_array: struc.AtomArray, atom_idx: int) -> list[int]:
    """Fix an overvalent carbon atom.

    Tries multiple strategies in order and returns after the first that applies:

    1. **Bonded H removal**: remove a directly bonded H (e.g., a CCD ghost H on a
       non-polymer cap such as ACE/1j8z that is displaced when the covalent bond
       forms).
    2. **C=O decrement**: decrement a C=O double bond to single and set the oxygen
       charge to -1 (nucleophilic addition pushes lone pair to O; only applies when
       no H is available to remove — e.g., 4PA/CAI in 1tqh where SER-94 OG attacks
       the carbonyl carbon).

    Returns:
        List of atom indices to remove (caller must apply). Empty list means the fix
        was applied in-place (e.g., C=O decrement).
    """
    bonds_arr = atom_array.bonds.as_array()
    neighbors = _get_bond_neighbors(bonds_arr, atom_idx)

    o_neighbors = neighbors[atom_array.element[neighbors] == "O"]

    # --- Strategy 1: remove bonded H (explicit or implicit) ---
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()

    def _log_h_removal() -> None:
        logger.warning(
            f"Removed H from overvalent {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]} "
            f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
        )

    # ... implicit path
    if has_nhyd and atom_array.nhyd[atom_idx] > 0:
        atom_array.nhyd[atom_idx] -= 1
        _log_h_removal()
        return []

    # ... explicit path
    h_neighbors = neighbors[atom_array.element[neighbors] == "H"]
    if len(h_neighbors) > 0:
        _log_h_removal()
        return [int(h_neighbors[0])]

    # --- Strategy 2: C=O decrement ---
    for o_idx in o_neighbors:
        bond_mask = ((bonds_arr[:, 0] == atom_idx) & (bonds_arr[:, 1] == o_idx)) | (
            (bonds_arr[:, 0] == o_idx) & (bonds_arr[:, 1] == atom_idx)
        )
        if bond_mask.any() and bonds_arr[bond_mask][0, 2] == int(struc.BondType.DOUBLE):
            atom_array.bonds.remove_bond(atom_idx, int(o_idx))
            atom_array.bonds.add_bond(atom_idx, int(o_idx), struc.BondType.SINGLE)
            atom_array.charge[o_idx] = -1
            logger.warning(
                f"Decremented C=O bond order for {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]} "
                f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
            )
            return []

    return []


def _remove_ccd_leaving_atoms(atom_array: struc.AtomArray, atoms_with_inter_bonds: np.ndarray) -> struc.AtomArray:
    """Remove CCD-defined leaving atoms for atoms that form inter-residue bonds."""
    leaving_atom_mask = np.zeros(len(atom_array), dtype=bool)
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()
    _h_first = frozenset(s[0] for s in HYDROGEN_LIKE_SYMBOLS)

    # Pre-fetch CCD leaving info for every residue that makes an inter-residue bond.
    unique_res_names = np.unique(atom_array.res_name[atoms_with_inter_bonds])
    leaving_cache = {rn: get_chem_comp_leaving_atom_names(rn) for rn in unique_res_names}

    # Build per-atom residue slices for O(1) lookup inside the loop.
    _rs = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sizes = np.diff(_rs)
    atom_to_res_start = np.repeat(_rs[:-1], sizes)
    atom_to_res_stop = np.repeat(_rs[1:], sizes)

    for atom_idx in atoms_with_inter_bonds:
        leaving_names = leaving_cache[atom_array.res_name[atom_idx]].get(atom_array.atom_name[atom_idx], ())
        if not leaving_names:
            continue

        # Mark all leaving atoms within this residue for removal.
        res_start = int(atom_to_res_start[atom_idx])
        res_stop = int(atom_to_res_stop[atom_idx])
        leaving_set = set(leaving_names)
        is_leaving = np.isin(atom_array.atom_name[res_start:res_stop], list(leaving_set))
        leaving_atom_mask[res_start:res_stop] |= is_leaving

        # If all leaving atoms are implicit H tracked via nhyd, also decrement the
        # counter so the implicit H count stays consistent after removal.
        if has_nhyd and all(n[0] in _h_first for n in leaving_set if n):
            atom_array.nhyd[atom_idx] -= len(leaving_set)

    if np.any(leaving_atom_mask):
        # Sanity check: leaving atoms must only bond to their parent or each other.
        _validate_leaving_atoms_have_no_unexpected_bonds(atom_array, leaving_atom_mask, atoms_with_inter_bonds)
        atom_array = atom_array[~leaving_atom_mask]

    return atom_array


def _resolve_overvalent_atoms(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Fix atoms that remain over-valent after CCD leaving atom removal."""
    inter_bonds = _get_inter_residue_bonds(atom_array)
    if len(inter_bonds) == 0:
        return atom_array

    inter_atoms = np.unique(inter_bonds[:, :2])
    degree = get_bond_degree_per_atom(atom_array)
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()
    # Collect removals and apply in one batch — removing inside the loop would
    # invalidate the pre-computed inter_atoms index array.
    atoms_to_remove: list[int] = []

    for atom_idx in inter_atoms:
        expected = DEFAULT_VALENCE.get(atom_array.element[atom_idx])
        if expected is not None:
            expected += atom_array.charge[atom_idx]  # N+ → 4, O- → 1
        if expected is None or degree[atom_idx] <= expected:
            continue

        elem = atom_array.element[atom_idx]
        if elem == "C":
            atoms_to_remove.extend(_maybe_fix_overvalent_carbon(atom_array, atom_idx))
        elif elem == "N" or elem == "O":
            # N/N+: bonded to one extra heavy atom via inter-residue bond; shed H
            #     to restore valence (e.g. LYS NZ→PLP Schiff base, DLY NZ→PLP).
            # O:  formed a covalent bond (ester/ether) but CCD did not mark its
            #     hydroxyl H as a leaving atom; remove that H here.
            excess = int(degree[atom_idx] - expected)
            if has_nhyd and atom_array.nhyd[atom_idx] > 0:
                # Implicit H — in-place decrement, no index change needed.
                n_to_remove = min(excess, atom_array.nhyd[atom_idx])
                atom_array.nhyd[atom_idx] -= n_to_remove
                logger.warning(
                    f"Removed H from overvalent {atom_array.res_name[atom_idx]}/"
                    f"{atom_array.atom_name[atom_idx]} "
                    f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
                )
            else:
                bonded_h = _find_bonded_hydrogens(atom_array, atom_idx)
                atoms_to_remove.extend([int(h) for h in bonded_h[:excess]])
                if bonded_h[:excess].size:
                    logger.warning(
                        f"Removed H from overvalent {atom_array.res_name[atom_idx]}/"
                        f"{atom_array.atom_name[atom_idx]} "
                        f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
                    )

    if atoms_to_remove:
        keep = np.ones(len(atom_array), dtype=bool)
        keep[atoms_to_remove] = False
        atom_array = atom_array[keep]

    return atom_array


def resolve_leaving_atoms(atom_array: struc.AtomArray) -> tuple[struc.AtomArray, np.ndarray]:
    """Remove leaving atoms and fix valence for atoms involved in inter-residue bonds.

    Returns:
        ``(atom_array, impacted_mask)`` where ``impacted_mask`` marks atoms
        involved in inter-residue bonds, suitable for charge-correction.

    Raises:
        ValueError: If a leaving atom is bonded to atoms other than its parent
            or fellow leaving atoms.
    """
    inter_bonds = _get_inter_residue_bonds(atom_array)
    if len(inter_bonds) == 0:
        return atom_array, np.zeros(len(atom_array), dtype=bool)

    atoms_with_inter_bonds = np.unique(inter_bonds[:, :2])
    atom_array = _remove_ccd_leaving_atoms(atom_array, atoms_with_inter_bonds)
    atom_array = _resolve_overvalent_atoms(atom_array)
    return atom_array, get_inter_residue_atom_mask(atom_array)
