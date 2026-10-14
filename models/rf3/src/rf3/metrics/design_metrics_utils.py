"""Design metrics utilities, ported from RFdiffusion3 (original authors: see README.md).

Notes:
- No SS-conditioning adherence (rf3 has no SS conditioning).
- ``get_hotspot_contacts`` derives the diffused region from the C_CRD mask
  (``mask_coordinate_1_atom``); falls back to all atoms if absent.
"""

import itertools

import numpy as np
from biotite.structure import AtomArrayStack
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.conditions import C_DIS
from atomworks.ml.utils.token import apply_and_spread_token_wise
from biotite.structure import CellList, annotate_sse, gyration_radius

from rf3.conditions import HBondEnum, RasaEnum
from rf3.data.design_tasks.design_task_modifiers import calculate_hbonds

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
"""Protein backbone atom names. Used to derive backbone vs sidechain masks
inline in metrics, avoiding the need for an explicit ``is_backbone`` /
``is_sidechain`` feature in the network input."""

STANDARD_AA = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)
"""20 canonical amino acids (3-letter codes) — used for sidechain topology matching."""

STANDARD_CACA_DIST = 3.8
"""Standard CA-CA distance (Å) between consecutive residues in a polymer."""


def get_motif_features(atom_array, include=["int_coord", "dist"]):
    """
    Equivalent to get_motif_features in conditioning_base.py but in the new conditioning world.

    Args:
        only_get: list[str] (names of the features to get).
            The definition of "is_motif_atom" in the old conditioning world is
                is_motif_atom = is_fixed_coord | is_fixed_seq | is_unindexed
            In the new conditioning world, the default is
                is_motif_atom = is_fixed_coord | is_fixed_dist | is_fixed_seq | is_unindexed
    """
    # is_fixed_intcoord = C_INTCRD.mask(atom_array) # TODO: commentout once C_INTCRD is in
    is_fixed_dist = C_DIS.mask(atom_array).as_dense_array(False).any(axis=0)
    is_motif_atom = np.zeros(atom_array.array_length(), dtype=bool)
    
    # Motif atom if has any conditioning
    if "int_coord" in include:
        is_motif_atom |= is_fixed_dist
    if "dist" in include:
        is_motif_atom |= is_fixed_dist
    is_motif_token = apply_and_spread_token_wise(
        atom_array, is_motif_atom, function=lambda x: np.any(x)
    )  # Has any atoms with conditioning

    return {"is_motif_atom": is_motif_atom, "is_motif_token": is_motif_token}


def get_ss_metrics_and_rg(atom_array):
    """Compute secondary structure composition and radius of gyration.

    Uses biotite's P-SEA SSE assignment. Returns helix/sheet/loop fractions,
    number of disjoint SS elements, and radius of gyration.
    """
    sse_array = annotate_sse(atom_array)
    sse_array_prot_only = sse_array[sse_array != ""]

    if len(sse_array_prot_only) == 0:
        # No protein residues — return zeros / NaN as appropriate
        return {
            "non_loop_fraction": 0.0,
            "loop_fraction": 0.0,
            "helix_fraction": 0.0,
            "sheet_fraction": 0.0,
            "num_ss_elements": 0,
            "radius_of_gyration": float(gyration_radius(atom_array))
            if len(atom_array) > 0
            else 0.0,
        }

    helix_pct = float(np.mean(sse_array_prot_only == "a"))
    sheet_pct = float(np.mean(sse_array_prot_only == "b"))
    loop_pct = float(np.mean(sse_array_prot_only == "c"))

    num_ss = 0
    for k, _ in itertools.groupby(sse_array):
        if k not in ("", "c"):
            num_ss += 1

    return {
        "non_loop_fraction": helix_pct + sheet_pct,
        "loop_fraction": loop_pct,
        "helix_fraction": helix_pct,
        "sheet_fraction": sheet_pct,
        "num_ss_elements": num_ss,
        "radius_of_gyration": float(gyration_radius(atom_array)),
    }


def _get_diffused_mask(atom_array):
    """Return a boolean mask of "diffused" atoms (i.e., NOT motif-fixed).

    Prefers the atomworks C_CRD annotation ``mask_coordinate_1_atom``; falls
    back to "all atoms" when the annotation is absent.
    """
    if "mask_coordinate_1_atom" in atom_array.get_annotation_categories():
        motif_mask = atom_array.get_annotation("mask_coordinate_1_atom").astype(bool)
        return ~motif_mask
    return np.ones(len(atom_array), dtype=bool)


def get_hotspot_contacts(atom_array, hotspot_mask, distance_cutoff=4.5):
    """Fraction of hotspot atoms with at least one inter-chain contact to a diffused atom.

    "Diffused" = ``~mask_coordinate_1_atom`` if present, else all atoms.
    """
    cell_list = CellList(atom_array, cell_size=distance_cutoff)
    hotspot_array = atom_array[hotspot_mask]

    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        hotspot_array.coord, cell_list, len(atom_array), distance_cutoff
    )  # (n_hotspots, n_atoms)

    interchain_mask = (
        hotspot_array.pn_unit_iid[:, None] != atom_array.pn_unit_iid[None]
    )
    interchain_contacts_mask = full_contacting_atom_mask & interchain_mask

    diffused_mask = _get_diffused_mask(atom_array)
    diffused_interchain_contacts_mask = interchain_contacts_mask[:, diffused_mask]

    contacted_hotspots_mask = np.any(diffused_interchain_contacts_mask, axis=1)
    return float(contacted_hotspots_mask.mean())


def get_rasa_metrics(
    atom_array,
    rasa_condition: np.ndarray,
    probe_radius: float = 1.4,
    atom_radii: str = "ProtOr",
    point_number: int = 100,
    rasa_max: float = 0.2,
) -> dict[str, float]:
    """Fraction of RASA-conditioned atoms whose predicted RASA bin matches the condition.

    ``rasa_condition`` is the per-atom ``RasaEnum`` array (C_RAS annotation). SASA is
    computed over the conditioned atoms only, the same ``atom_filter`` that
    ``add_rasa_condition`` / ``apply_rasa_selection`` used to make the labels.
    """
    conditioned = rasa_condition != RasaEnum.UNSPECIFIED
    rasa = calculate_atomwise_rasa(
        atom_array, probe_radius, atom_radii, point_number, atom_filter=conditioned
    )
    # digitize bins 1/2/3 equal RasaEnum BURIED/INTERMEDIATE/EXPOSED; NaN -> EXPOSED as in training
    predicted = np.digitize(rasa, np.linspace(0, rasa_max, 3), right=False)
    out = {}
    for enum in (RasaEnum.BURIED, RasaEnum.INTERMEDIATE, RasaEnum.EXPOSED):
        idx = rasa_condition == enum
        out[f"{enum.name.lower()}_conditioning_adherence"] = (
            float((predicted[idx] == enum).mean()) if idx.any() else np.nan
        )
    return out


"""Hydrogen-bond metrics, ported from RFdiffusion3 (original authors: see README.md).

Compares H-bond donor / acceptor predictions atom-by-atom against the GT
``hbond_gt`` annotation set by the rf3 design-task pipeline. Both GT and
predicted annotations are computed with the same hbplus-based code path
(``rf3.data.design_tasks.design_task_modifiers.calculate_hbonds``) 

- If hbplus is unavailable (``HBPLUS_PATH`` unset) or fails for any reason,
  the metric returns an empty dict for the whole batch rather than raising.
- The metric exposes an ``enabled`` switch to disable it entirely from config
  in CI / hbplus-less environments.
"""

def _has_hbond_gt(atom_array) -> bool:
    return "hbond_gt" in atom_array.get_annotation_categories()


def _is_donor(hbond_enum_array: np.ndarray) -> np.ndarray:
    return (hbond_enum_array == HBondEnum.DONOR) | (hbond_enum_array == HBondEnum.BOTH)


def _is_acceptor(hbond_enum_array: np.ndarray) -> np.ndarray:
    return (hbond_enum_array == HBondEnum.ACCEPTOR) | (
        hbond_enum_array == HBondEnum.BOTH
    )


def _calculate_hbond_stats(
    gt_stack: AtomArrayStack,
    pred_stack: AtomArrayStack,
    cutoff_HA_dist: float,
    cutoff_DA_distance: float,
) -> tuple[float, float, float, int]:
    """Return (donor_pct, acceptor_pct, n_hbonds, n_valid_samples)."""
    total_donor_pct = 0.0
    total_acceptor_pct = 0.0
    total_n_hbonds = 0
    n_valid = 0

    for gt_arr, pred_arr in zip(gt_stack, pred_stack):
        if not _has_hbond_gt(gt_arr):
            continue

        gt_hbond = np.asarray(gt_arr.get_annotation("hbond_gt"), dtype=int)
        gt_donor = _is_donor(gt_hbond)
        gt_acceptor = _is_acceptor(gt_hbond)
        n_gt_donor = int(gt_donor.sum())
        n_gt_acceptor = int(gt_acceptor.sum())
        if n_gt_donor == 0 and n_gt_acceptor == 0:
            continue

        # Run hbplus on the predicted structure.
        # `calculate_hbonds` expects a 1D bool ndarray (atom-level), not the dict
        # returned by `get_motif_features` — pull out the atom-level mask explicitly.
        motif_mask = get_motif_features(pred_arr)["is_motif_atom"]

        annotated_pred, _hbonds, n_motif_hbonds, failed = calculate_hbonds(
            pred_arr,
            motif_mask=motif_mask,
            cutoff_HA_dist=cutoff_HA_dist,
            cutoff_DA_distance=cutoff_DA_distance,
            raise_on_failure=False,
        )
        if failed or annotated_pred is None:
            continue

        pred_hbond = np.asarray(
            annotated_pred.get_annotation("hbond_gt"), dtype=int
        )
        if pred_hbond.shape != gt_hbond.shape:
            # Atom-array length mismatch (shouldn't happen since they share the same
            # template) — skip rather than guess at correspondence
            continue

        pred_donor = _is_donor(pred_hbond)
        pred_acceptor = _is_acceptor(pred_hbond)

        donor_pct = (
            float((gt_donor & pred_donor).sum()) / n_gt_donor
            if n_gt_donor > 0
            else 1.0
        )
        acceptor_pct = (
            float((gt_acceptor & pred_acceptor).sum()) / n_gt_acceptor
            if n_gt_acceptor > 0
            else 1.0
        )

        total_donor_pct += donor_pct
        total_acceptor_pct += acceptor_pct
        total_n_hbonds += int(n_motif_hbonds or 0)
        n_valid += 1

    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        total_donor_pct / n_valid,
        total_acceptor_pct / n_valid,
        total_n_hbonds / n_valid,
        n_valid,
    )

