import logging
from typing import Any, ClassVar

import torch
from biotite.structure import AtomArray
import numpy as np

logger = logging.getLogger(__name__)
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.enums import ChainType
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.mirror_transform import MIROR_IMAGE_MAPPING
from atomworks.ml.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_af3_token_representative_masks,
    spread_token_wise,
)
from biotite.structure import AtomArray

class AnnotateTokenRepresentatives(Transform):
    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array","symmetry_resolution"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data['atom_array']

        # 1 - unique identifier for each chain (atom level)
        unique_iids, inverse = np.unique(atom_array.chain_iid, return_inverse=True)
        chain_iid_by_atom = torch.tensor(inverse, dtype=torch.int64)

        # 2 - representative atom indices
        token_representative_masks = get_af3_token_representative_masks(atom_array)
        token_to_repatom_index = torch.where(torch.from_numpy(token_representative_masks))[0]

        data['symmetry_resolution']['chain_iid_by_atom'] = chain_iid_by_atom
        data['symmetry_resolution']['token_to_repatom_index'] = token_to_repatom_index

        return data


class CheckForNaNsInInputs(Transform):
    """
    Look for NaNs in inputs and asserts False if found.

    The coordinate check is skipped for missing_atom_behavior == rollout (nans still used)
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["noise"])

    def forward(self, data: dict) -> dict:
        #fd moved to inference sampler
        # During inference, replace coordinates with true noise
        #if data.get("is_inference", False):
        #    data["coord_atom_lvl_to_be_noised"] = torch.randn_like(
        #        data["coord_atom_lvl_to_be_noised"]
        #    )

        # In rollout mode, unresolved atoms are left as NaN here and filled later by
        # the model; skip the coordinate NaN check in that case.
        if data["missing_atom_behavior"] != "rollout":
            assert not torch.isnan(
                data["coord_atom_lvl_to_be_noised"]
            ).any(), "NaN found in network input"
        assert not torch.isnan(data["noise"]).any(), "NaN found in network noise"

        return data

# A version of SampleEDMNoise that permits custom params
class SampleEDMNoiseCustom(Transform):
    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        BatchStructuresForDiffusionNoising
    ]

    def __init__(
        self,
        sigma_data: float,
        diffusion_batch_size: int,
        pmean: float = -1.2,
        pstd: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_data = sigma_data
        self.pmean = pmean
        self.pstd = pstd
        self.diffusion_batch_size = diffusion_batch_size

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        assert (
            data["coord_atom_lvl_to_be_noised"].shape[0] == self.diffusion_batch_size
        ), "Must batch coordinates to be noised before applying this transform"

    def forward(self, data: dict) -> dict:
        """
        Apply EDM noise sampling to the coordinates that are to be noised.
        """
        t = self.sigma_data * torch.exp(
            torch.normal(
                mean=self.pmean, std=self.pstd, size=(self.diffusion_batch_size,)
            )
        )
        t_tiled = t[:, None, None].tile(
            1, data["coord_atom_lvl_to_be_noised"].shape[1], 3
        )
        noise = torch.normal(mean=0, std=1, size=t_tiled.shape) * t_tiled

        if data["missing_atom_behavior"] == "origin":
            # Unresolved atoms are placed at origin; they must receive no noise.
            unresolved = torch.from_numpy(data["atom_array"].occupancy == 0)  # [n_atoms]
            noise[:, unresolved, :] = 0.0

        data["t"] = t
        data["noise"] = noise
        return data


class PlaceUnresolvedTokenAtomsAtOrigin(Transform):
    """Place all unresolved atoms at the origin (0, 0, 0).

    An alternative to the two-step PlaceUnresolvedTokenAtomsOnRepresentativeAtom +
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence approach that requires no resolved
    neighbours. Both partially-unresolved tokens (e.g., missing side-chain atoms) and
    fully-unresolved tokens are handled uniformly.

    Args:
        annotation_to_update (str): The annotation to update. E.g. ``"coord"`` (ground-truth)
            or ``"coord_to_be_noised"`` (noised coordinates only).
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def __init__(self, annotation_to_update: str = "coord_to_be_noised") -> None:
        self.annotation_to_update = annotation_to_update

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        annotations_to_check = ["occupancy"]
        if self.annotation_to_update != "coord":
            annotations_to_check += [self.annotation_to_update]
        check_atom_array_annotation(data, annotations_to_check)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        unresolved_mask = atom_array.occupancy == 0

        if np.any(unresolved_mask):
            if self.annotation_to_update == "coord":
                atom_array.coord[unresolved_mask] = 0.0
            else:
                atom_array.get_annotation(self.annotation_to_update)[unresolved_mask] = 0.0

        return data


class CropToResolvedOnly(Transform):
    """Remove fully-unresolved tokens from the structure.

    A token is considered unresolved if every one of its atoms has ``occupancy == 0``.
    Such tokens are dropped before the main spatial/contiguous crop step (the ``'trim'``
    ``missing_atom_behavior``).  Partially-unresolved tokens (at least one atom with
    ``occupancy > 0``) are kept.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # A token is resolved if any of its atoms has occupancy > 0
        is_token_resolved = apply_token_wise(
            array=atom_array,
            data=atom_array.occupancy > 0,
            function=np.any,
        )

        # Spread token-level mask back to atom level
        is_atom_resolved = spread_token_wise(atom_array, is_token_resolved)

        crop_info = {
            "type": self.__class__.__name__,
            "requires_crop": not np.all(is_token_resolved),
            "crop_token_idxs": np.where(is_token_resolved)[0],
            "crop_atom_idxs": np.where(is_atom_resolved)[0],
        }

        data["crop_info"] = crop_info
        data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]

        return data
    
    
def flag_noncanonical_polymer_bonds(atom_array: AtomArray, p_atomize_disulfide: float, is_inference: bool) -> AtomArray:
    """
    Flags noncanonical polymer bonds for atomization and downstream processing. Ignores canonical bonds between adjacent
    amino acids or nucleotides.

    Args:
        atom_array (AtomArray): Current `AtomArray` within the Transform pipeline

    Returns:
        AtomArray: The modified `AtomArray` with updated annotations for covalent bonds betweeen polymer peptide units.

    NOTE: If `atomize` annotation is not present in the `AtomArray`, it will be added.
    """
    bonds = atom_array.bonds.as_array()

    # Identify polymer-polymer bonds in the entry
    polymer_polymer_bond_mask = (atom_array.is_polymer[bonds[:, 0]] & atom_array.is_polymer[bonds[:, 1]])

    # Identify interresidue bonds
    interresidue_bond_mask = (
        (atom_array.res_id[bonds[:, 0]] != atom_array.res_id[bonds[:, 1]]) |
        (atom_array.chain_id[bonds[:, 0]] != atom_array.chain_id[bonds[:, 1]])
    )

    # Identify noncanonical peptide bonds
    noncanonical_peptide_bond_mask = ~(
        ((atom_array.atom_name[bonds[:, 0]] == "C") &
         (atom_array.atom_name[bonds[:, 1]] == "N")) |
        ((atom_array.atom_name[bonds[:, 0]] == "N") &
         (atom_array.atom_name[bonds[:, 1]] == "C"))
    )

    # Identify noncanonical nucleic acid bonds
    noncanonical_nucleic_acid_bond_mask = ~(
        ((atom_array.atom_name[bonds[:, 0]] == "O3'") &
         (atom_array.atom_name[bonds[:, 1]] == "P")) |
        ((atom_array.atom_name[bonds[:, 0]] == "P") &
         (atom_array.atom_name[bonds[:, 1]] == "O3'"))
    )

    # Combine masks to identify noncanonical bonds
    noncanonical_polymer_polymer_bond_mask = (
        polymer_polymer_bond_mask &
        interresidue_bond_mask &
        noncanonical_peptide_bond_mask &
        noncanonical_nucleic_acid_bond_mask
    )

    # Identify disulfide bonds based on residue names and atom names, and randomly mark for atomization
    disulfide_bond_mask = (
        ((atom_array.res_name[bonds[:, 0]] == "CYS") &
         (atom_array.atom_name[bonds[:, 0]] == "SG") &
         (atom_array.res_name[bonds[:, 1]] == "CYS") &
         (atom_array.atom_name[bonds[:, 1]] == "SG")) 
    )
    if np.any(disulfide_bond_mask):
        # Randomly decide if we sample disulfides or not
        sample_disulfides = np.random.rand() < p_atomize_disulfide
        # Don't sample disulfides during validation
        if not sample_disulfides or is_inference:
            noncanonical_polymer_polymer_bond_mask = (
                noncanonical_polymer_polymer_bond_mask &
                ~disulfide_bond_mask
            )

    bonds_to_atomize = bonds[noncanonical_polymer_polymer_bond_mask]
    if "atomize" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("atomize", np.zeros(len(atom_array), dtype=bool))

    if len(bonds_to_atomize) == 0:
        return atom_array

    # Collect residue identifiers for all bonded atoms (both ends)
    bonded_atom_indices = np.concatenate([bonds_to_atomize[:, 0], bonds_to_atomize[:, 1]])
    
    bonded_residues = set(zip(
        atom_array.res_id[bonded_atom_indices],
        atom_array.chain_id[bonded_atom_indices],
        atom_array.pn_unit_iid[bonded_atom_indices],
    ))

    # Build one mask covering all bonded residues
    atomize_mask = np.zeros(len(atom_array), dtype=bool)
    for res_id, chain_id, pn_unit_iid in bonded_residues:
        res_mask = (
            (atom_array.res_id == res_id) &
            (atom_array.chain_id == chain_id) &
            (atom_array.pn_unit_iid == pn_unit_iid)
        )
        atomize_mask = atomize_mask | res_mask

    atom_array.atomize = atom_array.atomize | atomize_mask

    # Find and remove anything flagged that has occupancy 0 so we don't later place atoms that may have left as a result of the new bond
    unresolved_flagged_mask = atomize_mask & (atom_array.occupancy == 0)
    if np.any(unresolved_flagged_mask):
        atom_array = atom_array[~unresolved_flagged_mask]

    return atom_array


class FlagNoncanonicalPolymerBondsForAtomization(Transform):
    """
    Flags noncanonical polymer-polymer bonds for atomization and downstream processing. Ignores N-C and P-O3' bonds between adjacent residues.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, "AddGlobalTokenIdAnnotation"]


    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def __init__(self, p_atomize_disulfide: float = 0.05):
        super().__init__()
        self.p_atomize_disulfide = p_atomize_disulfide

    def forward(self, data: dict) -> dict:
        is_inference = data.get("is_inference", False)
        data["atom_array"] = flag_noncanonical_polymer_bonds(data["atom_array"], self.p_atomize_disulfide, is_inference)
        return data


# Build L and D amino acid name sets from the mirror mapping
_L_AA_NAMES = frozenset(k for k, v in MIROR_IMAGE_MAPPING.items() if k != "GLY")
_D_AA_NAMES = frozenset(v for k, v in MIROR_IMAGE_MAPPING.items() if v != "GLY")
# Bidirectional swap mapping: L->D and D->L
_CHIRALITY_SWAP = {k: v for k, v in MIROR_IMAGE_MAPPING.items() if k != "GLY"}
_CHIRALITY_SWAP.update({v: k for k, v in MIROR_IMAGE_MAPPING.items() if k != "GLY"})


def validate_chirality(atom_array: AtomArray, swap: bool = False) -> AtomArray:
    """Zero occupancy on amino acid residues whose chirality is incorrect.

    For each non-glycine L- or D-amino acid residue, computes the triple product
    ``(CB - CA) · ((N - CA) × (C - CA))``. L-amino acids should have a positive
    triple product and D-amino acids a negative one. Residues that violate this
    have all their atoms' occupancy set to 0, or their res_name swapped (L<->D)
    if ``swap`` is True.
    """
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    res_ids = atom_array.res_id
    chain_ids = atom_array.chain_id
    pn_unit_iids = atom_array.pn_unit_iid

    is_l = np.isin(res_names, list(_L_AA_NAMES))
    is_d = np.isin(res_names, list(_D_AA_NAMES))
    is_amino = is_l | is_d

    if not np.any(is_amino):
        return atom_array

    # Build residue keys for amino acid atoms only
    # We need N, CA, C, CB per residue — find them via vectorized masks
    is_N = is_amino & (atom_names == "N")
    is_CA = is_amino & (atom_names == "CA")
    is_C = is_amino & (atom_names == "C")
    is_CB = is_amino & (atom_names == "CB")

    # For each atom type, map (chain_id, res_id, pn_unit_iid) -> atom index
    def _build_lookup(mask):
        indices = np.where(mask)[0]
        return {
            (chain_ids[i], res_ids[i], pn_unit_iids[i]): i for i in indices
        }

    lookup_N = _build_lookup(is_N)
    lookup_CA = _build_lookup(is_CA)
    lookup_C = _build_lookup(is_C)
    lookup_CB = _build_lookup(is_CB)

    # Intersect: residues that have all four atoms
    common_keys = lookup_N.keys() & lookup_CA.keys() & lookup_C.keys() & lookup_CB.keys()

    if not common_keys:
        return atom_array

    # Gather coordinate indices in a single array for vectorized math
    common_keys_list = list(common_keys)
    idx_N = np.array([lookup_N[k] for k in common_keys_list])
    idx_CA = np.array([lookup_CA[k] for k in common_keys_list])
    idx_C = np.array([lookup_C[k] for k in common_keys_list])
    idx_CB = np.array([lookup_CB[k] for k in common_keys_list])

    coords = atom_array.coord
    v_CB = coords[idx_CB] - coords[idx_CA]  # (n_res, 3)
    v_N = coords[idx_N] - coords[idx_CA]    # (n_res, 3)
    v_C = coords[idx_C] - coords[idx_CA]    # (n_res, 3)

    # Triple product: v_CB · (v_N × v_C)
    triple = np.sum(v_CB * np.cross(v_N, v_C), axis=1)  # (n_res,)

    # Expected sign: +1 for L, -1 for D
    is_l_residue = np.array([res_names[idx_CA[i]] in _L_AA_NAMES for i in range(len(common_keys_list))])
    expected_sign = np.where(is_l_residue, 1.0, -1.0)

    wrong_chirality = (triple * expected_sign) < 0

    if not np.any(wrong_chirality):
        return atom_array

    # Build mask for atoms belonging to wrong-chirality residues
    bad_keys = {common_keys_list[i] for i in np.where(wrong_chirality)[0]}
    bad_mask = np.zeros(len(atom_array), dtype=bool)
    for chain, res, pn_unit in bad_keys:
        bad_mask |= (chain_ids == chain) & (res_ids == res) & (pn_unit_iids == pn_unit)

    if swap:
        new_res_names = res_names.copy()
        for i in range(len(new_res_names)):
            if bad_mask[i] and new_res_names[i] in _CHIRALITY_SWAP:
                new_res_names[i] = _CHIRALITY_SWAP[new_res_names[i]]
        atom_array.res_name = new_res_names
        for i in np.where(wrong_chirality)[0]:
            key = common_keys_list[i]
            rname = res_names[idx_CA[i]]
            logger.warning(
                "Swapped chirality for residue %s -> %s (chain=%s, res_id=%s)",
                rname, _CHIRALITY_SWAP[rname], key[0], key[1],
            )
    else:
        atom_array.occupancy[bad_mask] = 0.0
        for i in np.where(wrong_chirality)[0]:
            key = common_keys_list[i]
            rname = res_names[idx_CA[i]]
            logger.warning(
                "Set occupancy to 0 for residue %s with incorrect chirality (chain=%s, res_id=%s)",
                rname, key[0], key[1],
            )

    return atom_array


class ValidateProteinChirality(Transform):
    """Set occupancy to 0 for amino acid residues with incorrect chirality.

    If ``swap_pdb_codes`` is provided and the current example's PDB code (from
    ``data["example_id"]``) is in the whitelist, the residue name is swapped
    between L and D forms instead of zeroing occupancy.

    Args:
        swap_pdb_codes: PDB codes for which wrong-chirality residues should have
            their res_name swapped (L<->D) instead of being unoccupied.
    """

    def __init__(self, swap_pdb_codes: list[str] | None = None):
        super().__init__()
        self.swap_pdb_codes = set(swap_pdb_codes) if swap_pdb_codes else None

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict) -> dict:
        swap = False
        if self.swap_pdb_codes is not None:
            example_id = data.get("example_id", "")
            swap = any(pdb_code in example_id for pdb_code in self.swap_pdb_codes)
        data["atom_array"] = validate_chirality(data["atom_array"], swap=swap)
        return data


class AnnotateMSAStatsByChain(Transform):
    """Aggregate MSA loading stats for reporting
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["LoadPolymerMSAs"]
    MIN_RESIDUES: ClassVar[int] = 50

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["chain_info", "atom_array"])

    def forward(self, data: dict) -> dict:
        chain_info = data.get("chain_info", {})
        polymer_msas = data.get("polymer_msas_by_chain_id", {})
        present_chain_ids = set(np.unique(data["atom_array"].chain_id).tolist())

        n_with = 0
        n_without = 0
        for chain_id in present_chain_ids:
            info = chain_info.get(chain_id)
            if info is None:
                continue
            try:
                is_protein = ChainType.as_enum(info["chain_type"]).is_protein()
            except (KeyError, ValueError):
                continue
            if not is_protein:
                continue
            if len(info.get("processed_entity_canonical_sequence", "")) < self.MIN_RESIDUES:
                continue
            if chain_id in polymer_msas:
                n_with += 1
            else:
                n_without += 1

        data["n_protein_chains_gt50_with_msa"] = n_with
        data["n_protein_chains_gt50_without_msa"] = n_without
        return data
