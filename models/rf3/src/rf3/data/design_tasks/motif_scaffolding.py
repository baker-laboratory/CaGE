from collections import defaultdict
from collections.abc import Callable
from typing import Literal, ClassVar
from functools import partial

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.query import QueryExpression
from atomworks.io.utils.scatter import (
    apply_segment_wise,
    spread_segment_wise,
)
from atomworks.ml.utils.token import apply_and_spread_token_wise
from atomworks.ml.conditions import (
    C_CRD,
    C_DIS,
)
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.ml.transforms import mask_generator as mg
from atomworks.ml.transforms.crop import resize_crop_info_if_too_many_atoms
from atomworks.ml.transforms.design_task import DesignTaskABC
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.token import get_token_count, get_token_starts
from biotite.structure import AtomArray
from foundry.common import true_with_prob
from foundry.utils.ddp import RankedLogger
from scipy.spatial import ConvexHull
from scipy.special import softmax

from rf3.data.design_tasks.base import RFD3DesignTask
from rf3.data.design_tasks.distance_condition import sample_distance_condition

# ---------------------------------------------------------------------------
# Default potential functions (module-level for picklability under DDP)
# ---------------------------------------------------------------------------
def _default_tip_atom_potential(x: np.ndarray, scaling_factor: float = 8.0) -> np.ndarray:
    """Default tip atom sampling potential: exponential decay."""
    return np.exp(-x / scaling_factor)

def _default_lig_lig_potential(distances: np.ndarray) -> np.ndarray:
    """rfd3 default: ``exp(-d / 8)``. How close a ligand should be from already-sampled ligands."""
    return np.exp(-distances / 8.0)


def _default_tip_lig_potential(distances: np.ndarray) -> np.ndarray:
    """rfd3 default: ``exp(-d / 2)``. How close tip atoms should be to the sampled ligand."""
    return np.exp(-distances / 2.0)


ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class TipAtomActiveSiteTask(RFD3DesignTask):   
    """
    Designed to sample tip atoms around a protein pocket when no ligand is present.
    This is done by calculating the SASA of all atoms, and calculating the distance they lie from
    the convex hull of the protein chain. Atoms above a SASA threshold are sampled at rates
    proportional to their distance from the convex hull. In effect this samples tip atoms around protein
    pocket.

    Args:
        tip_atom_rasa_threshold: rASA threshold to consider tip atoms for sampling
        min_tip_atoms: minimum number of tip atoms to sample
        max_tip_atoms: maximum number of tip atoms to sample
        p_seq_conditioned: probability of providing sequence conditioning to atoms with tip-atoms sampled
            NOTE: In proteina, providing tip atoms inherently gives away sequence information because of Atom37!
        min_protein_length: minimum length of protein chain to sample
        p_unindex_all: probability of unindexing all atoms in the protein chain
        p_unindex_any: probability of unindexing individual sampled atom. Will only do this is p_index_all was sampled to False
        p_merge_unindexed_groups: probability of merging unindexed groups
        p_split_unindexed_groups: probability of splitting unindexed groups
        p_increase_group_id_gap: probability of increasing the gap between unindexed group ids
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }

    def __init__(
        self,
        tip_atom_rasa_threshold: float = 0.2,
        min_tip_atoms: int = 2,
        max_tip_atoms: int = 15,
        min_protein_length: int = 50,  # Avoid peptides
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tip_atom_rasa_threshold = tip_atom_rasa_threshold
        self.min_tip_atoms = min_tip_atoms
        self.max_tip_atoms = max_tip_atoms
        self.min_protein_length = min_protein_length

        self.sampled_chain = None
        self.sampled_tip_atom_ind = None

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_tip_atom", "is_protein", "token_min_occupancy")

        data["atom_array"] = atom_array
        return data

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        eligible_mask = atom_array.mask("is_protein & (token_min_occupancy > 0)")
        if "at_crop_boundary" in atom_array.get_annotation_categories():
            eligible_mask &= ~atom_array.mask("at_crop_boundary")

        # randomly choose one of the protein chains that is long enough to be our sampled protein chain
        possible_chains = np.unique(atom_array[eligible_mask].chain_id)
        chain_lengths = {
            str(chain_id): len(np.unique(atom_array[atom_array.chain_id == chain_id].res_id))
            for chain_id in possible_chains
        }
        eligible_chains = [chain_id for chain_id, length in chain_lengths.items() if length >= self.min_protein_length]

        if len(eligible_chains) == 0:
            return False

        sampled_chain = np.random.choice(eligible_chains)
        self.sampled_chain = sampled_chain

        # calculate rasa for all atoms
        rasa = calculate_atomwise_rasa(
            atom_array,
            probe_radius=1.4,
            atom_radii="ProtOr",
            point_number=100,
            atom_filter=eligible_mask & (atom_array.chain_id == self.sampled_chain) & atom_array.is_tip_atom,
        )

        atom_array.set_annotation("tip_atom_rasa", rasa)

        if (rasa > self.tip_atom_rasa_threshold).sum() < self.min_tip_atoms:
            return False

        return True

    def sample_protein_pocket(self, atom_array: AtomArray, eligible_tip_atom_mask: np.ndarray) -> int:
        """
        Calculates convex hull of the protein chain (self.sampled_chain) and samples a tip atom based on distance from the convex hull
        boundary and the rasa (stored in tip_atom_rasa annotation). Returns the index of the sampled tip atom.
        """
        # Get chain coordinates, filtering out NaN/Inf values and unresolved atoms
        chain_mask = atom_array.chain_id == self.sampled_chain
        # Also filter by occupancy if available to exclude unresolved atoms
        if "token_min_occupancy" in atom_array.get_annotation_categories():
            chain_mask = chain_mask & (atom_array.token_min_occupancy > 0)

        chain_coords = atom_array[chain_mask].coord
        # Use np.isfinite to catch both NaN and Inf values
        valid_coord_mask = np.isfinite(chain_coords).all(axis=1)

        if valid_coord_mask.sum() < 4:
            raise ValueError(f"Not enough valid coordinates for convex hull (need >= 4, got {valid_coord_mask.sum()})")

        # Make explicit copy to avoid any view issues
        valid_chain_coords = chain_coords[valid_coord_mask].copy()
        convex_hull = ConvexHull(valid_chain_coords)

        eligible_tip_atom_inds = np.where(eligible_tip_atom_mask)[0]
        tip_atom_coords = atom_array.coord[eligible_tip_atom_inds]

        # Filter out tip atoms with NaN/Inf coordinates
        valid_tip_mask = np.isfinite(tip_atom_coords).all(axis=1)
        if valid_tip_mask.sum() == 0:
            raise ValueError("No valid tip atom coordinates (all contain NaN/Inf)")

        eligible_tip_atom_inds = eligible_tip_atom_inds[valid_tip_mask]
        tip_atom_coords = tip_atom_coords[valid_tip_mask].copy()

        distances_to_facets = (
            tip_atom_coords @ convex_hull.equations[:, :-1].T + convex_hull.equations[:, -1]
        )  # (n_tip_atoms, n_facets)
        distances_to_hull_boundary = np.min(distances_to_facets, axis=1)

        probabilities = softmax(distances_to_hull_boundary)
        sampled_tip_atom_ind = np.random.choice(eligible_tip_atom_inds, p=probabilities)
        return sampled_tip_atom_ind

    def sample_mask(self, atom_array: AtomArray) -> np.ndarray:
        is_eligible_query = f"is_tip_atom & (chain_id == '{self.sampled_chain}') & (token_min_occupancy > 0) & (tip_atom_rasa > {self.tip_atom_rasa_threshold})"
        if "at_crop_boundary" in atom_array.get_annotation_categories():
            is_eligible_query += " & (at_crop_boundary == False)"

        eligible_tip_atom_mask = atom_array.mask(is_eligible_query)

        sampled_tip_atom_ind = self.sample_protein_pocket(atom_array, eligible_tip_atom_mask)
        sampled_tip_atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
        sampled_tip_atom_mask[sampled_tip_atom_ind] = True

        target_num_tip_atoms = np.random.randint(
            self.min_tip_atoms, min(self.max_tip_atoms, eligible_tip_atom_mask.sum()) + 1
        )

        fn_sample_seed = mg.SampleWithPotential(
            is_eligible=is_eligible_query,
            avoid_same=("atom_id"),  # Effectively no token collision avoidance
            potential=partial(_default_tip_atom_potential, scaling_factor=2.0),
        )

        # remember with seed grow merge, start with the sampled tip atom already as the first all_masks! Then it will sample from there
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            all_masks=[
                sampled_tip_atom_mask
            ],  # start with sampled tip atom as the first mask. Potential will grow around it
            fn_sample_seed=fn_sample_seed,
            fn_check_budget=mg.CheckAtomBudget(
                n_min_atoms=target_num_tip_atoms,
                n_max_atoms=target_num_tip_atoms,
            ),
        )
        return total_mask, all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_tip_atom", "tip_atom_rasa")
        assert self.sampled_chain is not None, "sampled_chain was not set"

        total_mask, all_masks = self.sample_mask(atom_array)

        # set coordinate conditions for sampled atoms
        C_CRD.set_mask(atom_array, total_mask)

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "total_mask": total_mask,
            "target_num_tip_atoms": total_mask.sum(),
        }

        data["atom_array"] = atom_array
        data = self.apply_design_task_modifiers(data)

        return data

class TipAtomPartialLigandTask(RFD3DesignTask):
    """Sample partial ligand atoms + tip-atom residues clustered around them."""

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }

    def __init__(
        self,
        *,
        n_max_ligands: int = 3,
        ligand_atom_max_fraction: float = 0.8,
        ligand_atom_min_fraction: float = 0.5,
        n_hops_expected_ligand: int = 5,
        n_max_residues: int = 14,
        n_min_residues: int = 2,
        n_hops_expected_tip_atom: int = 3,
        lig_lig_potential: Callable[[np.ndarray], np.ndarray] = _default_lig_lig_potential,
        tip_lig_potential: Callable[[np.ndarray], np.ndarray] = _default_tip_lig_potential,
        p_seq_conditioned: float = 0.5,
        p_knockout_seq: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_max_ligands = n_max_ligands
        self.ligand_atom_max_fraction = ligand_atom_max_fraction
        self.ligand_atom_min_fraction = ligand_atom_min_fraction
        self.n_hops_expected_ligand = n_hops_expected_ligand
        self.lig_lig_potential = lig_lig_potential
        self.lig_lig_potential_cutoff: float | None = None

        self.n_max_residues = n_max_residues
        self.n_min_residues = n_min_residues
        self.n_hops_expected_tip_atom = n_hops_expected_tip_atom
        self.tip_lig_potential = tip_lig_potential

        self.p_seq_conditioned = p_seq_conditioned
        self.p_knockout_seq = p_knockout_seq

        # Eligibility query strings.
        # NOTE: rfd3 source uses an extra ``is_pn_unit_start`` filter for ligand
        # counting, but that annotation is not in the rf3 atomworks fork's registry.
        # Counting is done via ``pn_unit_iid`` uniqueness instead
        self.lig_str = "is_ligand & (res_min_occupancy > 0)"
        self.res_str = "is_protein & is_standard_aa & (res_min_occupancy > 0)"
        self.res_start_str = self.res_str + " & res_has_tip_atom & is_res_start"

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(
            atom_array,
            "is_protein",
            "is_standard_aa",
            "is_res_start",
            "res_min_occupancy",
            "is_ligand",
            "res_has_tip_atom",
        )
        return data

    @staticmethod
    def _count_unique_lig_pn_units(atom_array: AtomArray, lig_mask: np.ndarray) -> int:
        """Count distinct ligand pn_units. atomworks' annotator registry does not
        provide ``is_pn_unit_start`` so we derive the count from ``pn_unit_iid``.
        """
        if not lig_mask.any():
            return 0
        return int(len(np.unique(atom_array.pn_unit_iid[lig_mask])))

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        lig_eligible_mask = atom_array.mask(self.lig_str)
        n_eligible_ligands = self._count_unique_lig_pn_units(
            atom_array, lig_eligible_mask
        )
        n_eligible_residues = int(np.sum(atom_array.mask(self.res_start_str)))
        return n_eligible_residues >= 1 and n_eligible_ligands >= 1

    def sample_mask(
        self,
        atom_array: AtomArray,
        cum_total_mask: np.ndarray | None = None,
        n_ligands_to_sample: int | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if cum_total_mask is None:
            cum_total_mask = np.zeros(atom_array.array_length(), dtype=bool)

        # ============== Ligand sampling ==============
        lig_eligible_mask = atom_array.mask(self.lig_str) & ~cum_total_mask
        n_eligible_ligands = self._count_unique_lig_pn_units(
            atom_array, lig_eligible_mask
        )
        if n_eligible_ligands == 0:
            ranked_logger.info("No more eligible ligands to sample")
            return cum_total_mask, []
        if n_ligands_to_sample is None:
            n_max_ligands = min(n_eligible_ligands, self.n_max_ligands)
            n_ligands_to_sample = int(np.random.randint(1, n_max_ligands + 1))

        _same_ligand = ("pn_unit_iid",)
        fn_sample_lig_seed = mg.SampleWithPotential(
            potential=self.lig_lig_potential,
            is_eligible=lig_eligible_mask,
            avoid_same=_same_ligand,
        )
        fn_grow_lig = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected_ligand,
            require_same_annotation=_same_ligand,
            atom_array=atom_array,
        )
        fn_lig_budget = mg.CheckResidueBudget(
            n_min_residues=n_ligands_to_sample,
            n_max_residues=n_ligands_to_sample,
            reduce=np.any,
        )

        # all_masks=[] emulates rfd3 reset_sampling_history=True:
        # The new ligand seed is sampled fresh, not biased toward
        # previously-accumulated tip-atom/ligand atoms in cum_total_mask.
        lig_mask, lig_all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            all_masks=[],
            fn_sample_seed=fn_sample_lig_seed,
            fn_grow_seed=fn_grow_lig,
            fn_check_budget=fn_lig_budget,
        )
        lig_mask = lig_mask & ~cum_total_mask
        if lig_mask.sum() == 0:
            ranked_logger.info("No more eligible ligands to sample")
            return cum_total_mask, []

        # ============== Residue budget ==============
        # Match rfd3: scale residue budget by the ACTUAL number of new ligands
        n_actual_new_ligands = int(len(np.unique(atom_array.pn_unit_iid[lig_mask])))
        n_eligible_res = int(
            np.sum(atom_array.mask(self.res_start_str) & ~cum_total_mask)
        )
        n_max_res = min(n_eligible_res, n_actual_new_ligands * self.n_max_residues)
        n_min_res = min(n_eligible_res, n_actual_new_ligands * self.n_min_residues)
        n_res_to_sample = int(
            np.random.randint(n_min_res, max(n_min_res, n_max_res) + 1)
        )

        # ============== Tip-atom residue sampling ==============
        _same_residue = ("chain_iid", "res_name", "res_id", "transformation_id")
        res_eligible_mask = (
            atom_array.mask(self.res_str + " & is_tip_atom") & ~cum_total_mask
        )
        fn_sample_res_seed = mg.SampleTipAtomWithPotential(
            potential=self.tip_lig_potential,
            is_eligible=res_eligible_mask,
            avoid_same=_same_residue,
        )
        fn_grow_res = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected_tip_atom,
            require_same_annotation=_same_residue,
            atom_array=atom_array,
        )
        fn_res_budget = mg.CheckResidueBudget(
            n_min_residues=n_res_to_sample,
            n_max_residues=n_res_to_sample,
            reduce=np.any,
        )

        res_mask, res_all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_res_seed,
            fn_grow_seed=fn_grow_res,
            fn_check_budget=fn_res_budget,
        )
        total_mask = lig_mask | res_mask
        return total_mask, lig_all_masks + res_all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        total_mask, all_masks = self.sample_mask(atom_array)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)


        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "all_masks": all_masks,
        }
        data = self.apply_design_task_modifiers(data)
        return data


class TipAtomTask(RFD3DesignTask):
    """Sample tip-atom residues on a single protein chain (enzyme active-site style).

    The rfd3 source samples a seed (tip-atom or whole-residue uniformly), grows
    along the bond graph, and budgets by residue count. A sequence-conditioning
    sub-step optionally exposes residue identities (with per-residue knockout).
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }

    def __init__(
        self,
        n_min_residues: int = 2,
        n_max_residues: int = 25,
        p_tip_atom: float = 0.8,
        n_hops_expected: int = 3,
        p_seq_conditioned: float = 0.5,
        p_knockout_seq: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_min_residues = n_min_residues
        self.n_max_residues = n_max_residues
        self.p_tip_atom = p_tip_atom
        self.n_hops_expected = n_hops_expected
        self.p_seq_conditioned = p_seq_conditioned
        self.p_knockout_seq = p_knockout_seq

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(
            atom_array,
            "is_protein",
            "is_standard_aa",
            "is_res_start",
            "res_min_occupancy",
            "is_tip_atom",
            "res_has_tip_atom",
        )
        n_eligible_residues = np.sum(
            atom_array.mask(
                "is_standard_aa & is_protein & is_res_start & (res_min_occupancy > 0)"
            )
        )
        return n_eligible_residues >= self.n_min_residues

    def sample_mask(
        self, atom_array: AtomArray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n_eligible_residues = int(
            np.sum(
                atom_array.mask(
                    "is_standard_aa & is_protein & is_res_start & (res_min_occupancy > 0)"
                )
            )
        )

        _same_residue = ("chain_iid", "res_name", "res_id", "transformation_id")

        def fn_sample_seed(atom_array, total_mask, all_masks):
            if true_with_prob(self.p_tip_atom):
                return mg.SampleTipAtomUniformly(
                    is_eligible="is_tip_atom & (res_min_occupancy > 0)",
                    avoid_same=_same_residue,
                )(atom_array, total_mask, all_masks)
            return mg.SampleUniformly(
                is_eligible="is_standard_aa & is_protein & (res_min_occupancy > 0)",
                avoid_same=_same_residue,
            )(atom_array, total_mask, all_masks)

        fn_grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected,
            require_same_annotation=_same_residue,
            atom_array=atom_array,
        )

        n_max_residues = min(n_eligible_residues, self.n_max_residues)
        target_n_residues = int(np.random.randint(self.n_min_residues, n_max_residues + 1))
        fn_check_budget = mg.CheckResidueBudget(
            n_min_residues=target_n_residues,
            n_max_residues=target_n_residues,
            reduce=np.any,
        )

        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )
        return total_mask, all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        total_mask, all_masks = self.sample_mask(atom_array)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "all_masks": all_masks,
        }
        data = self.apply_design_task_modifiers(data)
        return data

class MotifScaffoldTask(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }
    def __init__(
        self,
        min_tokens_per_motif: int = 1,
        max_tokens_per_motif: int = 50,
        min_coverage_fraction: float = 0.05,
        max_coverage_fraction: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_tokens_per_motif = min_tokens_per_motif
        self.max_tokens_per_motif = max_tokens_per_motif
        self.min_coverage_fraction = min_coverage_fraction
        self.max_coverage_fraction = max_coverage_fraction

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_token_start", "token_min_occupancy", "is_polymer", "token_id", "is_protein")
        return data

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        polymer_chain_iids = np.unique(atom_array.chain_iid[atom_array.is_polymer])
        for chain_iid in polymer_chain_iids:
            chain_length = atom_array.mask(f"is_token_start & (chain_iid == '{chain_iid}')").sum()
            if chain_length >= self.min_tokens_per_motif:
                return True

        return False

    def sample_mask(self, atom_array: AtomArray) -> tuple[np.ndarray, list[np.ndarray]]:
        # Sample coverage fraction per-example (not per-instance) for reproducibility across restarts
        self.coverage_fraction_goal = np.random.uniform(self.min_coverage_fraction, self.max_coverage_fraction)

        fn_sample_seed = mg.SampleUniformly(is_eligible="is_polymer & (token_min_occupancy > 0) & ~has_nan_coord()")
        fn_grow_seed = mg.GrowToTokenIsland(
            island_min_size=self.min_tokens_per_motif,
            island_max_size=self.max_tokens_per_motif,
        )

        total_eligible_tokens = atom_array.mask("is_polymer & is_token_start").sum()
        min_total_tokens = int((self.coverage_fraction_goal - 0.05) * total_eligible_tokens)
        max_total_tokens = int((self.coverage_fraction_goal + 0.05) * total_eligible_tokens)
        fn_check_budget = mg.CheckTokenBudget(
            n_min_tokens=min_total_tokens,
            n_max_tokens=max_total_tokens,
        )
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )

        return total_mask, all_masks

    def sample_unindexed_mask(
        self,
        all_masks: list[np.ndarray],
        token_starts: np.ndarray,
    ) -> np.ndarray:
        """Samples a mask of tokens to be unindexed, ensuring that only full coordinate-conditioned islands are unindexed.

        NOTE: This function assumes that the coordinate-conditioned islands do not contain partial tokens, which is always
        true for the MotifScaffoldTask.

        Args:
            all_masks: List of boolean masks for each coordinate-conditioned island (shape of each mask is (n_atoms,))
            token_starts: Boolean array indicating the start of each token (shape (n_atoms,))

        Returns:
            unindexed_mask: Boolean array of shape (n_atoms,) where True indicates the token is unindexed
        """

        unindexed_mask = np.zeros_like(all_masks[0], dtype=bool)
        unindexed_mask_indices = []
        target_num_islands_to_unindex = np.random.randint(1, len(all_masks) + 1)

        for _ in range(target_num_islands_to_unindex):
            valid_mask_indices = [
                i for i, mask in enumerate(all_masks) if i not in unindexed_mask_indices and 
                (mask | unindexed_mask)[token_starts].sum() <= self.max_unindexed_tokens
            ]
            if len(valid_mask_indices) == 0:
                break
            mask_index_to_add = np.random.choice(valid_mask_indices)
            unindexed_mask_indices.append(mask_index_to_add)
            unindexed_mask |= all_masks[mask_index_to_add]
        
        return unindexed_mask

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks = self.sample_mask(atom_array)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "coverage_fraction_goal": self.coverage_fraction_goal,
            "all_masks": all_masks,
        }
        data = self.apply_design_task_modifiers(data)

        return data


class PartialLigandTask(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }
    """
    Intended only for OMOL examples where we have ligands and maybe some tip atoms.
    In this case, we always coordinate condition the non-ligands and sequence condition everything.

    """

    def __init__(
        self,
        *,
        min_coverage_fraction: float = 0.0,
        max_coverage_fraction: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_coverage_fraction = min_coverage_fraction
        self.max_coverage_fraction = max_coverage_fraction

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_ligand", "is_res_start", "res_min_occupancy", "pn_unit_iid")
        n_eligible_ligands = np.sum(atom_array.mask("is_ligand & is_res_start & (res_min_occupancy > 0)"))
        return n_eligible_ligands >= 1

    def sample_ligand_mask(self, atom_array: AtomArray, ligand_mask: np.ndarray) -> np.ndarray:
        if not ligand_mask.any():
            return np.zeros(atom_array.array_length(), dtype=bool)

        sample_seed = mg.SampleUniformly(is_eligible=ligand_mask)

        grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=1,  # Hop one at a time, and rely on check budget to stop correctly
            require_same_annotation=("pn_unit_iid",),
            atom_array=atom_array,
        )

        # Check budget based on sampled coverage fraction
        n_ligand_atoms = ligand_mask.sum()
        coverage_fraction = np.random.uniform(self.min_coverage_fraction, self.max_coverage_fraction)

        check_budget = mg.CheckAtomBudget(
            n_min_atoms=int(n_ligand_atoms * coverage_fraction),  # rounds down
            n_max_atoms=n_ligand_atoms,
        )
        sampled_ligand_mask, _ = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=sample_seed,
            fn_grow_seed=grow_seed,
            fn_check_budget=check_budget,
        )
        return sampled_ligand_mask

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_ligand", "pn_unit_iid", "res_min_occupancy", "is_res_start")

        # Loop through available ligands and sample a mask for each
        coord_conditioned_mask = np.zeros(atom_array.array_length(), dtype=bool)
        ligand_pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.is_ligand])

        for ligand_pn_unit_iid in ligand_pn_unit_iids:
            ligand_mask = atom_array.pn_unit_iid == ligand_pn_unit_iid
            sampled_ligand_mask = self.sample_ligand_mask(atom_array, ligand_mask)
            coord_conditioned_mask |= sampled_ligand_mask


        # Do not coord condition nan coordinates
        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = coord_conditioned_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": coord_conditioned_mask,
        }
        data = self.apply_design_task_modifiers(data)

        return data
    
# ---------------------------------------------------------------------------
# DistanceTask
# ---------------------------------------------------------------------------
class TipAtomPartialLigandDistanceTask(TipAtomPartialLigandTask):
    """Iterative partial-ligand sampling with 2D distance conditioning."""

    def __init__(
        self,
        *,
        p_knockout_dist: float = 0.95,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.7,
        lig_lig_potential_cutoff: float = 10.0,
        max_dist_cutoff: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction
        self.lig_lig_potential_cutoff = lig_lig_potential_cutoff
        self.max_dist_cutoff = max_dist_cutoff

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        cum_total_mask = np.zeros(atom_array.array_length(), dtype=bool)
        all_masks: list[np.ndarray] = []

        n_eligible_ligands = self._count_unique_lig_pn_units(
            atom_array, atom_array.mask(self.lig_str)
        )
        n_sampled_ligands = min(n_eligible_ligands, self.n_max_ligands)
        for _ in range(n_sampled_ligands):
            total_mask, this_all_masks = self.sample_mask(
                atom_array,
                cum_total_mask,
                n_ligands_to_sample=n_sampled_ligands,
            )
            if len(this_all_masks) == 0:
                break

            this_total_mask = total_mask & ~cum_total_mask
            atom_array = sample_distance_condition(
                atom_array,
                can_have_condition=this_total_mask,
                knockout_p=self.p_knockout_dist,
                dropout_min_fraction=self.knockout_dist_min_fraction,
                dropout_max_fraction=self.knockout_dist_max_fraction,
                additive=cum_total_mask.sum() > 0,
                allow_overlap=True,
                threshold=self.max_dist_cutoff,
            )
            this_total_mask = (
                C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)
            )

            cum_total_mask = np.logical_or(cum_total_mask, this_total_mask)
            all_masks.extend(this_all_masks)


        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": cum_total_mask,
            "all_masks": all_masks,
        }
        data = self.apply_design_task_modifiers(data)
        return data

class TipAtomDistanceTask(TipAtomTask):
    """Same sampling as :class:`EnzymeTipAtomTask` but emits 2D pairwise distance
    conditioning (``C_DIS``) instead of 3D coordinate conditioning (``C_CRD``).
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }

    def __init__(
        self,
        p_knockout_dist: float = 0.8,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        total_mask, all_masks = self.sample_mask(atom_array)

        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=total_mask,
            knockout_p=self.p_knockout_dist,
            dropout_min_fraction=self.knockout_dist_min_fraction,
            dropout_max_fraction=self.knockout_dist_max_fraction,
        )
        # After dropout the effective conditioned set may be smaller;
        # recompute total_mask from the C_DIS pairs.
        total_mask = (
            C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)
        )

        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "all_masks": all_masks,
        }
        data = self.apply_design_task_modifiers(data)
        return data


class PartialLigandDistanceTask(PartialLigandTask):
    def __init__(
        self,
        p_knockout_dist: float = 0.8,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction

    def forward(self, data: dict) -> dict:
        data = super().forward(data)
        atom_array = data["atom_array"]
        ligand_mask = atom_array.mask("is_ligand")

        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=ligand_mask,
            knockout_p=self.p_knockout_dist,
            dropout_min_fraction=self.knockout_dist_min_fraction,
            dropout_max_fraction=self.knockout_dist_max_fraction,
        )
        total_mask = (
            C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)
        )

        data["atom_array"] = atom_array
        data["task"]["mask"] = total_mask
        return data


class MotifScaffoldDistanceTask(MotifScaffoldTask):
    def __init__(
        self,
        inter_motif_distance_p: float = 0.2,
        p_knockout_dist: float = 0.8,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inter_motif_distance_p = inter_motif_distance_p
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks = self.sample_mask(atom_array)
        if true_with_prob(self.inter_motif_distance_p):
            atom_array = sample_distance_condition(
                atom_array,
                can_have_condition=total_mask,
                knockout_p=self.p_knockout_dist,
                dropout_min_fraction=self.knockout_dist_min_fraction,
                dropout_max_fraction=self.knockout_dist_max_fraction,
                additive=False,
            )
            added_inter_motif_distances = True
        else:
            total_mask_so_far = np.zeros(atom_array.array_length(), dtype=bool)
            for motif_mask in all_masks:
                motif_mask = motif_mask & ~total_mask_so_far
                total_mask_so_far |= motif_mask
                atom_array = sample_distance_condition(
                    atom_array,
                    can_have_condition=motif_mask,
                    knockout_p=self.p_knockout_dist,
                    dropout_min_fraction=self.knockout_dist_min_fraction,
                    dropout_max_fraction=self.knockout_dist_max_fraction,
                    additive=True,
                )
            added_inter_motif_distances = False

        total_mask = C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "coverage_fraction_goal": self.coverage_fraction_goal,
            "all_masks": all_masks,
            "added_inter_motif_distances": added_inter_motif_distances,
        }

        data = self.apply_design_task_modifiers(data)

        return data