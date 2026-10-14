from collections import defaultdict
from collections.abc import Callable
from typing import Literal, ClassVar
from functools import partial

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.enums import ChainTypeInfo
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
    C_IDX,
    C_SEQ,
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

from rf3.conditions import C_HOT, C_ITYPE, ItypeEnum
from rf3.data.design_tasks.ppi_utils import (
    crop_spatial_keep_full_binder,
    get_hotspot_atoms,
    sample_hotspots,
)
from rf3.data.design_tasks.base import RFD3DesignTask

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

_INTERFACE_TYPE_TO_ITYPE = {
    "protein-protein": ItypeEnum.PPI,
    "protein-ligand": ItypeEnum.PLI,
    "protein-na": ItypeEnum.PNA,
}


def _classify_interface(atom_array: AtomArray, target_pn_unit_iid: str) -> str:
    """One of: protein-protein / protein-ligand / protein-dna / protein-rna,
    based on the target chain's ``chain_type`` annotation."""
    target_chain_types = set(
        np.unique(atom_array.chain_type[atom_array.pn_unit_iid == target_pn_unit_iid])
    )

    votes: dict[str, int] = {
        "protein-protein": sum([x in ChainTypeInfo.PROTEINS for x in target_chain_types]),
        "protein-ligand": sum([x in ChainTypeInfo.NON_POLYMERS for x in target_chain_types]),
        "protein-na": sum([x in ChainTypeInfo.NUCLEIC_ACIDS for x in target_chain_types]),
    }

    # return the key with the highest vote
    return max(votes, key=votes.get)


class PPIHotspot(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
    }

    def __init__(
        self,
        max_hotspots_frac_to_provide: float = 0.2,
        hotspot_max_distance: float = 7.0,
        annotate_interface_type: bool = False,
        enable_during_validation: bool = False,
        **kwargs,
    ):
        """
        This task samples a set of hotspots (atom level) and provides as 1D conditioning to the model.
        Based on an earlier internal PPI hotspot transform by Rafael Brent; see README.md.

        The logic is as follows:
        1. Decide which chain is the binder. We do this by randomly choosing one
           of the ``query_pn_unit_iids`` since we assume these chains form a valid
           interface. The other query chain is the target.
        2. Classify the interface type from the target chain's ``chain_type`` into one
           of three classes (protein-protein / protein-ligand / protein-na, where
           protein-na covers both DNA and RNA). PPI gets the hotspot mask on the
           target side; PLI / PNA gets it on the protein binder side.
        3. Find all candidate hotspots within ``hotspot_max_distance`` across
           the binder/target pair and sample a subset up to
           ``max_hotspots_frac_to_provide``.

        Args:
            max_hotspots_frac_to_provide: maximum fraction of the pocket to sample as hotspots
            hotspot_max_distance: interchain distance cutoff (Angstroms) that defines a candidate hotspot
            annotate_interface_type: if True, set C_ITYPE annotation on hotspot atoms based on interface classification
            enable_during_validation: if True, allow hotspot conditioning to be provided to the model during
                validation/inference. Validation loaders do not populate ``query_pn_unit_iids``, so when this
                flag is on we backfill it from ``np.unique(atom_array.pn_unit_iid)`` so the existing
                binder/target selection logic can run. Default False (no behavioral change at training time).
        """
        super().__init__(**kwargs)
        self.max_hotspots_frac_to_provide = max_hotspots_frac_to_provide
        self.hotspot_max_distance = hotspot_max_distance
        self.annotate_interface_type = annotate_interface_type
        self.enable_during_validation = enable_during_validation

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(
            atom_array,
            "is_protein",
        )
        data["atom_array"] = atom_array
        return data

    def can_apply(self, data: dict) -> bool:
        # we need:
        #   a) two chains in query
        #   b) at least 1 protein chain
        atom_array = data["atom_array"]
        if self.enable_during_validation and data.get("query_pn_unit_iids") is None and data.get("is_inference", False):
            query_pn_unit_iids = list(np.unique(atom_array.pn_unit_iid))
        else:
            query_pn_unit_iids = data.get("query_pn_unit_iids")
        if (query_pn_unit_iids is None or len(query_pn_unit_iids) != 2):
            return False

        found_protein_chain = False
        for pn_unit_iid in query_pn_unit_iids:
            # make sure pn_unit_iid is actually in the atom array
            if atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'").sum() == 0:
                found_protein_chain = False
            if atom_array[atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'")].mask("is_protein").any():
                return True  # found a protein chain, return
        return found_protein_chain

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if self.enable_during_validation and data.get("query_pn_unit_iids") is None:
            query_pn_unit_iids = list(np.unique(atom_array.pn_unit_iid))
        else:
            query_pn_unit_iids = data["query_pn_unit_iids"]  # existence guaranteed via can_apply()

        # Select binder chain (binder must be protein, and we've already ensured there is at least one protein chain)
        protein_pn_unit_iids = [
            pn_unit_iid for pn_unit_iid in query_pn_unit_iids
            if atom_array[atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'")].mask("is_protein").any()
        ]
        if len(protein_pn_unit_iids) > 1:
            binder_pn_unit_iid = str(np.random.choice(protein_pn_unit_iids))
        else:
            binder_pn_unit_iid = protein_pn_unit_iids[0]
        target_pn_unit_iid = next(iid for iid in query_pn_unit_iids if iid != binder_pn_unit_iid)

        # Classify the interface from the target chain's chain_type so we can pick the hotspot side.
        # PPI: mask on the non-binder protein (target). PLI / PNA: mask on the protein binder.
        interface_type = _classify_interface(atom_array, target_pn_unit_iid)
        hotspot_side = (
            "binder"
            if interface_type in {"protein-ligand", "protein-na"}
            else "target"
        )

        # sample_hotspots is scoped to the explicit binder/target pair; cofactors/ions on other chains are ignored.
        hotspot_atom_mask = sample_hotspots(
            atom_array,
            binder_pn_unit_iid,
            target_pn_unit_iid,
            self.max_hotspots_frac_to_provide,
            self.hotspot_max_distance,
            hotspot_side=hotspot_side,
        )
        if hotspot_atom_mask.sum() == 0:
            ranked_logger.warning(
                f"No hotspots found for binder chain {binder_pn_unit_iid} on example_id {data['example_id']}"
            )

        # ... set the conditions for the task
        C_HOT.set_annotation(atom_array=atom_array, array=hotspot_atom_mask)
        C_HOT.set_mask(atom_array, hotspot_atom_mask, 1, "atom")

        # ... update the now condition-annotated atom array
        data["atom_array"] = atom_array

        # ... add the task mask
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            # required fields
            "class": self.__class__.__name__,
            "mask": hotspot_atom_mask,
            # optional fields
            "binder_pn_unit_iid": binder_pn_unit_iid,
            "num_hotspots_to_sample": hotspot_atom_mask.sum(),
        }

        if self.annotate_interface_type:
            itype_array = np.full(len(atom_array), ItypeEnum.NONE, dtype=int)
            itype_array[hotspot_atom_mask] = _INTERFACE_TYPE_TO_ITYPE[interface_type]
            C_ITYPE.set_annotation(atom_array=atom_array, array=itype_array)

        return data
