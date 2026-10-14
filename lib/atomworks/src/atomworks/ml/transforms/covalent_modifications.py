"""Transforms to handle covalent modifications"""

from typing import ClassVar

from biotite.structure import AtomArray

from atomworks.io.transforms.atomize import flag_and_reassign_covalent_modifications
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform


class FlagAndReassignCovalentModifications(Transform):
    """Handles covalent modifications within the AtomArray.

    Covalent modifications, e.g., glycosylation, are handled by the following algorithm:

    for polymer residues with atoms covalently bound to a NON-POLYMER:
        for ALL atoms in the polymer residue:
            set the pn_unit_iid and pn_unit_id identifying annotations to that of the NON-POLYMER polymer/non-polymer unit
            set atomize = true (thus, this transform must be run before the Atomize transform)
            set is_covalent_modification = true (for the entire pn_unit)

    TODO: Break into two Transforms - one that flags, one that reassigns. Atomizing covalent modifications is a design choice
    that may not be desired in all pipelines. Annotating covalent modifications, however, is broadly useful.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, "AddGlobalTokenIdAnnotation"]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = flag_and_reassign_covalent_modifications(data["atom_array"])
        return data
