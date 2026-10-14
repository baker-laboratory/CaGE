import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from biotite.structure import AtomArray

from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_DIS
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms._checks import check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray
from rf3.conditions import C_HB, C_HOT, C_ITYPE, C_RAS
from torch import Tensor
from torch.nn.functional import one_hot


def add_conditions_as_features(
    atom_array: AtomArray,
) -> dict[str, Tensor]:
    """Add conditions as features to the data dictionary."""

    # coordinate_conditioning = C_CRD.annotation(atom_array)
    # coordinate_conditioning = np.nan_to_num(coordinate_conditioning, 0.0)  # change nans to 0.0

    # Drop the "no condition" class (index 0) so unconditioned residues are
    # all-zeros and contribute exactly nothing through the bias-free input Linear.
    rasa_conditioning = torch.tensor(C_RAS.annotation(atom_array))  # L
    rasa_conditioning = one_hot(rasa_conditioning, num_classes=4)[..., 1:]  # L x 3

    hbond_conditioning = torch.tensor(C_HB.annotation(atom_array))  # L
    hbond_conditioning = one_hot(hbond_conditioning, num_classes=4)[..., 1:]  # L x 3

    itype_conditioning = torch.tensor(C_ITYPE.annotation(atom_array))  # L
    itype_conditioning = one_hot(itype_conditioning, num_classes=4)[..., 1:]  # L x 3
    
    # 2D atom-level distance conditioning. 
    atom_array_plus = as_atom_array_plus(atom_array)
    distance_annotation = C_DIS.annotation(atom_array_plus, default="generate")
    distance_dense = distance_annotation.as_dense_array(default=0.0)  # [L, L] float
    distance_mask = (
        C_DIS.mask(atom_array_plus, default="generate").as_dense_array(default=False)
    )  # [L, L] bool

    feats = {}
    # feats[C_CRD.get_full_name()] = torch.tensor(coordinate_conditioning)  # L x 3 # TODO: replace this with C_INTCRD once it's in
    feats[C_HOT.get_full_name()] = torch.tensor(C_HOT.annotation(atom_array))  # L,
    feats[C_HB.get_full_name()] = hbond_conditioning  # L x 3,
    feats[C_RAS.get_full_name()] = rasa_conditioning  # L x 3,
    feats[C_ITYPE.get_full_name()] = itype_conditioning  # L x 3
    feats[C_DIS.get_full_name()] = torch.tensor(distance_dense, dtype=torch.float32)  # L x L
    feats[C_DIS.get_mask_name()] = torch.tensor(distance_mask, dtype=torch.bool)  # L x L

    return feats


class AddConditionsAsFeatures(Transform):
    """Add design task features to the data dictionary."""

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        """Add design task features to the data dictionary."""
        feats = add_conditions_as_features(data["atom_array"])
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(feats)
        return data
