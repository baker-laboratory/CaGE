import logging

import numpy as np
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
import torch
from atomworks.constants import STANDARD_AA

logger = logging.getLogger("atomworks.ml")

class MaskTokenFeatures(Transform):
    """
    Transform that masks a fraction of the tokens in the MSA for masked token recovery training. 
    """

    # need to do it after this transform, because we want to include poly-poly bonds, which AF3TokenBondFeatures does not.
    requires_previous_transforms = [
        "LoadPolymerMSAs",
        "PairAndMergePolymerMSAs",
        "EncodeMSA",
        "FillFullMSAFromEncoded"
    ]

    # Reference-conformer features that masking zeroes (and that we optionally copy for diffusion).
    _REF_FEATURES = ["ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars"]

    def __init__(self, mask_prob: float, skip_prob: float, mask_token: str, sequence_encoding,
                 provide_diffusion_ref_copy: bool = False):
        super().__init__()
        self.mask_prob = mask_prob
        self.skip_prob = skip_prob
        self.mask_encoding = sequence_encoding.encode([mask_token])
        self.provide_diffusion_ref_copy = provide_diffusion_ref_copy

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        restype = data["feats"]["restype"]  # [n_tokens_across_chains, 32]
        true_token_ids = restype.argmax(dim=-1)  # [n_tokens_across_chains]

        # Real (unmasked) reference conformer for the diffusion encoder; pairformer keeps the masked originals.
        if self.provide_diffusion_ref_copy:
            for name in self._REF_FEATURES:
                data["feats"][f"{name}_diff"] = data["feats"][name].clone()

        if np.random.rand()<=self.skip_prob:
            data["ground_truth"]["masked_tokens"] = torch.zeros(restype.shape[0], dtype=torch.bool)  # [n_tokens_across_chains]
            data["ground_truth"]["true_token_ids"] = true_token_ids  # [n_tokens_across_chains]
            return data

        # Identify maskable positions
        token_starts = get_token_starts(atom_array, add_exclusive_stop=True)
        is_protein = np.isin(atom_array.res_name, STANDARD_AA)
        maskable_positions = is_protein & ~atom_array.atomize # only select unatomized protein residues for masking

        # Sample masked tokens
        n_tokens = restype.shape[0]
        masked_tokens = np.random.rand(n_tokens) < self.mask_prob  # [n_tokens_across_chains]
        masked_tokens = masked_tokens & maskable_positions[token_starts[:-1]]  # [n_tokens_across_chains]

        data["ground_truth"]["masked_tokens"] = torch.from_numpy(masked_tokens)  # [n_tokens_across_chains]
        data["ground_truth"]["true_token_ids"] = true_token_ids  # [n_tokens_across_chains]

        if masked_tokens.sum() == 0:
            return data  # no tokens masked


        masked_atoms = np.zeros(atom_array.shape[0], dtype=bool)  # [n_atoms_across_chains]
        for idx in masked_tokens.nonzero()[0]:
            start = token_starts[idx]
            end = token_starts[idx + 1]
            masked_atoms[start:end] = True

        masked_tokens = torch.from_numpy(masked_tokens)
        masked_atoms = torch.from_numpy(masked_atoms)

        n_restypes = restype.shape[-1]
        n_msa_types = data["feats"]["msa_stack"].shape[-1]

        mask_encoding = torch.from_numpy(self.mask_encoding)
        restype_mask = torch.nn.functional.one_hot(
            mask_encoding, num_classes=n_restypes
        ).squeeze(0)
        msa_mask = torch.nn.functional.one_hot(
            mask_encoding, num_classes=n_msa_types
        ).squeeze(0)

        # We provide the mask token for restype, profile, and the msa, while leaving the deletion_mean untouched.
        data["feats"]["restype"][masked_tokens, :] = restype_mask
        data["feats"]["profile"][masked_tokens, :] = restype_mask.float()
        data["feats"]["msa_stack"][:, :, masked_tokens, :] = msa_mask.float()

        # We provide no information on the reference conformers, which prevents the model from inferring the true number of atoms per masked token.
        data["feats"]["ref_pos"][masked_atoms, :] = 0
        data["feats"]["ref_charge"][masked_atoms] = 0
        data["feats"]["ref_mask"][masked_atoms] = 0
        data["feats"]["ref_element"][masked_atoms, :] = 0
        data["feats"]["ref_atom_name_chars"][masked_atoms, :, :] = 0

        # The ADD-policy native conformer (ref_pos_ground_truth) in another path where residue identity can be leaked
        #  -> ensure masked residues hide this as well
        if "ref_pos_ground_truth" in data["feats"]:
            data["feats"]["ref_pos_ground_truth"][masked_atoms, :] = 0
        if "ref_pos_is_ground_truth" in data["feats"]:
            data["feats"]["ref_pos_is_ground_truth"][masked_atoms] = 0

        return data
