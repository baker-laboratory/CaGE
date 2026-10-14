import torch

from biotite.structure import AtomArray
from dataclasses import dataclass, field
from jaxtyping import Float, Int, Bool
from typing import Any, Dict, Optional
from functools import cached_property
import numpy as np
from atomworks.ml.conditions import C_CRD

from rf3.data.schema.common import DataSample, Features
from rf3.conditions import C_HOT

@dataclass
class RF3Features(Features):
    # distogram templating features
    distogram_condition_noise_scale: Float[torch.Tensor, "I"]
    has_distogram_condition: Bool[torch.Tensor, "I I"]
    distogram_condition: Float[torch.Tensor, "I I 64"]
    # positional and token features
    residue_index: Int[torch.Tensor, "I"]
    token_index: Int[torch.Tensor, "I"]
    asym_id: Int[torch.Tensor, "I"]
    entity_id: Int[torch.Tensor, "I"]
    sym_id: Int[torch.Tensor, "I"]
    restype: Int[torch.Tensor, "I"]
    is_protein: Bool[torch.Tensor, "I"]
    is_rna: Bool[torch.Tensor, "I"]
    is_dna: Bool[torch.Tensor, "I"]
    is_ligand: Bool[torch.Tensor, "I"]
    is_atomized: Bool[torch.Tensor, "I"]
    # reference conformer features
    ref_pos: Float[torch.Tensor, "L 3"]
    ref_mask: Bool[torch.Tensor, "L"]
    ref_element: Int[torch.Tensor, "L 128"]
    ref_charge: Float[torch.Tensor, "L"]
    ref_atom_name_chars: Int[torch.Tensor, "L 4 64"]
    ref_space_uid: Int[torch.Tensor, "L"]
    # atom-level templating
    is_atomized_atom_level: Bool[torch.Tensor, "L"]
    ref_pos_ground_truth: Float[torch.Tensor, "L 3"]
    ref_pos_is_ground_truth: Bool[torch.Tensor, "L"]
    # atom-level templating
    # condition_coordinate_1_atom: Bool[torch.Tensor, "L"] # TODO: replace this with C_INTCRD
    mask_hotspot_1_atom: Bool[torch.Tensor, "L"]
    # indexing
    atom_to_token_map: Int[torch.Tensor, "L"]
    # chiral features
    chiral_feats: Float[torch.Tensor, "num_chiral_centers 5"]
    chiral_centers: Float[torch.Tensor, "num_chiral_centers 4"]
    chiral_center_dihedral_angles: Float[torch.Tensor, "num_chiral_centers"]
    # bond features
    token_bonds: Int[torch.Tensor, "I I"]
    cyclic_asym_ids: list[int]
    # MSA features
    msa_stack: Float[torch.Tensor, "num_recycles S I 35"]
    profile: Float[torch.Tensor, "I 32"]
    deletion_mean: Float[torch.Tensor, "I"]
    # Conditioning features
    # condition_coordinate_1_atom: Float[torch.Tensor, "L 3"] # TODO: replace this with C_INTCRD
    mask_hotspot_1_atom: Bool[torch.Tensor, "L"]
    condition_hbond_1_atom: Int[torch.Tensor, "L 3"]
    condition_rasa_1_atom: Int[torch.Tensor, "L 3"]
    condition_itype_1_atom: Int[torch.Tensor, "L 3"]
    # 2D atom-level distance conditioning. Sparse pair list densified
    condition_distance_2_atom: Float[torch.Tensor, "L L"]
    mask_distance_2_atom: Bool[torch.Tensor, "L L"]
    # real reference conformer for the diffusion encoder (masked tokens stay real here)
    ref_pos_diff: Float[torch.Tensor, "L 3"] = None
    ref_charge_diff: Float[torch.Tensor, "L"] = None
    ref_mask_diff: Bool[torch.Tensor, "L"] = None
    ref_element_diff: Int[torch.Tensor, "L 128"] = None
    ref_atom_name_chars_diff: Int[torch.Tensor, "L 4 64"] = None


    @cached_property
    def get_default_padding_value_overrides(self) -> Dict[str, Any]:
        return {
            "atom_to_token_map": -1,
        }


@dataclass
class RF3GroundTruth(Features):
    coord_atom_lvl: Float[torch.Tensor, "L 3"]
    mask_atom_lvl: Bool[torch.Tensor, "L"]
    coord_token_lvl: Float[torch.Tensor, "I 3"]
    mask_token_lvl: Bool[torch.Tensor, "I"]
    chain_iid_token_lvl: Int[torch.Tensor, "I"]
    rep_atom_idxs: Int[torch.Tensor, "I"]
    masked_tokens: Bool[torch.Tensor, "I"]
    true_token_ids: Int[torch.Tensor, "I"]
    interfaces_to_score: list = None
    pn_units_to_score: list = None


@dataclass
class RF3Confidence(Features):
    rf2aa_seq: Int[torch.Tensor, "I"]
    atom_frames: Int[torch.Tensor, "Q 3 2"]
    is_real_atom: Bool[torch.Tensor, "I 36"]
    pae_frame_idx_token_lvl_from_atom_lvl: Int[torch.Tensor, "I 3"]


@dataclass
class RF3DiffusionFeatures(Features):
    t: Float[torch.Tensor, "D"]
    noise: Float[torch.Tensor, "D L 3"]
    coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"]


@dataclass
class RF3SymmetryResolution(Features):
    automorphisms: list[np.ndarray]
    molecule_entity: Int[torch.Tensor, "uncropped_L"]
    molecule_iid: Int[torch.Tensor, "uncropped_L"]
    crop_mask: Int[torch.Tensor, "L"]
    coord_atom_lvl: Float[torch.Tensor, "uncropped_L 3"]
    mask_atom_lvl: Bool[torch.Tensor, "uncropped_L"]
    chain_iid_by_atom: Int[torch.Tensor, "L"]
    token_to_repatom_index: Int[torch.Tensor, "I"]

    @cached_property
    def get_default_padding_value_overrides(self) -> Dict[str, Any]:
        return {
            "crop_mask": -1,
            "token_to_repatom_index": -1,
        }


@dataclass
class RF3PipelineTracking(Features):
    """Per-example dataloading diagnostics (non-tensor scalars; collate -> list[int])."""

    n_fallbacks_used: int = 0
    n_protein_chains_gt50_with_msa: int = 0
    n_protein_chains_gt50_without_msa: int = 0


@dataclass
class RF3DataSample(DataSample):
    example_id: str
    features: RF3Features
    ground_truth: RF3GroundTruth
    diffusion_features: RF3DiffusionFeatures
    symmetry_resolution: RF3SymmetryResolution
    atom_array: AtomArray
    missing_atom_behavior: str
    confidence: Optional[RF3Confidence] = None
    pipeline_tracking: RF3PipelineTracking = field(default_factory=RF3PipelineTracking)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RF3DataSample":
        data["diffusion_features"] = {
            "t": data["t"],
            "noise": data["noise"],
            "coord_atom_lvl_to_be_noised": data["coord_atom_lvl_to_be_noised"],
        }
        data["symmetry_resolution"].update(
            {
                "automorphisms": data["automorphisms"],
            }
        )

        for k in ("interfaces_to_score", "pn_units_to_score"):
            if k in data and k not in data["ground_truth"]:
                v = data[k]
                if isinstance(v, str):
                    v = eval(v) if v else []
                data["ground_truth"][k] = v

        return cls(
            example_id=data["example_id"],
            features=RF3Features.from_dict(data["feats"]),
            ground_truth=RF3GroundTruth.from_dict(data["ground_truth"]),
            confidence=RF3Confidence.from_dict(data["confidence_feats"])
            if "confidence_feats" in data
            else None,
            diffusion_features=RF3DiffusionFeatures.from_dict(
                data["diffusion_features"]
            ),
            symmetry_resolution=RF3SymmetryResolution.from_dict(
                data["symmetry_resolution"]
            ),
            atom_array=data["atom_array"],
            missing_atom_behavior=data["missing_atom_behavior"],
            pipeline_tracking=RF3PipelineTracking(
                n_fallbacks_used=data.get("n_fallbacks_used", 0),
                n_protein_chains_gt50_with_msa=data.get("n_protein_chains_gt50_with_msa", 0),
                n_protein_chains_gt50_without_msa=data.get("n_protein_chains_gt50_without_msa", 0),
            ),
        )

    def pad_to_max_lengths(self, max_lengths: Dict[str, int]) -> "RF3DataSample":
        return RF3DataSample(
            example_id=self.example_id,
            features=self.features.pad_to_max_lengths(max_lengths),
            ground_truth=self.ground_truth.pad_to_max_lengths(max_lengths),
            confidence=self.confidence.pad_to_max_lengths(max_lengths)
            if self.confidence is not None
            else None,
            diffusion_features=self.diffusion_features.pad_to_max_lengths(max_lengths),
            symmetry_resolution=self.symmetry_resolution.pad_to_max_lengths(
                max_lengths
            ),
            atom_array=self.atom_array,  # Assuming atom_array does not need padding
            missing_atom_behavior=self.missing_atom_behavior,
            pipeline_tracking=self.pipeline_tracking.pad_to_max_lengths(max_lengths),
        )
