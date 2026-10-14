import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.common import exists
from atomworks.constants import (
    AF3_EXCLUDED_LIGANDS,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, AF3SequenceEncoding
from atomworks.ml.transforms.af3_reference_molecule import (
    GetAF3ReferenceMoleculeFeatures,
    GroundTruthConformerPolicy,
    RandomApplyGroundTruthConformerByChainType,
)
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalResIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
)
from atomworks.ml.transforms.atom_frames import (
    AddAtomFrames,
    AddIsRealAtom,
    AddPolymerFrameIndices,
)
from atomworks.ml.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from atomworks.ml.transforms.base import (
    AddData,
    ApplyFunction,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import (
    AddAF3TokenBondFeatures,
)
from atomworks.ml.transforms.cached_residue_data import (
    LoadCachedResidueLevelData,
    RandomSubsampleCachedConformers,
)
from atomworks.ml.transforms.center_random_augmentation import CenterRandomAugmentation
from atomworks.ml.transforms.chirals import AddAF3ChiralFeatures
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.design_task import SampleDesignTask
from atomworks.ml.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from atomworks.ml.transforms.dna.pad_dna import PadDNA

# from atomworks.ml.transforms.diffusion.edm import SampleEDMNoise
from atomworks.ml.transforms.encoding import (
    EncodeAF3TokenLevelFeatures,
    EncodeAtomArray,
)
from atomworks.ml.transforms.feature_aggregation.af3 import AggregateFeaturesLikeAF3
from atomworks.ml.transforms.feature_aggregation.confidence import (
    PackageConfidenceFeats,
)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RandomlyRemoveLigands,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)
from atomworks.ml.transforms.mirror_transform import RandomlyMirrorInputs
from atomworks.ml.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeAF3,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from atomworks.ml.transforms.random_atomize_residues import RandomAtomizeResidues
from atomworks.ml.transforms.rdkit_utils import GetRDKitChiralCenters
from atomworks.ml.transforms.symmetry import FindAutomorphismsWithNetworkX
from omegaconf import DictConfig
from rf3.data.cyclic_transform import AddCyclicBonds
from rf3.data.extra_xforms import (
    CheckForNaNsInInputs,
    SampleEDMNoiseCustom,
    AnnotateTokenRepresentatives,
    AnnotateMSAStatsByChain,
    PlaceUnresolvedTokenAtomsAtOrigin,
    CropToResolvedOnly,
    FlagNoncanonicalPolymerBondsForAtomization,
    ValidateProteinChirality,
)
from rf3.data.masked_token_transform import MaskTokenFeatures
from rf3.data.pipeline_utils import (
    annotate_post_crop_hash,
    annotate_pre_crop_hash,
    build_ground_truth_distogram_transform,
    set_to_occupancy_0_where_crop_hashes_differ,
)
from rf3.data.condition_featurization import AddConditionsAsFeatures


logger = logging.getLogger(__name__)


def TrainingRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )


def _get_design_task_name(data: dict) -> str | None:
    """Get the design task name from the data dict."""
    return (data.get("task") or {}).get("name")


def build_af3_transform_pipeline(
    *,
    # Training or inference (required)
    is_inference: bool,  # If True, we skip cropping, etc.
    # MSA dirs
    protein_msa_dirs: list[dict],
    rna_msa_dirs: list[dict],
    # Recycles
    n_recycles: int = 5,
    # Crop params
    crop_size: int = 384,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    max_atoms_in_crop: int | None = None,
    # Undesired res names
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Conformer generation params
    conformer_generation_timeout: float = 5.0,  # seconds
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    # MSA parameters
    max_msa_sequences: int = 10_000,  # Paper: 16,000, but we only have 10K stored on disk
    n_msa: int = 10_000,  # Paper: ?? I think ~12K?
    dense_msa: bool = True,  # True for AF3
    add_residue_is_paired_feature: bool = False,
    # Cache paths
    msa_cache_dir: PathLike | str | None = None,
    residue_cache_dir: PathLike | str | None = None,
    # Diffusion parameters
    sigma_data: float = 16.0,
    diffusion_batch_size: int = 48,
    # Whether to include features for confidence head
    run_confidence_head: bool = False,
    return_atom_array: bool = True,
    # Design tasks
    design_tasks: dict[str, dict[str, Any]] = {},
    # DNA
    pad_dna_p_skip: float = 0.0,
    b_factor_min: float | None = None,
    b_factor_max: float | None = None,
    # ------ Atom-level conditioning ------ #
    p_unconditional: float = 1.0,  # Show no conditioning, anywhere (i.e., unconditional)
    template_noise_scales: dict | DictConfig = {
        "atomized": 1e-5,  # No noise (for atomized tokens)
        "not_atomized": 0.2,  # Up to 0.2A of noise (for non-atomized tokens)
    },
    allowed_chain_types_for_conditioning: list[int | str | ChainType]
    | None = ChainType.get_all_types(),  # All chain types (None = no conditioning)
    p_condition_per_token: float = 0.4,  # When sampling with conditions, X% of tokens are conditioned (e.g., X^2% of pairs have conditions)
    p_provide_inter_molecule_distances: float = 0.05,  # When sampling with conditions, X% of the time, show any inter-molecule distances
    # (Reference Conformer)
    p_give_non_polymer_ref_conf: float = 0.0,  # When sampling with conditions, X% of non-polymer chains get a ground-truth reference conformer
    p_give_polymer_ref_conf: float = 0.0,  # When sampling with conditions, X% of polymer chains get a ground-truth reference conformer
    # -------------------------------------- #
    take_first_chiral_subordering: bool = False,
    mirror_prob: float = 0.0,
    input_contains_explicit_msa: bool = False,
    atomization_prob: float = 0.0,
    ligand_dropout_prob: float = 0.0,
    raise_if_missing_msa_for_protein_of_length_n: int | None = None,
    mask_crop_edges: bool = False,
    add_cyclic_bonds: bool = True,
    metrics_tags: list[str] | set[str] | None = None,
    pmean: float = -1.2,
    pstd: float = 1.5,
    masked_token_prob: float = 0.0,
    masked_token_frac_ex_to_mask: float = 0.0,
    provide_diffusion_ref_copy: bool = False,
    mask_token: str = "<G>",
    missing_atom_behavior: str = "nearest",  # "nearest" | "origin" | "trim" | "rollout"
    p_atomize_disulfide: float = 0.0,
    check_chirality: bool = True,
    chirality_swap_pdb_codes: list[str] | None = None,
):
    """Build the AF3 pipeline with specified parameters.

    This function constructs a pipeline of transforms for processing protein structures
    in a manner similar to AlphaFold 3. The pipeline includes steps for removing hydrogens,
    adding annotations, atomizing residues, cropping, adding templates, encoding features,
    and generating reference molecule features.

    Args:
        crop_size (int, optional): The size of the crop. Defaults to 384.
        crop_center_cutoff_distance (float, optional): The cutoff distance for spatial cropping.
            Defaults to 15.0.
        crop_contiguous_probability (float, optional): The probability of using contiguous cropping.
            Defaults to 0.5.
        crop_spatial_probability (float, optional): The probability of using spatial cropping.
            Defaults to 0.5.
        conformer_generation_timeout (float, optional): The timeout for conformer generation in seconds.
            Defaults to 10.0.
        metrics_tags (list[str] | set[str] | None, optional): Tags to use for determining which Metrics apply.
            Defaults to None (tags not added).

    Returns:
        Transform: A composed pipeline of transforms.

    Raises:
        AssertionError: If the crop probabilities do not sum to 1.0, if the crop size is not positive,
        or if the crop center cutoff distance is not positive.

    Note:
        The cropping method is chosen randomly based on the provided probabilities.
        The pipeline includes steps for processing the structure, adding annotations,
        and generating features required for AF3-like predictions.

    References:
        - AlphaFold 3 Supplementary Information.
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    _VALID_MISSING_ATOM_BEHAVIORS = ("nearest", "origin", "trim", "rollout")
    assert missing_atom_behavior in _VALID_MISSING_ATOM_BEHAVIORS, (
        f"Invalid missing_atom_behavior {missing_atom_behavior!r}. "
        f"Must be one of {_VALID_MISSING_ATOM_BEHAVIORS}."
    )

    if (
        crop_contiguous_probability > 0 or crop_spatial_probability > 0
    ) and not is_inference:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert (
            crop_center_cutoff_distance > 0
        ), "Crop center cutoff distance must be greater than 0"

    af3_sequence_encoding = AF3SequenceEncoding()
    rf2aa_sequence_encoding = RF2AA_ATOM36_ENCODING

    transforms = [
        AddData(
            {"is_inference": is_inference, "run_confidence_head": run_confidence_head,
             "missing_atom_behavior": missing_atom_behavior}
        ),
        # ... unconditional vs. conditional (training only; skipped at inference via
        # TrainingRoute so explicit template/relative selections are honored instead of
        # being zeroed out by is_unconditional=True). Mirrors the earlier internal pipeline.
        TrainingRoute(
            RandomRoute(
                transforms=[
                    AddData({"is_unconditional": True}),
                    AddData({"is_unconditional": False}),
                ],
                probs=[p_unconditional, 1 - p_unconditional],
            ),
        ),
        RemoveHydrogens(),
        TrainingRoute(
            FilterToSpecifiedPNUnits(
                extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
            ),
        ),
    ]

    if exists(metrics_tags):
        transforms.append(AddData({"metrics_tags": metrics_tags}))

    if check_chirality:
        transforms.append(ValidateProteinChirality(swap_pdb_codes=chirality_swap_pdb_codes))

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: RandomlyMirrorInputs(mirror_prob),
            },
        )
    )

    transforms += [
        RemoveTerminalOxygen(),
        TrainingRoute(
            SetOccToZeroOnBfactor(b_factor_min, b_factor_max),
        ),
        TrainingRoute(RemoveUnresolvedPNUnits()),
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                # UNX causes RDKit to crash (element is "X"), so we exclude even at inference
                True: HandleUndesiredResTokens(undesired_res_tokens=["UNX"]),
                False: HandleUndesiredResTokens(
                    undesired_res_tokens=undesired_res_names,
                    undesired_tokens_never_map=(UNKNOWN_AA, UNKNOWN_RNA, UNKNOWN_DNA),
                ),
            },
        ),
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        FlagNoncanonicalPolymerBondsForAtomization(p_atomize_disulfide=p_atomize_disulfide),
    ]

    # NOTE: PadDNA is only used in training, and requires X3DNA to be installed.
    # We skip it entirely during inference to avoid the X3DNA dependency.
    if not is_inference and pad_dna_p_skip < 1.0:
        if os.environ.get("X3DNA"):
            transforms.append(PadDNA(p_skip=pad_dna_p_skip))
        else:
            logger.warning(
                "X3DNA is not set; skipping PadDNA. DNA duplexes will not be structurally padded "
                "(set X3DNA to reproduce the training / design-validation pipeline exactly)."
            )

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: RandomAtomizeResidues(atomization_prob),
            },
        )
    )

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: RandomlyRemoveLigands(ligand_dropout_prob),
            },
        )
    )

    transforms += [
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA
            + STANDARD_RNA
            + STANDARD_DNA
            + (UNKNOWN_AA, UNKNOWN_RNA, UNKNOWN_DNA),
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # For 'trim' mode: crop to only resolved tokens before spatial cropping
    if missing_atom_behavior == "trim":
        transforms.append(CropToResolvedOnly())

    # Crop

    # ... crop around our query pn_unit(s) early, since we don't need the full structure moving forward
    cropping_transform = Identity()
    if crop_size is not None:
        cropping_transform = RandomRoute(
            transforms=[
                CropContiguousLikeAF3(
                    crop_size=crop_size,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                ),
                CropSpatialLikeAF3(
                    crop_size=crop_size,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                    raise_if_missing_crop_center=False,
                ),
            ],
            probs=[crop_contiguous_probability, crop_spatial_probability],
        )

    transforms += [
        TrainingRoute(ApplyFunction(annotate_pre_crop_hash)),
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: cropping_transform,
                # Default to Identity during inference (`is_inference == True`)
            },
        ),
        TrainingRoute(ApplyFunction(annotate_post_crop_hash)),
    ]

    if mask_crop_edges:
        transforms += [
            TrainingRoute(ApplyFunction(set_to_occupancy_0_where_crop_hashes_differ)),
        ]

    # +-----------------------------------------------------------+
    # +-------------- Sample and Apply DesignTasks ---------------+
    # +-----------------------------------------------------------+

    if design_tasks:
        transforms.append(SampleDesignTask(design_tasks=design_tasks))

        # ... create the design task
        transforms.append(ConditionalRoute(
            condition_func=_get_design_task_name,
            transform_map={name: task["transform"] for name, task in design_tasks.items()},
        ))

    # TODO: Eventually templates and reference conformers should be part of DesignTask and Condition framework
    # +-----------------------------------------------------------+
    # +------------------ GROUND TRUTH TEMPLATE ------------------+
    # +-----------------------------------------------------------+

    # Ground truth template noising (for training)
    transforms.append(
        build_ground_truth_distogram_transform(
            template_noise_scales=template_noise_scales,
            allowed_chain_types_for_conditioning=allowed_chain_types_for_conditioning,
            p_condition_per_token=p_condition_per_token,
            p_provide_inter_molecule_distances=p_provide_inter_molecule_distances,
            is_inference=is_inference,
        )
    )

    # +----------------------------------------------------------------------+
    # +------------------ GROUND TRUTH REFERENCE CONFORMER ------------------+
    # +----------------------------------------------------------------------+

    transforms.append(
        RandomApplyGroundTruthConformerByChainType(
            chain_type_probabilities={
                tuple(ChainType.get_polymers()): p_give_polymer_ref_conf,
                tuple(ChainType.get_non_polymers()): p_give_non_polymer_ref_conf,
            },
            policy=GroundTruthConformerPolicy.ADD,
        )
    )

    transforms += [
        AddGlobalTokenIdAnnotation(allow_overwrite=True),  # required for reference molecule features and TokenToAtomMap
        AddGlobalResIdAnnotation(),
        LoadCachedResidueLevelData(
            dir=Path(residue_cache_dir) if exists(residue_cache_dir) else None,
            sharding_depth=1,
        ),
        RandomSubsampleCachedConformers(n_conformers=1),
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        GetAF3ReferenceMoleculeFeatures(
            conformer_generation_timeout=conformer_generation_timeout,
            use_element_for_atom_names_of_atomized_tokens=use_element_for_atom_names_of_atomized_tokens,
        ),
        FindAutomorphismsWithNetworkX(),  # Adds the  "automorphisms" key to the data dictionary
        ComputeAtomToTokenMap(),
        GetRDKitChiralCenters(),
        AddAF3ChiralFeatures(
            take_first_chiral_subordering=take_first_chiral_subordering
        )
    ]

    transforms += [
        # ... load and pair MSAs
        LoadPolymerMSAs(
            protein_msa_dirs=protein_msa_dirs,
            rna_msa_dirs=rna_msa_dirs,
            max_msa_sequences=max_msa_sequences,  # maximum number of sequences to load (we later subsample further)
            msa_cache_dir=Path(msa_cache_dir) if exists(msa_cache_dir) else None,
            use_paths_in_chain_info=True,  # if there are paths specified in the `chain_info` for a given chain, use them
            raise_if_missing_msa_for_protein_of_length_n=raise_if_missing_msa_for_protein_of_length_n,
        ),
        # collect MSA stats
        AnnotateMSAStatsByChain(),
        PairAndMergePolymerMSAs(
            dense=dense_msa, add_residue_is_paired_feature=add_residue_is_paired_feature
        ),
    ]

    transforms += [
        # ... encode MSA to AF-3 format
        EncodeMSA(
            encoding=af3_sequence_encoding,
            token_to_use_for_gap=af3_sequence_encoding.token_to_idx["<G>"],
        ),
        # ... fill MSA, indexing into only the portions of the polymers that are present in the cropped structure
        FillFullMSAFromEncoded(
            pad_token=af3_sequence_encoding.token_to_idx["<G>"],
            add_residue_is_paired_feature=add_residue_is_paired_feature,
        ),
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: AddAF3TokenBondFeatures(),
                False: AddAF3TokenBondFeatures(),
            },
        ),
    ]

    if add_cyclic_bonds:
        transforms += [
            AddCyclicBonds(),
        ]

    transforms += [
        # ... featurize MSA
        ConvertToTorch(
            keys=[
                "encoded",
                "feats",
                "full_msa_details",
            ]
        ),
        FeaturizeMSALikeAF3(
            encoding=af3_sequence_encoding,
            n_recycles=n_recycles,
            n_msa=n_msa,
        ),
        # Prepare coordinates for noising (without modifying the ground truth)
        # ... add placeholder coordinates for noising
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),
        # ... handling of unresolved residues (note that these Transforms create the "atom_array_to_noise" dictionary, if not already present)
    ]
    if missing_atom_behavior == "nearest":
        transforms += [
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(
                annotation_to_update="coord_to_be_noised"
            ),
            PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(
                annotation_to_update="coord_to_be_noised",
                annotation_to_copy="coord_to_be_noised",
            ),
        ]
    elif missing_atom_behavior == "origin":
        transforms += [
            PlaceUnresolvedTokenAtomsAtOrigin(
                annotation_to_update="coord_to_be_noised"
            ),
        ]
    elif missing_atom_behavior == "trim":
        transforms += [
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(
                annotation_to_update="coord_to_be_noised"
            ),
        ]
    # "rollout": no transforms added
    transforms += [
        # Condition feature aggregation
        AddConditionsAsFeatures(),
        # Feature aggregation
        AggregateFeaturesLikeAF3(),
        # ... batching and noise sampling for diffusion
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        CenterRandomAugmentation(batch_size=diffusion_batch_size),
        SampleEDMNoiseCustom(
            sigma_data=sigma_data,
            diffusion_batch_size=diffusion_batch_size,
            pmean=pmean,
            pstd=pstd,
        ),
        AnnotateTokenRepresentatives(),
        CheckForNaNsInInputs(),
    ]

    confidence_transforms = Compose(
        [
            # Additions required for confidence calculation
            EncodeAtomArray(rf2aa_sequence_encoding),
            AddAtomFrames(),
            AddIsRealAtom(rf2aa_sequence_encoding),
            AddPolymerFrameIndices(),
            # wrap it all together
            PackageConfidenceFeats(),
        ]
    )


    transforms += [
        MaskTokenFeatures(
            mask_prob=masked_token_prob,
            skip_prob=(1.0-masked_token_frac_ex_to_mask),
            mask_token=mask_token,
            sequence_encoding=af3_sequence_encoding,
            provide_diffusion_ref_copy=provide_diffusion_ref_copy,
        )
    ]

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("run_confidence_head", False),
            transform_map={
                True: confidence_transforms,
                False: Identity(),
            },
        )
    )

    keys_to_keep = [
        "example_id",
        "feats",
        "t",
        "noise",
        "ground_truth",
        "coord_atom_lvl_to_be_noised",
        "automorphisms",
        "symmetry_resolution",
        "extra_info",
        "missing_atom_behavior",
        "n_protein_chains_gt50_with_msa",
        "n_protein_chains_gt50_without_msa",
        # Loader puts these at top level (see
        # `create_loader_with_interfaces_and_pn_units_to_score` in
        # atomworks/ml/datasets/loaders/cif.py); RF3DataSample.from_dict
        # then forwards them into ground_truth so the per-interface
        # validation metrics (by_type_lddt, interface_hits) receive them.
        "interfaces_to_score",
        "pn_units_to_score",
    ]

    if run_confidence_head:
        keys_to_keep.append("confidence_feats")

    if return_atom_array:  # and is_inference:
        keys_to_keep.append("atom_array")

    transforms += [
        # Subset to only keys necessary
        SubsetToKeys(keys_to_keep)
    ]

    # ... compose final pipeline
    pipeline = Compose(transforms)

    return pipeline
