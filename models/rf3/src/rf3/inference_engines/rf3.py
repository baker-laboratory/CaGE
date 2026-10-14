import json
import logging
import re
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.preprocessing.msa.finding import get_msa_dirs
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from biotite.structure import AtomArray, AtomArrayStack
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from foundry.inference_engines.base import BaseInferenceEngine
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
from rf3.data.collate import collate_batch
from rf3.data.schema.rf3 import RF3DataSample
from rf3.guidance import (
    MotifGuidanceCfg,
    MotifGuidanceEngine,
    build_motif_guidance_feats,
    load_motif_guidance_reference,
)
from rf3.model.RF3 import ShouldEarlyStopFn
from rf3.utils.inference import (
    CFG_CONDITION_FIELDS,
    CONDITION_FIELDS,
    InferenceInput,
    InferenceInputDataset,
    InputOptions,
    cfg_axis_fields_to_keep,
    prepare_inference_inputs_from_paths,
)
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_diffusion_trajectories,
    get_sharded_output_path,
)
from rf3.utils.model_output import normalize_inference_network_output
from rf3.utils.predicted_error import (
    annotate_atom_array_b_factor_with_plddt,
    compile_af3_style_confidence_outputs,
    get_mean_atomwise_plddt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

# Default metrics configuration for RF3 inference (ptm, iptm, clashing chains)
DEFAULT_RF3_METRICS_CFG = {
    "ptm": {"_target_": "rf3.metrics.predicted_error.ComputePTM"},
    "iptm": {"_target_": "rf3.metrics.predicted_error.ComputeIPTM"},
    "count_clashing_chains": {
        "_target_": "rf3.metrics.clashing_chains.CountClashingChains"
    },
}


def dump_json_compact_arrays(obj: dict, f: TextIO) -> None:
    """Dump JSON with indented structure but compact arrays (AF3 style).

    Arrays are written on single lines instead of one element per line.
    """
    # First dump with indent to get structure
    json_str = json.dumps(obj, indent=2)
    # Collapse arrays onto single lines using regex
    # Match arrays that span multiple lines and collapse them
    pattern = re.compile(r"\[\s*\n\s*([^\[\]]*?)\s*\n\s*\]", re.DOTALL)
    while pattern.search(json_str):
        json_str = pattern.sub(
            lambda m: "["
            + ",".join(item.strip() for item in m.group(1).split(","))
            + "]",
            json_str,
        )
    f.write(json_str)


def inject_metrics_into_summary(
    summary_confidences: dict, metrics: dict | None, sample_idx: int
) -> dict:
    """Add ptm / iptm / has_clash / ranking_score for one sample, in place.

    ``metrics`` is a MetricManager result; absent metrics leave the summary untouched
    (so `metrics_cfg=null` runs carry only what the confidence head itself produced).
    """
    if not metrics:
        return summary_confidences

    ptm_val = metrics.get(f"ptm.ptm_{sample_idx}")
    iptm_val = metrics.get(f"iptm.iptm_{sample_idx}")
    has_clash = bool(metrics.get(f"count_clashing_chains.has_clash_{sample_idx}", 0))

    # Convert to native Python floats for JSON serialization
    ptm = float(ptm_val) if ptm_val is not None else None
    iptm = float(iptm_val) if iptm_val is not None else None

    summary_confidences["ptm"] = ptm
    summary_confidences["iptm"] = iptm
    summary_confidences["has_clash"] = has_clash
    # With neither ptm nor iptm there is nothing to rank on -- say so instead of
    # reporting the 0.0 the formula would produce, which reads like a real score.
    summary_confidences["ranking_score"] = (
        None
        if ptm is None and iptm is None
        else round(compute_ranking_score(iptm=iptm, ptm=ptm, has_clash=has_clash), 4)
    )
    return summary_confidences


def compute_ranking_score(
    iptm: float | None,
    ptm: float | None,
    has_clash: bool,
) -> float:
    """Compute ranking score.

    Formula: 0.8 * ipTM + 0.2 * pTM - 100 * has_clash

    For single-chain predictions where ipTM is None, uses pTM only.
    """
    if iptm is None:
        # Single chain - use pTM only
        iptm = ptm if ptm is not None else 0.0
    if ptm is None:
        ptm = 0.0
    return 0.8 * iptm + 0.2 * ptm - 100 * int(has_clash)


@dataclass
class RF3Output:
    """Output container for RF3 predictions, analogous to RFD3Output.

    Stores predicted structures and confidence metrics in AlphaFold3-compatible format.
    """

    example_id: str
    atom_array: AtomArray
    summary_confidences: dict = field(default_factory=dict)
    confidences: dict | None = None
    #: {stream_name: {"summary_confidences": ..., "confidences": ...}} for every stream
    #: beyond "main" (which lives in the two fields above). See `confidence_streams`.
    extra_confidences: dict[str, dict] = field(default_factory=dict)
    #: Which stream's ranking_score orders samples and picks the top-ranked model.
    #: Falls back to "main" when the requested stream was not produced.
    ranking_stream: str = "main"
    sample_idx: int = 0
    seed: int = 0

    @property
    def resolved_ranking_stream(self) -> str:
        """`ranking_stream` if that stream exists for this sample, else "main"."""
        if (
            self.ranking_stream != "main"
            and self.ranking_stream in self.extra_confidences
        ):
            return self.ranking_stream
        return "main"

    @property
    def ranking_score(self) -> float | None:
        """The ranking score from `resolved_ranking_stream`."""
        stream = self.resolved_ranking_stream
        summary = (
            self.summary_confidences
            if stream == "main"
            else self.extra_confidences[stream]["summary_confidences"]
        )
        return summary.get("ranking_score")

    def dump(
        self,
        out_dir: Path,
        file_type: str = "cif",
        dump_full_confidences: bool = True,
    ) -> None:
        """Save output to disk in AlphaFold3-compatible format.

        Args:
            out_dir: Directory to save outputs to.
            file_type: File type for structure output ("cif" or "cif.gz").
            dump_full_confidences: Whether to save full per-atom confidences.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_name = f"{self.example_id}_seed-{self.seed}_sample-{self.sample_idx}"
        base_path = out_dir / sample_name

        # Save structure
        to_cif_file(
            self.atom_array,
            f"{base_path}_model",
            file_type=file_type,
            include_entity_poly=True,
            include_nan_coords=False,
        )

        # Save summary_confidences.json
        with open(f"{base_path}_summary_confidences.json", "w") as f:
            dump_json_compact_arrays(self.summary_confidences, f)

        # Save confidences.json (optional, for full per-atom data)
        if dump_full_confidences and self.confidences:
            with open(f"{base_path}_confidences.json", "w") as f:
                dump_json_compact_arrays(self.confidences, f)

        # Same pair of files per extra confidence stream, suffixed by stream name.
        for stream_name, conf in self.extra_confidences.items():
            with open(f"{base_path}_summary_confidences_{stream_name}.json", "w") as f:
                dump_json_compact_arrays(conf["summary_confidences"], f)
            if dump_full_confidences and conf.get("confidences"):
                with open(f"{base_path}_confidences_{stream_name}.json", "w") as f:
                    dump_json_compact_arrays(conf["confidences"], f)


def dump_ranking_scores(
    outputs: list[RF3Output],
    out_dir: Path,
    example_id: str,
) -> None:
    """Write {example_id}_ranking_scores.csv with ranking scores for all samples."""
    rows = [
        {
            "seed": o.seed,
            "sample": o.sample_idx,
            "ranking_score": o.ranking_score,
            "ranking_stream": o.resolved_ranking_stream,
        }
        for o in outputs
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{example_id}_ranking_scores.csv", index=False)


def dump_top_ranked_outputs(
    outputs: list[RF3Output],
    out_dir: Path,
    example_id: str,
    file_type: str = "cif",
) -> RF3Output:
    """Copy the top-ranked model and summary to the top-level directory.

    Returns the top-ranked RF3Output.
    """
    # Find the output with the highest ranking score
    best_output = max(
        outputs,
        key=lambda o: (
            o.ranking_score if o.ranking_score is not None else float("-inf")
        ),
    )

    # Save top-ranked model at top level
    to_cif_file(
        best_output.atom_array,
        out_dir / f"{example_id}_model",
        file_type=file_type,
        include_entity_poly=True,
        include_nan_coords=False,
    )

    # Save top-ranked summary_confidences at top level
    with open(out_dir / f"{example_id}_summary_confidences.json", "w") as f:
        dump_json_compact_arrays(best_output.summary_confidences, f)

    # Save top-ranked full confidences at top level (if present)
    if best_output.confidences:
        with open(out_dir / f"{example_id}_confidences.json", "w") as f:
            dump_json_compact_arrays(best_output.confidences, f)

    return best_output


#: Confidence streams the engine knows how to score. See ``base.yaml`` for what each means.
CONFIDENCE_STREAMS = ("main", "conditions", "merged")


def _validate_confidence_streams(
    confidence_streams: list[str] | None, cfg_enabled: bool
) -> tuple[str, ...]:
    """Normalize the requested streams; "main" is always scored (it is the primary dump)."""
    requested = tuple(confidence_streams or ("main",))
    unknown = [s for s in requested if s not in CONFIDENCE_STREAMS]
    if unknown:
        raise ValueError(
            f"Unknown confidence stream(s) {unknown}; expected a subset of "
            f"{list(CONFIDENCE_STREAMS)}"
        )
    if not cfg_enabled and set(requested) - {"main"}:
        # Without guidance there are no reference streams and nothing to extrapolate.
        ranked_logger.warning(
            "confidence_streams requested "
            f"{sorted(set(requested) - {'main'})} but classifier-free guidance is off; "
            "only 'main' will be written."
        )
        requested = ("main",)
    return tuple(dict.fromkeys(("main",) + requested))


def should_early_stop_by_mean_plddt(
    threshold: float, is_real_atom: torch.Tensor, max_value_of_plddt: float
) -> ShouldEarlyStopFn:
    """Returns a closure that triggers early stopping when mean pLDDT falls below the specified threshold."""

    def fn(confidence_outputs: dict, **kwargs):
        # `plddt_logits` is predict_plddt(S_trunk_I), i.e. already
        # [B, n_token, NHEAVY * n_bins] -- the 3-D shape get_mean_atomwise_plddt
        # asserts on. It used to be unsqueezed here, which made it 4-D.
        mean_plddt = get_mean_atomwise_plddt(
            plddt_logits=confidence_outputs["plddt_logits"],
            is_real_atom=is_real_atom,
            max_value=max_value_of_plddt,
        )
        return (mean_plddt < threshold).item(), {
            "mean_plddt": mean_plddt.item(),
            "threshold": threshold,
        }

    return fn


class RF3InferenceEngine(BaseInferenceEngine):
    """RF3 inference engine.

    Separates model setup, weight loading, and inference.

    Basic usage (setup model, run inference):
        engine = RF3InferenceEngine(ckpt_path="rf3_latest.pt")
        results = engine.run(inputs="path/to/input.cif")
    """

    def __init__(
        self,
        # Model parameters
        n_recycles: int = 10,
        diffusion_batch_size: int = 5,
        num_steps: int = 50,
        # Templating, MSAs, etc.
        template_noise_scale: float = 1e-5,
        # Compute control
        num_nodes: int = 1,
        devices_per_node: int | list[int] = 1,
        # Output control
        compress_outputs: bool = False,
        # Weight loading control
        load_weights: bool = True,
        # Debug
        print_config: bool = False,
        raise_if_missing_msa_for_protein_of_length_n: int | None = None,
        early_stopping_plddt_threshold: float | None = None,
        # Metrics
        metrics_cfg: dict | OmegaConf | MetricManager | str | None = "default",
        **kwargs,
    ):
        """Initialize inference engine and optionally load model weights.

        Model config is loaded from checkpoint and overridden with parameters provided here.

        Args:
          n_recycles: Number of recycles. Defaults to ``10``.
          diffusion_batch_size: Number of structures to generate per input. Defaults to ``5``.
          num_steps: Number of diffusion steps. Defaults to ``50``.
          template_noise_scale: Noise scale for template coordinates. Defaults to ``1e-5``.
          num_nodes: Number of nodes for distributed inference. Defaults to ``1``.
          devices_per_node: Number of devices per node, or list of device indices. Defaults to ``1``.
          compress_outputs: Whether to gzip output files. Defaults to ``True``.
          load_weights: Whether to load model weights during initialization.
              If False, call load_weights() before first inference. Defaults to ``True``.
          print_config: Whether to print config trees. Defaults to ``False``.
          raise_if_missing_msa_for_protein_of_length_n: Debug flag for MSA checking. Defaults to ``None``.
          early_stopping_plddt_threshold: Stop early if pLDDT below threshold. Defaults to ``None``.
          metrics_cfg: Metrics configuration. Can be:
              - "default" to use standard RF3 metrics (ptm, iptm, clashing chains)
              - dict/OmegaConf with Hydra configs
              - Pre-instantiated MetricManager
              - None (no metrics).
              Defaults to ``"default"``.
          **kwargs: Additional arguments passed to BaseInferenceEngine:
              - ckpt_path (PathLike, required): Path to model checkpoint.
              - model_seeds (list[int] | None): List of random seeds for multi-seed inference.
                If empty or None, uses external RNG state. Multiple seeds will run inference
                for each seed and select best model. Defaults to ``None``.
              - num_nodes (int): Number of nodes for distributed inference. Defaults to ``1``.
              - devices_per_node (int): Number of devices per node. Defaults to ``1``.
              - verbose (bool): If True, show detailed logging and config trees. Defaults to ``False``.
        """
        # Build base transform overrides
        transform_overrides = {
            "diffusion_batch_size": diffusion_batch_size,
            "n_recycles": n_recycles,
            "raise_if_missing_msa_for_protein_of_length_n": raise_if_missing_msa_for_protein_of_length_n,
            "undesired_res_names": [],
            "template_noise_scales": {
                "atomized": template_noise_scale,
                "not_atomized": template_noise_scale,
            },
            "allowed_chain_types_for_conditioning": None,
            "rna_msa_dirs": [],
            # (Paranoia - in validation, these should be set correctly anyhow)
            "p_give_polymer_ref_conf": 0.0,
            "p_give_non_polymer_ref_conf": 0.0,
            "use_element_for_atom_names_of_atomized_tokens": True,
        }

        # Only override protein_msa_dirs if env var is set; otherwise use config defaults
        protein_msa_dirs = get_msa_dirs("PROTEIN_MSA_DIRS", raise_if_not_set=False)
        if protein_msa_dirs:
            transform_overrides["protein_msa_dirs"] = protein_msa_dirs
            ranked_logger.debug(
                f"Using MSA directories from environment variable: {[d['dir'] for d in protein_msa_dirs]}"
            )
        else:
            ranked_logger.debug(
                "PROTEIN_MSA_DIRS env var not set; using config defaults"
            )

        # Map print_config to base class verbose parameter
        kwargs.setdefault("verbose", print_config)

        super().__init__(
            transform_overrides=transform_overrides,
            inference_sampler_overrides={
                "num_timesteps": num_steps,
            },
            **kwargs,
        )

        # Remove loss override — keep loss config from checkpoint
        self.overrides["trainer"].pop("loss", None)

        # Store config for use in initialize()
        self._metrics_cfg = metrics_cfg
        self.early_stopping_plddt_threshold = early_stopping_plddt_threshold
        self.compress_outputs = compress_outputs

    def initialize(self):
        cfg = super().initialize()

        if cfg is not None:
            self.cfg = cfg

            # Set trainer metrics directly based on what was requested
            # This bypasses the OmegaConf merge issue with empty dicts
            if isinstance(self._metrics_cfg, MetricManager):
                self.trainer.metrics = self._metrics_cfg
            elif self._metrics_cfg == "default":
                self.trainer.metrics = MetricManager.instantiate_from_hydra(
                    metrics_cfg=DEFAULT_RF3_METRICS_CFG
                )
            elif self._metrics_cfg is not None:
                self.trainer.metrics = MetricManager.instantiate_from_hydra(
                    metrics_cfg=self._metrics_cfg
                )
            else:
                self.trainer.metrics = None

            # Conf-head checkpoints carry the training rollout cache dir in their
            # embedded train_cfg (typically another user's scratch path). Caching a
            # frozen trunk+diffusion rollout is a training-time optimization keyed
            # only by (base weights, example_id), so it is wrong here. Cleared on the
            # instance rather than via config overrides, which would inject the key
            # into checkpoints whose train_cfg does not carry it.
            self.trainer.val_rollout_cache_dir = None

        return cfg

    def run(
        self,
        inputs: (
            InferenceInput
            | list[InferenceInput]
            | AtomArray
            | list[AtomArray]
            | PathLike
            | list[PathLike]
        ),
        # Output control
        out_dir: PathLike | None = None,
        dump_predictions: bool = True,
        dump_trajectories: bool = False,
        one_model_per_file: bool = False,
        annotate_b_factor_with_plddt: bool = False,
        sharding_pattern: str | None = None,
        skip_existing: bool = False,
        # Selection overrides (applied to all input types)
        template_selection: list[str] | str | None = None,
        relative_template_selection: dict[str, list[str]]
        | list[str]
        | str
        | None = None,
        ground_truth_conformer_selection: list[str] | str | None = None,
        # Design-task conditions
        hotspot_selection: list[str] | str | None = None,
        interface_type: dict[str, str] | str | None = None,
        rasa_selection: dict[str, str] | list[str] | str | None = None,
        hbond_selection: dict[str, str] | list[str] | str | None = None,
        distance_template_selection: dict | list | str | None = None,
        # Motif guidance (sampler-side projection on x0; see rf3.guidance). Explicit
        # opt-in: a selection alone never arms it, so an input that merely carries one
        # still folds unguided and stays comparable to its own baseline.
        motif_guidance_selection: dict | list | str | None = None,
        motif_guidance_reference: list[dict] | dict | str | None = None,
        inference_motif_guidance: bool = False,
        inference_motif_guidance_mu: float = 1.0,
        inference_motif_guidance_inner_steps: int = 1,
        inference_motif_guidance_rmsd_threshold: float = 5.0,
        inference_motif_guidance_t_high: float = 0.5,
        inference_motif_guidance_t_low: float = 0.0,
        cyclic_chains: list[str] = [],
        # Parsing overrides
        keep_all: bool = True,
        inference_classifier_free_guidance: bool = False,
        inference_cfg_scale: float | dict = 1.0,
        # Guidance interval
        inference_cfg_sigma_min: float = 0.0,
        inference_cfg_sigma_max: float | None = None,
        # Which trunk streams to score with the confidence head; see base.yaml
        confidence_streams: list[str] | None = None,
        # Which of them orders samples / picks the top-ranked model
        ranking_stream: str = "merged",
    ) -> dict[str, dict] | None:
        """Run inference on inputs.

        Requires model weights to be loaded. Call load_weights() first if needed.

        Args:
          inputs: Single/list of InferenceInput objects, AtomArray objects, file paths, or directory.
          out_dir: Output directory. If None, returns results as an AtomArray and dictionaries of metrics. Defaults to ``None``.
          dump_predictions: Whether to save predicted structures. Defaults to ``True``.
          dump_trajectories: Whether to save each sample's diffusion trajectory, including the
            per-stream and guided x0 estimates under CFG (see ``dump_diffusion_trajectories``).
            Requires ``out_dir``. Defaults to ``False``.
          one_model_per_file: Save each model in separate file. Defaults to ``False``.
          annotate_b_factor_with_plddt: Write pLDDT to B-factor column. Defaults to ``False``.
          sharding_pattern: Sharding pattern for output organization. Defaults to ``None``.
          skip_existing: Skip inputs with existing outputs. Requires ``out_dir`` to be set. If ``True`` when ``out_dir=None``, a warning is logged and skipping is disabled. Defaults to ``False``.
          template_selection: Template selection override. Defaults to ``None``.
          ground_truth_conformer_selection: Conformer selection override. Defaults to ``None``.
          cyclic_chains: List of chain IDs to cyclize. Defaults to ``[]``.
          keep_all: If True (default), keep waters and crystallization aids in
            parsed structures. If False, remove them. Only applies
            when ``inputs`` are file paths or AtomArrays. Defaults to ``True``.

        Returns:
          If ``out_dir`` is None: Dict mapping example_id to list of RF3Output objects.
          If ``out_dir`` is set: None (results saved to disk).
        """
        # Ensure engine is initialized and weights are loaded
        if not self._weights_loaded:
            self.load_weights()

        self.initialize()

        # Setup output directory if provided
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ranked_logger.info(f"Outputs will be written to {out_dir.resolve()}.")
        if not out_dir:
            ranked_logger.warning(
                "out_dir is None - results will be returned in memory! If you want to save to disk, please provide an out_dir."
            )

        # Validate skip_existing configuration
        if skip_existing and out_dir is None:
            ranked_logger.warning(
                "skip_existing=True requires out_dir to be set. "
                "Disabling skip_existing for in-memory inference mode."
            )
            skip_existing = False

        # Determine file type based on compression setting
        file_type = "cif.gz" if self.compress_outputs else "cif"

        # Bundle selection/parsing overrides into config
        input_options = InputOptions(
            template_selection=template_selection,
            relative_template_selection=relative_template_selection,
            ground_truth_conformer_selection=ground_truth_conformer_selection,
            hotspot_selection=hotspot_selection,
            interface_type=interface_type,
            rasa_selection=rasa_selection,
            hbond_selection=hbond_selection,
            distance_template_selection=distance_template_selection,
            motif_guidance_selection=motif_guidance_selection,
            motif_guidance_reference=motif_guidance_reference,
            keep_all=keep_all,
        )
        motif_guidance_cfg = (
            MotifGuidanceCfg(
                mu=float(inference_motif_guidance_mu),
                inner_steps=int(inference_motif_guidance_inner_steps),
                rmsd_threshold=float(inference_motif_guidance_rmsd_threshold),
                t_threshold_high=float(inference_motif_guidance_t_high),
                t_threshold_low=float(inference_motif_guidance_t_low),
            )
            if inference_motif_guidance
            else None
        )

        # Convert inputs to InferenceInput objects
        if isinstance(inputs, InferenceInput):
            inference_inputs = [inputs]
        elif isinstance(inputs, list) and all(
            isinstance(i, InferenceInput) for i in inputs
        ):
            inference_inputs = inputs
        elif isinstance(inputs, AtomArray):
            # Single AtomArray - convert to InferenceInput
            inference_inputs = [
                InferenceInput.from_atom_array(
                    inputs,
                    input_options=input_options,
                )
            ]
        elif isinstance(inputs, list) and all(isinstance(i, AtomArray) for i in inputs):
            # List of AtomArrays - convert each to InferenceInput
            inference_inputs = [
                InferenceInput.from_atom_array(
                    arr,
                    example_id=f"inference_{i}",
                    input_options=input_options,
                )
                for i, arr in enumerate(inputs)
            ]
        elif isinstance(inputs, (str, Path)) or (
            isinstance(inputs, list) and isinstance(inputs[0], (str, Path))
        ):
            inference_inputs = prepare_inference_inputs_from_paths(
                inputs=inputs,
                existing_outputs_dir=out_dir if skip_existing else None,
                sharding_pattern=sharding_pattern,
                disable_multiprocessing=torch.distributed.is_initialized(),
                input_options=input_options,
            )
        else:
            raise ValueError(f"Unsupported inputs type: {type(inputs)}")

        confidence_streams = _validate_confidence_streams(
            confidence_streams, inference_classifier_free_guidance
        )
        if ranking_stream != "main" and ranking_stream not in confidence_streams:
            # Not an error: the default asks for "merged" and most runs never produce it.
            ranked_logger.info(
                f"ranking_stream={ranking_stream!r} is not among the scored streams "
                f"{list(confidence_streams)}; ranking by 'main'."
            )

        # Flag chains for cyclization if specified
        if cyclic_chains:
            for input_spec in inference_inputs:
                input_spec.cyclic_chains = cyclic_chains

        # make InferenceInputDataset
        inference_dataset = InferenceInputDataset(inference_inputs)
        ranked_logger.info(f"Found {len(inference_dataset)} structures to predict!")

        # make LoadBalancedDistributedSampler
        sampler = LoadBalancedDistributedSampler(
            dataset=inference_dataset,
            key_to_balance=inference_dataset.key_to_balance,
            num_replicas=self.trainer.fabric.world_size,
            rank=self.trainer.fabric.global_rank,
            drop_last=False,
        )

        loader = DataLoader(
            dataset=inference_dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=0,  # multiprocessing is disabled since it shouldn't be hard to read InferenceInput objects
            collate_fn=lambda x: x,  # no collation since we're not batching
            pin_memory=True,
            drop_last=False,
        )

        # Prepare results dict (if returning in-memory)
        results = {} if out_dir is None else None

        # Main inference loop
        for batch_idx, input_spec in enumerate(loader):
            input_spec = input_spec[
                0
            ]  # since we're not batching, the loader returns a list of length 1
            ranked_logger.info(
                f"Predicting structure {batch_idx + 1}/{len(loader)}: {input_spec.example_id}"
            )

            # Register custom residues from this input
            from atomworks.io.utils.ccd import register_custom_residues_from_atom_array

            register_custom_residues_from_atom_array(input_spec.atom_array)

            # Create output directory for this example if saving to disk
            if out_dir:
                example_out_dir = get_sharded_output_path(
                    input_spec.example_id, out_dir, sharding_pattern
                )
                example_out_dir.mkdir(parents=True, exist_ok=True)

            # Run through Transform pipeline (expensive - do once per stream).
            # The pipeline is NOT seed-independent: a ligand with no cached conformer has
            # its reference conformer generated by RDKit off the global numpy RNG, and the
            # chirals feature is read off that conformer's 3D coordinates, so its length
            # changes from call to call. Under CFG each stream is a separate pipeline call,
            # so reset the RNG to the same state before each one. Otherwise the streams
            # differ in every random featurization choice (reference conformer, MSA
            # subsample, template noise, ref_pos augmentation) on top of their
            # conditioning.
            featurization_seed = self.model_seeds[0] if self.model_seeds else 0
            self._seed_for_inference(featurization_seed)
            pipeline_output = self.pipeline(input_spec.to_pipeline_input())

            # `inference_cfg_scale` is a float (one all-conditioning stream, scale =
            # the scalar) or a dict {condition: signed_scale}.
            cfg_condition_fields = CFG_CONDITION_FIELDS
            _condition_fields = list(CONDITION_FIELDS)
            main_pipeline_output = (
                pipeline_output  # non-CFG: main is the conditional input
            )
            cfg_guidance_pipeline_outputs = (
                None  # list of (pipeline_output, scale, name)
            )
            if inference_classifier_free_guidance:
                # One (input_spec, scale, name) per guidance stream: keep only that condition's selection, drop all others.
                guidance_specs = []
                # Conditions named in the dict at scale 0. They are still *claimed*: the
                # main stream drops them like any other guided condition, so scale 0 is
                # the zero point of the guidance axis (delta_uncond) rather than a
                # fallback to the fully conditional input. Tracked separately because
                # 0 * (delta_i - delta_uncond) == 0 makes their stream pointless to build.
                zero_scale_conditions = []
                unselected_conditions = []
                if OmegaConf.is_dict(inference_cfg_scale) or isinstance(
                    inference_cfg_scale, dict
                ):
                    for cond_name, scale in inference_cfg_scale.items():
                        field = cfg_condition_fields.get(cond_name)
                        if field is None:
                            raise ValueError(
                                f"Unknown CFG condition '{cond_name}'; expected one of "
                                f"{list(cfg_condition_fields)}"
                            )
                        if not getattr(input_spec, field):
                            # Nothing selected for this condition. Recorded so the
                            # fallback below can name it: with a single-condition dict
                            # this is the difference between the axis' zero point and
                            # the fully conditional input.
                            unselected_conditions.append(str(cond_name))
                            continue
                        if not scale:
                            zero_scale_conditions.append(str(cond_name))
                            continue
                        # Keep this axis' own field plus any prerequisite whose atoms it
                        # is written onto, so the stream actually carries the condition.
                        keep = cfg_axis_fields_to_keep(input_spec, cond_name, field)
                        spec_i = replace(
                            input_spec, **{f: None for f in _condition_fields}
                        )
                        spec_i = replace(
                            spec_i, **{f: getattr(input_spec, f) for f in keep}
                        )
                        guidance_specs.append((spec_i, float(scale), str(cond_name)))
                elif any(getattr(input_spec, f) for f in _condition_fields):
                    # float scale: a single all-conditioning guidance stream
                    guidance_specs.append(
                        (input_spec, float(inference_cfg_scale), "all")
                    )

                if guidance_specs or zero_scale_conditions:
                    input_spec_uncond = replace(
                        input_spec, **{f: None for f in _condition_fields}
                    )
                    # Same RNG state for every stream (see featurization_seed above), so
                    # the streams differ only in the conditioning they carry.
                    self._seed_for_inference(featurization_seed)
                    main_pipeline_output = self.pipeline(
                        input_spec_uncond.to_pipeline_input()
                    )
                    cfg_guidance_pipeline_outputs = []
                    for spec, scale, name in guidance_specs:
                        self._seed_for_inference(featurization_seed)
                        cfg_guidance_pipeline_outputs.append(
                            (self.pipeline(spec.to_pipeline_input()), scale, name)
                        )
                    if zero_scale_conditions:
                        ranked_logger.info(
                            f"{input_spec.example_id}: {sorted(zero_scale_conditions)} given "
                            "scale 0; dropped from the main stream and contributing no "
                            "guidance. With no other guided condition every condition "
                            "*field* is empty -- note this is not the same input as a "
                            "structure carrying no conditions at all, because annotations "
                            "already written onto the atom_array (notably the residue "
                            "atomization that distance_template_selection applies) are "
                            "part of the topology and survive."
                        )
                else:
                    # Guidance was asked for but no named condition is actually selected
                    # by this input. Downstream that is indistinguishable from a dropped
                    # kwarg, so say it here: the run continues, conditioned but unguided.
                    # Naming the conditions matters -- in a scale sweep this is the
                    # difference between the axis' zero point and the *most* conditioned
                    # run, e.g. a JSON entry that is missing the selection being swept.
                    detail = (
                        f" ({sorted(unselected_conditions)} named but not selected by this "
                        "input)"
                        if unselected_conditions
                        else ""
                    )
                    ranked_logger.warning(
                        f"{input_spec.example_id}: classifier-free guidance requested "
                        f"but no guidance conditions were found{detail}; running the "
                        "fully conditional input without guidance."
                    )

            # Setup early stopping function if configured (seed-independent)
            should_early_stop_fn = None
            if (
                "confidence_feats" in main_pipeline_output
                and self.early_stopping_plddt_threshold
                and self.early_stopping_plddt_threshold > 0
            ):
                # Keyed off the main stream, which under CFG is the unconditional input
                # the model actually rolls out.
                should_early_stop_fn = should_early_stop_by_mean_plddt(
                    self.early_stopping_plddt_threshold,
                    main_pipeline_output["confidence_feats"]["is_real_atom"],
                    self.cfg.trainer.loss.confidence_loss.plddt.max_value,
                )

            # Motif-guidance feats, built once per example and shared across seeds.
            # They MUST come from main_pipeline_output: under CFG that is a second
            # pipeline pass over the unconditional spec, and it is the array whose
            # order is the sampler's L axis. Resolving against `pipeline_output`
            # instead would index a different array and pull the wrong atoms.
            motif_feats = None
            if motif_guidance_cfg is not None:
                if not input_spec.motif_guidance_selection:
                    raise ValueError(
                        f"{input_spec.example_id}: inference_motif_guidance is on but no "
                        "motif_guidance_selection was given"
                    )
                motif_feats = build_motif_guidance_feats(
                    main_pipeline_output["atom_array"],
                    input_spec.motif_guidance_selection,
                    load_motif_guidance_reference(input_spec.motif_guidance_reference),
                )
                ranked_logger.info(
                    f"{input_spec.example_id}: motif guidance over "
                    f"{motif_feats['motif_idx'].numel()} atom(s), "
                    f"{motif_feats['anchor_idx_in_motif'].numel()} anchor(s), "
                    f"groups={motif_feats['group_names']}, "
                    f"mu={motif_guidance_cfg.mu}, "
                    f"window=[{motif_guidance_cfg.t_threshold_low}, "
                    f"{motif_guidance_cfg.t_threshold_high}]"
                )

            # Determine seeds to run (multi-seed support)
            seeds_to_run = self.model_seeds if self.model_seeds else [None]

            # Collect all RF3Output objects across all seeds for this input
            all_rf3_outputs: list[RF3Output] = []
            early_stopped_all_seeds = True  # Track if all seeds early-stopped

            # Inner loop: iterate over seeds
            for seed in seeds_to_run:
                # Seed before each forward pass
                self._seed_for_inference(seed)

                # Fresh engine per seed so the per-sample armed latch cannot leak
                # across trajectories; the print fires only on the arming transition,
                # so a leaked latch would guide an unconverged x0 with no log line.
                motif_guidance = None
                if motif_feats is not None:
                    motif_guidance = MotifGuidanceEngine(
                        motif_guidance_cfg, motif_feats
                    ).to(device=self.trainer.fabric.device)
                    motif_guidance.reset()

                # Model inference
                with torch.no_grad():
                    pipeline_output_device = self.trainer.fabric.to_device(
                        main_pipeline_output
                    )
                    # Collate the (unconditional Δ∅) main example into the batched format expected by validation_step.
                    # pipeline_output_device is kept for
                    # atom_array / confidence access.
                    collated_batch = self.trainer.fabric.to_device(
                        collate_batch([main_pipeline_output], schema=RF3DataSample)
                    )
                    # Classifier-free guidance: collate each guidance stream separately
                    # and pass the list into the non-confidence validation_step.
                    cfg_guidance_batches = None
                    if cfg_guidance_pipeline_outputs:
                        cfg_guidance_batches = [
                            {
                                "batch": self.trainer.fabric.to_device(
                                    collate_batch(
                                        [guidance_output], schema=RF3DataSample
                                    )
                                ),
                                "scale": scale,
                                "name": name,
                            }
                            for guidance_output, scale, name in cfg_guidance_pipeline_outputs
                        ]
                    # Single call site: both kwarg groups always go through. Branching
                    # here used to drop the CFG kwargs whenever early stopping was on,
                    # silently running unguided.
                    valid_step_outs = self.trainer.validation_step(
                        batch=collated_batch,
                        batch_idx=0,
                        compute_metrics=True,
                        should_early_stop_fn=should_early_stop_fn,
                        cfg_guidance_batches=cfg_guidance_batches,
                        inference_classifier_free_guidance=inference_classifier_free_guidance,
                        cfg_sigma_min=inference_cfg_sigma_min,
                        cfg_sigma_max=inference_cfg_sigma_max,
                        motif_guidance=motif_guidance,
                        confidence_streams=confidence_streams,
                    )
                    network_output = normalize_inference_network_output(
                        valid_step_outs["network_output"]
                    )
                    metrics_output = valid_step_outs["metrics_output"]

                # Handle early stopping for this seed
                if network_output.get("early_stopped", False):
                    ranked_logger.warning(
                        f"Early stopping triggered for {input_spec.example_id} (seed={seed}) "
                        f"with mean pLDDT {network_output['mean_plddt']:.2f} < "
                        f"{self.early_stopping_plddt_threshold:.2f}!"
                    )
                    continue  # Skip this seed, try next

                early_stopped_all_seeds = False

                # Build predicted structures for this seed
                atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                    network_output["X_L"], pipeline_output_device["atom_array"]
                )
                # The model places every atom, so occupancy in a prediction is 1.0
                atom_array_stack.set_annotation(
                    "occupancy",
                    np.ones(atom_array_stack.array_length(), dtype=np.float32),
                )
                num_samples = (
                    len(atom_array_stack)
                    if isinstance(atom_array_stack, AtomArrayStack)
                    else 1
                )

                # Build RF3Output objects for each sample (for this seed)
                for sample_idx in range(num_samples):
                    # Get atom array for this sample
                    if isinstance(atom_array_stack, AtomArrayStack):
                        sample_atom_array = atom_array_stack[sample_idx]
                    else:
                        sample_atom_array = atom_array_stack

                    # Compile confidence outputs in AF3 format (if available)
                    summary_confidences = {}
                    confidences = None
                    extra_confidences = {}
                    if "plddt" in network_output:

                        def _compile(plddt_logits, pae_logits, pde_logits):
                            return compile_af3_style_confidence_outputs(
                                plddt_logits=plddt_logits,
                                pae_logits=pae_logits,
                                pde_logits=pde_logits,
                                chain_iid_token_lvl=pipeline_output_device[
                                    "ground_truth"
                                ]["chain_iid_token_lvl"],
                                is_real_atom=pipeline_output_device["confidence_feats"][
                                    "is_real_atom"
                                ],
                                atom_array=pipeline_output_device["atom_array"],
                                confidence_loss_cfg=self.cfg.trainer.loss.confidence_loss,
                                batch_idx=sample_idx,
                            )

                        conf_outs = _compile(
                            network_output["plddt"],
                            network_output["pae"],
                            network_output["pde"],
                        )
                        summary_confidences = conf_outs["summary_confidences"]
                        confidences = conf_outs["confidences"]

                        # Extra trunk streams, scored against the same structure.
                        # Their ptm / iptm / ranking_score are filled in below from the
                        # per-stream metrics the trainer computed.
                        for stream_name, logits in (
                            network_output.get("confidence_by_stream") or {}
                        ).items():
                            stream_outs = _compile(
                                logits["plddt_logits"],
                                logits["pae_logits"],
                                logits["pde_logits"],
                            )
                            stream_summary = stream_outs["summary_confidences"]
                            stream_summary["stream"] = stream_name
                            extra_confidences[stream_name] = {
                                "summary_confidences": stream_summary,
                                "confidences": stream_outs["confidences"],
                            }

                        # Annotate b-factor with pLDDT if requested
                        if annotate_b_factor_with_plddt:
                            atom_array_list = annotate_atom_array_b_factor_with_plddt(
                                atom_array_stack,
                                conf_outs["plddt"],
                                pipeline_output_device["confidence_feats"][
                                    "is_real_atom"
                                ],
                            )
                            sample_atom_array = atom_array_list[sample_idx]

                    # Add metrics (ptm, iptm, has_clash) to summary_confidences --
                    # for the main stream and, from the per-stream metrics the trainer
                    # computed, for every extra stream too.
                    inject_metrics_into_summary(
                        summary_confidences, metrics_output, sample_idx
                    )
                    stream_metrics = (metrics_output or {}).get(
                        "confidence_streams"
                    ) or {}
                    for stream_name, conf in extra_confidences.items():
                        inject_metrics_into_summary(
                            conf["summary_confidences"],
                            stream_metrics.get(stream_name),
                            sample_idx,
                        )

                    all_rf3_outputs.append(
                        RF3Output(
                            example_id=input_spec.example_id,
                            atom_array=sample_atom_array,
                            summary_confidences=summary_confidences,
                            confidences=confidences,
                            extra_confidences=extra_confidences,
                            ranking_stream=ranking_stream,
                            sample_idx=sample_idx,
                            seed=seed if seed is not None else 0,
                        )
                    )

                # Per seed: the trajectories are only held for this seed's network_output
                if out_dir and dump_trajectories:
                    dump_diffusion_trajectories(
                        network_output,
                        atom_array=pipeline_output_device["atom_array"],
                        out_dir=example_out_dir,
                        seed=seed if seed is not None else 0,
                        file_type=file_type,
                    )

            # Handle case where all seeds early-stopped
            if early_stopped_all_seeds:
                if out_dir:
                    # Save early stop info to disk
                    df_to_save = pd.DataFrame([{"early_stopped": True}])
                    df_to_save.to_csv(example_out_dir / "score.csv", index=False)
                else:
                    # Store in results dict
                    results[input_spec.example_id] = {
                        "early_stopped": True,
                    }
                continue

            # Use all_rf3_outputs for saving/returning (collected across all seeds)
            rf3_outputs = all_rf3_outputs

            # Save or return results
            if out_dir:
                # Save to disk in AlphaFold3-style directory structure
                # Top-level: ranking_scores.csv, best model, best summary
                dump_ranking_scores(rf3_outputs, example_out_dir, input_spec.example_id)
                dump_top_ranked_outputs(
                    rf3_outputs,
                    example_out_dir,
                    input_spec.example_id,
                    file_type=file_type,
                )

                # Per-sample subdirectories
                if dump_predictions:
                    for rf3_out in rf3_outputs:
                        sample_subdir = (
                            example_out_dir
                            / f"seed-{rf3_out.seed}_sample-{rf3_out.sample_idx}"
                        )
                        rf3_out.dump(
                            out_dir=sample_subdir,
                            file_type=file_type,
                            dump_full_confidences=True,
                        )

                ranked_logger.info(
                    f"Outputs for {input_spec.example_id} written to {example_out_dir}!"
                )
            else:
                # Store in memory - return list of RF3Output objects
                results[input_spec.example_id] = rf3_outputs

        # merge results across ranks (only if RF3 launched its own distributed context)
        if not self._external_distributed_context:
            self.trainer.fabric.barrier()
            if results is not None and dist.is_initialized():
                gathered_results = [None] * self.trainer.fabric.world_size
                dist.all_gather_object(
                    gathered_results, results
                )  # returns a list of dicts, need to combine them
                gathered_results = {
                    k: v for result in gathered_results for k, v in result.items()
                }  # combine the dicts into a single dict
                results = gathered_results

        return results
