#!/usr/bin/env python3

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rf3._config import get_rf3_config_dir

from foundry.utils.logging import suppress_warnings

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
try:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except (FileNotFoundError, OSError):
    # Installed as a wheel: no .project-root marker on disk.
    pass


_config_path = get_rf3_config_dir()


@hydra.main(
    config_path=_config_path,
    config_name="inference",
    version_base="1.3",
)
def run_inference(cfg: DictConfig) -> None:
    """Execute RF3 inference pipeline."""

    # Extract run() parameters from config
    # Preserve string inputs, convert other sequence-like inputs to a Python list (None -> [])
    inputs_param = cfg.inputs if isinstance(cfg.inputs, str) else list(cfg.inputs or [])

    run_params = {
        "inputs": inputs_param,
        "out_dir": str(cfg.out_dir) if cfg.get("out_dir") else None,
        "dump_predictions": cfg.get("dump_predictions", True),
        "dump_trajectories": cfg.get("dump_trajectories", False),
        "one_model_per_file": cfg.get("one_model_per_file", False),
        "annotate_b_factor_with_plddt": cfg.get("annotate_b_factor_with_plddt", False),
        "sharding_pattern": cfg.get("sharding_pattern", None),
        "skip_existing": cfg.get("skip_existing", False),
        "template_selection": cfg.get("template_selection", None),
        "relative_template_selection": cfg.get("relative_template_selection", None),
        "ground_truth_conformer_selection": cfg.get(
            "ground_truth_conformer_selection", None
        ),
        "hotspot_selection": cfg.get("hotspot_selection", None),
        "interface_type": cfg.get("interface_type", None),
        "rasa_selection": cfg.get("rasa_selection", None),
        "hbond_selection": cfg.get("hbond_selection", None),
        "distance_template_selection": cfg.get("distance_template_selection", None),
        # Motif guidance: a sampler-side constraint, not a trained condition. Explicit
        # opt-in via inference_motif_guidance; a selection alone never arms it.
        "motif_guidance_selection": cfg.get("motif_guidance_selection", None),
        "motif_guidance_reference": cfg.get("motif_guidance_reference", None),
        "inference_motif_guidance": cfg.get("inference_motif_guidance", False),
        "inference_motif_guidance_mu": cfg.get("inference_motif_guidance_mu", 1.0),
        "inference_motif_guidance_inner_steps": cfg.get(
            "inference_motif_guidance_inner_steps", 1
        ),
        "inference_motif_guidance_rmsd_threshold": cfg.get(
            "inference_motif_guidance_rmsd_threshold", 5.0
        ),
        "inference_motif_guidance_t_high": cfg.get(
            "inference_motif_guidance_t_high", 0.5
        ),
        "inference_motif_guidance_t_low": cfg.get(
            "inference_motif_guidance_t_low", 0.0
        ),
        "cyclic_chains": cfg.get("cyclic_chains", []),
        "keep_all": cfg.get("keep_all", True),
        # Classifier-free guidance (Phase A: single shared scale over template + GT-conformer)
        "inference_classifier_free_guidance": cfg.get(
            "inference_classifier_free_guidance", False
        ),
        "inference_cfg_scale": cfg.get("inference_cfg_scale", 1.0),
        "inference_cfg_sigma_min": cfg.get("inference_cfg_sigma_min", 0.0),
        "inference_cfg_sigma_max": cfg.get("inference_cfg_sigma_max", None),
        "confidence_streams": cfg.get("confidence_streams", None),
        "ranking_stream": cfg.get("ranking_stream", "merged"),
    }

    # Hydra hands nested overrides through as OmegaConf nodes, but the design-condition
    # parsers in `rf3.utils.inference` branch on plain `dict`/`list` (a DictConfig is
    # neither), so a `{selection: label}` form given on the command line would be taken
    # for a bare scalar. Normalize once here rather than at every branch.
    run_params = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in run_params.items()
    }

    # Create init config with only __init__ params
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_param_keys = set(run_params.keys())
    init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in run_param_keys}
    init_cfg = OmegaConf.create(init_cfg_dict)

    # Instantiate engine (only __init__ params)
    inference_engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)

    # Run inference
    with suppress_warnings(is_inference=True):
        inference_engine.run(**run_params)


if __name__ == "__main__":
    run_inference()
