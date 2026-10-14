#!/usr/bin/env python3

import logging
import os

import hydra
import pathlib

import rootutils
from omegaconf import DictConfig

from foundry.utils.logging import suppress_warnings

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
try:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except (FileNotFoundError, OSError):
    # Installed as a wheel: no .project-root marker on disk.
    pass

def _rf3_configs_dir() -> str:
    """rf3 configs, whether installed beside the package or in the source tree."""
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here / "configs", here.parents[1] / "configs"):
        if candidate.is_dir():
            return str(candidate)
    raise FileNotFoundError("rf3 configs not found next to the package or in the source tree")


_config_path = _rf3_configs_dir()

_spawning_process_logger = logging.getLogger(__name__)


@hydra.main(config_path=_config_path, config_name="validate", version_base="1.3")
def validate(cfg: DictConfig) -> None:
    # ==============================================================================
    # Import dependencies and resolve Hydra configuration
    # ==============================================================================

    _spawning_process_logger.info("Importing dependencies...")

    # Lazy imports to make config generation fast
    from functools import partial

    import torch
    from lightning.fabric import seed_everything
    from lightning.fabric.loggers import Logger

    # If training on L40, set precision of matrix multiplication to balance speed and accuracy
    # Reference: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    from foundry.callbacks.callback import BaseCallback  # noqa
    from foundry.utils.instantiators import instantiate_loggers, instantiate_callbacks  # noqa
    from foundry.utils.logging import (
        print_config_tree,
        log_hyperparameters_with_all_loggers,
    )  # noqa
    from foundry.utils.ddp import RankedLogger, set_accelerator_based_on_availability  # noqa
    from foundry.utils.ddp import is_rank_zero  # noqa
    from foundry.utils.datasets import assemble_val_loader_dict  # noqa
    from rf3.data.collate import collate_batch  # noqa

    set_accelerator_based_on_availability(cfg)

    ranked_logger = RankedLogger(__name__, rank_zero_only=True)
    _spawning_process_logger.info("Completed dependency imports ...")

    # ... print the configuration tree (NOTE: Only prints for rank 0)
    print_config_tree(cfg, resolve=True)

    # ==============================================================================
    # Logging and Callback instantiation
    # ==============================================================================

    # Reduce the logging level for all dataset and sampler loggers (unless rank 0)
    # We will still see messages from Rank 0; they are identical, since all ranks load and sample from the same datasets
    if not is_rank_zero():
        dataset_logger = logging.getLogger("datasets")
        sampler_logger = logging.getLogger("atomworks.ml.samplers")
        dataset_logger.setLevel(logging.WARNING)
        sampler_logger.setLevel(logging.ERROR)

    # ... seed everything (NOTE: By setting `workers=True`, we ensure that the dataloaders are seeded as well)
    # (`PL_GLOBAL_SEED` environment varaible will be passed to the spawned subprocessed; e.g., through `ddp_spawn` backend)
    if cfg.get("seed"):
        ranked_logger.info(f"Seeding everything with seed={cfg.seed}...")
        seed_everything(cfg.seed, verbose=True)
    else:
        ranked_logger.warning("No seed provided - Not seeding anything!")

    ranked_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    ranked_logger.info("Instantiating callbacks...")
    callbacks: list[BaseCallback] = instantiate_callbacks(cfg.get("callbacks"))

    # ==============================================================================
    # Trainer and model instantiation
    # ==============================================================================

    # ... instantiate the trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=loggers or None,
        callbacks=callbacks or None,
        _convert_="partial",
        _recursive_=False,
    )
    # (Store the Hydra configuration in the trainer state)
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})

    # ... spawn processes for distributed training
    # (We spawn here, rather than within `fit`, so we can use Fabric's `init_module` to efficiently initialize the model on the appropriate device)
    ranked_logger.info(
        f"Spawning {trainer.fabric.world_size} processes from {trainer.fabric.global_rank}..."
    )
    trainer.fabric.launch()

    # ... construct the model
    trainer.construct_model()

    # ==============================================================================
    # Dataset instantiation
    # ==============================================================================

    # Compose the validation loader(s) — must use the same collate as train.py
    # so each batch is a PaddedBatchedDataSample (with `data_samples` and
    # `padding_masks` fields), not a raw list of dicts.
    val_loaders = assemble_val_loader_dict(
        cfg=cfg.datasets.val,
        rank=trainer.fabric.global_rank,
        world_size=trainer.fabric.world_size,
        loader_cfg=cfg.dataloader["val"],
        collate_fn=partial(
            collate_batch,
            schema=hydra.utils.get_class(cfg.datasets.input_feature_schema),
            fixed_dimensions=None,
        ),
    )

    # Eagerly init the wandb run (and any other Fabric loggers) so the run shows
    # up immediately rather than only after `on_validation_epoch_end` flushes
    # metrics — useful for long val passes that may be killed by SLURM before
    # finishing the epoch.
    ranked_logger.info("Logging hyperparameters...")
    log_hyperparameters_with_all_loggers(
        trainer=trainer, cfg=cfg, model=trainer.state["model"]
    )

    # ... load the checkpoint configuration, regardless of whether it's a path or a config.
    # `None` is allowed and means "validate the freshly initialized model as-is" (useful for
    # debugging the val pipeline / metric registration without depending on a checkpoint).
    ckpt_path = None
    if "ckpt_path" in cfg and cfg.ckpt_path:
        ckpt_path = cfg.ckpt_path
    elif "ckpt_config" in cfg and cfg.ckpt_config:
        assert (
            "path" in cfg.ckpt_config
        ), "No checkpoint path provided in `ckpt_config`!"
        ckpt_path = cfg.ckpt_config.path

    # Optional second checkpoint layered on top of `ckpt_path` -- for trunk-annealed
    # confidence_head_only checkpoints (see
    # confidence_head_training/trunk_anneal/README.md), which never contain trunk weights:
    # `ckpt_path` should point at the matching trunk checkpoint, and this at the
    # confidence_head_only checkpoint to overlay on top. Unset (None) for every other eval.
    ckpt_path_overlay = cfg.get("ckpt_path_overlay")

    # ... validate the model
    ranked_logger.info("Validating model...")
    with suppress_warnings():
        trainer.validate(
            val_loaders=val_loaders,
            ckpt_path=ckpt_path,
            ckpt_path_overlay=ckpt_path_overlay,
        )

    ranked_logger.info("Validation complete!")


if __name__ == "__main__":
    validate()
