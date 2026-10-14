import hydra
import torch
from dataclasses import asdict, is_dataclass
from beartype.typing import Any
from einops import repeat
from jaxtyping import Float, Int
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig
from pathlib import Path
from rf3.loss.af3_losses import (
    ResidueSymmetryResolution,
    SubunitSymmetryResolution,
)
from rf3.model.RF3 import ShouldEarlyStopFn
from rf3.model.rollout_cache import compute_base_fingerprint
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords
from rf3.utils.recycling import get_recycle_schedule

from foundry.common import exists
from foundry.metrics.losses import Loss
from foundry.metrics.metric import MetricManager
from foundry.trainers.fabric import FabricTrainer
from foundry.training.EMA import EMA
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans, assert_same_shape
from foundry.utils.weights import WeightLoadingConfig, WeightLoadingPolicy

#torch.autograd.set_detect_anomaly(True)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _remap_outputs(
    xyz: Float[torch.Tensor, "D L 3"], mapping: Int[torch.Tensor, "D L"]
) -> Float[torch.Tensor, "D L 3"]:
    """Helper function to remap outputs using a mapping tensor."""
    for i in range(xyz.shape[0]):
        xyz[i, mapping[i]] = xyz[i].clone()
    return xyz


def _batch_to_example(batch: Any) -> dict:
    """Normalize a batch into the example dict expected downstream.

    Handles three forms produced by different collators: a plain dict,
    a dataclass instance, or a list/tuple of length 1 (from the default
    `lambda x: x` collate_fn used by the val loader).
    """
    if isinstance(batch, dict):
        return batch
    if is_dataclass(batch):
        return asdict(batch)
    if isinstance(batch, (list, tuple)) and len(batch) == 1:
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


class RF3Trainer(FabricTrainer):
    """Standard Trainer for AF3-style models"""

    def __init__(
        self,
        *,
        n_recycles_train: int | None = None,
        symmetry_resolution_in_training: bool = False,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | MetricManager | None = None,
        use_torch_compile: bool = False,
        seed=None,  # dumped
        val_rollout_cache_dir: str | None = None,
        mock_fixed_batch: bool = False,
        **kwargs,
    ):
        """See `FabricTrainer` for the additional initialization arguments.

        Args:
            n_recycles_train: Maximum number of recycles (per-batch), for models that support recycling. During training, the model will be recycled a
                random number of times between 1 and `n_recycles_train`. During inference, we determine the number of recycles from the MSA stack shape. However,
                for training, we must sample the number of recycles upfront, so all GPUs within a distributed batch can sample the same number of recycles.
            loss: Configuration for the loss function. If None, the loss function will not be instantiated.
            metrics: Metrics configuration. Can be:
                - DictConfig/dict with Hydra configs (instantiated internally)
                - Pre-instantiated MetricManager
                - None (no metrics)
            val_rollout_cache_dir: Directory for confidence-head validation rollout cache
                (see :mod:`rf3.model.rollout_cache`). ``None`` disables caching.
            mock_fixed_batch: Debug-only. When True, the raw dataloader batch is cached on
                the first training step and reused verbatim on every subsequent step,
                instead of consuming whatever the dataloader/transform pipeline produces
                that step. Bypasses ALL per-step stochastic data augmentation (crop,
                mirror/atomization probabilities, symmetry-ambiguous frame/atom
                reassignment, MSA/template subsampling), giving a perfectly stationary
                training example -- pair with the model's ``mock_fixed_diffusion_rollout``
                for a fully frozen target end to end.
        """
        super().__init__(**kwargs)

        # torch.compile configuration
        self.use_torch_compile = use_torch_compile

        # Confidence-head validation rollout cache; None -> always recompute rollouts.
        self.val_rollout_cache_dir = val_rollout_cache_dir

        # Debug-only: freeze the raw training batch across steps (see docstring above).
        self.mock_fixed_batch = mock_fixed_batch
        self._mock_batch_cache = None

        # (Initialize recycle schedule upfront so all GPU's can sample the same number of recycles within a batch)
        self.symmetry_resolution_in_training = symmetry_resolution_in_training
        self.n_recycles_train = n_recycles_train
        self.recycle_schedule = get_recycle_schedule(
            max_cycle=n_recycles_train,
            n_epochs=self.max_epochs,  # Set by FabricTrainer
            n_train=self.n_examples_per_epoch,  # Set by FabricTrainer
            world_size=self.fabric.world_size,
        )  # [n_epochs, n_examples_per_epoch // world_size]

        # Metrics
        if isinstance(metrics, MetricManager):
            # Already instantiated
            self.metrics = metrics
        elif metrics is not None:
            # Hydra config - instantiate
            self.metrics = MetricManager.instantiate_from_hydra(metrics_cfg=metrics)
        else:
            # No metrics
            self.metrics = None

        # Loss
        self.loss = Loss(**loss) if loss else None

        # (Symmetry resolution)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()

    def construct_model(self):
        """Construct the model and optionally wrap with EMA."""
        # ... instantiate model with Hydra and Fabric
        with self.fabric.init_module():
            ranked_logger.info("Instantiating model...")

            model = hydra.utils.instantiate(
                self.state["train_cfg"].model.net,
                _recursive_=False,
            )

            # Optionally, wrap the model with EMA
            if self.state["train_cfg"].model.ema is not None:
                ranked_logger.info("Wrapping model with EMA...")
                model = EMA(model, **self.state["train_cfg"].model.ema)

            # Optionally, compile the model with torch.compile
            if self.use_torch_compile:
                ranked_logger.info("Compiling model with torch.compile...")
                model = torch.compile(model)

        self.initialize_or_update_trainer_state({"model": model})

    def _assemble_network_inputs(self, example: dict) -> dict:
        """Assemble and validate the network inputs."""
        data = example["data_samples"]

        assert_same_shape(
            data["diffusion_features"]["coord_atom_lvl_to_be_noised"],
            data["diffusion_features"]["noise"],
        )

        ##fd 1
        # ground truth, extend on batch dim
        diffusion_batch_size = data["diffusion_features"][
            "coord_atom_lvl_to_be_noised"
        ].shape[1]
        X_gt_L = repeat(
            data["ground_truth"]["coord_atom_lvl"],
            " b l c -> b d l c",
            d=diffusion_batch_size,
        )  # [L, 3] -> [D, L, 3] with broadcasting
        crd_mask_L = repeat(
            data["ground_truth"]["mask_atom_lvl"],
            "b l -> b d l",
            d=diffusion_batch_size,
        )  # [L] -> [D, L] with broadcasting
        ground_truth = {
            "X_gt_L": X_gt_L,  # [B, D, L, 3]
            "crd_mask_L": crd_mask_L,  # [B, D, L]
        }

        # The pipeline stamps `missing_atom_behavior` onto each example and the schema
        # carries it through collation. Required (no default): a regression that drops
        # it must raise loudly rather than silently switching to "nearest".
        mode = data["missing_atom_behavior"]
        if isinstance(mode, (list, tuple)):
            # Collation turns the per-example string into a list[str]. All examples in
            # a batch share the same config-driven mode, so take the first.
            mode = mode[0]

        # now assemble all inputs + gt + symminfo
        network_input = {
            #"X_noisy_L": data["diffusion_features"]["coord_atom_lvl_to_be_noised"]
            #+ data["diffusion_features"]["noise"],
            "X_unnoised_L": data["diffusion_features"]["coord_atom_lvl_to_be_noised"],
            "noise_L": data["diffusion_features"]["noise"],
            "t": data["diffusion_features"]["t"],
            "f": data["features"],
            "symmetry_resolution": data["symmetry_resolution"],
            "ground_truth": ground_truth,
            "missing_atom_behavior": mode,
        }

        # In rollout mode, X_unnoised_L intentionally contains NaNs for unresolved atoms
        # (they are filled later by the model using the mini-rollout prediction), so skip
        # this check entirely — nan_to_num here would corrupt the fill logic downstream.
        if network_input["missing_atom_behavior"] != "rollout":
            try:
                assert_no_nans(
                    network_input["X_unnoised_L"],
                    msg=f"network_input (X_unnoised_L) in example_ids: {data['example_id']}",
                )
            except AssertionError as e:
                if self.state["model"].training:
                    # In some cases, we may indeed have NaNs in the the noisy coordinates; we can safely replace them with zeros,
                    # and begin noising of those coordinates (which will not have their loss computed) from the origin.
                    # Such a situation could occur if there was a chain in the crop with no resolved residues (but that contained resolved
                    # residues outside the crop); we then would not be able to resolve the missing coordinates to their "closest resolved neighbor"
                    # within the same chain.
                    network_input["X_unnoised_L"] = torch.nan_to_num(
                        network_input["X_unnoised_L"]
                    )
                    ranked_logger.warning(str(e))
                else:
                    # During validation, since we do not crop, there should be no NaN's in the coordinates to noise
                    # (They were either removed, as is done with fully unresolved chains, or resolved accoring to our pipeline's rules)
                    raise e

        assert_no_nans(
            network_input["f"],
            msg=f"NaN detected in `feats` in example_ids: {data['example_id']}",
        )

        return network_input

    def _assemble_loss_extra_info(self, example: dict, network_output: dict) -> dict:
        """Assembles metadata arguments to the loss function (incremental to the network inputs and outputs)."""
        # grab the realigned natives from network output (based on rollout)
        B,D = network_output['X_L'].shape[:2]

        if 'X_gt_align_L' in network_output and network_output['X_gt_align_L'] is not None:
            atom2token = example["symmetry_resolution"]["token_to_repatom_index"]

            # rollout
            X_gt_align_L = network_output['X_gt_align_L'].expand(-1,D,-1,-1)
            mask_gt_align_L = network_output['mask_gt_align_L'].expand(-1,D,-1)
            
            # update token-level native features (used for disto loss)
            Bidx = torch.arange(B, device=network_output['X_gt_align_L'].device)[:, None]
            coord_token_lvl = network_output['X_gt_align_L'][Bidx, 0, atom2token, : ]
            mask_token_lvl = network_output['mask_gt_align_L'][Bidx, 0, atom2token]

            # padding
            coord_token_lvl[atom2token<0] = 0.0
            mask_token_lvl[atom2token<0] = False
        else:
            # no rollout
            X_gt_align_L = example["ground_truth"]["coord_atom_lvl"][:,None].expand(-1,D,-1,-1)
            mask_gt_align_L = example["ground_truth"]["mask_atom_lvl"][:,None].expand(-1,D,-1)

            # update token-level native features (used for disto loss)
            coord_token_lvl = example["ground_truth"]["coord_token_lvl"]
            mask_token_lvl = example["ground_truth"]["mask_token_lvl"]


        X_gt_align_L_old = example["ground_truth"]["coord_atom_lvl"][:,None].expand(-1,D,-1,-1)
        mask_gt_align_L_old = example["ground_truth"]["mask_atom_lvl"][:,None].expand(-1,D,-1)
        coord_token_lvl_old = example["ground_truth"]["coord_token_lvl"]
        mask_token_lvl_old = example["ground_truth"]["mask_token_lvl"]

        loss_extra_info = {
            "X_gt_L": X_gt_align_L,
            "crd_mask_L": mask_gt_align_L,
            "coord_token_lvl": coord_token_lvl,
            "mask_token_lvl": mask_token_lvl,
            "true_token_ids": example["ground_truth"]["true_token_ids"],
            "masked_tokens": example["ground_truth"]["masked_tokens"],
        }
        return loss_extra_info


    def _assemble_metrics_extra_info(self, data: dict, network_output: dict) -> dict:
        """Prepares the extra info for the metrics"""
        # We need the same information as for the loss...
        metrics_extra_info = self._assemble_loss_extra_info(data, network_output)

        # ... and possibly some additional metadata from the example dictionary
        # TODO: Generalize, so we always use the `extra_info` key, rather than unpacking the ground truth as well
        metrics_extra_info.update(
            {
                # TODO: Remove, instead using `extra_info` for all keys
                **{
                    k: data["ground_truth"][k]
                    for k in [
                        "interfaces_to_score",
                        "pn_units_to_score",
                        "chain_iid_token_lvl",
                        "true_token_ids",
                        "masked_tokens",
                    ]
                    if k in data["ground_truth"]
                },
                "example_id": data[
                    "example_id"
                ],  # We require the example ID for logging
                # (From the parser)
                **data.get("extra_info", {}),
            }
        )

        # Record metrics_tags for this example
        metrics_extra_info["metrics_tags"] = data.get("metrics_tags", set())

        # (Create a shallow copy to avoid modifying the original dictionary)
        return {**metrics_extra_info}

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Training step, running forward and backward passes.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            is_accumulating: Whether we are accumulating gradients (i.e., not yet calling optimizer.step()).
                If this is the case, we should skip the synchronization during the backward pass.

        Returns:
            None; we call `loss.backward()` directly, and store the outputs in `self._current_train_return`.
        """
        model = self.state["model"]
        assert model.training, "Model must be training!"

        # Recycling
        # (Number of recycles for the current batch; shared across all GPUs within a distributed batch)
        n_cycle = self.recycle_schedule[self.state["current_epoch"], batch_idx].item()

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # (We assume batch size of 1 for structure predictions)
            example = _batch_to_example(batch)

            network_input = self._assemble_network_inputs(example)
            data = example["data_samples"]
            padding_masks = example["padding_masks"]

            network_output = model.forward(
                input=network_input, padding_masks=padding_masks, n_cycle=n_cycle,
                # needed for rollout
                coord_atom_lvl_to_be_noised=data["diffusion_features"][
                    "coord_atom_lvl_to_be_noised"
                ],
                tag=example["data_samples"]['example_id']
            )

            # process realigned native to use in loss calcs
            loss_extra_info = self._assemble_loss_extra_info(data, network_output)

            total_loss, loss_dict_batched = self.loss(
                network_input=network_input,
                network_output=network_output,
                loss_input=loss_extra_info,
            )

            # Backward pass
            self.fabric.backward(total_loss)

            # ... store the outputs without gradients for use in logging, callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                {
                    "total_loss": total_loss,
                    "loss_dict": loss_dict_batched,
                    "n_symm_res_failures": int(network_output.get("n_symm_res_failures", 0)),
                },
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
        # cfg_guidance_batches is a list of {"batch": collated per-condition example, "scale": float}
        # the main `batch` is the unconditional Δ∅.
        cfg_guidance_batches: list | None = None,
        inference_classifier_free_guidance: bool = False,
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        motif_guidance: Any = None,
        should_early_stop_fn: ShouldEarlyStopFn | None = None,
        confidence_streams: tuple[str, ...] | list[str] | None = None,
    ) -> dict:
        """Validation step, running forward pass and computing validation metrics.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            compute_metrics: Whether to compute metrics. If False, we will not compute metrics, and the output will be None.
                Set to False during the inference pipeline, where we need the network output but cannot compute metrics (since we
                do not have the ground truth).

        Returns:
            dict: Output dictionary containing the validation metrics and network output.
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        # Early stopping is decided from predicted pLDDT, so it only exists on the
        # confidence path. Fail loudly rather than silently ignoring the request.
        if should_early_stop_fn is not None:
            raise NotImplementedError(
                f"{type(self).__name__} has no confidence head and cannot honour "
                "should_early_stop_fn; use RF3TrainerWithConfidence or set "
                "early_stopping_plddt_threshold=0."
            )
        if confidence_streams is not None and set(confidence_streams) - {"main"}:
            raise NotImplementedError(
                f"{type(self).__name__} has no confidence head; confidence_streams="
                f"{list(confidence_streams)} cannot be produced."
            )

        example = _batch_to_example(batch)
        data = example["data_samples"]
        padding_masks = example["padding_masks"]

        network_input = self._assemble_network_inputs(example)

        # Classifier-free guidance: build one guidance input per collated per-condition
        # batch supplied by the inference engine, carrying its guidance scale.
        cfg_guidance_inputs = None
        if inference_classifier_free_guidance and cfg_guidance_batches:
            cfg_guidance_inputs = []
            for ref in cfg_guidance_batches:
                ref_example = _batch_to_example(ref["batch"])
                cfg_guidance_inputs.append(
                    {
                        "f": ref_example["data_samples"]["features"],
                        # Each stream was collated on its own, so it has its own padding
                        # lengths. Carry them: reusing the main stream's silently
                        # mismatches features whose length is example-dependent
                        # (num_chiral_centers).
                        "padding_masks": ref_example["padding_masks"],
                        "scale": ref["scale"],
                    }
                )

        # ... forward pass (with rollout)
        # (Note that forward() passes to the EMA/shadow model if the model is not training)
        network_output = model.forward(
            input=network_input,
            padding_masks=padding_masks,
            n_cycle=data["features"]["msa_stack"].shape[
                1
            ],  # Determine the number of recycles from the MSA stack shape, first dimension after batch
            coord_atom_lvl_to_be_noised=data["diffusion_features"][
                "coord_atom_lvl_to_be_noised"
            ],
            cfg_guidance_inputs=cfg_guidance_inputs,
            inference_classifier_free_guidance=inference_classifier_free_guidance,
            cfg_sigma_min=cfg_sigma_min,
            cfg_sigma_max=cfg_sigma_max,
            motif_guidance=motif_guidance,
        )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {''.join(data['example_id'])}",
        )

        metrics_output = {}
        if compute_metrics and exists(self.metrics):
            # symmetry resolution
            x_native_aln, x_native_mask = self.subunit_symm_resolve(
                data["symmetry_resolution"]["molecule_entity"], # mol_entities
                data["symmetry_resolution"]["molecule_iid"],    # mol_iid
                data["symmetry_resolution"]["crop_mask"],       # mask_crop
                data["symmetry_resolution"]["coord_atom_lvl"],  # x_native
                data["symmetry_resolution"]["mask_atom_lvl"],   # mask_native
                network_output["X_L"], # x_pred
                padding_masks["L"],
            )
            x_native_aln, x_native_mask = self.residue_symm_resolve(
                network_output["X_L"], 
                x_native_aln, 
                x_native_mask, 
                data["symmetry_resolution"]["automorphisms"]
            )

            # add symm-resolved native to network out
            network_output['X_gt_align_L'] = x_native_aln
            network_output['mask_gt_align_L'] = x_native_mask

            metrics_extra_info = self._assemble_metrics_extra_info(data, network_output)

            # TODO: enable this to work with batch size > 1
            metrics_output = self.metrics(
                network_input=network_input,
                network_output=network_output,
                extra_info=metrics_extra_info,
                # (Uses the permuted ground truth after symmetry resolution)
                ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    metrics_extra_info["X_gt_L"][0], data["atom_array"][0]
                ),
                predicted_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    network_output["X_L"][0], data["atom_array"][0]
                ),
            )

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        network_output = apply_to_collection(
            network_output, torch.Tensor, lambda x: x.detach()
        )

        return {"metrics_output": metrics_output, "network_output": network_output}


class RF3TrainerWithConfidence(RF3Trainer):
    """AF-3 trainer with rollout and confidence prediction"""

    def __init__(
        self,
        *,
        train_pae_head: bool = True,
        confidence_head_only_checkpoints: bool = False,
        **kwargs,
    ):
        """See `RF3Trainer` for the additional initialization arguments.

        Args:
            train_pae_head: If False, also freeze the confidence head's PAE output
                projection (``predict_pae``), alongside the (always-frozen) trunk.
                The rest of the confidence head (pLDDT/PDE/resolved + the shared
                confidence Pairformer stack) still trains normally. Combine with
                ``trainer.loss.confidence_loss.pae.weight: 0`` so no PAE gradient
                reaches the shared confidence-head layers either.
            confidence_head_only_checkpoints: If True, ``save_checkpoint`` persists
                only ``*confidence_head*`` weights (~30x smaller; the trunk/diffusion
                weights are frozen and redundant with whatever base checkpoint is
                loaded via ``ckpt_config`` or trunk annealing). Optimizer/scheduler
                state is saved in full, but is already confidence_head-only in size
                since those are the only trainable parameters.
        """
        self.train_pae_head = train_pae_head
        self.confidence_head_only_checkpoints = confidence_head_only_checkpoints
        super().__init__(**kwargs)

    def save_checkpoint(self) -> None:
        if not self.confidence_head_only_checkpoints:
            super().save_checkpoint()
            return

        if not self.output_dir:
            ranked_logger.warning(
                "No output directory specified; skipping model checkpointing of state dictionary."
            )
            return

        self.fabric.call("on_save_checkpoint", trainer=self, state=self.state)
        checkpoint_file = (
            self.output_dir / "ckpt" / f"epoch-{self.state['current_epoch']:04d}.ckpt"
        )
        self.fabric.save(
            checkpoint_file,
            self.state,
            filter={"model": lambda name, _param: "confidence_head" in name},
        )
        ranked_logger.info(
            f"Saved confidence_head-only checkpoint to: {checkpoint_file}"
        )

    def construct_model(self):
        super().construct_model()
        self._freeze_non_confidence_parameters()

    def _unwrap_module(self, module):
        """Unwrap Fabric / DDP wrappers to the underlying nn.Module."""
        while hasattr(module, "module"):
            module = module.module
        return module

    def _freeze_non_confidence_parameters(self) -> None:
        """Train only the online confidence head; freeze trunk + EMA shadow.

        When ``self.train_pae_head`` is False, the PAE output projection
        (``predict_pae``) is frozen too.
        """
        for name, param in self.state["model"].named_parameters():
            name_parts = name.split(".")
            # EMA exposes ``model.*`` (online) and ``shadow.*``. Only online
            # confidence_head should receive grads; shadow is updated via EMA.update().
            is_online_confidence = (
                "confidence_head" in name and "shadow" not in name_parts
            )
            is_frozen_pae = not self.train_pae_head and "predict_pae" in name_parts
            param.requires_grad = is_online_confidence and not is_frozen_pae

        n_train = sum(
            p.numel() for p in self.state["model"].parameters() if p.requires_grad
        )
        ranked_logger.info(
            f"Frozen non-confidence params; {n_train:,} trainable confidence_head parameters "
            f"(train_pae_head={self.train_pae_head})."
        )

    def trunk_fingerprint(self) -> str:
        """Fingerprint of the current frozen trunk+diffusion weights (excludes confidence_head).

        See :func:`rf3.model.rollout_cache.compute_base_fingerprint`: the same fingerprint
        already used to namespace the validation rollout cache, reused here as the
        authoritative "did the trunk actually change" signal for trunk annealing.
        """
        return compute_base_fingerprint(self.state["model"])

    def load_trunk_only_checkpoint(self, checkpoint: Path | str) -> str:
        """Swap in a new trunk (non-confidence-head) checkpoint mid-run.

        Used for confidence-head "trunk annealing" (see
        ``rf3.callbacks.trunk_anneal.TrunkAnnealCallback``): periodically cycles the
        frozen trunk through a sequence of upstream structure-training checkpoints
        while the confidence head keeps training continuously against them. Unlike
        :meth:`load_checkpoint`, this never touches the confidence head (online or
        EMA shadow), optimizer/scheduler state, or the epoch/step counters -- only
        trunk tensors are copied in.

        Returns:
            The trunk fingerprint (see :meth:`trunk_fingerprint`) after loading, so
            callers can log/verify it without recomputing.

        Raises:
            RuntimeError: if the fingerprint is unchanged after loading -- this means
                the swap silently had no effect (e.g. a weight-loading pattern/policy
                bug), which would otherwise make an "annealed" run indistinguishable
                from hours of training against one static trunk without ever raising
                an error.
        """
        fingerprint_before = self.trunk_fingerprint()
        ranked_logger.info(
            f"[trunk-anneal] Loading trunk weights from: {checkpoint} "
            f"(fingerprint before: {fingerprint_before})"
        )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weight_loading_config = WeightLoadingConfig(
            param_policies={"*confidence_head*": WeightLoadingPolicy.REINIT},
        )
        self._load_model(ckpt, weight_loading_config)
        self._freeze_non_confidence_parameters()

        fingerprint_after = self.trunk_fingerprint()
        if fingerprint_after == fingerprint_before:
            raise RuntimeError(
                f"[trunk-anneal] Trunk fingerprint unchanged after loading {checkpoint} "
                f"(still {fingerprint_after}). The swap silently had no effect; refusing "
                "to continue training as if annealing were happening."
            )
        ranked_logger.info(f"[trunk-anneal] New trunk fingerprint: {fingerprint_after}")
        return fingerprint_after

    def _initialize_and_sync_confidence_head_ema(self) -> None:
        """Re-init online confidence_head and copy it into the EMA shadow.

        Structure-only checkpoints (e.g. run14) have no ``*confidence_head*`` keys.
        Weight loading keeps construction-time tensors via REINIT, but the online and
        shadow copies can drift if policies apply unevenly. Explicitly reset the online
        head and seed EMA from it so ``EMA.update()`` tracks a known shared start.
        """
        root = self._unwrap_module(self.state["model"])
        if not isinstance(root, EMA):
            ranked_logger.warning(
                "Model is not EMA-wrapped; skipping confidence_head EMA sync."
            )
            online = root
            if hasattr(online, "confidence_head"):
                online.confidence_head.reset_parameters()
                ranked_logger.info("Re-initialized online confidence_head (no EMA).")
            return

        online = root.model
        shadow = root.shadow
        if not hasattr(online, "confidence_head"):
            ranked_logger.warning("No confidence_head on online model; skip EMA sync.")
            return

        online.confidence_head.reset_parameters()
        shadow.confidence_head.load_state_dict(online.confidence_head.state_dict())
        for p in shadow.confidence_head.parameters():
            p.detach_()
            p.requires_grad = False

        n_online = sum(p.numel() for p in online.confidence_head.parameters())
        ranked_logger.info(
            f"Re-initialized online confidence_head ({n_online:,} params) and "
            "copied into EMA shadow as the starting EMA state."
        )

    @staticmethod
    def _checkpoint_has_confidence_head(checkpoint: Path | str | dict) -> bool:
        """Return True if ``checkpoint`` contains any confidence_head tensors."""
        if isinstance(checkpoint, dict):
            ckpt = checkpoint
        else:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(model_sd, dict):
            return False
        return any("confidence_head" in str(k) for k in model_sd)

    @staticmethod
    def _weight_policy_reinits_confidence_head(weight_loading_config) -> bool:
        """Return True if weight-loading policies explicitly reinit confidence_head."""
        if weight_loading_config is None:
            return False
        policies = getattr(weight_loading_config, "param_policies", None) or {}
        for pattern, policy in policies.items():
            if "confidence_head" not in str(pattern):
                continue
            policy_name = getattr(policy, "value", policy)
            if str(policy_name).lower() == "reinit":
                return True
        return False

    def load_checkpoint(
        self,
        checkpoint: Path | str | dict,
        weight_loading_config=None,
        reset_optimizer: bool = False,
    ) -> None:
        """Load checkpoint; reinit confidence_head only when bootstrapping a fine-tune.

        Structure-only trunks (e.g. run14) need a fresh confidence_head + EMA sync.
        Eval / resume of a confidence fine-tune ckpt must keep the loaded online and
        EMA shadow heads — otherwise ``validate.py`` silently evaluates random init
        (~uniform PAE ≈ 16 Å).
        """
        # Load once here when given a path so we can inspect keys without a
        # second full ``torch.load`` inside ``_checkpoint_has_confidence_head``.
        if isinstance(checkpoint, dict):
            ckpt_for_load: Path | str | dict = checkpoint
        else:
            ranked_logger.info(f"Loading checkpoint from: {checkpoint}...")
            ckpt_for_load = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )

        # Only wipe/re-seed the confidence head when the *checkpoint itself* has no
        # confidence_head tensors (structure-only trunk). Never use reset_optimizer or
        # weight-loading ``reinit`` policies as a signal here — those still apply during
        # ``super().load_checkpoint`` for missing keys, but must not destroy a trained
        # head on resume (Hydra often leaves ``*confidence_head*: reinit`` in the config).
        ckpt_has_confidence = self._checkpoint_has_confidence_head(ckpt_for_load)
        should_reinit_confidence = not ckpt_has_confidence

        if ckpt_has_confidence and self._weight_policy_reinits_confidence_head(
            weight_loading_config
        ):
            ranked_logger.warning(
                "Checkpoint has confidence_head weights but weight_loading_config "
                "still maps *confidence_head* -> reinit; forcing copy so resume/eval "
                "does not wipe the trained head."
            )
            # Mutate a shallow copy of policies so load copies ckpt weights.
            try:
                from copy import deepcopy

                from foundry.utils.weights import WeightLoadingPolicy

                weight_loading_config = deepcopy(weight_loading_config)
                policies = dict(getattr(weight_loading_config, "param_policies", {}) or {})
                for pattern in list(policies):
                    if "confidence_head" in str(pattern):
                        policies[pattern] = WeightLoadingPolicy.COPY
                weight_loading_config.param_policies = policies
                if hasattr(weight_loading_config, "_compiled_patterns"):
                    weight_loading_config._compiled_patterns = (
                        weight_loading_config._compile_patterns(policies)
                    )
            except Exception as e:
                ranked_logger.warning(f"Failed to override conf-head reinit policy: {e}")

        super().load_checkpoint(
            ckpt_for_load,
            weight_loading_config=weight_loading_config,
            reset_optimizer=reset_optimizer,
        )

        if should_reinit_confidence:
            self._initialize_and_sync_confidence_head_ema()
        else:
            ranked_logger.info(
                "Checkpoint already has confidence_head weights; keeping online + "
                "EMA shadow (skip reinit). Eval/resume will use EMA shadow under "
                "model.eval()."
            )

        # Re-apply freeze: load_state_dict does not change requires_grad, but
        # reset_parameters / sync can leave flags inconsistent across wrappers.
        self._freeze_non_confidence_parameters()

        # Structure-only ckpts (run14 etc.) carry a large current_epoch. For
        # confidence-head fine-tuning we always reset the counters so
        # ``max_epochs`` / schedulers describe *this* run, not the trunk's past.
        if reset_optimizer:
            ranked_logger.info(
                f"Resetting epoch/step for confidence fine-tune "
                f"(was epoch={self.state.get('current_epoch')}, "
                f"step={self.state.get('global_step')})."
            )
            self.state["current_epoch"] = 0
            self.state["global_step"] = 0

    def _assemble_network_inputs(self, example):
        # assemble the base network inputs...
        network_input = super()._assemble_network_inputs(example)
        #  ... and then add the confidence-specific inputs
        data = example["data_samples"]
        conf = data["confidence"]

        # Confidence head uses 1-D token indices (index_select); structure batch is 1.
        def _squeeze_batch(t):
            if torch.is_tensor(t) and t.ndim > 0 and t.shape[0] == 1:
                return t[0]
            return t

        network_input.update(
            {
                "seq": _squeeze_batch(conf["rf2aa_seq"]),
                "rep_atom_idxs": _squeeze_batch(data["ground_truth"]["rep_atom_idxs"]),
                "frame_atom_idxs": _squeeze_batch(
                    conf["pae_frame_idx_token_lvl_from_atom_lvl"]
                ),
            }
        )

        return network_input

    def _assemble_confidence_loss_extra_info(self, example: dict) -> dict:
        """Token-level features for :class:`~rf3.loss.af3_confidence_loss.ConfidenceLoss`."""
        data = example["data_samples"]
        conf = data["confidence"]

        # ConfidenceLoss expects unbatched token features (I, ...) when structure
        # batch size is 1 (the usual case). Squeeze a leading batch dim if present.
        def _squeeze_batch(t):
            if torch.is_tensor(t) and t.ndim > 0 and t.shape[0] == 1:
                return t[0]
            return t

        return {
            "seq": _squeeze_batch(conf["rf2aa_seq"]),
            "atom_frames": _squeeze_batch(conf["atom_frames"]),
            "tok_idx": _squeeze_batch(data["features"]["atom_to_token_map"]),
            "is_real_atom": _squeeze_batch(conf["is_real_atom"]),
            "rep_atom_idxs": _squeeze_batch(data["ground_truth"]["rep_atom_idxs"]),
            "frame_atom_idxs": _squeeze_batch(
                conf["pae_frame_idx_token_lvl_from_atom_lvl"]
            ),
        }

    def _assemble_metrics_extra_info(self, example, network_output):
        # Use RF3Trainer assembly (needs X_gt_align_* on network_output). Do not call
        # ``super()`` via ``self._assemble_loss_extra_info`` — that name is not
        # overridden here so parent dispatch is fine, but keep the call explicit.
        data = example["data_samples"]
        metrics_extra_info = RF3Trainer._assemble_metrics_extra_info(
            self, data, network_output
        )
        # PAE/pLDDT metrics need token-level confidence features (seq, frames, …).
        metrics_extra_info.update(self._assemble_confidence_loss_extra_info(example))
        metrics_extra_info.update(
            {
                "is_real_atom": data["confidence"]["is_real_atom"],
                "is_ligand": data["features"]["is_ligand"],
                "confidence_loss": self.state["train_cfg"].trainer.loss.confidence_loss,
            }
        )
        # Confidence head stacks diffusion samples on dim0; structure assembly keeps
        # a leading structure-batch dim. Squeeze B=1 so metric B matches D.
        for key in ("X_gt_L", "crd_mask_L"):
            t = metrics_extra_info.get(key)
            if torch.is_tensor(t) and t.ndim >= 3 and t.shape[0] == 1:
                metrics_extra_info[key] = t[0]
        return metrics_extra_info
    def _resolve_symmetry_for_confidence(
        self, data, padding_masks, network_output
    ):
        """Align GT to the rollout prediction (same API as structure validation).

        On rare subunit/sidechain resolve failures, fall back to identity GT so a
        single bad example cannot crash one DDP rank and hang the job.
        """
        example_id = data.get("example_id", "?")
        try:
            x_native_aln, x_native_mask = self.subunit_symm_resolve(
                data["symmetry_resolution"]["molecule_entity"],
                data["symmetry_resolution"]["molecule_iid"],
                data["symmetry_resolution"]["crop_mask"],
                data["symmetry_resolution"]["coord_atom_lvl"],
                data["symmetry_resolution"]["mask_atom_lvl"],
                network_output["X_L"],
                padding_masks["L"],
                tag=str(example_id),
            )
            x_native_aln, x_native_mask = self.residue_symm_resolve(
                network_output["X_L"],
                x_native_aln,
                x_native_mask,
                data["symmetry_resolution"]["automorphisms"],
            )
            return x_native_aln, x_native_mask
        except Exception as e:
            ranked_logger.warning(
                "Symmetry resolution failed for %s (%s: %s); using identity GT.",
                example_id,
                type(e).__name__,
                e,
            )
            return self._identity_gt_for_confidence(
                data, padding_masks, network_output
            )

    def _identity_gt_for_confidence(self, data, padding_masks, network_output):
        """Place unpermuted native coords onto the cropped atom axis of ``X_L``."""
        X_L = network_output["X_L"]
        # X_L: [B, D, L, 3] (confidence train) — fall back per structure batch item.
        if X_L.ndim == 3:
            X_L = X_L.unsqueeze(0)
        B, D, L, _ = X_L.shape
        device = X_L.device
        dtype = X_L.dtype

        x_native_aln = torch.zeros(B, D, L, 3, device=device, dtype=dtype)
        x_native_mask = torch.zeros(B, D, L, dtype=torch.bool, device=device)

        crop_mask = data["symmetry_resolution"]["crop_mask"]
        coord = data["symmetry_resolution"]["coord_atom_lvl"]
        mask_nat = data["symmetry_resolution"]["mask_atom_lvl"]
        is_pad_L = padding_masks["L"]
        for b in range(B):
            crop_b = crop_mask[b]
            valid_crop = crop_b >= 0
            src = crop_b[valid_crop]
            # Dest slots on the padded L axis (same order as model pred crop).
            if is_pad_L.ndim == 2:
                dest = (~is_pad_L[b]).nonzero(as_tuple=False).squeeze(-1)
            else:
                dest = (~is_pad_L).nonzero(as_tuple=False).squeeze(-1)
            n = min(int(src.numel()), int(dest.numel()))
            if n == 0:
                continue
            dest = dest[:n]
            src = src[:n]
            x_native_aln[b, :, dest] = coord[b, src].to(dtype).unsqueeze(0).expand(D, -1, -1)
            x_native_mask[b, :, dest] = mask_nat[b, src].unsqueeze(0).expand(D, -1)

        return x_native_aln, x_native_mask
    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Perform mini-rollout and assess gradient of the confidence head parameters with respect to the confidence loss."""
        model = self.state["model"]
        assert model.training, "Model must be training!"

        # Debug-only: reuse the exact same raw batch on every training step (computed
        # once, cached in memory) instead of consuming whatever the dataloader/transform
        # pipeline produces this step. See ``mock_fixed_batch`` docstring in __init__.
        if getattr(self, "mock_fixed_batch", False):
            if self._mock_batch_cache is None:
                ranked_logger.info(
                    "mock_fixed_batch=True: caching this training batch and reusing it "
                    "verbatim on every subsequent step (bypasses dataloader augmentation)."
                )
                self._mock_batch_cache = batch
            batch = self._mock_batch_cache

        # Recycling
        # (Number of recycles for the current batch; shared across all GPUs within a distributed batch)
        n_cycle = self.recycle_schedule[self.state["current_epoch"], batch_idx].item()

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # (We assume batch size of 1 for structure predictions)
            example = _batch_to_example(batch)
            data = example["data_samples"]
            padding_masks = example["padding_masks"]

            network_input = self._assemble_network_inputs(example)

            # Forward pass (with mini-rollout) — same padding_masks as structure trunk
            network_output = model.forward(
                input=network_input,
                padding_masks=padding_masks,
                n_cycle=n_cycle,
                coord_atom_lvl_to_be_noised=data["diffusion_features"][
                    "coord_atom_lvl_to_be_noised"
                ],
            )
            assert_no_nans(
                network_output,
                msg=f"network_output for example_id: {data['example_id']}",
            )

            # Remap X_L to the rollout X_L so ground truth matches rollout batch dimension
            network_output["X_L"] = network_output["X_pred_rollout_L"]

            x_native_aln, x_native_mask = self._resolve_symmetry_for_confidence(
                data, padding_masks, network_output
            )

            # ConfidenceLoss treats dim0 as the diffusion-sample batch (D), with
            # unbatched token features. Squeeze structure batch (B=1).
            X_pred = network_output["X_pred_rollout_L"]
            if X_pred.ndim == 4:
                X_pred = X_pred[0]  # [D, L, 3]
                x_native_aln = x_native_aln[0]
                x_native_mask = x_native_mask[0]

            # Strip fixed L-padding. ConfidenceLoss.calc_lddt assumes atom dim L
            # equals ``is_real_atom.sum()`` (no pad suffix). Pad is always a
            # trailing block (see PaddedBatchedDataSample.get_padding_masks).
            # Token features / conf logits keep the padded I=crop_size axis;
            # pad tokens are False in is_real_atom so they skip the loss.
            is_pad_L = padding_masks["L"]
            if is_pad_L.ndim == 2:
                is_pad_L = is_pad_L[0]
            if is_pad_L.any():
                valid_L = ~is_pad_L
                X_pred = X_pred[:, valid_L]
                x_native_aln = x_native_aln[:, valid_L]
                x_native_mask = x_native_mask[:, valid_L]

            network_output = {**network_output, "X_pred_rollout_L": X_pred}

            loss_extra_info = self._assemble_confidence_loss_extra_info(example)
            loss_extra_info["X_gt_L"] = x_native_aln
            loss_extra_info["crd_mask_L"] = x_native_mask

            # We only assess the confidence loss
            total_loss, loss_dict_batched = self.loss(
                network_input=network_input,
                network_output=network_output,
                loss_input=loss_extra_info,
            )

            # Backward pass
            self.fabric.backward(total_loss)

            # ... store the outputs without gradients for use in logging, callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                {
                    "total_loss": total_loss,
                    "loss_dict": loss_dict_batched,
                    "n_symm_res_failures": int(network_output.get("n_symm_res_failures", 0)),
                },
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
        should_early_stop_fn: ShouldEarlyStopFn | None = None,
        cfg_guidance_batches: list | None = None,
        inference_classifier_free_guidance: bool = False,
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        motif_guidance: Any = None,
        confidence_streams: tuple[str, ...] | list[str] | None = None,
    ) -> dict:
        """Validation step, running forward pass (with full rollout) and computing validation metrics, including confidence."""
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        example = _batch_to_example(batch)
        data = example["data_samples"]
        padding_masks = example["padding_masks"]

        network_input = self._assemble_network_inputs(example)

        # Same construction as RF3Trainer.validation_step: one guidance input per
        # collated per-condition batch, carrying its scale.
        cfg_guidance_inputs = None
        if inference_classifier_free_guidance and cfg_guidance_batches:
            cfg_guidance_inputs = []
            for ref in cfg_guidance_batches:
                ref_example = _batch_to_example(ref["batch"])
                cfg_guidance_inputs.append(
                    {
                        "f": ref_example["data_samples"]["features"],
                        "padding_masks": ref_example["padding_masks"],
                        "scale": ref["scale"],
                        "name": ref.get("name"),
                    }
                )

        # The rollout cache key is (base weight fingerprint, example_id) only -- it does
        # not cover conditioning, scales, or the sigma window, so a CFG run would replay
        # an unconditional rollout. Bypass the cache whenever guidance is active.
        rollout_cache_dir = self.val_rollout_cache_dir
        if cfg_guidance_inputs:
            rollout_cache_dir = None
        # Do not assert_no_nans on network_input: rollout mode leaves NaNs in
        # X_unnoised_L, and symmetry_resolution.coord_atom_lvl has NaNs for
        # unresolved atoms (same as structure RF3Trainer.validation_step).

        # ... forward pass (with FULL rollout; optional disk cache of trunk+diffusion)
        network_output = model.forward(
            input=network_input,
            padding_masks=padding_masks,
            n_cycle=data["features"]["msa_stack"].shape[1],
            coord_atom_lvl_to_be_noised=data["diffusion_features"][
                "coord_atom_lvl_to_be_noised"
            ],
            should_early_stop_fn=should_early_stop_fn,
            rollout_cache_dir=rollout_cache_dir,
            rollout_cache_key=str(data["example_id"]),
            cfg_guidance_inputs=cfg_guidance_inputs,
            inference_classifier_free_guidance=inference_classifier_free_guidance,
            cfg_sigma_min=cfg_sigma_min,
            cfg_sigma_max=cfg_sigma_max,
            motif_guidance=motif_guidance,
            confidence_streams=confidence_streams,
        )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {data['example_id']}",
        )

        # Remap X_L to the rollout X_L
        network_output["X_L"] = network_output.get("X_pred_rollout_L")

        metrics_output = {}
        if (
            compute_metrics
            and exists(self.metrics)
            and not network_output.get("early_stopped", False)
        ):
            x_native_aln, x_native_mask = self._resolve_symmetry_for_confidence(
                data, padding_masks, network_output
            )
            network_output["X_gt_align_L"] = x_native_aln
            network_output["mask_gt_align_L"] = x_native_mask

            metrics_extra_info = self._assemble_metrics_extra_info(
                example, network_output
            )

            # Collation wraps atom_array as a length-1 list; coords may be [B, D, L, 3].
            atom_array = data["atom_array"]
            if isinstance(atom_array, (list, tuple)):
                atom_array = atom_array[0]
            X_gt_for_stack = metrics_extra_info.get("X_gt_L", x_native_aln)
            X_pred_for_stack = network_output.get("X_pred_rollout_L", network_output["X_L"])
            if torch.is_tensor(X_gt_for_stack) and X_gt_for_stack.ndim == 4:
                X_gt_for_stack = X_gt_for_stack[0]
            if torch.is_tensor(X_pred_for_stack) and X_pred_for_stack.ndim == 4:
                X_pred_for_stack = X_pred_for_stack[0]
            # Align network_output diffusion-batch layout with confidence logits [D, ...].
            network_output = {
                **network_output,
                "X_L": X_pred_for_stack,
                "X_pred_rollout_L": X_pred_for_stack,
            }

            metrics_output = self.metrics(
                network_input=network_input,
                network_output=network_output,
                extra_info=metrics_extra_info,
                ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    X_gt_for_stack,
                    atom_array,
                ),
                predicted_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    X_pred_for_stack,
                    atom_array,
                ),
            )

            # Same metrics for every extra confidence stream. They all describe the same
            # structure, so only the pae/pde/plddt logits differ -- everything else
            # (ground truth, atom arrays, extra_info) is reused as-is. Structure-derived
            # metrics such as the clash count therefore just recompute the same value.
            stream_logits = network_output.get("confidence_by_stream") or {}
            if stream_logits and metrics_output is not None:
                metrics_output["confidence_streams"] = {
                    name: self.metrics(
                        network_input=network_input,
                        network_output={
                            **network_output,
                            "plddt": logits["plddt_logits"],
                            "pae": logits["pae_logits"],
                            "pde": logits["pde_logits"],
                        },
                        extra_info=metrics_extra_info,
                        ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                            X_gt_for_stack,
                            atom_array,
                        ),
                        predicted_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                            X_pred_for_stack,
                            atom_array,
                        ),
                    )
                    for name, logits in stream_logits.items()
                }

            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        network_output = (
            apply_to_collection(network_output, torch.Tensor, lambda x: x.detach())
            if network_output is not None
            else None
        )

        return {"metrics_output": metrics_output, "network_output": network_output}
