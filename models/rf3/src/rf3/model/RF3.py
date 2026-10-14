from collections import deque
from contextlib import ExitStack
import copy
import logging
import os

import hydra
import torch
import torch.utils.checkpoint as checkpoint
from beartype.typing import Any, Generator, Protocol
from omegaconf import DictConfig
from rf3.diffusion_samplers.inference_sampler import (
    SampleDiffusion,
    SamplePartialDiffusion,
)
from rf3.model.layers.embedders.feature_init import (
    FeatureInitializer,
)
from rf3.model.layers.diffusion.diffusion_module import DiffusionModule
from rf3.model.cfg_streams import extrapolate_trunk_streams, trunk_tensors
from rf3.model.rollout_cache import (
    compute_base_fingerprint,
    save_rollout_cache,
    try_load_rollout_cache,
)
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation

from rf3.loss.af3_losses import (
    ResidueSymmetryResolution,
    SubunitSymmetryResolution,
)

from rf3.model.recycler import DistogramHead, Recycler, MaskedTokenRecoveryHead
from rf3.model.layers.utils.layer_utils import NormBiasConfig
from torch import nn

from foundry.training.checkpoint import create_custom_forward

logger = logging.getLogger(__name__)

"""
Shape Annotation Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
    D: # diffusion structure batch dim

    C_s: # Token-level single reprentation channel dimension
    C_z: # Token-level pair reprentation channel dimension
    C_atom: # Atom-level single reprentation channel dimension
    C_atompair: # Atom-level pair reprentation channel dimension

Tensor Name Glossary:
    S: Token-level single representation (I, C_s)
    Z: Token-level pair representation (I, I, C_z)
    Q: Atom-level single representation (L, C_atom)
    P: Atom-level pair representation (L, L, C_atompair)
"""


def _guidance_stream_padding_masks(ref: dict, padding_masks: dict) -> dict:
    """Return a CFG guidance stream's own padding masks, falling back to the main stream's.

    Each stream is featurized and collated separately, so a feature whose length is
    example-dependent (num_chiral_centers) can be padded differently from the main
    stream. The token/atom counts must agree, though -- the streams share one set of
    coordinates -- so a mismatch there is a bug worth reporting rather than masking.
    """
    ref_masks = ref.get("padding_masks")
    if ref_masks is None:
        return padding_masks
    for key in ("I", "L"):
        # .get, not []: a hand-built stream dict (tests, external scripts) may carry
        # only some keys, and a KeyError there is harder to read than no check.
        ref_mask = ref_masks.get(key)
        main_mask = padding_masks.get(key)
        if ref_mask is None or main_mask is None:
            continue
        if ref_mask.shape != main_mask.shape:
            raise ValueError(
                f"CFG guidance stream '{ref.get('name') or '<unnamed>'}' has {key} "
                f"padding shape {tuple(ref_mask.shape)}, but the main stream has "
                f"{tuple(main_mask.shape)}; the streams must describe the same "
                "tokens and atoms and differ only in their conditioning."
            )
    return ref_masks


class ShouldEarlyStopFn(Protocol):
    def __call__(
        self, confidence_outputs: dict[str, Any], first_recycle_outputs: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Duck-typed function Protocol for early stopping based on confidence outputs.

        Returns:
            tuple: A pair containing:
                - should_stop (bool): Whether to stop early.
                - additional_data (dict): Metadata for the user, if any
        """
        ...


class RF3(nn.Module):
    """RF3 Network module.

    We adhere to the PyTorch Lightning Style Guide; see (1).

    References:
        (1) PyTorch Lightning Style Guide: https://lightning.ai/docs/pytorch/latest/starter/style_guide.html
    """

    def __init__(
        self,
        *,
        # Arguments for modules that will be instantiated
        feature_initializer: DictConfig | dict,
        recycler: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        distogram_head: DictConfig | dict,
        masked_token_recovery_head: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        mini_rollout_sampler: DictConfig | dict,
        norm_bias: DictConfig | dict | None = None,
        # Channel dimensions
        c_s: int,  # AF-3: 384,
        c_z: int,  # AF-3: 128,
        c_atom: int,  # AF-3: 128,
        c_atompair: int,  # AF-3: 16,
        c_s_inputs: int,  # AF-3: 449,
    ):
        """Initializes the AF3 model.

        Args:
            feature_initializer: Arguments for FeatureInitializer
            recycler: Arguments for Recycler
            diffusion_module: Arguments for DiffusionModule
            distogram_head: Arguments for DistogramHead
            masked_token_recovery_head: Arguments for DistogramHead
            inference_sampler: Arguments for the SampleDiffusion class, used for inference (contains no trainable parameters)
            c_s: Token-level single reprentation         norm_bias: DictConfig | dict | None = None,
channel dimension
            c_z: Token-level pair reprentation channel dimension
            c_atom: Atom-level single reprentation channel dimension
            c_atompair: Atom-level pair reprentation channel dimension
            c_s_inputs: Output dimension of the InputFeatureEmbedder
        """
        super().__init__()

        norm_bias_cfg = (
            hydra.utils.instantiate(norm_bias)
            if norm_bias is not None
            else NormBiasConfig()
        )

        # Use hydra.utils.instantiate so nested _target_ sub-configs
        # (e.g. the optional conditioning_2d_processor) are built recursively.
        self.feature_initializer = hydra.utils.instantiate(
            feature_initializer,
            _target_="rf3.model.layers.embedders.feature_init.FeatureInitializer",
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s_inputs=c_s_inputs,
            norm_bias=norm_bias_cfg,
        )

        # ... initialize the Recycler, which runs the trunk repeatedly with shared weights
        self.recycler = Recycler(c_s=c_s, c_z=c_z, norm_bias=norm_bias_cfg, **recycler)
        # Use hydra.utils.instantiate so nested _target_ sub-configs
        # (e.g. the optional conditioning_2d_processor) are built recursively.
        self.diffusion_module = hydra.utils.instantiate(
            diffusion_module,
            _target_="rf3.model.layers.diffusion.diffusion_module.DiffusionModule",
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            norm_bias=norm_bias_cfg,
        )
        self.distogram_head = DistogramHead(c_z=c_z, **distogram_head)

        if masked_token_recovery_head is not None:
            self.masked_token_recovery_head = MaskedTokenRecoveryHead(c_s=c_s, **masked_token_recovery_head)
        else:
            self.masked_token_recovery_head = None


        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = (
            SampleDiffusion(**inference_sampler)
            if not inference_sampler.get("partial_t", False)
            else SamplePartialDiffusion(**inference_sampler)
        )

        # mini rollout sampler (used in training)
        self.mini_rollout_sampler = SampleDiffusion(**mini_rollout_sampler)

        # The mini-rollout runs on a deep copy of the diffusion module (weights synced
        # every step). 
        # Stashed via object.__setattr__ so it is not registered as a submodule
        # (no DDP wrapping, no entry in state_dict).
        object.__setattr__(self, '_rollout_module', None)

        # (Symmetry resolution)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()

    def _get_rollout_module(self):
        """Return a no-grad eval-mode copy of self.diffusion_module with current weights.

        Lazily deepcopied once; weights are synced from the live module on every call.
        """
        rollout_module = object.__getattribute__(self, '_rollout_module')
        if rollout_module is None:
            rollout_module = copy.deepcopy(self.diffusion_module)
            object.__setattr__(self, '_rollout_module', rollout_module)

        # Sync weights from the live diffusion module (optimizer may have updated them).
        rollout_module.load_state_dict(self.diffusion_module.state_dict())
        rollout_module.eval()
        return rollout_module

    def _update_inputs(
        self,
        chain_iid_by_atom,      # Int   [B, L]
        token_to_repatom_index, # Int   [B, I]
        new_native,             # Float [B, 1, L, 3]
        new_mask,               # Bool  [B, 1, L],
        is_padding_L,           # Bool  [B, L]
        D                       # Int
    ):
        B, _, L, _ = new_native.shape
        device = new_native.device

        corrected = new_native.clone()  # [B, 1, L, 3]
        augmented = torch.empty(B, D, L, 3, device=device)  # [B, D, L, 3]
        all_atom_idx = torch.arange(L, device=device)

        for b in range(B):
            rep_indices  = token_to_repatom_index[b]           # [I]
            chain_at_rep = chain_iid_by_atom[b, rep_indices]   # [I]
            mask_at_rep  = new_mask[b, 0, rep_indices].bool()  # [I]
            atom_mask    = new_mask[b, 0].bool()               # [L]
            chain_ids    = chain_iid_by_atom[b]                # [L]

            for chain_id in chain_ids.unique():
                in_chain_atoms = (chain_ids == chain_id)        # [L]
                to_update = in_chain_atoms & ~atom_mask         # [L]

                if not to_update.any():
                    continue

                valid_rep_in_chain = (chain_at_rep == chain_id) & mask_at_rep  # [I]

                if valid_rep_in_chain.any():
                    valid_rep_positions = rep_indices[valid_rep_in_chain]  # [I_valid], sorted

                    insert = torch.searchsorted(
                        valid_rep_positions.contiguous(),
                        all_atom_idx.contiguous()
                    )  # [L]

                    lo = (insert - 1).clamp(0, len(valid_rep_positions) - 1)
                    hi = insert.clamp(0, len(valid_rep_positions) - 1)

                    lo_pos = valid_rep_positions[lo]
                    hi_pos = valid_rep_positions[hi]

                    nearest_valid = torch.where(
                        (all_atom_idx - lo_pos).abs() <= (all_atom_idx - hi_pos).abs(),
                        lo, hi
                    )  # [L]

                    source_indices = valid_rep_positions[nearest_valid]  # [L]
                    corrected[b, 0, to_update] = new_native[b, 0, source_indices[to_update]]
                else:
                    corrected[b, 0, to_update] = 0.0

            # Expand along D, center, and randomly augment
            corrected_stack     = corrected[b, 0].unsqueeze(0).expand(D, -1, -1).contiguous()  # [D, L, 3]
            mask_atom_lvl_stack = new_mask[b, 0].unsqueeze(0).expand(D, -1).contiguous()       # [D, L]

            # center
            corrected_stack = masked_center(corrected_stack, mask_atom_lvl_stack)

            # zero out padded
            corrected_stack[:,is_padding_L[b]] = 0.0

            # random rotation/translation
            augmented[b] = random_rigid_augmentation(corrected_stack, batch_size=D, s=1)    # [D, L, 3]

        return augmented

    def _do_mini_rollout(
        self,
        input,
        recycling_outputs,
        coord_atom_lvl_to_be_noised,
        is_padding_I,
        is_padding_L,
        is_padding_chiral,
        tag=''
    ):
        rollout_module = self._get_rollout_module()
        with torch.no_grad():
            sample_diffusion_outs = (
               self.mini_rollout_sampler.sample_diffusion_like_af3(
                   f={k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in input["f"].items()},
                   S_inputs_I=recycling_outputs["S_inputs_I"].detach(),
                   S_trunk_I=recycling_outputs["S_I"].detach(),
                   Z_trunk_II=recycling_outputs["Z_II"].detach(),
                   diffusion_module=rollout_module,
                   diffusion_batch_size=1,
                   coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.detach(),
                   is_padding_I=is_padding_I.detach(),
                   is_padding_L=is_padding_L.detach(),
                   is_padding_chiral=is_padding_chiral.detach(),
               )
            )

        # Fallback for NaN in the mini-rollout: use nan_to_num
        #fd: this loop is exceedingly rare
        n_nan = int(sample_diffusion_outs['X_L'].isnan().sum())
        if n_nan:
            logger.warning(
                "Mini-rollout produced %d non-finite coords%s; applying nan_to_num fallback.",
                n_nan, f" in {tag}" if tag else "",
            )
            sample_diffusion_outs['X_L'] = torch.nan_to_num(sample_diffusion_outs['X_L'])

        # subunit symmetry resolution, catching errors
        try:
            x_native_aln, x_native_mask = self.subunit_symm_resolve(
                input["symmetry_resolution"]["molecule_entity"],                   # mol_entities
                input["symmetry_resolution"]["molecule_iid"],                      # mol_iid
                input["symmetry_resolution"]["crop_mask"],                         # mask_crop
                input["symmetry_resolution"]["coord_atom_lvl"],  # x_native
                input["symmetry_resolution"]["mask_atom_lvl"],                     # mask_native
                sample_diffusion_outs["X_L"].detach(),      # x_pred
                is_padding_L.detach(),
                tag=tag
            )
        except Exception as e:
            print (f'WARNING: Symmetry resolution error in {tag}, no symm resolution will be done!')
            print (f'WARNING: {e}')
            return None, None, sample_diffusion_outs["X_L"]

        # sidechain symmetry resolution
        x_native_aln, x_native_mask = self.residue_symm_resolve(
           sample_diffusion_outs["X_L"].detach(),
            x_native_aln,
            x_native_mask,
            input["symmetry_resolution"]["automorphisms"]
        )

        # Mode-agnostic: return the symm-resolved native + the rollout prediction.
        # The caller (forward) assembles X_unnoised_L per missing_atom_behavior.
        return x_native_aln, x_native_mask, sample_diffusion_outs["X_L"]


    def _fill_missing_with_rollout(
        self,
        input,          # full network input dict (provides symmetry_resolution and noise_L shape)
        x_native_aln,   # Float [B, 1, L, 3] -- native coords (native frame, chain assignment resolved)
        x_native_mask,  # Bool  [B, 1, L]    -- True where native atoms are resolved
        X_rollout_L,    # Float [B, 1, L, 3]  -- rollout prediction (prediction frame)
        is_padding_L,   # Bool  [B, L]
    ):
        """Fill unresolved native atoms with rollout-predicted coordinates, then build X_unnoised_L.

        x_native_aln is in the native coordinate frame while X_rollout_L is in the prediction's
        (arbitrary) frame.  We therefore Kabsch-superimpose the prediction onto the native
        structure (using only the resolved atoms as anchors) before filling in the gaps.

        Args:
            input: Network input dict.
            x_native_aln: Native coordinates in native frame [B, 1, L, 3].
            x_native_mask: Boolean mask; True for resolved native atoms [B, 1, L]
            X_rollout_L: Raw rollout prediction in prediction frame [B, 1, L, 3]
            is_padding_L: Padding mask [B, L].

        Returns:
            X_unnoised_L: Filled and augmented coordinates [B, D, L, 3].
        """
        B = x_native_aln.shape[0]
        D = input['noise_L'].shape[1]

        assert X_rollout_L.shape[1] == 1, (
            f"_fill_missing_with_rollout assumes batch=1, got {X_rollout_L.shape[1]}"
        )

        filled_list = []
        for b in range(B):
            mask = x_native_mask[b, 0]  # [L] bool, True = resolved
            nat = x_native_aln[b, 0]   # [L, 3]
            pred = X_rollout_L[b, 0]   # [L, 3] -- this example's rollout prediction

            if mask.sum() > 3:
                # Kabsch needs to be float32
                with torch.amp.autocast('cuda', enabled=False):
                    nat_res = nat[mask].float()    # [N, 3]
                    pred_res = pred[mask].float()  # [N, 3]
                    pred_f = pred.float()          # [L, 3]

                    nat_mean = nat_res.mean(dim=0)
                    pred_mean = pred_res.mean(dim=0)
                    nat_c  = nat_res  - nat_mean
                    pred_c = pred_res - pred_mean

                    # Covariance: rotate pred -> native
                    C = pred_c.T @ nat_c    # [3, 3]
                    U, _, Vt = torch.linalg.svd(C)
                    # Correct for reflections
                    F = torch.eye(3, device=C.device)
                    F[2, 2] = torch.linalg.det(U @ Vt).sign()
                    R = U @ F @ Vt          # [3, 3]

                    # Apply rotation to the full prediction, translated into native frame
                    pred_aln = ((pred_f - pred_mean) @ R + nat_mean).to(pred.dtype)  # [L, 3]
            else:
                # Too few resolved atoms to align reliably -- use prediction as-is
                pred_aln = pred

            filled = torch.where(mask.unsqueeze(-1), nat, pred_aln)  # [L, 3]
            filled_list.append(filled.unsqueeze(0).unsqueeze(0))     # [1, 1, L, 3]

        filled_native = torch.cat(filled_list, dim=0)  # [B, 1, L, 3]

        # With all positions filled, the mask is now all-True.
        all_resolved_mask = torch.ones_like(x_native_mask, dtype=torch.bool)  # [B, 1, L]

        return self._update_inputs(
            input["symmetry_resolution"]["chain_iid_by_atom"],
            input["symmetry_resolution"]["token_to_repatom_index"],
            filled_native,
            all_resolved_mask,
            is_padding_L,
            D,
        )

    def forward(
        self,
        input: dict,
        padding_masks: dict,
        n_cycle: int,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        tag='',
        # Classifier-free guidance (composable, inference only). 
        # `cfg_guidance_inputs` is a list of per-condition inputs
        # Each is a dict {"f": features, "scale": float}.
        cfg_guidance_inputs: list[dict] | None = None,
        inference_classifier_free_guidance: bool = False,
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        # Motif guidance (inference only, independent of CFG). Duck-typed engine with
        # `.refine(x0, t_norm=float)`; None is an exact no-op.
        motif_guidance: Any = None,
    ) -> dict:
        """Complete forward pass of the model.

        Runs recycling with gradients only on final recycle.

        Args:
            input (dict): Dictionary of model inputs
            padding_masks (dict): Dictionary of padding masks
            n_cycle (int): Number of recycling cycles for the trunk
            coord_atom_lvl_to_be_noised (torch.Tensor): Atom-level coordinates to be noised further. Optional;
                only used during inference for partial denoising.

        Returns:
            dict: Dictionary of model outputs, including:
                - X_L: Predicted atomic coordinates [D, L, 3]
                - distogram: Predicted distogram [I, I, C], where C is the number of bins in the distogram
                - If not training, additional lists are returned, each of length T:
                    * X_noisy_L_traj: List of noisy atomic coordinates at each timestep [D, L, 3]
                    * X_denoised_L_traj: List of denoised atomic coordinates at each timestep [D, L, 3]
                    * t_hats: List of tensor scalars representing the noise schedule at each timestep
        """
        # Cast features to lower precision if autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype("cuda")
            for x in [
                "msa_stack",
                "profile",
                "template_distogram",
                "template_restype",
                "template_unit_vector",
            ]:
                if x in input["f"]:
                    input["f"][x] = input["f"][x].to(autocast_dtype)
                for ref in cfg_guidance_inputs or []:
                    if x in ref["f"]:
                        ref["f"][x] = ref["f"][x].to(autocast_dtype)

        # ... recycling
        # Gives dictionary of outputs S_inputs_I, S_init_I, Z_init_II, S_I, Z_II
        # (We use `deque` with maxlen=1 to ensure that we only keep the last output in memory)
        try:
            recycling_outputs = deque(
                self.trunk_forward_with_recycling(
                    f=input["f"], padding_masks=padding_masks, n_recycles=n_cycle
                ),
                maxlen=1,
            ).pop()
        except IndexError:
            # Handle the case where the generator is empty
            raise RuntimeError("Recycling generator produced no outputs")

        # Predict the distogram from the pair representation
        distogram_pred = self.distogram_head(recycling_outputs["Z_II"])

        # Predict the masked tokens from the single representation
        if self.masked_token_recovery_head is not None:
            masked_token_recovery_pred = self.masked_token_recovery_head(recycling_outputs["S_I"])
        else:
            masked_token_recovery_pred = None

        is_padding_I = padding_masks["I"]
        is_padding_L = padding_masks["L"]
        is_padding_chiral = padding_masks.get("num_chiral_centers", None)
        # ... post-recycling (diffusion module)
        if self.training:
            # mini rollout
            with torch.no_grad():
                x_native_aln, x_native_mask, X_rollout_L = self._do_mini_rollout(
                    input,
                    recycling_outputs,
                    coord_atom_lvl_to_be_noised,
                    is_padding_I,
                    is_padding_L,
                    is_padding_chiral,
                    tag=tag
                )

                # Assemble X_unnoised_L per missing_atom_behavior.
                # Required key (no default): the trainer guarantees it via the schema,
                # so a missing key is a regression we want to surface, not hide.
                if input["missing_atom_behavior"] == "rollout":
                    # Rollout: use rollout (aligned) to fill gaps
                    if x_native_aln is not None:
                        X_unnoised_L = self._fill_missing_with_rollout(
                            input, x_native_aln, x_native_mask, X_rollout_L, is_padding_L
                        )
                    else:
                        # Symm-res failed: resolve rollout + raw native
                        cm = input["symmetry_resolution"]["crop_mask"]                # [B, L]
                        valid = cm >= 0
                        safe_cm = cm.clamp(min=0)
                        b = torch.arange(cm.shape[0], device=cm.device)[:, None]      # [B, 1]
                        coord_un = input["symmetry_resolution"]["coord_atom_lvl"]     # [B, uncropped_L, 3]
                        mask_un  = input["symmetry_resolution"]["mask_atom_lvl"]      # [B, uncropped_L]
                        x_native_raw      = coord_un[b, safe_cm].unsqueeze(1)         # [B,1,L,3]
                        x_native_mask_raw = (mask_un[b, safe_cm] & valid).unsqueeze(1)  # [B,1,L]
                        X_unnoised_L = self._fill_missing_with_rollout(
                            input, x_native_raw, x_native_mask_raw, X_rollout_L, is_padding_L
                        )
                elif x_native_aln is not None:
                    # Non-rollout
                    X_unnoised_L = self._update_inputs(
                        input["symmetry_resolution"]["chain_iid_by_atom"],
                        input["symmetry_resolution"]["token_to_repatom_index"],
                        x_native_aln,
                        x_native_mask,
                        is_padding_L,
                        input['noise_L'].shape[1],
                    )
                else:
                    # symm resolution failed (non-rollout) or rollout failed
                    # use input (dataloader) params
                    X_unnoised_L = input['X_unnoised_L']

                # origin mode...
                #   ensure zero noise for missing coords
                if input["missing_atom_behavior"] == "origin" and x_native_mask is not None:
                    unresolved = ~x_native_mask  # [B, 1, L], True = unresolved
                    X_unnoised_L = X_unnoised_L.masked_fill(unresolved.unsqueeze(-1), 0.0)
                    input['noise_L'] = input['noise_L'].masked_fill(unresolved.unsqueeze(-1), 0.0)

            # Single denoising step
            with torch.amp.autocast('cuda', enabled=False):
                X_pred = self.diffusion_module(
                    X_noisy_L=X_unnoised_L + input['noise_L'],
                    t=input["t"],
                    f=input["f"],
                    S_inputs_I=recycling_outputs["S_inputs_I"],
                    S_trunk_I=recycling_outputs["S_I"],
                    Z_trunk_II=recycling_outputs["Z_II"],
                    is_padding_I=is_padding_I,
                    is_padding_L=is_padding_L,
                    is_padding_chiral=is_padding_chiral,
                )  # [D, L, 3]

            return dict(
                X_L=X_pred,
                distogram=distogram_pred,
                masked_token_logits=masked_token_recovery_pred,

                # realigned native
                X_gt_align_L=x_native_aln,
                mask_gt_align_L=x_native_mask,

                # diagnostics: 1 if symmetry resolution failed this step, else 0
                n_symm_res_failures=int(x_native_aln is None),
            )
        else:
            # Full diffusion rollout (no gradients, or will OOM)
            # Classifier-free guidance (composable): `input` is the unconditional Δ∅, so
            # `recycling_outputs` (its trunk outputs) is the sampler's main stream. For each
            # per-condition reference input, re-run the trunk (conditioning enters at the
            # trunk) and build one reference stream carrying its weight.
            cfg_guidance_streams = None
            if inference_classifier_free_guidance and cfg_guidance_inputs:
                cfg_guidance_streams = []
                for ref in cfg_guidance_inputs:
                    ref_padding_masks = _guidance_stream_padding_masks(ref, padding_masks)
                    try:
                        ref_recycling_outputs = deque(
                            self.trunk_forward_with_recycling(
                                f=ref["f"],
                                padding_masks=ref_padding_masks,
                                n_recycles=n_cycle,
                            ),
                            maxlen=1,
                        ).pop()
                    except IndexError:
                        raise RuntimeError("Reference recycling generator produced no outputs")
                    cfg_guidance_streams.append(
                        {
                            "f": ref["f"],
                            "S_inputs_I": ref_recycling_outputs["S_inputs_I"],
                            "S_trunk_I": ref_recycling_outputs["S_I"],
                            "Z_trunk_II": ref_recycling_outputs["Z_II"],
                            "is_padding_chiral": ref_padding_masks.get(
                                "num_chiral_centers"
                            ),
                            "scale": ref["scale"],
                            "name": ref.get("name"),
                        }
                    )

            sample_diffusion_outs = self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                S_inputs_I=recycling_outputs["S_inputs_I"],
                S_trunk_I=recycling_outputs["S_I"],
                Z_trunk_II=recycling_outputs["Z_II"],
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=input["t"].shape[
                    1
                ],  # Diffusion batch size is 2nd dim of t, first dimension is batch size of pairformer
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                is_padding_I=is_padding_I,
                is_padding_L=is_padding_L,
                is_padding_chiral=is_padding_chiral,
                inference_classifier_free_guidance=inference_classifier_free_guidance,
                cfg_guidance_streams=cfg_guidance_streams,
                cfg_sigma_min=cfg_sigma_min,
                cfg_sigma_max=cfg_sigma_max,
                motif_guidance=motif_guidance,
            )
            return dict(
                X_L=sample_diffusion_outs["X_L"],
                distogram=distogram_pred,
                masked_token_logits=masked_token_recovery_pred,

                # For reporting, inference (validation or testing) only
                X_noisy_L_traj=sample_diffusion_outs["X_noisy_L_traj"],
                X_denoised_L_traj=sample_diffusion_outs["X_denoised_L_traj"],
                # .get, not []: a cached or mocked rollout need not carry these keys,
                # which is why the confidence path below already uses .get for them.
                X_denoised_L_guided_traj=sample_diffusion_outs.get(
                    "X_denoised_L_guided_traj", []
                ),
                X_denoised_L_stream_trajs=sample_diffusion_outs.get(
                    "X_denoised_L_stream_trajs", {}
                ),
                t_hats=sample_diffusion_outs["t_hats"],
            )

    def trunk_forward_with_recycling(
        self, f: dict, padding_masks: dict, n_recycles: int
    ) -> Generator[dict[str, torch.Tensor]]:
        """Forward pass of the AF-3 trunk.

        (e.g., the recycling process, including the MSAModule, PairfomerStack, etc.).

        Notes:
            - We run with gradients ONLY on the final recycle
            - All recycles use shared weights (ResNet-style)
            - We yield results after reach recycle to support use cases such as e.g., early stopping during inference

        Args:
            f: Feature dictionary
            padding_masks: Dictionary of padding masks
            n_recycles: Number of recycles to run

        Returns:
            dict: Recycling outputs, with keys:
                - S_inputs_I: Token-level single representation input, prior to AtomAttention [I, c_s_inputs]
                - S_init_I: Token-level single representation initialization [I, c_s], after AtomAttention but before recycling stack
                - Z_init_II: Token-level pair representation initialization [I, I, c_z], after AtomAttention but before recycling stack
                - S_I: Token-level single representation [I, c_s], after recycling stack
                - Z_II: Token-level pair representation [I, I, c_z], after recycling stack
        """
        # ... initialize the recycling process (feature initialization)
        # Gives S_inputs_I, S_init_I, Z_init_II, S_I, Z_II
        initialized_features = self.pre_recycle(f, padding_masks)

        # ... collect the recycling inputs, which will be updated in place
        recycling_inputs = {
            **initialized_features,
            "f": f,
            "padding_masks": padding_masks,
        }

        for i_cycle in range(n_recycles):
            with ExitStack() as stack:
                # For the first n_recycles - 1 cycles (all but the last recycle), we run without gradients
                if i_cycle < n_recycles - 1:
                    stack.enter_context(torch.no_grad())

                # Clear the autocast cache if gradients are enabled (workaround for autocast bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()

                # Select the MSA for the current recycle (we sample an i.i.d. MSA for each recycle)
                recycling_inputs["f"]["msa"] = f["msa_stack"][:, i_cycle]

                # Run the model trunk (MSAModule, PairformerStack, etc.)
                # We alter the S_I and Z_II in place such that the next iteration uses the updated values
                recycling_inputs = self.recycle(**recycling_inputs)

                # Yield after each recycle
                yield {
                    "S_inputs_I": recycling_inputs["S_inputs_I"],
                    "S_init_I": recycling_inputs["S_init_I"],
                    "Z_init_II": recycling_inputs["Z_init_II"],
                    "S_I": recycling_inputs["S_I"],
                    "Z_II": recycling_inputs["Z_II"],
                }

    def pre_recycle(self, f: dict, padding_masks: dict) -> dict:
        """Prepare feature inputs for recycling.

        Includes:
            - Feature initialization (S_inputs_I, S_init_I, Z_init_II)
            - Initializing S_I and Z_II to zeros

        Returns:
            dict: Dictionary of recycling inputs, including:
                - S_inputs_I: Token-level single representation input (prior to AtomAttention) [I, c_s_inputs]
                - S_init_I: Token-level single representation initialization [I, c_s] (after round of AtomAttention)
                - Z_init_II: Token-level pair representation initialization [I, I, c_z] (after round of AtomAttention)
                - S_I: Token-level single representation [I, c_s], initialized to zeros
                - Z_II: Token-level pair representation [I, I, c_z], initialized to zeros
        """
        S_inputs_I, S_init_I, Z_init_II = self.feature_initializer(f, padding_masks)
        S_I = torch.zeros_like(S_init_I)
        Z_II = torch.zeros_like(Z_init_II)

        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
        )

    def recycle(
        self,
        # TODO: Jax typing
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
        f,
        padding_masks,
    ):
        is_padding_I = padding_masks["I"]
        is_padding_S = padding_masks["S"]
        S_I, Z_II = self.recycler(
            f=f,
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
            is_padding_I=is_padding_I,
            is_padding_S=is_padding_S,
        )
        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
            f=f,
            padding_masks=padding_masks,
        )


class RF3WithConfidence(RF3):
    """Model for training and inference with confidence metric computation"""

    def __init__(
        self,
        confidence_head: DictConfig | dict,
        mini_rollout_sampler: DictConfig | dict,
        mock_fixed_diffusion_rollout: bool = False,
        **kwargs,
    ):
        """
        Args:
            (... all arguments from the AF3 class)
            confidence_head: Hydra configuration for the confidence head architecture
            mini_rollout_sampler: Hydra configuration for the mini-rollout sampler (e.g., SampleDiffusion with 20 rather than
                200 timesteps. Note that the `inference_sampler` argument in the AF3 class will still be used for full
                rollouts during inference)
            mock_fixed_diffusion_rollout: Debug-only. When True, the frozen trunk+diffusion
                rollout is computed once (first training step) and reused verbatim on every
                subsequent training step, instead of resampling fresh diffusion noise each
                step. Isolates whether the confidence head can learn given a perfectly
                stationary target (no diffusion-sample-to-sample variation at all).
        """
        # (Lazy import)
        from rf3.model.layers.af3_auxiliary_heads import ConfidenceHead  # noqa

        # Pass mini_rollout_sampler through to RF3 (required kw-only); then
        # recreate from the confidence-specific config so 20-step rollout wins
        # even if a structure net default was merged differently.
        super().__init__(mini_rollout_sampler=mini_rollout_sampler, **kwargs)

        self.confidence_head = ConfidenceHead(**confidence_head)
        self.mini_rollout_sampler = SampleDiffusion(**mini_rollout_sampler)
        self.mock_fixed_diffusion_rollout = mock_fixed_diffusion_rollout
        self._mock_rollout_cache = None

    def _score_with_confidence_head(
        self,
        trunk_stream: dict,
        X_rollout_BDL: torch.Tensor,
        input: dict,
        is_padding_I: torch.Tensor,
    ) -> dict:
        """Run the confidence head over every diffusion sample for one trunk stream.

        Looped over D because the full batch OOMs; see the TODO at the call site.
        """
        S_inputs_I, S_trunk_I, Z_trunk_II = trunk_tensors(trunk_stream)
        confidence_stack = {}
        for i in range(X_rollout_BDL.shape[1]):
            confidence = checkpoint.checkpoint(
                create_custom_forward(
                    self.confidence_head,
                    frame_atom_idxs=input["frame_atom_idxs"],
                ),
                S_inputs_I,
                S_trunk_I,
                Z_trunk_II,
                X_rollout_BDL[:, i],  # [B, L, 3]
                input["seq"],
                input["rep_atom_idxs"],
                is_padding_I,
                use_reentrant=False,
            )
            for k, v in confidence.items():
                if k in confidence_stack:
                    confidence_stack[k] = torch.cat((confidence_stack[k], v), dim=0)
                else:
                    confidence_stack[k] = v
        return confidence_stack

    def _run_rollout_no_grad(
        self,
        input: dict,
        padding_masks: dict,
        n_cycle: int,
        coord_atom_lvl_to_be_noised: torch.Tensor | None,
        should_early_stop_fn: ShouldEarlyStopFn | None,
        diffusion_batch_size: int,
        is_padding_I: torch.Tensor,
        is_padding_L: torch.Tensor,
        is_padding_chiral: torch.Tensor | None,
        cfg_guidance_inputs: list[dict] | None = None,
        inference_classifier_free_guidance: bool = False,
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        motif_guidance: Any = None,
    ) -> dict:
        """Grad-free trunk recycling + diffusion rollout for confidence training/eval.

        Returns ``{"recycling_outputs", "sample_diffusion_outs", "distogram_pred"}``,
        or ``{"early_stopped": True, ...}`` if early-stop fires. These tensors are what
        the confidence head needs and what :mod:`rf3.model.rollout_cache` persists.
        """
        with torch.no_grad():
            recycling_output_generator = self.trunk_forward_with_recycling(
                f=input["f"], padding_masks=padding_masks, n_recycles=n_cycle
            )
            if should_early_stop_fn:
                assert (
                    not self.training
                ), "Early stopping is not supported during training!"
                first_recycle_outputs = next(recycling_output_generator)
                confidence_outputs = checkpoint.checkpoint(
                    create_custom_forward(
                        self.confidence_head,
                        frame_atom_idxs=input["frame_atom_idxs"],
                    ),
                    first_recycle_outputs["S_inputs_I"],
                    first_recycle_outputs["S_I"],
                    first_recycle_outputs["Z_II"],
                    None,  # Omit structure
                    input["seq"],
                    input["rep_atom_idxs"],
                    is_padding_I,
                    use_reentrant=False,
                )
                should_early_stop, early_stop_data = should_early_stop_fn(
                    confidence_outputs=confidence_outputs,
                    first_recycle_outputs=first_recycle_outputs,
                )
                if should_early_stop:
                    result = {"early_stopped": True}
                    return result | early_stop_data

            try:
                recycling_outputs = deque(recycling_output_generator, maxlen=1).pop()
            except IndexError:
                raise RuntimeError("Recycling generator produced no outputs")

            distogram_pred = self.distogram_head(recycling_outputs["Z_II"])

            sample_diffusion_kwargs = dict(
                f=input["f"],
                S_inputs_I=recycling_outputs["S_inputs_I"],
                S_trunk_I=recycling_outputs["S_I"],
                Z_trunk_II=recycling_outputs["Z_II"],
                diffusion_batch_size=diffusion_batch_size,
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                is_padding_I=is_padding_I,
                is_padding_L=is_padding_L,
                is_padding_chiral=is_padding_chiral,
            )

            # Classifier-free guidance: same construction as RF3.forward -- conditioning
            # enters at the trunk, so each reference input needs its own trunk pass.
            # Inference only; the training mini-rollout stays unguided. An empty
            # `cfg_guidance_inputs` means the engine found no conditions to split off and
            # left `input` conditional, so running it unguided here is correct (the engine
            # warns about that case, where the two are distinguishable).
            cfg_guidance_streams = None
            if (
                inference_classifier_free_guidance
                and cfg_guidance_inputs
                and not self.training
            ):
                cfg_guidance_streams = []
                for ref in cfg_guidance_inputs:
                    ref_padding_masks = _guidance_stream_padding_masks(ref, padding_masks)
                    try:
                        ref_recycling_outputs = deque(
                            self.trunk_forward_with_recycling(
                                f=ref["f"],
                                padding_masks=ref_padding_masks,
                                n_recycles=n_cycle,
                            ),
                            maxlen=1,
                        ).pop()
                    except IndexError:
                        raise RuntimeError(
                            "Reference recycling generator produced no outputs"
                        )
                    cfg_guidance_streams.append(
                        {
                            "f": ref["f"],
                            "S_inputs_I": ref_recycling_outputs["S_inputs_I"],
                            "S_trunk_I": ref_recycling_outputs["S_I"],
                            "Z_trunk_II": ref_recycling_outputs["Z_II"],
                            "is_padding_chiral": ref_padding_masks.get(
                                "num_chiral_centers"
                            ),
                            "scale": ref["scale"],
                            "name": ref.get("name"),
                        }
                    )
                sample_diffusion_kwargs.update(
                    inference_classifier_free_guidance=True,
                    cfg_guidance_streams=cfg_guidance_streams,
                    cfg_sigma_min=cfg_sigma_min,
                    cfg_sigma_max=cfg_sigma_max,
                )

            # Motif guidance gets its own guard rather than riding on the CFG block
            # above: it is independent of CFG, so folding it in there would make it
            # require CFG to take effect. It must also stay out of the base kwargs,
            # which the training mini-rollout below shares.
            if motif_guidance is not None:
                if self.training:
                    raise RuntimeError(
                        "motif_guidance is inference-only; the training mini-rollout "
                        "must stay unguided."
                    )
                sample_diffusion_kwargs.update(motif_guidance=motif_guidance)

            if self.training:
                sample_diffusion_outs = (
                    self.mini_rollout_sampler.sample_diffusion_like_af3(
                        diffusion_module=self.diffusion_module,
                        **sample_diffusion_kwargs,
                    )
                )
            else:
                sample_diffusion_outs = (
                    self.inference_sampler.sample_diffusion_like_af3(
                        diffusion_module=self.diffusion_module,
                        **sample_diffusion_kwargs,
                    )
                )

        result = {
            "recycling_outputs": recycling_outputs,
            "sample_diffusion_outs": sample_diffusion_outs,
            "distogram_pred": distogram_pred,
        }
        if cfg_guidance_streams:
            # Kept so the confidence head can score them without a second trunk pass.
            # Only set under guidance, which never hits the rollout cache.
            result["cfg_guidance_streams"] = cfg_guidance_streams
        return result

    def forward(
        self,
        input: dict,
        padding_masks: dict,
        n_cycle: int,
        coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
        should_early_stop_fn: ShouldEarlyStopFn | None = None,
        rollout_cache_dir: str | None = None,
        rollout_cache_key: str | None = None,
        cfg_guidance_inputs: list[dict] | None = None,
        inference_classifier_free_guidance: bool = False,
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        motif_guidance: Any = None,
        confidence_streams: tuple[str, ...] | list[str] | None = None,
    ) -> dict:
        """Complete forward pass of the model with confidence head.

        Notes:
            - Performs a mini-rollout without gradients during training (e.g., 20 timesteps) and a full rollout (e.g., 200 timesteps) during inference
            - Runs the trunk forward without gradients to conserve memory (which departs from the AF-3 implementation)
            - Runs the forward pass (with gradients) for the confidence model
            - Uses the same ``padding_masks`` (True = padding) as the structure trunk /
              diffusion path for recycling, rollout, and the confidence Pairformer
            - In eval, optional ``rollout_cache_dir`` / ``rollout_cache_key`` skip recomputing
              the frozen trunk+diffusion rollout (see :mod:`rf3.model.rollout_cache`)

        Args:
            input (dict): Dictionary of model inputs. In addition to the standard AF-3 model inputs, we expect:
                - rep_atom_idxs: TBD
                - frame_atom_idxs: TBD
            padding_masks: Collate padding masks (keys ``I``, ``L``, ``S``, ...),
                True = padding — same dict passed to :meth:`RF3.forward`.
            n_cycle (int): Number of recycling cycles for the trunk
            coord_atom_lvl_to_be_noised (torch.Tensor): Atom-level coordinates to be noised further. Optional;
                only used during inference for partial denoising.
            should_early_stop_fn(Callable): Function that takes the confidence and trunk outputs after the first recycle and returns a boolean
                indicating whether to stop early and a dictionary with additional information (e.g., value and threshold).
                If None, no early stopping is performed. Optional; only used during inference.
            rollout_cache_dir: Directory for validation rollout cache. ``None`` disables caching.
            rollout_cache_key: Stable example id used as the cache filename stem.

        Returns:
            dict: Dictionary of model outputs, including:
                - X_L: Predicted atomic coordinates [D, L, 3] (from the mini rollout during training or full rollout during inference)
                - plddt: TBD
                - pae: TBD
                - pde: TBD
                - exp_resolved: TBD
        """
        # Cast features to lower precision if autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype("cuda")
            for x in [
                "msa_stack",
                "profile",
                "template_distogram",
                "template_restype",
                "template_unit_vector",
            ]:
                if x in input["f"]:
                    input["f"][x] = input["f"][x].to(autocast_dtype)
                # Reference inputs get their own trunk pass, so they need the same cast.
                for ref in cfg_guidance_inputs or []:
                    if x in ref["f"]:
                        ref["f"][x] = ref["f"][x].to(autocast_dtype)

        # Same masks the structure forward uses (True = padding)
        is_padding_I = padding_masks["I"]
        is_padding_L = padding_masks["L"]
        is_padding_chiral = padding_masks.get("num_chiral_centers", None)

        # Diffusion batch size is 2nd dim of t; first dim is the structure/pairformer batch
        # (same convention as :meth:`RF3.forward`).
        t = input["t"]
        diffusion_batch_size = t.shape[1] if t.ndim > 1 else t.shape[0]

        # Debug-only: reuse the exact same frozen rollout on every training step (computed
        # once, cached in memory) instead of resampling fresh diffusion noise each step.
        # See ``mock_fixed_diffusion_rollout`` docstring in __init__.
        if self.training and getattr(self, "mock_fixed_diffusion_rollout", False):
            if self._mock_rollout_cache is None:
                logger.info(
                    "mock_fixed_diffusion_rollout=True: computing rollout once and "
                    "reusing it verbatim on every subsequent training step."
                )
                self._mock_rollout_cache = self._run_rollout_no_grad(
                    input=input,
                    padding_masks=padding_masks,
                    n_cycle=n_cycle,
                    coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                    should_early_stop_fn=should_early_stop_fn,
                    diffusion_batch_size=diffusion_batch_size,
                    is_padding_I=is_padding_I,
                    is_padding_L=is_padding_L,
                    is_padding_chiral=is_padding_chiral,
                )
            rollout = self._mock_rollout_cache
        else:
            rollout = None

        # Optional validation rollout cache: trunk + diffusion are frozen for conf-head
        # fine-tuning, so repeat evals can reuse S/Z/X and only rerun the confidence head.
        _use_rollout_cache = (
            rollout is None
            and rollout_cache_dir is not None
            and rollout_cache_key is not None
            and should_early_stop_fn is None
            and not self.training
            # The cache key covers (base weights, example_id) only, so ANY guided run
            # would hit an entry written by an unguided one -- and could write a guided
            # rollout for an unguided run to read back.
            and not inference_classifier_free_guidance
            and motif_guidance is None
        )
        _cache_ns = None
        if _use_rollout_cache:
            if getattr(self, "_rollout_base_fingerprint", None) is None:
                self._rollout_base_fingerprint = compute_base_fingerprint(self)
            _cache_ns = self._rollout_base_fingerprint

        if rollout is None and _use_rollout_cache:
            rollout = try_load_rollout_cache(
                rollout_cache_dir, _cache_ns, rollout_cache_key, device=t.device
            )
        if rollout is None:
            rollout = self._run_rollout_no_grad(
                input=input,
                padding_masks=padding_masks,
                n_cycle=n_cycle,
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                should_early_stop_fn=should_early_stop_fn,
                diffusion_batch_size=diffusion_batch_size,
                is_padding_I=is_padding_I,
                is_padding_L=is_padding_L,
                is_padding_chiral=is_padding_chiral,
                cfg_guidance_inputs=cfg_guidance_inputs,
                inference_classifier_free_guidance=inference_classifier_free_guidance,
                cfg_sigma_min=cfg_sigma_min,
                cfg_sigma_max=cfg_sigma_max,
                motif_guidance=motif_guidance,
            )
            if rollout.get("early_stopped"):
                return rollout
            if _use_rollout_cache:
                save_rollout_cache(
                    rollout_cache_dir, _cache_ns, rollout_cache_key, rollout
                )

        recycling_outputs = rollout["recycling_outputs"]
        sample_diffusion_outs = rollout["sample_diffusion_outs"]
        distogram_pred = rollout["distogram_pred"]

        # ... run non-batched confidence head
        # TODO: Write a version of the confidence head that splits into batches based on memory available
        # (Currently, we OOM with the full batch size, so we loop, which is slow)
        # X_L from the sampler is [B, D, L, 3]
        X_rollout_BDL = sample_diffusion_outs["X_L"]

        confidence = self._score_with_confidence_head(
            recycling_outputs, X_rollout_BDL, input, is_padding_I
        )

        # Optional extra streams. Every stream scores the *same* rolled-out structure;
        # what differs is the trunk representation the head conditions on. Under CFG the
        # main stream is the unconditional input, so `main` confidence says nothing about
        # the conditioning -- that is what these are for.
        # Extras only; the main stream's logits are the four top-level keys below.
        confidence_by_stream = {}
        extra_streams = set(confidence_streams or ()) - {"main"}
        guidance_streams = rollout.get("cfg_guidance_streams") or []
        if extra_streams and not guidance_streams:
            logger.warning(
                f"confidence_streams={sorted(extra_streams)} requested but this input "
                "produced no guidance streams; only the main stream will be written."
            )
        if extra_streams and guidance_streams:
            # `merged` needs every stream's trunk tensors, so score it before the
            # per-condition loop frees them.
            if "merged" in extra_streams:
                confidence_by_stream["merged"] = self._score_with_confidence_head(
                    extrapolate_trunk_streams(recycling_outputs, guidance_streams),
                    X_rollout_BDL,
                    input,
                    is_padding_I,
                )
            if "conditions" in extra_streams:
                for idx, stream in enumerate(guidance_streams):
                    name = stream.get("name") or f"condition_{idx}"
                    confidence_by_stream[f"condition-{name}"] = (
                        self._score_with_confidence_head(
                            stream, X_rollout_BDL, input, is_padding_I
                        )
                    )

        # Drop the retained trunk tensors as soon as nothing else needs them. Z_trunk_II
        # alone is [B, I, I, c_z] per stream, and these are otherwise held for the whole
        # confidence loop -- unlike before, when the sampler was their last reader.
        for stream in guidance_streams:
            for key in ("S_inputs_I", "S_trunk_I", "Z_trunk_II"):
                stream.pop(key, None)
        rollout.pop("cfg_guidance_streams", None)

        # TODO: Return outputs in a more structured way (e.g., a dataclass)
        return dict(
            early_stopped=False,
            # We return X_L from diffusion sampling as X_pred_rollout_L to support future joint training with the confidence head (where we would have both X_L and X_pred_rollout_L)
            X_L=None,
            distogram=distogram_pred,
            # For reporting, inference (validation or testing) only
            X_noisy_L_traj=sample_diffusion_outs["X_noisy_L_traj"],
            X_denoised_L_traj=sample_diffusion_outs["X_denoised_L_traj"],
            # Not in the rollout cache, like the trajectories above; only set under CFG,
            # which bypasses it
            X_denoised_L_guided_traj=sample_diffusion_outs.get("X_denoised_L_guided_traj", []),
            X_denoised_L_stream_trajs=sample_diffusion_outs.get("X_denoised_L_stream_trajs", {}),
            t_hats=sample_diffusion_outs["t_hats"],
            # Confidence outputs
            X_pred_rollout_L=sample_diffusion_outs["X_L"],
            plddt=confidence["plddt_logits"],
            pae=confidence["pae_logits"],
            pde=confidence["pde_logits"],
            exp_resolved=confidence["exp_resolved_logits"],
            # Extra streams only, as {stream_name: {plddt_logits, pae_logits, pde_logits,
            # exp_resolved_logits}}; the main stream is the four keys above.
            confidence_by_stream=confidence_by_stream,
        )
