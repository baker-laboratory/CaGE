import torch
import torch.nn as nn
import torch.nn.functional as F
from rf3.model.recycler import PairformerBlock, linearNoBias

# TODO: Get from RF2AA encoding instead
CHEM_DATA_LEGACY = {"NHEAVY": 23, "aa2num": {"UNK": 20, "GLY": 7, "MAS": 21}}


def discretize_distance_matrix(
    distance_matrix, num_bins=38, min_distance=3.25, max_distance=50.75
):
    # Calculate the bin width
    bin_width = (max_distance - min_distance) / num_bins
    bins = (
        torch.arange(num_bins, device=distance_matrix.device) * bin_width + min_distance
    )

    # Discretize distances into bins (bucketize automatically places out-of-range values in the last bin)
    binned_distances = torch.bucketize(distance_matrix, bins)

    return binned_distances


class ConfidenceHead(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        c_s,
        c_z,
        n_pairformer_layers,
        pairformer,
        n_bins_pae,
        n_bins_pde,
        n_bins_plddt,
        n_bins_exp_resolved,
        legacy_full_reinit: bool = False,
    ):
        super(ConfidenceHead, self).__init__()
        # See `reset_parameters` docstring. Default (False) is the fix; set True
        # in config only to bit-for-bit reproduce a prior run's (buggy) reinit.
        self.legacy_full_reinit = legacy_full_reinit
        self.process_s_inputs_right = linearNoBias(449, c_z)
        self.process_s_inputs_left = linearNoBias(449, c_z)
        self.process_pred_distances = linearNoBias(40, c_z)
        self.normalize_pair_features = nn.LayerNorm(c_z)
        self.normalize_trunk_features = nn.LayerNorm(c_s)
        self.normalize_input_features = nn.LayerNorm(449)

        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer)
                for _ in range(n_pairformer_layers)
            ]
        )

        self.predict_pae = linearNoBias(c_z, n_bins_pae)
        self.predict_pde = linearNoBias(c_z, n_bins_pde)
        self.predict_plddt = linearNoBias(
            c_s, CHEM_DATA_LEGACY["NHEAVY"] * n_bins_plddt
        )
        self.predict_exp_resolved = linearNoBias(
            c_s, CHEM_DATA_LEGACY["NHEAVY"] * n_bins_exp_resolved
        )

    def reset_parameters(self):
        """Reinitialize for a fresh confidence-head fine-tune (e.g. bootstrapping
        from a structure-only checkpoint that has no confidence_head weights).

        Each Pairformer sublayer (TriangleMultiplication, TriangleAttention,
        AttentionPairBiasPairformer, Transition) already implements the AF3
        residual-init discipline in its own ``reset_parameters``: symmetry-breaking
        Xavier/lecun-normal on QKV/bias-projection layers, but *zero*-init on the
        last linear before every residual add and gates opened via bias=1, so the
        stack starts close to the identity function. Blanket-overwriting every
        ``nn.Linear`` in the module (the old behavior, still available via
        ``legacy_full_reinit`` for reproducing prior runs) destroys that: it
        un-zeros every residual-output projection and un-opens every gate, so all
        ~20 residual branches (4 blocks x 5 updates) inject full-magnitude random
        noise simultaneously on step 1 instead of starting near-identity.

        ``init_lecun_normal`` (used by several submodules' own ``reset_parameters``,
        e.g. ``TriangleMultiplication``'s ``p_in``) replaces the ``nn.Parameter``
        object outright via ``module.weight = nn.Parameter(sample_truncated_normal(...))``
        rather than filling the existing tensor in-place -- the new tensor lands on
        the CPU default device. That's harmless when ``reset_parameters`` only ever
        runs once, inside ``__init__``, before the model is moved to its training
        device. This method is explicitly meant to be called again later (bootstrapping
        a confidence fine-tune *after* the checkpoint/device setup has already placed
        the model on GPU), so we snapshot the device/dtype up front and restore it at
        the end -- otherwise those replaced parameters silently end up on CPU while the
        rest of the model is on GPU, and the first forward pass hard-crashes.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if self.legacy_full_reinit:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.to(device=device, dtype=dtype)
            return

        # Layers this module owns directly: plain nn.Linear with no gate/residual
        # role, so they need symmetry-breaking (Xavier) like the rest of AF3's
        # non-residual embed/readout layers.
        for m in (
            self.process_s_inputs_right,
            self.process_s_inputs_left,
            self.process_pred_distances,
            self.predict_pae,
            self.predict_pde,
            self.predict_plddt,
            self.predict_exp_resolved,
        ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Pairformer sublayers: re-run their own reset_parameters() rather than
        # touching their internal nn.Linear leaves directly, so the zero-init/
        # gate-open residual discipline is preserved.
        #
        # Some of these (e.g. TriangleMultiplication) only explicitly re-init a
        # subset of their own nn.Linear leaves (its docstring: "we use default
        # PyTorch initialization for the other parameters") -- fine the first
        # time reset_parameters() runs right after nn.Linear.__init__, but NOT
        # safe here: this method runs *after* checkpoint weight-loading, so any
        # leaf a submodule doesn't explicitly touch could still hold whatever
        # was preloaded into it (e.g. a copy-policy match, or values carried
        # over from a prior confidence-head checkpoint with different shapes/
        # semantics). Give every leaf a fresh PyTorch-default init first, so
        # there is no preloaded value left for a submodule's partial reset to
        # silently inherit -- then apply the submodule's own AF3-specific
        # corrections (lecun-normal, zero-before-residual, gate biases) on top.
        for block in self.pairformer:
            for submodule in (
                block.tri_mul_outgoing,
                block.tri_mul_incoming,
                block.tri_attn_start,
                block.tri_attn_end,
                block.z_transition,
                block.s_transition,
                block.attention_pair_bias,
            ):
                for leaf in submodule.modules():
                    if isinstance(leaf, (nn.Linear, nn.LayerNorm)):
                        leaf.reset_parameters()
                submodule.reset_parameters()

        self.to(device=device, dtype=dtype)

    @staticmethod
    def _mask_pad_features(S_I, Z_II, is_padding_I):
        """Zero pad token singles/pairs. ``is_padding_I`` True = padding (collate polarity)."""
        valid_I = (~is_padding_I).to(dtype=Z_II.dtype)
        Z_II = Z_II * valid_I[:, :, None, None] * valid_I[:, None, :, None]
        S_I = S_I * valid_I[..., None]
        return S_I, Z_II, valid_I

    def forward(
        self,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        X_pred_L,
        seq,
        rep_atoms,
        is_padding_I,
        frame_atom_idxs=None,
    ):
        # detach inputs (trunk/diffusion run under no_grad; only this head trains)
        S_inputs_I = S_inputs_I.detach().float()  # B, I, 449
        S_trunk_I = S_trunk_I.detach().float()  # B, I, 384
        Z_trunk_II = Z_trunk_II.detach().float()  # B, I, I, 128
        is_padding_I = is_padding_I.detach().bool()  # B, I; True = padding
        if X_pred_L is not None:
            X_pred_L = X_pred_L.detach().float()  # B, L, 3

        assert is_padding_I.shape[:2] == S_trunk_I.shape[:2], (
            f"is_padding_I shape {tuple(is_padding_I.shape)} "
            f"!= S_trunk_I batch/token {tuple(S_trunk_I.shape[:2])}"
        )

        # Zero pad before LayerNorm so pad tokens do not dominate LN stats when
        # most of the fixed-pad axis is empty (crop_size >> real tokens).
        S_trunk_I, Z_trunk_II, valid_I = self._mask_pad_features(
            S_trunk_I, Z_trunk_II, is_padding_I
        )
        S_inputs_I = S_inputs_I * valid_I[..., None]

        # normalize Zs,Ss
        S_trunk_I = self.normalize_trunk_features(S_trunk_I)
        S_inputs_I = self.normalize_input_features(S_inputs_I)
        Z_trunk_II = self.normalize_pair_features(Z_trunk_II)

        # embed S_inputs_I twice
        S_inputs_I_right = self.process_s_inputs_right(S_inputs_I)
        S_inputs_I_left = self.process_s_inputs_left(S_inputs_I)

        # add outer product of two linear embeddings of S_inputs_I  to Z_II
        Z_trunk_II = Z_trunk_II + (
            S_inputs_I_right.unsqueeze(-2) + S_inputs_I_left.unsqueeze(-3)
        )

        # if there are input coords, embed pairwise distances
        if X_pred_L is not None:
            # rep_atoms (freshly computed per-example from the current atom_array) and
            # X_pred_L (the diffusion rollout, possibly reused from a rollout cache --
            # see rollout_cache.py) must agree on atom count/ordering. A mismatch here
            # otherwise surfaces as an opaque, asynchronous
            # "CUDA error: device-side assert triggered" from index_select several lines
            # later, with no indication of the actual cause -- seen in practice from a
            # stale val_rollout_cache entry computed under an earlier version of the
            # data pipeline than the one now producing rep_atoms, for a handful of the
            # largest/most complex validation structures.
            if rep_atoms.numel() > 0 and (
                int(rep_atoms.max()) >= X_pred_L.shape[1] or int(rep_atoms.min()) < 0
            ):
                raise RuntimeError(
                    "rep_atoms out of bounds for X_pred_L: rep_atoms range "
                    f"[{int(rep_atoms.min())}, {int(rep_atoms.max())}], "
                    f"X_pred_L.shape={tuple(X_pred_L.shape)} (valid atom index range "
                    f"[0, {X_pred_L.shape[1] - 1}]). If X_pred_L came from a rollout "
                    "cache, this usually means the cached entry predates a data-pipeline "
                    "change that altered this example's atom count -- delete its cache "
                    "entry (rollout_cache.py's <cache_dir>/<fingerprint>/<example_id>.pt) "
                    "and retry."
                )
            X_pred_rep_I = X_pred_L.index_select(1, rep_atoms)
            dist = torch.cdist(X_pred_rep_I, X_pred_rep_I)

            dist_one_hot = F.one_hot(
                discretize_distance_matrix(
                    dist, min_distance=3.25, max_distance=50.75, num_bins=39
                ),
                num_classes=40,
            )

            Z_trunk_II = Z_trunk_II + self.process_pred_distances(dist_one_hot.float())

        # Re-zero after LN/embeds: LayerNorm rejuvenates pad rows from batch stats.
        # Pairformer triangle / attention then use the same is_padding_I as the trunk
        # (True = padding → cueq mask = ~pad). Without this, pad QKV can hit all-masked
        # attention rows and bleed NaN/garbage grads into valid tokens via residuals.
        S_trunk_I, Z_trunk_II, _ = self._mask_pad_features(
            S_trunk_I, Z_trunk_II, is_padding_I
        )

        # process with pairformer stack (same is_padding_I as structure trunk).
        # Activation-checkpoint each block during training -- same pattern as the
        # structure trunk's own Pairformer stack (recycler.py) -- so a deeper stack
        # (e.g. n_pairformer_layers=8) doesn't retain all blocks' activations for
        # backward simultaneously.
        for n in range(len(self.pairformer)):
            if torch.is_grad_enabled():
                S_trunk_I, Z_trunk_II = torch.utils.checkpoint.checkpoint(
                    self.pairformer[n], S_trunk_I, Z_trunk_II, is_padding_I, use_reentrant=False
                )
            else:
                S_trunk_I, Z_trunk_II = self.pairformer[n](
                    S_trunk_I, Z_trunk_II, is_padding_I
                )
            # Transition / residual can reintroduce pad content — clear again.
            S_trunk_I, Z_trunk_II, _ = self._mask_pad_features(
                S_trunk_I, Z_trunk_II, is_padding_I
            )

        pde_logits = self.predict_pde(Z_trunk_II)
        pde_logits = 0.5 * (pde_logits + pde_logits.transpose(-2, -3))

        pae_logits = self.predict_pae(Z_trunk_II)
        plddt_logits = self.predict_plddt(S_trunk_I)
        exp_resolved_logits = self.predict_exp_resolved(S_trunk_I)

        return dict(
            pde_logits=pde_logits,
            pae_logits=pae_logits,
            plddt_logits=plddt_logits,
            exp_resolved_logits=exp_resolved_logits,
        )


def calc_Cb_distances(X_pred_L, seq, rep_atoms, frame_atom_idxs):
    frame_atom_idxs = frame_atom_idxs.unsqueeze(0).expand(X_pred_L.shape[0], -1, -1)

    N = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
    )
    Ca = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
    )
    C = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
    )
    Cb = X_pred_L.index_select(1, rep_atoms)

    is_valid_Cb = (
        (seq != CHEM_DATA_LEGACY.aa2num["UNK"])
        & (seq != CHEM_DATA_LEGACY.aa2num["GLY"])
        & (seq != CHEM_DATA_LEGACY.aa2num["MAS"])
    )

    def _legacy_is_protein(seq):
        return (seq >= 0).all() & (seq < 20).all()

    is_valid_Cb = is_valid_Cb & _legacy_is_protein(seq)

    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)

    ideal_Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    Cb_distances = torch.norm(Cb - ideal_Cb, dim=-1)
    Cb_distances[:, ~is_valid_Cb] = 0.0

    return Cb_distances
