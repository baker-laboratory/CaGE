import torch
import torch.nn as nn
from rf3.model.layers.utils.chirals import calc_chiral_grads_flat_impl
from rf3.model.layers.diffusion.atom_transformer_diffusion import (
    AtomTransformer,
)
from rf3.model.layers.utils.layer_utils import (
    NormBiasConfig,
    linearNoBias,
)
from rf3.model.layers.components.mlff import (
    ConformerEmbeddingWeightedAverage,
)
from rf3.conditioning_2d import compute_scales

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.torch import device_of


class AtomAttentionEncoderDiffusion(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_tokenpair,
        c_s,
        atom_1d_features,
        c_atom_1d_features,
        atom_transformer,
        use_chiral_features,
        no_grad_on_chiral_center,
        ref_pos_feature_name: str = "ref_pos",
        use_atom_level_embedding: bool = False,
        atom_level_embedding_dim: int = 384,
        normalize_chiral_grads: bool = True,
        norm_bias: NormBiasConfig = NormBiasConfig(),
        conditioning_2d_processor: nn.Module | None = None,
        fourier_min_scale: float = 1.0 / 9.0,
        fourier_max_scale: float = 64.0,
        fourier_num_scales: int = 32,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.c_s = c_s
        self.atom_1d_features = atom_1d_features
        self.ref_pos_feature_name = ref_pos_feature_name
        self.use_chiral_features = use_chiral_features
        self.no_grad_on_chiral_center = no_grad_on_chiral_center
        self.use_atom_level_embedding = use_atom_level_embedding
        self.atom_level_embedding_dim = atom_level_embedding_dim
        self.normalize_chiral_grads = normalize_chiral_grads

        # Add 2D atom-level distance conditioning (C_DIS) on P_LL
        self.conditioning_2d_processor = conditioning_2d_processor
        if self.conditioning_2d_processor is not None:
            # Project back to P_LL; Linear is zero-init'd in reset_parameters
            # so the residual starts as a no-op.
            # bias=False on the LayerNorm so distance_pair_proj(0) == 0, keeping
            # the unconditioned-position contribution exactly zero.
            self.distance_pair_proj = nn.Sequential(
                nn.LayerNorm(c_atompair, bias=False),
                linearNoBias(c_atompair, c_atompair),
            )
            scales = compute_scales(
                min_scale=fourier_min_scale,
                max_scale=fourier_max_scale,
                num_scales=fourier_num_scales,
                spacing="log",
            )
            # Buffer so it follows the module's device/dtype but is not trained.
            self.register_buffer("fourier_scales", scales, persistent=False)

        self.process_input_features = linearNoBias(c_atom_1d_features, c_atom)

        self.process_d = linearNoBias(3, c_atompair)  # x,y,z

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

        self.process_s_trunk = nn.Sequential(
            nn.LayerNorm(c_s, bias=norm_bias.projection), linearNoBias(c_s, c_atom)
        )
        self.process_z = nn.Sequential(
            nn.LayerNorm(c_tokenpair, bias=norm_bias.projection), linearNoBias(c_tokenpair, c_atompair)
        )
        self.process_r = linearNoBias(3, c_atom)
        if self.use_chiral_features:
            self.process_ch = linearNoBias(3, c_atom)

        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
        )

        self.process_q = nn.Sequential(
            linearNoBias(c_atom, c_token),
            nn.ReLU(),
        )

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, norm_bias=norm_bias, **atom_transformer
        )

        if self.use_atom_level_embedding:
            self.process_atom_level_embedding = ConformerEmbeddingWeightedAverage(
                atom_level_embedding_dim=self.atom_level_embedding_dim,
                c_atompair=c_atompair,
                c_atom=c_atom,
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_chiral_features:
            nn.init.zeros_(self.process_ch.weight)
        # Zero-init, enabling 2D conditioning on a pretrained checkpoint
        # starts as a no-op residual.
        if self.conditioning_2d_processor is not None:
            nn.init.zeros_(self.distance_pair_proj[1].weight)

    def add_2d_conditioning_to_P_LL(
        self,
        f: dict,
        P_LL: torch.Tensor,
    ) -> torch.Tensor:
        """Adds atom-based 2D conditioning to P_LL.  
        Stay sparse as much as possible.
        """
        if self.conditioning_2d_processor is None:
            return P_LL

        raw_dist = f["condition_distance_2_atom"]  # [B, L, L]
        mask = f["mask_distance_2_atom"]  # [B, L, L] bool

        idx = mask.nonzero(as_tuple=True)
        scales_bf16 = self.fourier_scales.to(torch.bfloat16)

        if idx[0].numel() == 0:
            # Run a 1-element dummy through the full 2D-conditioning stack and
            # add zero to P_LL so DDP sees grads for distance_pair_proj +
            # conditioning_2d_processor on ranks where the mask is empty.
            dummy_dist = raw_dist.new_zeros(1, dtype=torch.bfloat16)
            with torch.autocast(device_type=raw_dist.device.type, dtype=torch.bfloat16):
                dummy_sparse = self.conditioning_2d_processor.forward_sparse(
                    masked_dist=dummy_dist,
                    fourier_scales=scales_bf16,
                )
            dummy_proj = self.distance_pair_proj(dummy_sparse.to(P_LL.dtype))
            return P_LL + 0.0 * dummy_proj.sum()

        masked_dist = raw_dist[idx].to(torch.bfloat16)
        with torch.autocast(device_type=raw_dist.device.type, dtype=torch.bfloat16):
            sparse_out = self.conditioning_2d_processor.forward_sparse(
                masked_dist=masked_dist,
                fourier_scales=scales_bf16,
            )
        sparse_proj = self.distance_pair_proj(sparse_out.to(P_LL.dtype))  # [n_valid, c_atompair]

        # add to dense
        return torch.index_put(P_LL, idx, sparse_proj, accumulate=True)

    def mean_atom_features_tokenwise(self, processed_Q_L, tok_idx, I, eps=1e-3):
        """
        Aggregate per-atom representation to per-token representation while handling padding tokens.
        """
        valid_mask = tok_idx >= 0  # [B, L]
        tok_idx = torch.where(
            valid_mask, tok_idx, torch.zeros_like(tok_idx)
        )  # replace -1 with 0 for one-hot
        one_hot = torch.nn.functional.one_hot(
            tok_idx.long(), num_classes=I
        ).to(processed_Q_L.dtype)  #

        one_hot = one_hot * valid_mask.unsqueeze(-1).to(
            one_hot.dtype
        )  # zero out invalid positions
        token_sums = torch.einsum("bdlc,bdlt->bdtc", processed_Q_L, one_hot[:, None])
        token_counts = one_hot.sum(dim=-2)
        A_I = token_sums / (token_counts[:, None, ..., None] + eps)
        return A_I

    def forward(
        self,
        f,  # Dict (Input feature dictionary)
        R_L,  # [B, D, L, 3]
        S_trunk_I,  # [B, I, C_S_trunk] [...,I,C_S_trunk]
        Z_II,  # [B, I, I, C_Z] [...,I,I,C_Z]
        is_padding_L,  # [B, L]
        is_padding_chiral,  # [B, num_chiral_centers]
    ):
        assert R_L is not None

        tok_idx = f["atom_to_token_map"]
        B, L = tok_idx.shape
        I = f["profile"].shape[1]
        D = R_L.shape[1]

        # Create the atom single conditioning: Embed per-atom meta data
        C_L = self.process_input_features(
            torch.cat(
                tuple(
                    f[feature_name].reshape(B, L, -1)
                    for feature_name in self.atom_1d_features
                ),
                dim=-1,
            )
        )

        if self.use_atom_level_embedding:
            assert "atom_level_embedding" in f
            C_L = C_L + self.process_atom_level_embedding(f["atom_level_embedding"])

        # Embed offsets between atom reference positions
        D_LL = f[self.ref_pos_feature_name].unsqueeze(-2) - f[self.ref_pos_feature_name].unsqueeze(-3)
        V_LL = (
            f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
        ).unsqueeze(-1)
        P_LL = self.process_d(D_LL) * V_LL

        mean_atom_features = self.mean_atom_features_tokenwise

        @activation_checkpointing
        def embed_atom_feats(R_L, C_L, D_LL, V_LL, P_LL):
            # Embed pairwise inverse squared distances, and the valid mask
            if self.training:
                P_LL = (
                    P_LL
                    + self.process_inverse_dist(
                        1 / (1 + torch.sum(D_LL * D_LL, dim=-1, keepdim=True))
                    )
                    * V_LL
                )

                P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
            else:
                P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                    1
                    / (
                        1
                        + torch.sum(
                            D_LL[V_LL[..., 0]] * D_LL[V_LL[..., 0]],
                            dim=-1,
                            keepdim=True,
                        )
                    )
                )

                P_LL[V_LL[..., 0]] += self.process_valid_mask(
                    V_LL[V_LL[..., 0]].to(P_LL.dtype)
                )

            # Initialise the atom single representation as the single conditioning.
            Q_L = C_L

            # If provided, add trunk embeddings and noisy positions.
            # Broadcast the single and pair embedding from the trunk.
            tok_idx_no_padding = tok_idx.clamp(min=0)
            S_trunk_embed_I = self.process_s_trunk(S_trunk_I)
            S_trunk_embed_L = torch.gather(
                S_trunk_embed_I,
                dim=1,
                index=tok_idx_no_padding[..., None].expand(-1, -1, self.c_atom).long(),
            )

            C_L = C_L + S_trunk_embed_L
            assert not (C_L == Q_L).all()
            tok_idx_rows = tok_idx_no_padding[..., None, None].expand(
                -1, -1, I, self.c_atompair
            )
            tok_idx_cols = tok_idx_no_padding[..., None, :, None].expand(
                -1, L, -1, self.c_atompair
            )
            processed_Z_II = self.process_z(Z_II)
            processed_Z_LL = torch.gather(
                torch.gather(
                    processed_Z_II,
                    dim=1,
                    index=tok_idx_rows.long(),
                ),
                dim=2,
                index=tok_idx_cols.long(),
            )

            P_LL = P_LL + processed_Z_LL

            # Add the noisy positions.
            Q_L = Q_L[:, None] + self.process_r(R_L)

            # Add chirality gradients
            if self.use_chiral_features:
                with torch.autocast(device_type=device_of(self).type, enabled=False):
                    # Do not pass grads through grad calc
                    R_L = calc_chiral_grads_flat_impl(
                        R_L.detach(),  # padding atom positions zeroed out
                        f["chiral_centers"],
                        f["chiral_center_dihedral_angles"],
                        self.no_grad_on_chiral_center,
                        is_padding_chiral,
                        is_padding_L,
                    ).nan_to_num()
                # Soft per-structure normalization: divide by the max per-atom
                # gradient norm within each (batch, diffusion) sample + 1.
                # Preserves relative magnitudes across atoms while bounding the
                # scale regardless of noise level or dihedral error magnitude.
                if self.normalize_chiral_grads:
                    per_atom_norm = R_L.norm(dim=-1, keepdim=True)          # [B, D, L, 1]
                    max_norm = per_atom_norm.amax(dim=2, keepdim=True) + 1.0  # [B, D, 1, 1]
                    R_L = R_L / max_norm
                Q_L = self.process_ch(R_L) + Q_L

            # Add the combined single conditioning to the pair representation.
            P_LL = P_LL + (
                self.process_single_l(C_L).unsqueeze(-2)
                + self.process_single_m(C_L).unsqueeze(-3)
            )

            # Optional 2D atom-level distance conditioning (C_DIS). 
            # No-op when ``conditioning_2d_processor`` is not configured.
            P_LL = self.add_2d_conditioning_to_P_LL(f, P_LL)

            # Run a small MLP on the pair activations
            P_LL = P_LL + self.pair_mlp(P_LL)

            # combine the B, D dimensions for atom transformer
            C_L = C_L[:, None]
            P_LL = P_LL[:, None]
            # Cross attention transformer.
            Q_L = self.atom_transformer(Q_L, C_L, P_LL, is_padding_L)

            #fd: checkpoint L x L op (needed for natom>3400 on L40)
            processed_Q_L = activation_checkpointing(self.process_q)(
                Q_L
            )  # [L, C_atom] -> [L, C_token]

            A_I = mean_atom_features(processed_Q_L, tok_idx, I)  # [I, C_token]

            return A_I, Q_L, C_L, P_LL

        return embed_atom_feats(R_L, C_L, D_LL, V_LL, P_LL)
