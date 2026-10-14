import torch

from rf3.model.layers.components.atom_transformer import AtomTransformer
from rf3.model.layers.utils.layer_utils import (
    NormBiasConfig,
    linearNoBias,
)
from rf3.model.layers.components.mlff import (
    ConformerEmbeddingWeightedAverage,
)

from rf3.model.layers.embedders.relative_positional_encoding import (
    RelativePositionEncoding,
)
from rf3.conditioning_2d import compute_scales
from torch import nn

from foundry.training.checkpoint import activation_checkpointing


class AtomAttentionEncoderPairformer(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_tokenpair,
        atom_1d_features,
        c_atom_1d_features,
        atom_transformer,
        use_atom_level_embedding: bool = False,
        atom_level_embedding_dim: int = 384,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.atom_1d_features = atom_1d_features

        self.process_input_features = linearNoBias(c_atom_1d_features, c_atom)

        self.process_d = linearNoBias(3, c_atompair)
        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

        self.use_atom_level_embedding = use_atom_level_embedding

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
                atom_level_embedding_dim=atom_level_embedding_dim,
                c_atompair=c_atompair,
                c_atom=c_atom,
            )

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
        token_sums = torch.einsum("blc,blt->btc", processed_Q_L, one_hot)
        token_counts = one_hot.sum(dim=-2)
        A_I = token_sums / (token_counts[..., None] + eps)
        return A_I

    def forward(
        self,
        f,  # Dict (Input feature dictionary)
        is_padded_L,
    ):
        # ... get the number of atoms and tokens
        tok_idx = f["atom_to_token_map"]
        B, L = tok_idx.shape  # N_batch, N_atom
        I = f["profile"].shape[1]  # N_token

        # Atom single conditioning (C_L): Linearly embed concatenated per-atom features
        # (e.g., ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)
        C_L = (
            self.process_input_features(
                torch.cat(
                    tuple(
                        f[feature_name].reshape(B, L, -1)
                        for feature_name in self.atom_1d_features
                    ),
                    dim=-1,
                )
            )
            * ~is_padded_L[..., None]
        )  # [L, C_atom]

        if self.use_atom_level_embedding:
            assert "atom_level_embedding" in f
            C_L = (
                C_L
                + self.process_atom_level_embedding(f["atom_level_embedding"])
                * ~is_padded_L[..., None]
            )

        # Now, we have the single conditioning (C_L) for each atom. We will:
        # 1. Use C_L to initialize the pair atom representation
        # 2. Pass C_L as a skip connection to the diffusion module

        # Embed offsets between atom reference positions
        # ref_pos is of shape [L, 3], so ref_pos.unsqueeze(-2) is of shape [L, 1, 3] and ref_pos.unsqueeze(-3) is of shape [1, L, 3]
        # We then take the outer difference between these two tensors to get a tensor of shape [L, L, 3] (via broadcasting both to shape [L, L, 3], and then taking the difference)
        # TODO: prefetch only the relevant indices to avoid 0(L^2) memory usage
        D_LL = f["ref_pos"].unsqueeze(-2) - f["ref_pos"].unsqueeze(
            -3
        )  # [L, 1, 3] - [1, L, 3] -> [L, L, 3]

        # Create a mask indicating if two atoms are on the same chain AND the same residue (e.g., the same ref_space_uid)
        # AND both atoms are not padding
        # (We add a singleton dimension to the mask to make it broadcastable with D_LL, which will be useful later)
        V_LL = (
            f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
        ).unsqueeze(-1)  # [B, L, 1] == [B, 1, L] -> [B, L, L, 1]

        # Mask out pairs where either atom is padding
        is_valid_L = ~is_padded_L  # [B, L]
        V_LL = (
            V_LL
            & is_valid_L.unsqueeze(-1).unsqueeze(-1)
            & is_valid_L.unsqueeze(-2).unsqueeze(-1)
        )  # [B, L, L, 1]

        mean_atom_features = self.mean_atom_features_tokenwise

        @activation_checkpointing
        def embed_features(C_L, D_LL, V_LL):
            P_LL = self.process_d(D_LL) * V_LL  # [L, L, 3] -> [L, L, C_atompair]

            # Embed pairwise inverse squared distances, and the valid mask
            P_LL += (
                self.process_inverse_dist(
                    1 / (1 + torch.sum(D_LL * D_LL, dim=-1, keepdim=True))
                )
                * V_LL
            )  # [L, L, 1] -> [L, L, C_atompair]

            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL

            # Initialise the atom single representation as the single conditioning.
            # NOTE: We create a new view on the tensor, so that the original tensor is not modified (unless we perform an in-place operation)
            Q_L = C_L.clone()  # [L, C_atom]

            # Add the combined single conditioning to the pair representation.
            # (With a residual connection)
            P_LL = (
                P_LL
                + (
                    self.process_single_l(C_L).unsqueeze(-2)
                    + self.process_single_m(C_L).unsqueeze(-3)
                )
                * V_LL
            )  # [B, L, 1, C_atompair] + [B, 1, L, C_atompair] -> [B, L, L, C_atompair]

            # Run a small MLP on the pair activations
            # (With a residual connection)
            P_LL = P_LL + self.pair_mlp(
                P_LL
            )  # [L, L, C_atompair] -> [L, L, C_atompair]

            # Cross attention transformer
            Q_L = self.atom_transformer(Q_L, C_L, P_LL, is_padded_L)  # [L, C_atom]

            # Aggregate per-atom representation to per-token representation
            # (Set the per-token representation to be the mean activation of all atoms in the token)
            processed_Q_L = self.process_q(Q_L)  # [L, C_atom] -> [L, C_token]
            # Ensure dtype consistency for index_reduce
            # processed_Q_L = processed_Q_L.to(Q_L.dtype)

            A_I = mean_atom_features(processed_Q_L, tok_idx, I)  # [I, C_token]
            return A_I

        return embed_features(C_L, D_LL, V_LL)


class FeatureInitializer(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        c_s_inputs,
        input_feature_embedder,
        relative_position_encoding,
        conditioning_2d_processor: nn.Module | None = None,
        fourier_min_scale: float = 1.0 / 9.0,
        fourier_max_scale: float = 64.0,
        fourier_num_scales: int = 32,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.input_feature_embedder = InputFeatureEmbedder(
            c_atom=c_atom, c_atompair=c_atompair, norm_bias=norm_bias, **input_feature_embedder
        )
        self.to_s_init = linearNoBias(c_s_inputs, c_s)
        self.to_z_init_i = linearNoBias(c_s_inputs, c_z)
        self.to_z_init_j = linearNoBias(c_s_inputs, c_z)
        self.relative_position_encoding = RelativePositionEncoding(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

        # Optional 2D atom-level distance conditioning (C_DIS) injected into trunk Z_II.
        # Mirrors rfd3 ``DiffusionTokenEncoder.add_2d_conditioning_to_Z_II``
        self.conditioning_2d_processor = conditioning_2d_processor
        if self.conditioning_2d_processor is not None:
            self.atom_pair_to_z = linearNoBias(c_atompair, c_z)
            nn.init.zeros_(self.atom_pair_to_z.weight)
            scales = compute_scales(
                min_scale=fourier_min_scale,
                max_scale=fourier_max_scale,
                num_scales=fourier_num_scales,
                spacing="log",
            )
            self.register_buffer("fourier_scales", scales)

    def add_2d_conditioning_to_Z_II(
        self,
        f: dict,
        Z_II: torch.Tensor,
    ) -> torch.Tensor:
        """RFD3's DiffusionTokenEncoder.add_2d_conditioning_to_Z_II
        but stays sparse end-to-end.
        """
        if self.conditioning_2d_processor is None:
            return Z_II

        raw_dist = f["condition_distance_2_atom"]  # [B, L, L]
        mask = f["mask_distance_2_atom"]  # [B, L, L] bool
        tok_map = f["atom_to_token_map"]  # [B, L]

        idx = mask.nonzero(as_tuple=True)  # (b_idx, m_idx, n_idx), each [n_valid]
        masked_dist = raw_dist[idx].to(torch.bfloat16)
        scales_bf16 = self.fourier_scales.to(torch.bfloat16)
        with torch.autocast(device_type=raw_dist.device.type, dtype=torch.bfloat16):
            sparse_out = self.conditioning_2d_processor.forward_sparse(
                masked_dist=masked_dist,
                fourier_scales=scales_bf16,
            )  # [n_valid, c_atompair]
        sparse_out = sparse_out.to(Z_II.dtype)

        b_idx, m_idx, n_idx = idx
        B = mask.shape[0]
        I = Z_II.shape[-2]
        C = sparse_out.shape[-1]

        # Sparse atom -> sparse token
        tok_i = tok_map[b_idx, m_idx]
        tok_j = tok_map[b_idx, n_idx]
        keep = (tok_i >= 0) & (tok_j >= 0)
        b_idx = b_idx[keep]
        tok_i = tok_i[keep].long()
        tok_j = tok_j[keep].long()
        sparse_out = sparse_out[keep]

        # Sparse token -> dense token (counts and sums)
        token_sum = torch.zeros(B, I, I, C, dtype=sparse_out.dtype, device=sparse_out.device)
        token_sum.index_put_((b_idx, tok_i, tok_j), sparse_out, accumulate=True)

        token_count = torch.zeros(B, I, I, dtype=sparse_out.dtype, device=sparse_out.device)
        ones = torch.ones(b_idx.shape[0], dtype=sparse_out.dtype, device=sparse_out.device)
        token_count.index_put_((b_idx, tok_i, tok_j), ones, accumulate=True)

        # average dist over token
        token_dist = token_sum / token_count.clamp(min=1.0).unsqueeze(-1)

        return Z_II + self.atom_pair_to_z(token_dist)  # [B, I, I, c_z]

    def forward(
        self,
        f,
        is_padded_masks,
    ):
        is_padded_L = is_padded_masks["L"]
        is_padded_I = is_padded_masks["I"]  # Token-level padding mask

        S_inputs_I = self.input_feature_embedder(f, is_padded_L)
        S_init_I = self.to_s_init(S_inputs_I)
        Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + self.to_z_init_j(
            S_inputs_I
        ).unsqueeze(-2)

        # Add relative position encoding and mask padding tokens
        rel_pos_encoding = self.relative_position_encoding(f)
        # Mask out pairs where either token is padding
        is_valid_I = ~is_padded_I  # [B, I]
        is_valid_II = (is_valid_I.unsqueeze(-1) & is_valid_I.unsqueeze(-2)).unsqueeze(
            -1
        )  # [B, I, I, 1]
        Z_init_II = Z_init_II + rel_pos_encoding * is_valid_II.to(
            rel_pos_encoding.dtype
        )

        # Add token bonds (also mask)
        token_bonds = self.process_token_bonds(
            f["token_bonds"].unsqueeze(-1).to(torch.float)
        )
        Z_init_II = Z_init_II + token_bonds * is_valid_II.to(token_bonds.dtype)

        # Optional 2D atom-level distance conditioning (C_DIS)
        Z_init_II = self.add_2d_conditioning_to_Z_II(f, Z_init_II)

        return S_inputs_I, S_init_I, Z_init_II


class InputFeatureEmbedder(nn.Module):
    def __init__(self, features, c_atom, c_atompair, atom_attention_encoder, norm_bias: NormBiasConfig = NormBiasConfig()):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoderPairformer(
            c_atom=c_atom, c_atompair=c_atompair, norm_bias=norm_bias, **atom_attention_encoder
        )
        self.features = features
        self.features_to_unsqueeze = ["deletion_mean"]

    def forward(
        self,
        f,
        is_padded_L,
    ):
        A_I = self.atom_attention_encoder(f, is_padded_L)
        S_I = torch.cat(
            [A_I]
            + [
                f[feature].unsqueeze(-1)
                if feature in self.features_to_unsqueeze
                else f[feature]
                for feature in self.features
            ],
            dim=-1,
        )
        return S_I
