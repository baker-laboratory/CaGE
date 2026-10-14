import torch
import torch.nn as nn
import math

from einops import rearrange
from rf3.model.layers.utils.layer_utils import (
    AdaLN,
    LinearBiasInit,
    NormBiasConfig,
    linearNoBias,
)
from rf3.util_module import init_lecun_normal


class AtomTransformer(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        n_block,
        atom_transformer_block,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AtomTransformerBlock(
                    c_atom=c_atom,
                    c_atompair=c_atompair,
                    norm_bias=norm_bias,
                    **atom_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(
        self,
        Q_L,  # [B, L, C_atom]
        C_L,  # [B, L, C_atom]
        P_LL,  # [B, L, L, C_atompair]
        is_padding_L,
    ):
        for block in self.blocks:
            Q_L = block(
                Q_L,
                C_L,
                P_LL,
                is_padding_L=is_padding_L,
            )
        return Q_L


class AtomTransformerBlock(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        n_head,
        key_window_size: int,
        query_window_size: int,
        periodicity: int,
        qk_norm: bool,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.attention = WindowedAtomAttention(
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_head=n_head,
            key_window_size=key_window_size,
            query_window_size=query_window_size,
            periodicity=periodicity,
            qk_norm=qk_norm,
            norm_bias=norm_bias,
        )
        self.transition = ConditionedAtomTransition(
            c_atom=c_atom,
        )

    def forward(
        self,
        Q_L,  # [B, L, C_atom]
        C_L,  # [B, L, C_atom]
        P_LL,  # [B, L, L, C_atompair]
        is_padding_L,  # [B, L]
    ):
        Q_L = Q_L + self.attention(
            Q_L,
            C_L,
            P_LL,
            is_padding_L,
        )
        Q_L = Q_L + self.transition(
            Q_L,
            C_L,
        )
        return Q_L


class WindowedAtomAttention(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        n_head,
        qk_norm: bool,
        key_window_size: int, # number of atoms in key window
        query_window_size: int, # number of atoms in query window
        periodicity: int, # periodicity for window placement, should be equal to query_window_size
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.c = c_atom // n_head
        self.n_head = n_head
        self.key_window_size = key_window_size
        self.query_window_size = query_window_size
        self.periodicity = periodicity
        self.qk_norm = qk_norm

        assert query_window_size == periodicity, (
            "For now, only support query_window_size equal to periodicity"
        )
        assert query_window_size % 2 == 0, "For now, only support even query_window_size"
        assert key_window_size % 2 == 0, "For now, only support even key_window_size"

        self.ada_ln_1 = AdaLN(c_a=c_atom, c_s=c_atom)
        self.ln_pair = nn.LayerNorm((c_atompair,), bias=norm_bias.attention_io)

        self.to_qkv = nn.Linear(c_atom, 3 * self.n_head * self.c, bias=False)

        if self.qk_norm:
            self.q_norm = nn.LayerNorm((self.n_head * self.c,), bias=norm_bias.qk)
            self.k_norm = nn.LayerNorm((self.n_head * self.c,), bias=norm_bias.qk)

        self.to_bias = nn.Linear(c_atompair, n_head, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(c_atom, c_atom), nn.Sigmoid())

        self.to_out = nn.Linear(c_atom, c_atom, bias=False)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_atom, c_atom, biasinit=-2.0),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.to_qkv.weight)
        init_lecun_normal(self.to_bias)
        nn.init.zeros_(self.to_gate[0].weight)
        nn.init.ones_(self.to_gate[0].bias)
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        Q_L,  # [B, L, C_atom]
        C_L,  # [B, L, C_atom]
        P_LL,  # [B, L, L, C_atompair]
        is_padding_L,  # [B, L]
    ):
        Q_L = self.ada_ln_1(Q_L, C_L)
        B, L = C_L.shape[:2]

        Q_LH, K_LH, V_LH = [t.view(B, L, self.n_head, self.c) for t in self.to_qkv(Q_L).chunk(3, dim=-1)]

        G_LH = self.to_gate(Q_L)

        if self.qk_norm:
            Q_LH = self._qk_norm(Q_LH, self.q_norm)
            K_LH = self._qk_norm(K_LH, self.k_norm)

        # potential optimization: precompute the window masks and flattened indices for P_LL to avoid
        # materializing the full distance matrix ever
        bias = self.to_bias(self.ln_pair(P_LL))
        query_window_indices, key_window_indices = self._get_flattened_window_indices(
            L,
            self.query_window_size,
            self.key_window_size,
            periodicity=self.periodicity,
            device=Q_L.device,
        )

        out = self._flattened_dot_product_attention(
            Q_LH,
            K_LH,
            V_LH,
            bias,
            query_window_indices,
            key_window_indices,
            is_padding_L,
        )  # [B, n_head, query_window_size, num_windows, key_window_size]

        out = G_LH * out.view(B, L, self.n_head * self.c)

        out = self.to_out(out.view(B, L, -1))
        out = self.linear_output_project(C_L) * out
        return out

    def _qk_norm(self, tensor, norm_layer):
        """Apply LayerNorm across head and channel dimensions."""
        h = tensor.shape[-2]  # num_heads dimension
        tensor = rearrange(tensor, "... h c -> ... (h c)")
        tensor = norm_layer(tensor)
        tensor = rearrange(tensor, "... (h c) -> ... h c", h=h)
        return tensor

    @staticmethod
    def _get_flattened_window_indices(
        L: int,
        query_window_size: int,
        key_window_size: int,
        periodicity: int,
        device: torch.device,
    ):
        """Get flattened indices for windowed attention.

        Args:
            L: Length of the sequence (number of atoms).
            query_window_size: Size of the query attention window.
            key_window_size: Size of the key attention window.
            periodicity: Periodicity for window placement.
            device: Device to create the indices on.
        Returns:
            A tensor of shape [query_window_size, num_windows] containing flattened indices for the queries and
            a tensor of shape [key_window_size, num_windows] containing flattened indices for the keys/values.
        """
        assert (
            query_window_size == periodicity
        ), "For now, only support query_window_size equal to periodicity"
        assert (
            query_window_size % 2 == 0
        ), "For now, only support even query_window_size"
        assert key_window_size % 2 == 0, "For now, only support even key_window_size"

        max_number_of_windows = math.ceil(L / periodicity)
        subset_centers = ((periodicity - 1) / 2) + torch.arange(
            max_number_of_windows, device=device
        ) * periodicity

        # for each subset center, get the indices within the window
        query_window_representatives = torch.arange(
            query_window_size, device=device
        ) - (query_window_size // 2)
        key_window_representatives = torch.arange(key_window_size, device=device) - (
            key_window_size // 2
        )

        # for each subset center, get the indices within the window
        query_window_indices = (
            subset_centers[:, None] + query_window_representatives[None, :]
        )
        key_window_indices = (
            subset_centers[:, None] + key_window_representatives[None, :]
        )

        # subset centers are decimals so use floor to convert to int
        return query_window_indices, key_window_indices

    def _flattened_dot_product_attention(
        self,
        Q,  # [B, n_head, L, c]
        K,  # [B, n_head, L, c]
        V,  # [B, n_head, L, c]
        bias,  # [B, n_head, L, L]
        query_window_indices,  # [query_window_size, num_windows]
        key_window_indices,  # [key_window_size, num_windows]
        is_padding_L,  # [B, L]
    ):
        B, L, n_head, c = Q.shape

        # the window indices may be decimals, so we ceil them to get the correct indices
        query_window_indices, key_window_indices = (
            torch.ceil(query_window_indices),
            torch.ceil(key_window_indices),
        )

        # some of the indices are invalid since the input is not necessarily divisible by num_windows
        # (less than 0 or greater than L-1) so we create a mask to ignore them later on
        is_invalid_query = (query_window_indices < 0) | (
            query_window_indices > L - 1
        )  # [num_windows, query_window_size]
        is_invalid_key = (key_window_indices < 0) | (
            key_window_indices > L - 1
        )  # [num_windows, key_window_size]

        query_window_indices = torch.clamp(query_window_indices.int(), 0, L - 1)
        key_window_indices = torch.clamp(key_window_indices.int(), 0, L - 1)

        query_subset = Q[
            :, query_window_indices
        ]  # [B, n_head, query_window_size, num_windows, c]
        key_subset = K[
            :, key_window_indices
        ]  # [B, n_head, key_window_size, num_windows, c]
        attention = torch.einsum(
            "...ihd,...jhd->...ijh",
            query_subset,
            key_subset,
        )
        attention = attention / (self.c**0.5)

        # Add pair bias is where it gets complicated
        # query_window_indices: [num_windows, query_window_size]
        # key_window_indices: [num_windows, key_window_size]
        # We need to get bias_subset of shape [B, n_head, num_windows, query_window_size, key_window_size]
        bias_subset = bias[
            :,  # batch dimension
            query_window_indices[:, :, None],  # num_windows, num_query_windows, 1
            key_window_indices[:, None, :],  # num_windows, 1, num_key_windows
        ]  # [B, n_head, num_windows, query_window_size, key_window_size]

        # There is no guarantee the system is perfectly divisible into windows, so some indices may be invalid. We mask those out by setting their bias to a large negative value.
        # We also update the attention to ignore indices that are padding tokens.
        is_invalid_query_mask = (
            is_invalid_query[None] | is_padding_L[:, query_window_indices]
        )  # [B, num_windows, query_window_size]
        is_invalid_key_mask = (
            is_invalid_key[None] | is_padding_L[:, key_window_indices]
        )  # [B, num_windows, key_window_size]
        is_invalid_bias_mask = (
            is_invalid_query_mask[:, :, :, None, None]
            | is_invalid_key_mask[:, :, None, :, None]
        )  # [1, 1, num_windows, query_window_size, key_window_size]

        attention = attention + bias_subset
        attention = attention.masked_fill(is_invalid_bias_mask, torch.finfo(attention.dtype).min)

        attention = torch.softmax(attention, dim=-2)
        value_subset = V[
            :, key_window_indices
        ]  # [B, n_head, key_window_size, num_windows, c]

        out = torch.einsum(
            "...ijh,...jhc->...ihc",
            attention,
            value_subset,
        )
        out = out[
            :, ~is_invalid_query
        ]  # NOTE: this removes invalid query positions that were added to make the windows even sized, NOT padding tokens from batching

        return out

    @staticmethod
    def _get_window_mask(
        L: int,
        query_window_size: int,
        key_window_size: int,
        periodicity: int,
        device: torch.device,
    ):
        """Generate a window mask for attention.

        Args:
            L: Length of the sequence (number of atoms).
            window_size: Size of the attention window.
            device: Device to create the mask on.
        Returns:
            A boolean mask of shape [L, L] where True indicates positions within the window.
        """
        l, m = torch.arange(L, device=device), torch.arange(L, device=device)
        max_number_of_windows = math.ceil(
            L / periodicity
        )  # TODO: check if using math is bad for performance
        subset_centers = ((periodicity - 1) / 2) + torch.arange(
            max_number_of_windows, device=device
        ) * periodicity

        l_within_subset_window = (
            torch.abs(l[:, None] - subset_centers[None]) < query_window_size / 2
        )
        m_within_subset_window = (
            torch.abs(m[:, None] - subset_centers[None]) < key_window_size / 2
        )
        return torch.any(
            l_within_subset_window[:, None] * m_within_subset_window[None, :], dim=-1
        )

    def _naive_dot_product_attention(
        self,
        Q,  # [B, n_head, L, c]
        K,  # [B, n_head, L, c]
        V,  # [B, n_head, L, c]
        bias,  # [B, n_head, L, L]
        window_mask,  # [L, L]
        is_padding_L,  # [B, L]
    ):
        """
        Compute standard dot-product attention with bias and padding mask.
        Args:
            Q: Query tensor of shape [B, n_head, L, c]
            K: Key tensor of shape [B, n_head, L, c]
            V: Value tensor of shape [B, n_head, L, c]
            bias: Bias tensor of shape [B, n_head, L, L]
            is_padding_L: Padding mask of shape [B, L]
        Returns:
            Output tensor of shape [B, L, n_head, c]
        """
        B, L, n_head, c = Q.shape

        attention = torch.einsum(
            "...ihd,...jhd->...ijh",
            Q,
            K,
        )
        attention = attention / (self.c**0.5)

        attention = attention + bias
        attention = attention.masked_fill(~window_mask[None, :, :, None], torch.finfo(attention.dtype).min)
        # TODO: Apply padding mask
        is_padding_LL = (
            is_padding_L[:, None, None] * is_padding_L[None, :, None]
        )  # [B, L, L, 1]
        attention = attention.masked_fill(is_padding_LL.bool(), torch.finfo(attention.dtype).min)

        attention = torch.softmax(attention, dim=-2)
        out = torch.einsum(
            "...ijh,...jhc->...ihc",
            attention,
            V,
        )
        return out


class ConditionedAtomTransition(nn.Module):
    """
    Transition block for atom-level embeddings conditioned on scalar features.
    """

    def __init__(self, c_atom, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_atom, c_s=c_atom)
        self.linear_1 = linearNoBias(c_atom, c_atom * n)
        self.linear_2 = linearNoBias(c_atom, c_atom * n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_atom, c_atom, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_atom * n, c_atom)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_lecun_normal(self.linear_1)
        init_lecun_normal(self.linear_2)
        nn.init.zeros_(self.linear_3.weight)

    def forward(
        self,
        Q_L,  # [B, L, C_token]
        C_L,  # [B, L, C_token]
    ):
        Q_L = self.ada_ln(Q_L, C_L)
        B_L = torch.nn.functional.silu(self.linear_1(Q_L)) * self.linear_2(Q_L)

        # Output projection (from adaLN-Zero)
        return self.linear_output_project(C_L) * self.linear_3(B_L)
