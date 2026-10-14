import torch
import torch.nn as nn
from einops import rearrange
import foundry
import math
from abc import ABC, abstractmethod

from rf3.model.layers.components.atom_transformer import AtomTransformer
from rf3.model.layers.utils.layer_utils import linearNoBias, AdaLN, LinearBiasInit, NormBiasConfig
from rf3.util_module import init_lecun_normal
from rf3.model.layers.components.mlff import (
    ConformerEmbeddingWeightedAverage,
)
from foundry.training.checkpoint import activation_checkpointing

if foundry.SHOULD_USE_CUEQUIVARIANCE:
    import cuequivariance_torch as cuet


class AbstractAttentionPairBias(nn.Module, ABC):
    """Attention with pair bias for Pairformer blocks (token-level, dense).


    Routes to either cuEquivariance or vanilla implementation based on configuration.

    Args:
      c_a: Single representation dimension.
      c_s: Legacy parameter, kept for backwards compatibility (unused).
      c_pair: Pair representation dimension.
      n_head: Number of attention heads.
      use_cuequivariance: Use cuEquivariance fused kernel when available.
    """

    def __init__(self, c_a, c_pair, n_head, use_cuequivariance: bool, qk_norm: bool, norm_bias: NormBiasConfig = NormBiasConfig()):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head
        self.use_cuequivariance = use_cuequivariance
        self.qk_norm = qk_norm

        # (optional) QK norm
        if self.qk_norm:
            self.ln_k = nn.LayerNorm((n_head * self.c,), bias=norm_bias.qk)
            self.ln_q = nn.LayerNorm((n_head * self.c,), bias=norm_bias.qk)

        # Query, Key, Value projections (no bias)
        self.to_qkv = nn.Linear(c_a, 3 * n_head * self.c, bias=False)

        # Pair bias projection
        self.to_b = linearNoBias(c_pair, n_head)

        # Gating mechanism
        self.to_g = nn.Linear(c_a, n_head * self.c, bias=False)

        # Output projection
        self.to_a = linearNoBias(c_a, c_a)

        # Normalization layers
        self.ln_0 = nn.LayerNorm((c_pair,), bias=norm_bias.attention_io)
        self.ln_1 = nn.LayerNorm((c_a,), bias=norm_bias.attention_io)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qkv.weight)
        init_lecun_normal(self.to_b)
        nn.init.zeros_(self.to_g.weight)
        nn.init.zeros_(self.to_a.weight)

    @abstractmethod
    def get_input_normalization(self):
        """Get the input normalization layer.

        Returns:
            The LayerNorm module applied to the single representation input.
        """
        pass

    @abstractmethod
    def forward(
        self,
    ):
        """
        Attention pair bias is a common operation used in different contexts (e.g., Pairformer blocks, Diffusion blocks).
        This class serves as an abstract base class for these different implementations, providing shared functionality
        """
        pass

    def _forward_vanilla(self, S_I_ln, Z_II, is_padding_I):
        """Vanilla PyTorch implementation."""
        # expects pre-normalized A_I

        # Project Q, K, V and reshape
        Q_IH, K_IH, V_IH = self.to_qkv(S_I_ln).chunk(3, dim=-1)
        if self.qk_norm:
            Q_IH = self.ln_q(Q_IH)
            K_IH = self.ln_k(K_IH)

        Q_IH = rearrange(Q_IH, "... (h c) -> ... h c", h=self.n_head)
        K_IH = rearrange(K_IH, "... (h c) -> ... h c", h=self.n_head)
        V_IH = rearrange(V_IH, "... (h c) -> ... h c", h=self.n_head)

        # Project pair bias
        B_IIH = self.to_b(self.ln_0(Z_II))

        # Compute gating and reshape
        G_IH = torch.sigmoid(
            rearrange(self.to_g(S_I_ln), "... (h c) -> ... h c", h=self.n_head)
        )

        # Scale queries
        Q_IH = Q_IH / torch.sqrt(
            torch.tensor(self.c, dtype=Q_IH.dtype, device=Q_IH.device)
        )

        is_padding_II = is_padding_I.unsqueeze(-2) | is_padding_I.unsqueeze(-1)
        B_IIH = B_IIH.masked_fill(is_padding_II.unsqueeze(-1), torch.finfo(B_IIH.dtype).min)
        # Compute attention: softmax(Q @ K.T + B) @ V
        A_IIH = torch.softmax(
            torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2
        )  # softmax over j

        # Apply attention to values
        ## G_IH: [..., I, H, C]
        ## A_IIH: [..., I, I, H]
        ## V_IH: [..., I, H, C]
        A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
        A_I = G_IH * A_I  # [..., I, H, C]
        A_I = A_I.flatten(start_dim=-2)  # [..., I, Ca]

        # Output projection
        A_I = self.to_a(A_I)

        return A_I

    def _forward_cuequivariance(self, S_I_ln, Z_II, is_padding_I):
        """cuEquivariance fused kernel implementation.

        Args:
          S_I: Single representation (queries/keys/values source). Shape (I, c_a) or (B, I, c_a).
          Z_II: Pair representation (bias source). Shape (I, I, c_pair).

        Returns:
          Updated single representation after attention. Shape (I, c_a) or (B, I, c_a).
        """

        # Handle batch dimension: cuEquivariance expects (B, I, c_a) format
        # If input is (I, c_a), add batch dimension -> (1, I, c_a)
        needs_unsqueeze = S_I_ln.ndim == 2
        if needs_unsqueeze:
            S_I_ln = S_I_ln.unsqueeze(0)  # (I, c_a) -> (1, I, c_a)
            # Same for Z_II
            if Z_II.ndim == 3:
                Z_II = Z_II.unsqueeze(0)  # (I, I, c_pair) -> (1, I, I, c_pair)

        # Get batch size and sequence length
        B = S_I_ln.shape[0]
        I = Z_II.shape[-2]

        # Ensure Z_II has batch dimension matching S_I_ln
        # Use .expand().contiguous() to avoid stride issues with cuEquivariance
        if Z_II.ndim == 3:
            # Z_II is [I, I, c_pair], needs batch dim
            Z_II = Z_II.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        elif Z_II.shape[0] == 1 and B > 1:
            # Z_II is [1, I, I, c_pair] but B > 1, broadcast
            Z_II = Z_II.expand(B, -1, -1, -1).contiguous()
        elif Z_II.shape[0] != B:
            # Mismatch - broadcast to match batch size
            Z_II = Z_II.expand(B, -1, -1, -1).contiguous()

        # Create boolean mask: all positions valid (no masking)
        # cuEquivariance expects mask shape (B, I) where True = valid, False = masked
        cueq_mask = ~is_padding_I.bool()

        # Project Q, K, V and reshape to cuEquivariance format
        q, k, v = self.to_qkv(S_I_ln).chunk(3, dim=-1)
        if self.qk_norm:
            q = self.ln_q(q)
            k = self.ln_k(k)

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = torch.bfloat16

        S_I_ln = S_I_ln.to(dtype=dtype)
        Z_II = Z_II.to(dtype=dtype)

        # cuEquivariance expects: (B, H, I, C) for q/k/v where H=n_head, C=c (head dim)
        q = rearrange(q.to(dtype=dtype), "... i (h c) -> ... h i c", h=self.n_head)
        k = rearrange(k.to(dtype=dtype), "... i (h c) -> ... h i c", h=self.n_head)
        v = rearrange(v.to(dtype=dtype), "... i (h c) -> ... h i c", h=self.n_head)

        # Cast weights to match autocast dtype for backward pass compatibility
        w_proj_z = self.to_b.weight.to(dtype=dtype)
        w_proj_g = self.to_g.weight.to(dtype=dtype)
        w_proj_o = self.to_a.weight.to(dtype=dtype)
        w_ln_z = self.ln_0.weight.to(dtype=dtype)
        b_ln_z = self.ln_0.bias.to(dtype=dtype) if self.ln_0.bias is not None else torch.zeros_like(w_ln_z)

        # Call cuEquivariance fused attention_pair_bias kernel
        # This kernel internally performs:
        # 1. Projects pair bias: LayerNorm(z) @ w_proj_z -> (B, I, I, n_head)
        # 2. Computes attention: softmax(q @ k^T / sqrt(c) + pair_bias) @ v
        # 3. Computes gating: sigmoid(s @ w_proj_g)
        # 4. Applies gating: gate * attention_output
        # 5. Projects output: gated_output @ w_proj_o -> (B, I, c_a)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, _ = cuet.attention_pair_bias(
                s=S_I_ln,  # Single representation (B, I, c_a)
                q=q,  # Queries (B, H, I, C)
                k=k,  # Keys (B, H, I, C)
                v=v,  # Values (B, H, I, C)
                z=Z_II,  # Pair representation (B, I, I, c_pair)
                mask=cueq_mask,  # Boolean mask (B, I)
                num_heads=self.n_head,  # Number of attention heads
                w_proj_z=w_proj_z,  # Pair bias projection weight (n_head, c_pair)
                w_proj_g=w_proj_g,  # Gating projection weight (n_head * c, c_a)
                w_proj_o=w_proj_o,  # Output projection weight (c_a, c_a)
                w_ln_z=w_ln_z,  # Pair LayerNorm weight (c_pair,)
                b_ln_z=b_ln_z,  # Pair LayerNorm bias (c_pair,)
                attn_scale=1.0 / math.sqrt(self.c),  # Attention scaling factor
            )

        # Remove batch dimension if we added it at the start
        if needs_unsqueeze:
            output = output.squeeze(0)  # (1, I, c_a) -> (I, c_a)

        return output


class AttentionPairBiasPairformer(AbstractAttentionPairBias):
    """Attention with pair bias for Pairformer blocks (token-level, dense).

    Uses standard LayerNorm without adaptive components.

    Routes to either cuEquivariance or vanilla implementation based on configuration.

    Args:
      c_a: Single representation dimension.
      c_s: Legacy parameter, kept for backwards compatibility (unused).
      c_pair: Pair representation dimension.
      n_head: Number of attention heads.
      use_cuequivariance: Use cuEquivariance fused kernel when available.
    """

    def __init__(self, c_a, c_pair, n_head, use_cuequivariance, qk_norm, norm_bias: NormBiasConfig = NormBiasConfig()):
        super(AttentionPairBiasPairformer, self).__init__(
            c_a=c_a, c_pair=c_pair, n_head=n_head, use_cuequivariance=use_cuequivariance, qk_norm=qk_norm, norm_bias=norm_bias
        )
        # Normalization layers
        self.ln_1 = nn.LayerNorm((c_a,), bias=norm_bias.attention_io)

    def get_input_normalization(self, S_I):
        """Get the input normalization layer.

        Returns:
            The LayerNorm module applied to the single representation input.
        """
        return self.ln_1(S_I)

    def forward(
        self,
        S_I,  # [I, C_a] or [B, I, C_a]
        Z_II,  # [I, I, C_z]
        is_padding_I,
    ):
        """Forward pass of Pairformer-style attention with pair bias.

        Args:
          S_I: Single representation (queries/keys/values source).
          Z_II: Pair representation (bias source).

        Returns:
          Updated single representation after attention.
        """
        S_I = self.get_input_normalization(S_I)
        # Route to appropriate implementation
        if self.use_cuequivariance and foundry.SHOULD_USE_CUEQUIVARIANCE:
            return self._forward_cuequivariance(S_I, Z_II, is_padding_I)
        else:
            return self._forward_vanilla(S_I, Z_II, is_padding_I)


class AttentionPairBiasDiffusion(AbstractAttentionPairBias):
    def __init__(self, c_a, c_s, c_pair, n_head, use_cuequivariance, qk_norm, norm_bias: NormBiasConfig = NormBiasConfig()):
        super(AttentionPairBiasDiffusion, self).__init__(
            c_a=c_a, c_pair=c_pair, n_head=n_head, use_cuequivariance=use_cuequivariance, qk_norm=qk_norm, norm_bias=norm_bias
        )
        # Normalization layers
        self.ln_1 = AdaLN(c_a, c_s)

        # in diffusion (unlike pairformer) outputs are sigmoid gated
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_a, biasinit=-2.0),
            nn.Sigmoid(),
        )

    def get_input_normalization(self, A_I, S_I):
        """Get the input normalization layer.

        Returns:
            The LayerNorm module applied to the single representation input.
        """
        return self.ln_1(A_I, S_I)

    def forward(
        self,
        A_I,  # [B, D, I, C_a]
        S_I,  # [B, D, I, C_s]
        Z_II,  # [B, I, I, C_z]
        is_padding_I,
    ):
        """Forward pass of Diffusion-style attention with pair bias.
        Args:
          A_I: Single representation (queries/keys/values source).
          S_I: Conditioning single representation for AdaLN.
          Z_II: Pair representation (bias source).
        Returns:
          Updated single representation after attention.
        """
        B, D = A_I.shape[0], A_I.shape[1]
        A_I_ln = self.get_input_normalization(A_I, S_I)

        A_I_ln = rearrange(A_I_ln, "b d ... -> (b d) ...")
        Z_II = Z_II[:, None, :, :].expand(-1, D, -1, -1, -1).contiguous()
        Z_II = rearrange(Z_II, "b d i j c -> (b d) i j c")
        is_padding_I = is_padding_I[:, None, :].expand(-1, D, -1).contiguous()
        is_padding_I = rearrange(is_padding_I, "b d i -> (b d) i")
        # Route to appropriate implementation
        if self.use_cuequivariance and foundry.SHOULD_USE_CUEQUIVARIANCE:
            A_I = self._forward_cuequivariance(A_I_ln, Z_II, is_padding_I)
        else:
            A_I = self._forward_vanilla(A_I_ln, Z_II, is_padding_I)
        A_I = rearrange(A_I, "(b d) ... -> b d ...", b=B, d=D)
        A_I = self.linear_output_project(S_I) * A_I
        return A_I
