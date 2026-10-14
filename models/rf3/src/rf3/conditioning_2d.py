"""2D pairwise conditioning helpers for atom-level distance conditioning.

Ported from rfd3 (``models/rfd3/src/rfd3/model/conditioning_2d_utils.py`` and
``models/rfd3/src/rfd3/model/layers/block_utils.py``). Used by:

* the diffusion atom-attention encoder, to add ``C_DIS`` distance conditioning
  to the atom-level pair representation ``P_LL``;
* the trunk feature initializer, to mean-pool the atom-pair distance signal
  to token-level and add it to ``Z_init_II`` (mirrors rfd3's
  ``DiffusionTokenEncoder.add_2d_conditioning_to_Z_II``).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Conditioning2DProcessor(nn.Module):
    """Process pairwise Fourier features + a binary "has-condition" mask.

    Pipeline (per pair (i, j)):
        1. ``Linear(64 -> c_out) + ReLU`` on Fourier-embedded distance features
        2. ``LayerNorm``
        3. concat with the boolean mask (cast to float)
        4. small MLP ``(c_out + 1) -> c_out -> c_out -> c_out``

    The output is meant to be added (after a final projection) to the atom-pair
    representation ``P_LL`` of shape ``[..., L, L, c_atompair]``.
    """

    def __init__(self, c_out: int, in_features: int = 64) -> None:
        super().__init__()
        self.in_features = in_features

        self.sparse_projection = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=c_out, bias=True),
            nn.ReLU(),
        )

        self.layer_norm = nn.LayerNorm(c_out)
        # Bias-free so mlp(0) == 0: an unconditioned position (zero input) must
        # produce a zero contribution to P_LL.
        self.mlp = nn.Sequential(
            nn.Linear(c_out + 1, c_out, bias=False),
            nn.ReLU(),
            nn.Linear(c_out, c_out, bias=False),
            nn.ReLU(),
            nn.Linear(c_out, c_out, bias=False),
        )

    def forward(
        self,
        sparse_features: Float[Tensor, "... L L D"],
        mask: Float[Tensor, "... L L"] | Bool[Tensor, "... L L"],
    ) -> Float[Tensor, "... L L c_out"]:
        projected = self.sparse_projection(sparse_features)
        projected = self.layer_norm(projected)
        if mask.dtype == torch.bool:
            mask = mask.to(projected.dtype)
        concatenated = torch.cat([projected, mask.unsqueeze(-1)], dim=-1)
        return self.mlp(concatenated)

    def forward_sparse(
        self,
        masked_dist: Float[Tensor, "n_valid"],
        fourier_scales: Float[Tensor, "d"],
    ) -> Float[Tensor, "n_valid c_out"]:

        c_out = self.layer_norm.normalized_shape[0]

        sparse_fourier = fourier_embed_scales(masked_dist, fourier_scales)
        proj = self.sparse_projection(sparse_fourier)   # [n_valid, c_out]
        proj = self.layer_norm(proj)                    # [n_valid, c_out]

        ones = torch.ones(proj.shape[0], 1, dtype=proj.dtype, device=proj.device)
        masked_input = torch.cat([proj, ones], dim=-1)                 # [n_valid, c_out+1]
        zero_row = torch.zeros(1, c_out + 1, dtype=proj.dtype, device=proj.device)

        # add a dummy row to keep DDP happy
        all_input = torch.cat([masked_input, zero_row], dim=0)         # [n_valid+1, c_out+1]
        return self.mlp(all_input)[:-1]                                # [n_valid, c_out]

def compute_scales(
    min_scale: float = 1.0 / 9.0,
    max_scale: float = 64.0,
    num_scales: int = 32,
    spacing: Literal["log", "linear"] = "log",
    device: torch.device | str = "cpu",
) -> Float[Tensor, "d"]:
    """Build the scale vector used by :func:`fourier_embed_scales`.

    Returns a tensor of length ``num_scales`` spanning ``[min_scale, max_scale]``
    on either a log or linear spacing.
    """
    if spacing == "log":
        return torch.exp(
            torch.linspace(np.log(min_scale), np.log(max_scale), num_scales, device=device)
        )
    if spacing == "linear":
        return torch.linspace(min_scale, max_scale, num_scales, device=device)
    raise ValueError(f"Invalid spacing: {spacing}")


def fourier_embed_scales(
    x: Float[Tensor, "... L"],
    scales: Float[Tensor, "d"],
) -> Float[Tensor, "... L 2*d"]:
    """Sin/cos Fourier feature embedding.

    Formula (scales :math:`\\sigma = [\\sigma_1, \\sigma_2, \\dots]`):

        :math:`\\phi(x) = [\\sin(2 \\pi x / \\sigma), \\cos(2 \\pi x / \\sigma)]`

    Args:
        x: Input tensor with arbitrary leading dims (e.g. raw distances ``[B, L, L]``).
        scales: 1-D tensor of length ``d``; the resulting embedding has dim ``2*d``.

    Returns:
        Tensor with shape ``(*x.shape, 2*d)``.
    """
    inv_scales = (2 * math.pi) / scales  # [d], same dtype as scales
    proj = x.unsqueeze(-1) * inv_scales  # [..., L, d], same dtype as x
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def pairwise_mean_pool(
    pairwise_atom_features: Float[Tensor, "batch n_atoms n_atoms d_hidden"],
    atom_to_token_map: Int[Tensor, "batch n_atoms"],
    I: int,
    dtype: torch.dtype,
    pairwise_mask: Optional[Float[Tensor, "batch n_atoms n_atoms"]] = None,
) -> Float[Tensor, "batch n_tokens n_tokens d_hidden"]:
    """Mean-pool atom-pair features to token-pair features, with optional masking.

    Adapted from rfd3's ``pairwise_mean_pool`` to use a per-example
    ``atom_to_token_map``; the one-hot tensor and einsums carry a batch axis.

    Args:
        pairwise_atom_features: ``(B, L, L, D)`` atom-pair features.
        atom_to_token_map: ``(B, L)`` per-example map from atom index to token index.
        I: number of tokens (max across batch after padding).
        dtype: dtype to cast the one-hot map to (matches feature dtype).
        pairwise_mask: optional ``(B, L, L)`` (or ``(L, L)`` for broadcast) float mask;
            invalid pairs do not contribute to the sum or the count.

    Returns:
        ``(B, I, I, D)`` token-pair features. Token pairs with zero valid atom
        pairs are returned as zeros.
    """
    B, L, _, D = pairwise_atom_features.shape

    # Apply the pairwise mask if provided (zero out invalid pairs).
    if pairwise_mask is not None:
        if pairwise_mask.dim() == 2:
            pairwise_mask = pairwise_mask.unsqueeze(0)  # (1, L, L) for broadcast
        pairwise_atom_features = pairwise_atom_features * pairwise_mask[..., None]

    # Per-batch one-hot atom→token assignment: (B, L, I).
    valid_atom_mask = atom_to_token_map >= 0  # (B, L)
    safe_token_idx = torch.where(
        valid_atom_mask, atom_to_token_map, torch.zeros_like(atom_to_token_map)
    )
    atom_to_token_onehot = F.one_hot(safe_token_idx.long(), num_classes=I).to(dtype)
    atom_to_token_onehot = atom_to_token_onehot * valid_atom_mask.unsqueeze(-1).to(dtype)

    # temp[b, i, n, d] = sum_m onehot[b, m, i] * features[b, m, n, d]
    temp = torch.einsum(
        "bmi,bmnd->bind", atom_to_token_onehot, pairwise_atom_features
    )  # (B, I, L, D)
    del pairwise_atom_features
    # token_features_sum[b, i, j, d] = sum_n onehot[b, n, j] * temp[b, i, n, d]
    token_features_sum = torch.einsum(
        "bnj,bind->bijd", atom_to_token_onehot, temp
    )  # (B, I, I, D)
    del temp

    # Count valid atom pairs per token pair 
    if pairwise_mask is not None:
        temp_mask = torch.einsum(
            "bmi,bmn->bin", atom_to_token_onehot, pairwise_mask
        )  # (B, I, L)
        token_pair_counts = torch.einsum(
            "bnj,bin->bij", atom_to_token_onehot, temp_mask
        )  # (B, I, I)
    else:
        atom_counts_per_token = atom_to_token_onehot.sum(dim=-2)  # (B, I)
        token_pair_counts = (
            atom_counts_per_token.unsqueeze(-1) * atom_counts_per_token.unsqueeze(-2)
        )  # (B, I, I)


    is_zero = token_pair_counts == 0
    token_pair_counts_safe = torch.where(
        is_zero, torch.ones_like(token_pair_counts), token_pair_counts
    )
    token_pairwise_features = token_features_sum / token_pair_counts_safe.unsqueeze(-1)
    token_pairwise_features = torch.where(
        is_zero.unsqueeze(-1).expand_as(token_pairwise_features),
        torch.zeros_like(token_pairwise_features),
        token_pairwise_features,
    )
    return token_pairwise_features
