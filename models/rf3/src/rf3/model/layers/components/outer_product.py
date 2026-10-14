import torch
import torch.nn as nn

from foundry.training.checkpoint import activation_checkpointing
from rf3.util_module import init_lecun_normal
from rf3.model.layers.utils.layer_utils import NormBiasConfig


class OuterProductMean_AF3(nn.Module):
    def __init__(self, c_msa_embed, c_outer_product, c_out, norm_bias: NormBiasConfig = NormBiasConfig()):
        super(OuterProductMean_AF3, self).__init__()
        self.norm = nn.LayerNorm(c_msa_embed, bias=norm_bias.projection)
        self.proj_left = nn.Linear(c_msa_embed, c_outer_product, bias=False)
        self.proj_right = nn.Linear(c_msa_embed, c_outer_product, bias=False)
        self.proj_out = nn.Linear(c_outer_product * c_outer_product, c_out, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init_lecun_normal(self.proj_left)
        init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_out.weight)

    @activation_checkpointing
    def forward(self, msa, is_padding_S):
        B, N, I = msa.shape[:3]
        msa_norm = self.norm(msa)
        left = self.proj_left(msa_norm)
        right = self.proj_right(msa_norm)

        # Mask out padding sequences in the MSA num aligned sequence dimension
        left = left * (~is_padding_S[..., None, None]).to(left.dtype)
        right = right * (~is_padding_S[..., None, None]).to(right.dtype)

        # NOTE: dividing by N here makes the sum in the einsum a mean
        # Looks like a bug but is mathematically equivalent to dividing after the sum in the einsum
        num_sequences_non_padding = (
            (~is_padding_S).sum(dim=-1).clamp(min=1)[..., None, None, None]
        )
        right = right / num_sequences_non_padding.to(right.dtype)

        out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, I, I, -1)
        out = self.proj_out(out)

        return out
