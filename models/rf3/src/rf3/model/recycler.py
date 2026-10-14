import logging

import torch
import torch.nn as nn

from rf3.model.layers.utils.layer_utils import NormBiasConfig, Transition, linearNoBias
from rf3.model.layers.trunk.pairformer_layers import (
    PairformerBlock,
)
from rf3.model.layers.embedders.template_embedder import (
    RF3TemplateEmbedder,
)
from rf3.model.layers.embedders.msa import MSAModule




logger = logging.getLogger(__name__)

"""
Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
    D: # diffusion structure batch dim
"""


class DistogramHead(nn.Module):
    def __init__(
        self,
        c_z,
        bins,
    ):
        super().__init__()
        self.predictor = nn.Linear(c_z, bins)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        Z_II,
    ):
        return self.predictor(
            Z_II + Z_II.transpose(-2, -3)  # symmetrize pair features
        )


class MaskedTokenRecoveryHead(nn.Module):
    def __init__(
        self,
        c_s,
        n_classes=21,
    ):
        super().__init__()
        self.predictor = nn.Linear(c_s, n_classes)  # 20 amino acids + mask
        self.reset_parameters()

    def reset_parameters(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        S_I,
    ):
        return self.predictor(S_I)


class Recycler(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        template_embedder,
        msa_module,
        n_pairformer_blocks,
        pairformer_block,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.c_z = c_z
        self.process_zh = nn.Sequential(
            nn.LayerNorm(c_z, bias=norm_bias.projection),
            linearNoBias(c_z, c_z),
        )
        self.template_embedder = RF3TemplateEmbedder(c_z=c_z, norm_bias=norm_bias, **template_embedder)
        self.msa_module = MSAModule(norm_bias=norm_bias, **msa_module)
        self.process_sh = nn.Sequential(
            nn.LayerNorm(c_s, bias=norm_bias.projection),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, norm_bias=norm_bias, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

    def forward(
        self,
        f,
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
        is_padding_I,  # padding on the token dimension
        is_padding_S,  # padding on the MSA sequence dimension
    ):
        Z_II = Z_init_II + self.process_zh(Z_II)
        Z_II = Z_II + self.template_embedder(f, Z_II, is_padding_I)
        # NOTE: Implementing bugfix from the Protenix Technical report, where residual-connecting the MSA module is redundant
        # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
        Z_II = self.msa_module(f, Z_II, S_inputs_I, is_padding_I, is_padding_S)
        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            if torch.is_grad_enabled():
                S_I, Z_II = torch.utils.checkpoint.checkpoint(
                    block, S_I, Z_II, is_padding_I, use_reentrant=False
                )
            else:
                S_I, Z_II = block(S_I, Z_II, is_padding_I)
        return S_I, Z_II
