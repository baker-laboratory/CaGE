import torch
import torch.nn as nn
from rf3.model.layers.utils.layer_utils import (
    AdaLN,
    LinearBiasInit,
    NormBiasConfig,
    linearNoBias,
)
from rf3.util_module import init_lecun_normal
from rf3.model.layers.components.attention_pair_bias_cuet import (
    AttentionPairBiasDiffusion,
)

from foundry.training.checkpoint import activation_checkpointing


class DiffusionTransformer(nn.Module):
    def __init__(self, c_token, c_s, c_tokenpair, n_block, diffusion_transformer_block, norm_bias: NormBiasConfig = NormBiasConfig()):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c_token=c_token,
                    c_s=c_s,
                    c_tokenpair=c_tokenpair,
                    norm_bias=norm_bias,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(
        self,
        A_I,  # [..., I, C_token]
        S_I,  # [..., I, C_token]
        Z_II,  # [..., I, I, C_tokenpair]
        is_padding_I,  # [..., I]
    ):
        for block in self.blocks:
            A_I = block(A_I, S_I, Z_II, is_padding_I)
        return A_I


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        c_token,
        c_s,
        c_tokenpair,
        n_head,
        qk_norm,
        use_cuequivariance,
        compile,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()

        self.attention_pair_bias = AttentionPairBiasDiffusion(
            c_a=c_token, c_s=c_s, c_pair=c_tokenpair, n_head=n_head, qk_norm=qk_norm, use_cuequivariance=use_cuequivariance, norm_bias=norm_bias
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            c_token=c_token, c_s=c_s
        )

        # Compiled in training, eager in eval. Implemented as a real method (not a
        # captured closure on self.forward) so it survives EMA's deepcopy.
        self._use_compile = compile
        self._compiled_impl = None

    def forward(self, *args, **kwargs):
        if self.training and self._use_compile:
            if self._compiled_impl is None:
                self._compiled_impl = torch.compile(self._forward_impl, mode="default")
            return self._compiled_impl(*args, **kwargs)
        return self._forward_impl(*args, **kwargs)

    @activation_checkpointing
    def _forward_impl(
        self,
        A_I,  # [..., I, C_token]
        S_I,  # [..., I, C_s]
        Z_II,  # [..., I, I, C_tokenpair]
        is_padding_I,  # [..., I]
    ):
        #A_I = A_I + self.attention_pair_bias(A_I, S_I, Z_II, is_padding_I)
        #A_I = A_I + self.conditioned_transition_block(A_I, S_I)
        A_I = A_I + self.attention_pair_bias(A_I, S_I, Z_II, is_padding_I) + self.conditioned_transition_block(A_I, S_I)

        return A_I


class ConditionedTransitionBlock(nn.Module):
    """SwiGLU transition block with adaptive layernorm"""

    def __init__(self, c_token, c_s, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_token, c_s=c_s)
        self.linear_1 = linearNoBias(c_token, c_token * n)
        self.linear_2 = linearNoBias(c_token, c_token * n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_token, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_token * n, c_token)

        self.reset_parameters()

    def reset_parameters(self):
        init_lecun_normal(self.linear_1)
        init_lecun_normal(self.linear_2)
        nn.init.zeros_(self.linear_3.weight)

    def forward(
        self,
        Ai,  # [B, I, C_token]
        Si,  # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        Bi = torch.nn.functional.silu(self.linear_1(Ai)) * self.linear_2(Ai)

        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)
