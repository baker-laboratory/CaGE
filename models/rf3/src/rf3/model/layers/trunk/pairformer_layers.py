import torch

from rf3.model.layers.components.triangle import (
    TriangleAttention,
    TriangleMultiplication,
)
from rf3.model.layers.utils.layer_utils import (
    NormBiasConfig,
    Transition,
    linearNoBias,
)
from rf3.model.layers.components.attention_pair_bias_cuet import (
    AttentionPairBiasPairformer,
)
from foundry.model.layers.blocks import Dropout
from torch import nn
from torch.nn.functional import one_hot




class PairformerBlock(nn.Module):
    """
    Attempt to replicate AF3 architecture from scratch.
    """

    def __init__(
        self,
        c_s,
        c_z,
        p_drop,
        triangle_multiplication,
        triangle_attention,
        attention_pair_bias,
        compile,
        n_transition=4,
        norm_bias: NormBiasConfig = NormBiasConfig(),
        #**kwargs,  #fd: let's not....
    ):
        super().__init__()


        self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        # `use_cuequivariance` is deliberately not forwarded: honouring the config
        # (`use_cueq_trimult: false`) routes these blocks to TriangleMultiplication's
        # vanilla path, which means-normalizes over non-padded tokens while the cuEq
        # kernel does not. Both the run14 trunk weights and the conf head were trained
        # with this block on the cuEq path.
        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="outgoing",
            bias=norm_bias.triangle,
        )
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="incoming",
            bias=norm_bias.triangle,
        )

        self.tri_attn_start = TriangleAttention(
            c_z,
            **triangle_attention,
            start_node=True,
            norm_bias=norm_bias,
        )
        self.tri_attn_end = TriangleAttention(
            c_z,
            **triangle_attention,
            start_node=False,
            norm_bias=norm_bias,
        )

        self.z_transition = Transition(c=c_z, n=n_transition, norm_bias=norm_bias)

        self.s_transition = Transition(c=c_s, n=n_transition, norm_bias=norm_bias)

        self.attention_pair_bias = AttentionPairBiasPairformer(
            c_a=c_s, c_pair=c_z, norm_bias=norm_bias, **attention_pair_bias
        )

        # Compiled in training, eager in eval. Implemented as a real method (not a
        # captured closure on self.forward) so it survives EMA's deepcopy: the shadow's
        # forward binds to the shadow at call time, and its eval path runs plain eager.
        self._use_compile = compile
        self._compiled_impl = None

    def forward(self, *args, **kwargs):
        if self.training and self._use_compile:
            if self._compiled_impl is None:
                self._compiled_impl = torch.compile(self._forward_impl, mode="default")
            return self._compiled_impl(*args, **kwargs)
        return self._forward_impl(*args, **kwargs)

    def _forward_impl(self, S_I, Z_II, is_padding_I):
        Z_II = Z_II + self.drop_row(self.tri_mul_outgoing(Z_II, is_padding_I))

        Z_II = Z_II + self.drop_row(self.tri_mul_incoming(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_row(self.tri_attn_start(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_col(self.tri_attn_end(Z_II, is_padding_I))
        Z_II = Z_II + self.z_transition(Z_II)

        S_I = S_I + self.attention_pair_bias(S_I, Z_II, is_padding_I)
        S_I = S_I + self.s_transition(S_I)

        return S_I, Z_II
