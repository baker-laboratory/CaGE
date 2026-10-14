import torch
import torch.nn as nn
import torch.nn.functional as F

from rf3.model.layers.components.outer_product import (
    OuterProductMean_AF3,
)
from rf3.model.layers.components.triangle import (
    TriangleAttention,
    TriangleMultiplication,
)
from rf3.model.layers.utils.layer_utils import NormBiasConfig, Transition
from rf3.util_module import init_lecun_normal
from foundry.model.layers.blocks import Dropout


from foundry.training.checkpoint import activation_checkpointing


class MSAModule(nn.Module):
    def __init__(
        self,
        n_block,
        c_m,
        p_drop_msa,
        p_drop_pair,
        msa_subsample_embedder,
        outer_product,
        msa_pair_weighted_averaging,
        msa_transition,
        triangle_multiplication_outgoing,
        triangle_multiplication_incoming,
        triangle_attention_starting,
        triangle_attention_ending,
        pair_transition,
        compile,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.n_block = n_block
        self.msa_embedder = MSAEmbedder(**msa_subsample_embedder)

        # Final block does not update MSA features, so we use a different class
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(
                    outer_product=outer_product,
                    msa_pair_weighted_averaging=msa_pair_weighted_averaging,
                    msa_transition=msa_transition,
                    triangle_multiplication_outgoing=triangle_multiplication_outgoing,
                    triangle_multiplication_incoming=triangle_multiplication_incoming,
                    triangle_attention_starting=triangle_attention_starting,
                    triangle_attention_ending=triangle_attention_ending,
                    pair_transition=pair_transition,
                    p_drop_msa=p_drop_msa,
                    p_drop_pair=p_drop_pair,
                    norm_bias=norm_bias,
                    compile=compile,
                )
                for _ in range(n_block - 1)
            ]
        )

        self.final_block = MSAModuleFinalBlock(
            outer_product=outer_product,
            msa_pair_weighted_averaging=msa_pair_weighted_averaging,
            msa_transition=msa_transition,
            triangle_multiplication_outgoing=triangle_multiplication_outgoing,
            triangle_multiplication_incoming=triangle_multiplication_incoming,
            triangle_attention_starting=triangle_attention_starting,
            triangle_attention_ending=triangle_attention_ending,
            pair_transition=pair_transition,
            p_drop_msa=p_drop_msa,
            p_drop_pair=p_drop_pair,
            norm_bias=norm_bias,
            compile=compile,
        )

    def forward(
        self,
        f,
        Z_II,
        S_inputs_I,
        is_padding_I,
        is_padding_S,
    ):
        msa = f["msa"]
        msa_SI = self.msa_embedder(msa, S_inputs_I)

        for i in range(self.n_block - 1):
            if torch.is_grad_enabled():
                msa_SI, Z_II = torch.utils.checkpoint.checkpoint(
                    self.blocks[i], msa_SI, Z_II, is_padding_I, is_padding_S,
                    use_reentrant=False,
                )
            else:
                msa_SI, Z_II = self.blocks[i](msa_SI, Z_II, is_padding_I, is_padding_S)

        if torch.is_grad_enabled():
            Z_II = torch.utils.checkpoint.checkpoint(
                self.final_block, msa_SI, Z_II, is_padding_I, is_padding_S,
                use_reentrant=False,
            )
        else:
            Z_II = self.final_block(msa_SI, Z_II, is_padding_I, is_padding_S)

        return Z_II


class MSAModuleBlock(nn.Module):
    def __init__(
        self,
        outer_product,
        msa_pair_weighted_averaging,
        msa_transition,
        triangle_multiplication_outgoing,
        triangle_multiplication_incoming,
        triangle_attention_starting,
        triangle_attention_ending,
        pair_transition,
        p_drop_msa,
        p_drop_pair,
        norm_bias: NormBiasConfig = NormBiasConfig(),
        compile: bool = False,
    ):
        super().__init__()
        # Compiled in training, eager in eval — as a real method (see forward) so it
        # survives EMA's deepcopy.
        self._use_compile = compile
        self._compiled_impl = None
        self.outer_product = OuterProductMean_AF3(**outer_product, norm_bias=norm_bias)
        self.msa_pair_weighted_averaging = MSAPairWeightedAverage(
            **msa_pair_weighted_averaging, norm_bias=norm_bias
        )
        self.msa_transition = Transition(**msa_transition, norm_bias=norm_bias)

        self.drop_row_msa = Dropout(broadcast_dim=-2, p_drop=p_drop_msa)
        self.drop_row_pair = Dropout(broadcast_dim=-2, p_drop=p_drop_pair)
        self.drop_col_pair = Dropout(broadcast_dim=-3, p_drop=p_drop_pair)

        # NOTE: MSA triangle-multiplication norms intentionally keep bias=False
        # (they were never biased in the reference checkpoint, so the triangle
        #  flag deliberately does not touch them).
        self.tri_mult_outgoing = TriangleMultiplication(
            d_pair=triangle_multiplication_outgoing["d_pair"],
            d_hidden=triangle_multiplication_outgoing["d_hidden"],
            direction="outgoing",
            bias=False,
            use_cuequivariance=triangle_multiplication_outgoing["use_cuequivariance"],
        )
        self.tri_mult_incoming = TriangleMultiplication(
            d_pair=triangle_multiplication_incoming["d_pair"],
            d_hidden=triangle_multiplication_incoming["d_hidden"],
            direction="incoming",
            bias=False,
            use_cuequivariance=triangle_multiplication_incoming["use_cuequivariance"],
        )
        self.tri_attn_start = TriangleAttention(
            **triangle_attention_starting, start_node=True, norm_bias=norm_bias
        )
        self.tri_attn_end = TriangleAttention(
            **triangle_attention_ending, start_node=False, norm_bias=norm_bias
        )
        self.pair_transition = Transition(**pair_transition, norm_bias=norm_bias)

    def forward(self, *args, **kwargs):
        if self.training and self._use_compile:
            if self._compiled_impl is None:
                self._compiled_impl = torch.compile(self._forward_impl, mode="default")
            return self._compiled_impl(*args, **kwargs)
        return self._forward_impl(*args, **kwargs)

    def _forward_impl(
        self,
        msa_SI,
        Z_II,
        is_padding_I,
        is_padding_S,
    ):
        # update MSA features
        Z_II = Z_II + self.outer_product(msa_SI, is_padding_S)
        msa_SI = msa_SI + self.drop_row_msa(
            self.msa_pair_weighted_averaging(msa_SI, Z_II, is_padding_I, is_padding_S)
        )

        msa_SI = msa_SI + self.msa_transition(msa_SI)
        # update pair features
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_outgoing(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_incoming(Z_II, is_padding_I))

        Z_II = Z_II + self.drop_row_pair(self.tri_attn_start(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_col_pair(self.tri_attn_end(Z_II, is_padding_I))
        Z_II = Z_II + self.pair_transition(Z_II)

        return msa_SI, Z_II


class MSAModuleFinalBlock(MSAModuleBlock):
    """
    Final block of MSA module that does not update MSA features
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_pair_weighted_averaging = None
        self.msa_transition = None
        self.drop_row_msa = None

    # NOTE: inherits the train/eval-dispatching forward from MSAModuleBlock, which
    # calls this overridden _forward_impl.
    def _forward_impl(
        self,
        msa_SI,
        Z_II,
        is_padding_I,
        is_padding_S,
    ):
        # update pair features only
        Z_II = Z_II + self.drop_row_pair(self.outer_product(msa_SI, is_padding_S))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_outgoing(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_incoming(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_row_pair(self.tri_attn_start(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_col_pair(self.tri_attn_end(Z_II, is_padding_I))
        Z_II = Z_II + self.pair_transition(Z_II)
        return Z_II


class MSAEmbedder(nn.Module):
    def __init__(self, num_sequences, dim_raw_msa, c_msa_embed, c_s_inputs):
        super(MSAEmbedder, self).__init__()
        self.num_sequences = num_sequences
        self.emb_msa = nn.Linear(dim_raw_msa, c_msa_embed, bias=False)
        self.emb_S_inputs = nn.Linear(c_s_inputs, c_msa_embed, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_msa.weight)
        nn.init.xavier_uniform_(self.emb_S_inputs.weight)

    @activation_checkpointing
    def forward(
        self,
        msa_SI,  # (S, I, 34) (32 tokens + has_deletion + deletion value)
        S_inputs,  # (L, S_dim)
    ):
        # Embed the subsampled MSA
        # (NOTE: We subsample in the data loader to avoid memory issues)
        msa_SI = self.emb_msa(msa_SI)
        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)[:, None]
        return msa_SI


class MSAPairWeightedAverage(nn.Module):
    """implements Algorithm 10 from AF3 paper"""

    def __init__(
        self,
        c_weighted_average,
        n_heads,
        c_msa_embed,
        c_z,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super(MSAPairWeightedAverage, self).__init__()
        self.weighted_average_channels = c_weighted_average
        self.n_heads = n_heads
        self.msa_channels = c_msa_embed
        self.pair_channels = c_z
        self.norm_msa = nn.LayerNorm(self.msa_channels, eps=1e-3, bias=norm_bias.projection)
        self.to_v = nn.Linear(
            self.msa_channels, self.n_heads * self.weighted_average_channels, bias=False
        )
        self.norm_pair = nn.LayerNorm(self.pair_channels, bias=norm_bias.projection)
        self.to_bias = nn.Linear(self.pair_channels, self.n_heads, bias=False)

        self.to_gate = nn.Linear(
            self.msa_channels,
            self.weighted_average_channels * self.n_heads,
            bias=False,
        )

        self.to_out = nn.Linear(
            self.weighted_average_channels * self.n_heads, self.msa_channels, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        init_lecun_normal(self.to_bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_out.weight)

    @activation_checkpointing
    def forward(self, msa_SI, pair_II, is_padding_I, is_padding_S):
        """
        The way the padding is handled here is slightly complicated:
        First, any values that correspond to padding tokens in either the sequence or residue dimension are zeroed out
        when constructing the value tensor. This ensures that these positions do not contribute to the weighted average.
        Second, when choosing the weights for the weighted average, we set the logits corresponding to padding tokens
        to a large negative value (-1e31), so that after applying softmax, these positions have negligible weight.
        This two-step process ensures that padding tokens do not affect the output of the weighted average.
        """
        B, S, I = msa_SI.shape[:3]

        # normalize inputs
        msa_norm_SI = self.norm_msa(msa_SI)

        # construct values, bias and weights
        v_SIH = self.to_v(msa_norm_SI).reshape(
            B, S, I, self.n_heads, self.weighted_average_channels
        )

        # zero out contribution from padding tokens on length dimension and sequence dimension
        is_padding_SI = is_padding_S.unsqueeze(-1) | is_padding_I.unsqueeze(-2)
        v_SIH = v_SIH * (~is_padding_SI[..., None, None]).to(v_SIH.dtype)

        # construct biases
        bias_IIH = self.to_bias(self.norm_pair(pair_II))

        # padding tokens do not contribute to the Softmax (e.g. the unpadded token probabilities should add to 1.0)
        is_padding_II = is_padding_I[..., None] | is_padding_I[..., None, :]
        bias_IIH = bias_IIH.masked_fill(is_padding_II[..., None], torch.finfo(bias_IIH.dtype).min)

        # compute probabilities for weighted average
        w_IIH = F.softmax(bias_IIH, dim=-2)

        # construct gate
        gate_SIH = torch.sigmoid(self.to_gate(msa_SI))
        gate_SIH = gate_SIH * (~is_padding_SI[..., None]).to(gate_SIH.dtype)

        # compute weighted average & apply gate
        weights = torch.einsum("bijh,bsjhc->bsihc", w_IIH, v_SIH).reshape(B, S, I, -1)
        o_SIH = gate_SIH * weights

        # concatenate heads and project
        msa_update_SI = self.to_out(o_SIH.reshape(B, S, I, -1))
        return msa_update_SI
