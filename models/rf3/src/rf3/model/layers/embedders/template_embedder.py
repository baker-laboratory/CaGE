import torch
import torch.nn as nn
import torch.nn.functional as F

from rf3.data.ground_truth_template import (
    af3_noise_scale_to_noise_level,
)
from rf3.model.layers.components.triangle import (
    TriangleAttention,
    TriangleMultiplication,
)
from rf3.model.layers.utils.layer_utils import NormBiasConfig, Transition
from rf3.util_module import init_lecun_normal
from foundry.model.layers.blocks import Dropout



class PairformerBlockNoSingle(nn.Module):
    """
    Attempt to replicate AF3 architecture from scratch.
    """

    def __init__(
        self,
        c_z,
        p_drop,
        triangle_multiplication,
        triangle_attention,
        n_transition=4,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()

        self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="outgoing",
            bias=norm_bias.triangle,
            use_cuequivariance=triangle_multiplication['use_cuequivariance'],
        )
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="incoming",
            bias=norm_bias.triangle,
            use_cuequivariance=triangle_multiplication['use_cuequivariance'],
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

    def forward(self, Z_II, is_padding_I):
        Z_II = Z_II + self.drop_row(self.tri_mul_outgoing(Z_II, is_padding_I))

        Z_II = Z_II + self.drop_row(self.tri_mul_incoming(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_row(self.tri_attn_start(Z_II, is_padding_I))
        Z_II = Z_II + self.drop_col(self.tri_attn_end(Z_II, is_padding_I))
        Z_II = Z_II + self.z_transition(Z_II)

        return Z_II


class RF3TemplateEmbedder(nn.Module):
    """
    Template track that enables conditioning on noisy ground-truth templates at the token level.
    Supports all chain types.
    """

    def __init__(
        self,
        n_block,
        raw_template_dim,
        c_z,
        c,
        p_drop,
        use_cueq_trimult,
        use_cueq_triattn,
        norm_bias: NormBiasConfig = NormBiasConfig(),
        gate_output_by_distogram_condition: bool = False,
    ):
        super().__init__()
        self.c = c
        self.gate_output_by_distogram_condition = gate_output_by_distogram_condition
        self.emb_pair = nn.Linear(c_z, c, bias=False)
        self.norm_pair_before_pairformer = nn.LayerNorm(c_z, bias=norm_bias.projection)
        self.norm_after_pairformer = nn.LayerNorm(c, bias=norm_bias.projection)
        self.emb_templ = nn.Linear(raw_template_dim, c, bias=False)

        # template pairformer does not operate on sequence representation
        self.pairformer = nn.ModuleList(
            [
                PairformerBlockNoSingle(
                    c_z=c,
                    p_drop=p_drop,
                    triangle_multiplication=dict(d_hidden=c, use_cuequivariance=use_cueq_trimult),
                    triangle_attention=dict(d_hidden=c, n_head=4, use_cuequivariance=use_cueq_triattn),
                    n_transition=4,
                    norm_bias=norm_bias,
                )
                for _ in range(n_block)
            ]
        )

        # NOTE: this is not consistent with AF3 paper which outputs this tensor in the template_channel dimension
        # In Algorithm 1, line 9, the outputs of this function are added to the Z_II tensor which has dimensions [B, I, I, C_z]
        # so we make the outputs of this module also has those dimensions
        self.agg_emb = nn.Linear(c, c_z, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_pair.weight)
        nn.init.xavier_uniform_(self.emb_templ.weight)
        nn.init.zeros_(self.agg_emb.weight)

    def forward(self, f, Z_II, is_padding_I):
        has_distogram_condition = f["has_distogram_condition"]
        distogram_condition_noise_scale = f["distogram_condition_noise_scale"]
        distogram_condition = f["distogram_condition"]

        # Transform noise scale to reasonable range
        joint_noise_scale = (
            distogram_condition_noise_scale[:, None, :] ** 2
            + distogram_condition_noise_scale[:, :, None] ** 2
        ).sqrt()
        joint_noise_level = af3_noise_scale_to_noise_level(joint_noise_scale)

        template_feats = torch.cat(
            [
                distogram_condition,
                has_distogram_condition.unsqueeze(-1),
                joint_noise_level.unsqueeze(-1),
            ],
            dim=-1,
        )
        template_feats = template_feats * has_distogram_condition.unsqueeze(-1)
        template_channels = self.emb_templ(template_feats)

        v_II = self.emb_pair(self.norm_pair_before_pairformer(Z_II)) + template_channels
        for block in self.pairformer:
            if torch.is_grad_enabled():
                v_II = torch.utils.checkpoint.checkpoint(
                    block, v_II, is_padding_I, use_reentrant=False
                )
            else:
                v_II = block(v_II, is_padding_I)
        v_II = self.norm_after_pairformer(v_II)

        out = self.agg_emb(F.relu(v_II))
        if self.gate_output_by_distogram_condition:
            out = out * has_distogram_condition.unsqueeze(-1).to(out.dtype)
        return out
