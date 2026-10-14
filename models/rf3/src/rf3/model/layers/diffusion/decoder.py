import torch
import torch.nn as nn
from rf3.model.layers.diffusion.atom_transformer_diffusion import (
    AtomTransformer,
)
from rf3.model.layers.utils.layer_utils import (
    NormBiasConfig,
    linearNoBias,
)
from rf3.util_module import init_lecun_normal

from foundry.training.checkpoint import activation_checkpointing


class AtomAttentionDecoder(nn.Module):
    def __init__(self, c_token, c_atom, c_atompair, atom_transformer, norm_bias: NormBiasConfig = NormBiasConfig()):
        super().__init__()
        self.c_token = c_token
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, norm_bias=norm_bias, **atom_transformer
        )
        self.linear_1 = linearNoBias(c_token, c_atom)
        self.to_r_update = nn.Sequential(
            nn.LayerNorm((c_atom,), bias=norm_bias.projection), linearNoBias(c_atom, 3)
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_lecun_normal(self.linear_1)
        nn.init.zeros_(self.to_r_update[1].weight)

    def forward(
        self,
        f,
        Ai,  # [B, D, L, C_token]
        Ql_skip,  # [B, D, L, C_atom]
        Cl_skip,  # [B, D, L, C_atom]
        Plm_skip,  # [B, L, L, C_atompair]
        is_padding_L,
    ):
        tok_idx = f["atom_to_token_map"]
        D = Ai.shape[1]

        @activation_checkpointing
        def atom_decoder(Ai, Ql_skip, Cl_skip, Plm_skip, tok_idx):
            # Broadcast per-token activiations to per-atom activations and add the skip connection
            tok_idx_no_padding = tok_idx.clamp(min=0)

            A_L = torch.gather(
                Ai,
                dim=2,
                index=tok_idx_no_padding[:, None]
                .unsqueeze(-1)
                .expand(-1, D, -1, self.c_token)
                .long(),
            )
            Ql = self.linear_1(A_L) + Ql_skip

            # Cross attention transformer.
            Ql = self.atom_transformer(Ql, Cl_skip, Plm_skip, is_padding_L)

            # Map to positions update
            Rl_update = self.to_r_update(Ql)

            return Rl_update

        return atom_decoder(Ai, Ql_skip, Cl_skip, Plm_skip, tok_idx)
