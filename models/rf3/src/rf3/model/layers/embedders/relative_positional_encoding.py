import torch
import torch.nn as nn

from rf3.model.layers.utils.layer_utils import linearNoBias
from rf3.util_module import init_lecun_normal

from torch.nn.functional import one_hot


class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear = linearNoBias(
            2 * (2 * self.r_max + 2) + (2 * self.s_max + 2) + 1, c_z
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_lecun_normal(self.linear)

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)

        # Handle cyclic chains
        cyclic_asym_ids = f.get("cyclic_asym_ids", [])
        if len(cyclic_asym_ids) > 0:
            offset = f["residue_index"].unsqueeze(-1) - f["residue_index"].unsqueeze(-2)

            for cyclic_asym_id in cyclic_asym_ids:
                len_cyclic_chain = (
                    f["residue_index"][f["asym_id"] == cyclic_asym_id].unique().shape[0]
                )
                cyclic_chain_mask = (f["asym_id"].unsqueeze(-1) == cyclic_asym_id) & (
                    f["asym_id"].unsqueeze(-2) == cyclic_asym_id
                )

                # cyclic offset
                if len_cyclic_chain > 0:
                    offset_plus = offset + len_cyclic_chain
                    offset_minus = offset - len_cyclic_chain
                    abs_offset = offset.abs()
                    abs_offset_plus = offset_plus.abs()
                    abs_offset_minus = offset_minus.abs()

                    choice_plus_or_minus = torch.where(
                        abs_offset_plus <= abs_offset_minus, offset_plus, offset_minus
                    )
                    c_offset = torch.where(
                        (abs_offset <= abs_offset_plus)
                        & (abs_offset <= abs_offset_minus),
                        offset,
                        choice_plus_or_minus,
                    )
                    offset = torch.where(cyclic_chain_mask, c_offset, offset)

            offset = (offset + self.r_max).clamp(0, 2 * self.r_max)
            d_residue_II = torch.where(
                b_samechain_II, offset, (2 * self.r_max + 1) * torch.ones_like(offset)
            )

        else:
            d_residue_II = torch.where(
                b_samechain_II,
                torch.clip(
                    f["residue_index"].unsqueeze(-1)
                    - f["residue_index"].unsqueeze(-2)
                    + self.r_max,
                    0,
                    2 * self.r_max,
                ),
                2 * self.r_max + 1,
            )

        A_relpos_II = one_hot(d_residue_II.long(), 2 * self.r_max + 2)
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                f["token_index"].unsqueeze(-1)
                - f["token_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        A_reltoken_II = one_hot(d_token_II, 2 * self.r_max + 2)
        d_chain_II = torch.where(
            # NOTE: Implementing bugfix from the Protenix Technical report, where we use `same_entity` instead of `not same_chain` (as in the AF-3 pseudocode)
            # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
            b_same_entity_II,
            torch.clip(
                f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max,
                0,
                2 * self.s_max,
            ),
            2 * self.s_max + 1,
        )
        A_relchain_II = one_hot(d_chain_II.long(), 2 * self.s_max + 2)
        return self.linear(
            torch.cat(
                [
                    A_relpos_II,
                    A_reltoken_II,
                    b_same_entity_II.unsqueeze(-1),
                    A_relchain_II,
                ],
                dim=-1,
            ).to(torch.float)
        )
