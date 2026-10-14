"""Motif-anchor potential with per-group rigid alignment.

Energy:
    E(x) = 0.5 * sum_{g} sum_{i in motif_g} ||x_i - y'_{g,i}||^2,
    y'_{g,i} = R_g @ y_{g,i} + t_g,  where (R_g, t_g) is the rigid transform
    that maps the GT anchor subset of group g onto the current predicted
    anchor subset of group g (per-group Kabsch).

Each group in ``feats["group_starts"]`` is aligned independently. Inter-group
relative pose is NOT constrained — useful when e.g. the protein motif residues
and the ligand should each be kept rigid internally but their relative pose
can drift.

For a linear constraint x_i = y'_i, the protenix-style minimum-norm projection
reduces to a single hard snap: x_i <- y'_i (no iteration needed).

R, t are computed inside torch.no_grad() and treated as constants when taking
gradients of the energy: this makes the per-atom gradient an independent
"anchor force" and avoids SVD gradient singularities at near-degenerate
anchor configurations.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MotifAnchorPotential:
    @staticmethod
    def _kabsch(P: Tensor, Q: Tensor) -> tuple[Tensor, Tensor]:
        """Fit rigid (R, t) such that R @ Q + t aligns with P.

        Args:
            P: predicted anchor coords, shape [B, A, 3]
            Q: GT anchor coords,        shape [A, 3]

        Returns:
            R: [B, 3, 3]
            t: [B, 3]   (cast back to P's dtype)
        """
        out_dtype = P.dtype
        device_type = "cuda" if P.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            P32 = P.float()
            Q32 = Q.float()

            P_mean = P32.mean(dim=-2, keepdim=True)  # [B, 1, 3]
            Q_mean = Q32.mean(dim=-2, keepdim=True)  # [1, 3]
            Pc = P32 - P_mean  # [B, A, 3]
            Qc = Q32 - Q_mean  # [A, 3]

            H = torch.einsum("ad,bae->bde", Qc, Pc)  # [B, 3, 3]
            U, _, Vt = torch.linalg.svd(H)
            Vt_T = Vt.transpose(-2, -1)
            U_T = U.transpose(-2, -1)
            det_sign = torch.sign(torch.linalg.det(torch.matmul(Vt_T, U_T)))
            D = (
                torch.eye(3, device=H.device, dtype=H.dtype)
                .unsqueeze(0)
                .repeat(H.shape[0], 1, 1)
            )
            D[..., 2, 2] = det_sign
            R = torch.matmul(Vt_T, torch.matmul(D, U_T))
            t = P_mean.squeeze(-2) - torch.matmul(
                R, Q_mean.squeeze(-2).unsqueeze(-1)
            ).squeeze(-1)
        return R.to(out_dtype), t.to(out_dtype)

    @staticmethod
    def _iter_groups(feats: dict):
        """Yield (group_start, group_end, anchor_local) per group.

        ``anchor_local`` are the positions inside [group_start, group_end)
        used as Kabsch anchors for that group, expressed as 1-D LongTensor
        of offsets relative to ``group_start``.

        Falls back to a single group spanning the whole motif when
        ``group_starts`` is not present in ``feats`` — keeps backward
        compatibility with callers that haven't been updated to emit
        per-group boundaries.
        """
        motif_idx = feats["motif_idx"]  # [M]
        anchor_global = feats["anchor_idx_in_motif"]  # [A]
        group_starts = feats.get("group_starts")
        if group_starts is None:
            yield 0, int(motif_idx.numel()), anchor_global
            return
        for g in range(group_starts.numel() - 1):
            gs = int(group_starts[g].item())
            ge = int(group_starts[g + 1].item())
            mask = (anchor_global >= gs) & (anchor_global < ge)
            anchor_local = anchor_global[mask] - gs
            yield gs, ge, anchor_local

    def align_target(self, x0: Tensor, feats: dict) -> Tensor:
        """Map GT target into current frame per group; result has shape [..., M, 3].

        R, t are detached, so this is treated as a constant when differentiating
        downstream losses w.r.t. x.
        """
        with torch.no_grad():
            x_det = x0.detach()
            motif_idx = feats["motif_idx"]  # [M]
            target = feats["target"].to(
                device=x_det.device, dtype=x_det.dtype
            )  # [M, 3]

            x_motif = x_det.index_select(-2, motif_idx)  # [B, M, 3]
            # Build per-group aligned targets, concat along the M axis.
            aligned_groups: list[Tensor] = []
            for gs, ge, anchor_local in self._iter_groups(feats):
                x_group = x_motif[..., gs:ge, :]  # [B, M_g, 3]
                y_group = target[gs:ge]  # [M_g, 3]
                x_anchor = x_group.index_select(-2, anchor_local)  # [B, A_g, 3]
                y_anchor = y_group.index_select(0, anchor_local)  # [A_g, 3]

                R, t = self._kabsch(x_anchor, y_anchor)  # [B, 3, 3], [B, 3]
                y_aligned_g = torch.einsum("bij,mj->bmi", R, y_group) + t.unsqueeze(-2)
                aligned_groups.append(y_aligned_g)
            y_aligned = torch.cat(aligned_groups, dim=-2)  # [B, M, 3]
        return y_aligned

    def hard_project(self, x0: Tensor, feats: dict) -> Tensor:
        """Return delta s.t. (x0 + delta)[motif_idx] == y_aligned exactly."""
        y_aligned = self.align_target(x0, feats)
        motif_idx = feats["motif_idx"]
        x_motif = x0.index_select(-2, motif_idx)
        diff = y_aligned - x_motif  # [B, M, 3]
        delta = torch.zeros_like(x0)
        delta.index_add_(-2, motif_idx, diff)
        return delta

    def energy_and_grad(self, x: Tensor, feats: dict) -> tuple[Tensor, Tensor]:
        """E and dE/dx (each motif atom's gradient points to its own aligned target)."""
        y_aligned = self.align_target(x, feats)
        motif_idx = feats["motif_idx"]
        diff = x.index_select(-2, motif_idx) - y_aligned  # [B, M, 3]
        e = 0.5 * (diff**2).sum(dim=(-1, -2))  # [B]
        g = torch.zeros_like(x)
        g.index_add_(-2, motif_idx, diff)
        return e, g

    def anchor_rmsd(self, x0: Tensor, feats: dict) -> Tensor:
        """Max-over-groups anchor RMSD between current x and aligned GT, per batch.

        Max over groups (rather than mean) keeps the activation gate honest:
        a worst-group above threshold blocks arming, so guidance does not
        kick in until every group's anchors have converged.
        """
        y_aligned = self.align_target(x0, feats)
        motif_idx = feats["motif_idx"]
        x_motif = x0.index_select(-2, motif_idx)  # [B, M, 3]

        rmsds: list[Tensor] = []
        for gs, ge, anchor_local in self._iter_groups(feats):
            x_anchor = x_motif[..., gs:ge, :].index_select(-2, anchor_local)
            y_anchor = y_aligned[..., gs:ge, :].index_select(-2, anchor_local)
            d = x_anchor - y_anchor
            rmsds.append((d**2).sum(-1).mean(-1).sqrt())  # [B]
        return torch.stack(rmsds, dim=-1).max(dim=-1).values  # [B]
