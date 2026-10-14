"""Motif guidance: per-sample gate + mu projection on the clean estimate x0.

This is a sampler-side constraint, NOT one of the trained design conditions. It
never reaches the denoiser; it edits x0 between the CFG blend and the Euler step.

Activation (per diffusion sample, independently)
------------------------------------------------
Time window:  silent outside [t_threshold_low, t_threshold_high] on the
              normalized step fraction t_norm (1.0 at the first step, ~0 at the
              last). Re-evaluated every step, so leaving the window re-silences
              guidance even for a sample that had already armed.
Arming:       inside the window, a sample arms once its own anchor RMSD (max over
              groups) drops below rmsd_threshold, and stays armed thereafter.

Why per-sample: the engine folds D diffusion samples into one trajectory, each
with its own noise draw and its own random frame (``centre_random_augmentation``
is per-sample). A batch-wide ``.all()`` would let the slowest sample withhold
guidance from the rest, and then force-snap every sample -- including unconverged
ones -- on the step the last one happens to converge. At D == 1 the two are
identical, which is the regime the reference implementation ran in.

Refinement (armed samples only)
-------------------------------
Per inner step: x <- x - mu * dE/dx, where dE/dx is exactly (x_i - y'_i) on motif
atoms and 0 elsewhere, so the update is a linear interpolation toward the
per-group Kabsch-aligned target:  x_i <- (1 - mu) * x_i + mu * y'_i.
y' is refit every inner step, so the residual gap goes as ~(1 - mu)^inner_steps.
Defaults (mu=1.0, inner_steps=1) are therefore an exact hard snap, not a soft pull.

mu <= 0 or inner_steps <= 0 makes ``refine`` an exact no-op that returns its
argument unchanged (same object), which callers rely on to skip the delta update.

Call ``reset()`` between independent trajectories so armed flags do not leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rf3.guidance.potential import MotifAnchorPotential


@dataclass
class MotifGuidanceCfg:
    # Anchor-subset RMSD (A) below which a sample arms.
    rmsd_threshold: float = 5.0
    # Active window on the normalized step fraction t_norm, which runs 1.0
    # (first step, high noise) -> ~0.0 (last step). Guidance does not apply when
    # t_norm is above t_threshold_high or below t_threshold_low, so the default
    # 0.5 leaves the noisier first half of the trajectory unguided, and raising
    # t_threshold_low leaves the final steps to the denoiser.
    t_threshold_high: float = 0.5
    t_threshold_low: float = 0.0
    # Fraction of the motif-atom -> aligned-target gap closed per inner step.
    # Dimensionless: no noise-level weighting, no step-size schedule, no clipping.
    mu: float = 1.0
    inner_steps: int = 1


class MotifGuidanceEngine:
    """Per-job engine; attach one per (example, seed) and ``reset()`` before use."""

    def __init__(self, cfg: MotifGuidanceCfg, feats: dict[str, Tensor]):
        self.cfg = cfg
        self.feats = feats
        self.potential = MotifAnchorPotential()
        # None until the first refine sees the sample count; then a [S] bool mask.
        self.armed: Tensor | None = None

    def reset(self) -> None:
        self.armed = None

    def to(self, *, device, dtype=None) -> "MotifGuidanceEngine":
        new_feats = {}
        for k, v in self.feats.items():
            if not isinstance(v, Tensor):
                # Pass through non-tensor metadata (e.g. group_names: list[str]).
                new_feats[k] = v
                continue
            target_dtype = (
                dtype if (dtype is not None and v.is_floating_point()) else v.dtype
            )
            new_feats[k] = v.to(device=device, dtype=target_dtype)
        self.feats = new_feats
        if self.armed is not None:
            self.armed = self.armed.to(device=device)
        return self

    def refine(self, x0: Tensor, *, t_norm: float) -> Tensor:
        """Project armed samples' motif atoms toward their aligned targets.

        ``x0`` may carry any number of leading sample dims; they are flattened to
        a single axis S for the potential, which fits one Kabsch per (sample, group).
        Returns ``x0`` itself when nothing is applied.
        """
        cfg = self.cfg
        if t_norm > cfg.t_threshold_high or t_norm < cfg.t_threshold_low:
            return x0

        xf = x0.reshape(-1, x0.shape[-2], x0.shape[-1])  # [S, L, 3]
        n_samples = xf.shape[0]
        if (
            self.armed is None
            or self.armed.numel() != n_samples
            or self.armed.device != xf.device
        ):
            self.armed = torch.zeros(n_samples, dtype=torch.bool, device=xf.device)

        if not bool(self.armed.all()):
            # A sample whose coords have gone non-finite must never arm: its Kabsch
            # fit is meaningless and would propagate NaN into every motif atom.
            finite = torch.isfinite(xf).all(dim=-1).all(dim=-1)
            with torch.no_grad():
                rmsd = self.potential.anchor_rmsd(xf.detach(), self.feats)  # [S]
            newly = (rmsd < cfg.rmsd_threshold) & finite & ~self.armed
            if bool(newly.any()):
                self.armed = self.armed | newly
                print(
                    "[motif-guidance] armed samples "
                    f"{newly.nonzero(as_tuple=False).flatten().tolist()} "
                    f"at t_norm={t_norm:.3f} "
                    f"anchor_rmsd={[round(v, 3) for v in rmsd.flatten().tolist()]} "
                    f"armed_now={self.armed.tolist()}",
                    flush=True,
                )

        if not bool(self.armed.any()) or not (cfg.mu > 0.0 and cfg.inner_steps > 0):
            return x0

        # Index the armed samples rather than multiplying by a 0/1 mask: masking
        # would still run the Kabsch SVD on unconverged samples and fold their
        # result in as 0 * value.
        sel = self.armed.nonzero(as_tuple=False).squeeze(-1)
        xs = xf.index_select(0, sel)
        for _ in range(cfg.inner_steps):
            _, grad = self.potential.energy_and_grad(xs, self.feats)
            xs = xs - cfg.mu * grad
        # index_copy, not index_copy_: never mutate the caller's tensor.
        return xf.index_copy(0, sel, xs).reshape(x0.shape)
