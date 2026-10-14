"""Motif guidance: per-sample arming and the mu projection.

Exact invariants only -- nothing here depends on a sampler seed, model weights or
the noise schedule, so a failure always means the mechanism changed.
"""

import torch
from rf3.guidance import MotifGuidanceCfg, MotifGuidanceEngine

L, M = 12, 6
MOTIF_IDX = torch.arange(M)


def _feats(n_groups: int = 1) -> dict:
    """Synthetic feats: motif atoms 0..M-1, every motif atom also a Kabsch anchor."""
    torch.manual_seed(0)
    target = torch.randn(M, 3) * 4.0
    if n_groups == 1:
        group_starts = torch.tensor([0, M], dtype=torch.long)
        group_names = ["default"]
    else:
        group_starts = torch.tensor([0, M // 2, M], dtype=torch.long)
        group_names = ["g0", "g1"]
    return {
        "motif_idx": MOTIF_IDX.clone(),
        "target": target,
        "anchor_idx_in_motif": torch.arange(M, dtype=torch.long),
        "group_starts": group_starts,
        "group_names": group_names,
    }


def _engine(feats: dict, **kw) -> MotifGuidanceEngine:
    cfg = MotifGuidanceCfg(**{"t_threshold_high": 1.0, "t_threshold_low": 0.0, **kw})
    eng = MotifGuidanceEngine(cfg, feats)
    eng.reset()
    return eng


def _x(feats: dict, converged: list[bool], *, offset: float = 7.0) -> torch.Tensor:
    """Build [1, D, L, 3] whose motif block is rigid-shifted GT (converged) or scrambled."""
    torch.manual_seed(1)
    d = len(converged)
    x = torch.randn(1, d, L, 3) * 2.0
    for i, ok in enumerate(converged):
        if ok:
            # Pure translation of GT: Kabsch residual is ~0, so this sample arms.
            x[0, i, :M] = feats["target"] + offset
        else:
            # Internal geometry destroyed: Kabsch cannot fit, residual stays large.
            x[0, i, :M] = feats["target"] * 9.0 + 400.0
    return x


def test_rank4_runs_and_shape_is_preserved():
    feats = _feats()
    eng = _engine(feats, rmsd_threshold=1e9)
    x = _x(feats, [True] * 5)
    out = eng.refine(x, t_norm=0.5)
    assert out.shape == x.shape


def test_all_armed_matches_per_sample_calls():
    """With every sample armed, batched refine == refining each sample alone."""
    feats = _feats()
    x = _x(feats, [True] * 5)

    batched = _engine(feats, rmsd_threshold=1e9).refine(x, t_norm=0.5)
    per_sample = torch.cat(
        [
            _engine(feats, rmsd_threshold=1e9).refine(x[:, d : d + 1], t_norm=0.5)
            for d in range(5)
        ],
        dim=1,
    )
    # allclose, not equal: index_select changes the SVD batch size, which moves the
    # result by ~1e-6 A. The stock scalar engine shows the same wobble.
    assert torch.allclose(batched, per_sample, atol=1e-5)


def test_unarmed_samples_are_bitwise_untouched():
    feats = _feats()
    converged = [True, False, True, False, True]
    x = _x(feats, converged)
    out = _engine(feats).refine(x, t_norm=0.5)

    unarmed = [i for i, ok in enumerate(converged) if not ok]
    armed = [i for i, ok in enumerate(converged) if ok]
    assert torch.equal(out[:, unarmed], x[:, unarmed])
    assert not torch.equal(out[:, armed], x[:, armed])
    assert torch.isfinite(out).all()


def test_mu1_snaps_armed_motif_onto_aligned_target():
    """mu=1, inner_steps=1 is an exact snap: post-refine anchor RMSD ~ 0."""
    feats = _feats()
    eng = _engine(feats, rmsd_threshold=1e9, mu=1.0, inner_steps=1)
    x = _x(feats, [True] * 3)
    out = eng.refine(x, t_norm=0.5)
    rmsd = eng.potential.anchor_rmsd(out.reshape(-1, L, 3), feats)
    assert rmsd.max().item() < 1e-3


def test_mu_half_contracts_the_gap():
    feats = _feats()
    eng = _engine(feats, rmsd_threshold=1e9, mu=0.5, inner_steps=1)
    x = _x(feats, [True] * 3)
    before = eng.potential.anchor_rmsd(x.reshape(-1, L, 3), feats)
    out = eng.refine(x, t_norm=0.5)
    after = eng.potential.anchor_rmsd(out.reshape(-1, L, 3), feats)
    # One-sided: y' is refit each inner step, so 0.5x is an upper bound.
    assert (after <= 0.5 * before + 1e-4).all()


def test_inert_configs_return_the_same_object():
    feats = _feats()
    x = _x(feats, [True] * 3)
    # Outside the window.
    eng = _engine(feats, rmsd_threshold=1e9)
    assert eng.refine(x, t_norm=1.5) is x
    # mu == 0 and inner_steps == 0 are both exact no-ops.
    assert _engine(feats, rmsd_threshold=1e9, mu=0.0).refine(x, t_norm=0.5) is x
    assert _engine(feats, rmsd_threshold=1e9, inner_steps=0).refine(x, t_norm=0.5) is x
    # Nothing arms.
    assert _engine(feats, rmsd_threshold=1e-9).refine(x, t_norm=0.5) is x


def test_arming_is_latched_per_sample():
    feats = _feats()
    eng = _engine(feats)
    eng.refine(_x(feats, [True, False]), t_norm=0.5)
    assert eng.armed.tolist() == [True, False]

    # Nothing is converged now, but sample 0 stays armed and is still refined.
    x2 = _x(feats, [False, False])
    out = eng.refine(x2, t_norm=0.4)
    assert eng.armed.tolist() == [True, False]
    assert not torch.equal(out[:, 0], x2[:, 0])
    assert torch.equal(out[:, 1], x2[:, 1])


def test_sample_count_change_resets_the_mask():
    feats = _feats()
    eng = _engine(feats, rmsd_threshold=1e9)
    eng.refine(_x(feats, [True] * 5), t_norm=0.5)
    assert eng.armed.numel() == 5
    eng.refine(_x(feats, [True] * 2), t_norm=0.5)
    assert eng.armed.numel() == 2


def test_groups_align_independently():
    """Two groups must each snap onto their own Kabsch fit, not a shared one."""
    feats = _feats(n_groups=2)
    eng = _engine(feats, rmsd_threshold=1e9, mu=1.0, inner_steps=1)
    x = _x(feats, [True])
    # Displace the second group so a single shared transform cannot satisfy both.
    x[0, 0, M // 2 : M] += 25.0
    out = eng.refine(x, t_norm=0.5)
    rmsd = eng.potential.anchor_rmsd(out.reshape(-1, L, 3), feats)
    assert rmsd.max().item() < 1e-3
    assert feats["group_starts"].numel() - 1 == len(feats["group_names"])
