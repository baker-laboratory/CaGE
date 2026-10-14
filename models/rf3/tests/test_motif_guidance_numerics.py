"""Pin motif guidance against an INDEPENDENT target, including the gradient's direction.

A fixture that builds the motif block as an exact rigid image of the reference (say a
pure translation) makes `align_target` correctly return the input unchanged, so every
post-refine RMSD assertion is already satisfied before refine runs -- and a sign flip in
`energy_and_grad` or in `xs = xs - mu * grad` passes. That sign is exactly where the
earlier reconstruction-guidance implementation went wrong, turning descent into ascent.

So here the anchors are an exact image of the reference under a NON-TRIVIAL rotation
while a pulled non-anchor atom is displaced by a known amount. The correct answer differs
from the input, which is what makes the direction observable.

Equivariance alone does not catch a sign error (an ascent step is just as frame-covariant),
so direction and equivariance are pinned separately.
"""

import math

import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion
from rf3.guidance import MotifGuidanceCfg, MotifGuidanceEngine

L, M = 14, 6
# group 0 = motif rows 0..3 (anchors 0,1,2; row 3 is pulled-only); group 1 = rows 4..5.
GROUP_STARTS = torch.tensor([0, 4, 6], dtype=torch.long)
ANCHORS = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long)


def _rot(axis, deg):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    th = math.radians(deg)
    K = torch.tensor(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64
    )
    return (
        torch.eye(3, dtype=torch.float64)
        + math.sin(th) * K
        + (1 - math.cos(th)) * (K @ K)
    ).float()


def _case(perturb: float = 9.0):
    """Return (feats, x[1,1,L,3], expected_motif[M,3])."""
    torch.manual_seed(7)
    target = torch.randn(M, 3) * 5.0
    R0, t0 = _rot((0.3, -0.7, 0.5), 37.0), torch.tensor([11.0, -4.0, 2.5])
    R1, t1 = _rot((1.0, 0.2, -0.4), -118.0), torch.tensor([-6.0, 3.0, 20.0])
    expected = torch.cat([target[:4] @ R0.T + t0, target[4:] @ R1.T + t1], dim=0)
    x = torch.randn(1, 1, L, 3) * 3.0
    x[0, 0, :M] = expected
    x[0, 0, 3] = expected[3] + perturb  # pulled-only atom, off by `perturb`
    feats = {
        "motif_idx": torch.arange(M),
        "target": target,
        "anchor_idx_in_motif": ANCHORS.clone(),
        "group_starts": GROUP_STARTS.clone(),
        "group_names": ["g0", "g1"],
    }
    return feats, x, expected


def _eng(feats, **kw):
    return MotifGuidanceEngine(
        MotifGuidanceCfg(
            **{
                "t_threshold_high": 1.0,
                "t_threshold_low": 0.0,
                "rmsd_threshold": 1e9,
                **kw,
            }
        ),
        feats,
    )


def test_refine_lands_on_the_independently_computed_rigid_image():
    """mu=1 must put every motif atom on R_g @ target_g + t_g, fitted from anchors only."""
    feats, x, expected = _case()
    out = _eng(feats).refine(x, t_norm=0.5)
    assert torch.allclose(out[0, 0, :M], expected, atol=2e-3), out[0, 0, :M] - expected
    assert torch.equal(out[0, 0, M:], x[0, 0, M:])  # non-motif untouched


def test_refine_reduces_the_displaced_atoms_error():
    """The displaced pulled atom must move TOWARD the reference, not away from it.

    This is the assertion a flipped gradient sign fails.
    """
    feats, x, expected = _case(perturb=9.0)
    before = (x[0, 0, 3] - expected[3]).norm().item()
    out = _eng(feats).refine(x, t_norm=0.5)
    after = (out[0, 0, 3] - expected[3]).norm().item()
    assert before > 10.0 and after < 1e-2, (before, after)


def test_mu_half_halves_a_real_gap():
    feats, x, expected = _case(perturb=9.0)
    before = (x[0, 0, 3] - expected[3]).norm().item()
    out = _eng(feats, mu=0.5).refine(x, t_norm=0.5)
    after = (out[0, 0, 3] - expected[3]).norm().item()
    assert after == pytest.approx(0.5 * before, rel=0.02), (before, after)


def test_non_anchor_perturbation_does_not_move_the_fit():
    """Row 3 is pulled-only: the anchors alone determine R, t."""
    feats, x_small, _ = _case(perturb=0.5)
    feats2, x_big, _ = _case(perturb=40.0)
    a = _eng(feats).refine(x_small, t_norm=0.5)[0, 0, :M]
    b = _eng(feats2).refine(x_big, t_norm=0.5)[0, 0, :M]
    assert torch.allclose(a, b, atol=2e-3)


def test_refine_is_equivariant_under_a_random_rigid_frame():
    """centre_random_augmentation gives every sample its own frame."""
    feats, x, _ = _case()
    R, t = _rot((0.2, 0.9, -0.3), 73.0), torch.tensor([30.0, -12.0, 5.0])
    plain = _eng(feats).refine(x, t_norm=0.5)
    moved = _eng(feats).refine(x @ R.T + t, t_norm=0.5)
    assert torch.allclose(moved, plain @ R.T + t, atol=5e-3)


def test_reflected_anchors_are_not_fitted_with_a_reflection():
    """A mirrored prediction must still get a proper rotation (det R == +1)."""
    feats, x, _ = _case(perturb=0.0)
    x_mirror = x.clone()
    x_mirror[0, 0, :M, 0] *= -1.0
    out = _eng(feats).refine(x_mirror, t_norm=0.5)
    d_out = torch.cdist(out[0, 0, :4], out[0, 0, :4])
    d_ref = torch.cdist(feats["target"][:4], feats["target"][:4])
    assert torch.allclose(d_out, d_ref, atol=2e-3)  # rigid, not mirrored
    # and it is genuinely NOT the mirrored input
    assert (out[0, 0, :4] - x_mirror[0, 0, :4]).norm() > 1e-2


def test_non_finite_sample_never_arms():
    feats, x, _ = _case()
    x2 = torch.cat([x, x.clone()], dim=1)
    x2[0, 1, L - 1, 0] = float("nan")  # a NON-motif atom
    eng = _eng(feats, rmsd_threshold=1e9)
    out = eng.refine(x2, t_norm=0.5)
    assert eng.armed.tolist() == [True, False]
    assert torch.equal(
        out[:, 1][torch.isfinite(out[:, 1])], x2[:, 1][torch.isfinite(x2[:, 1])]
    )


def test_t_threshold_low_silences_guidance():
    feats, x, _ = _case()
    eng = _eng(feats, t_threshold_low=0.3, t_threshold_high=0.8)
    assert eng.refine(x, t_norm=0.29) is x  # below the window
    assert eng.refine(x, t_norm=0.81) is x  # above the window
    assert eng.refine(x, t_norm=0.3) is not x  # inclusive lower edge
    assert eng.refine(x, t_norm=0.8) is not x  # inclusive upper edge


def test_reset_clears_the_armed_latch():
    feats, x, _ = _case()
    eng = _eng(feats)
    eng.refine(x, t_norm=0.5)
    assert bool(eng.armed.any())
    eng.reset()
    assert eng.armed is None


def test_feats_from_the_resolver_drive_the_potential():
    """The CSR group_starts / global anchor rows the resolver emits must be consumable."""
    import numpy as np
    from biotite.structure import AtomArray
    from rf3.guidance import build_motif_guidance_feats

    arr = AtomArray(5)
    arr.chain_id = np.array(["A", "A", "A", "X", "X"])
    arr.res_id = np.array([10, 10, 10, 99, 99])
    arr.res_name = np.array(["HIS", "HIS", "HIS", "LIG", "LIG"])
    arr.atom_name = np.array(["N", "CA", "CB", "C1", "C2"])
    arr.element = np.array(["N", "C", "C", "C", "C"])
    arr.coord = np.arange(15, dtype=np.float32).reshape(5, 3)
    feats = build_motif_guidance_feats(
        arr,
        {
            "prot": {"atoms": ["A"], "anchor": ["A/HIS/10/N", "A/HIS/10/CA"]},
            "lig": ["X"],
        },
    )
    assert feats["group_starts"].tolist() == [0, 3, 5]
    assert feats["anchor_idx_in_motif"].tolist() == [0, 1, 3, 4]
    x = torch.zeros(1, 1, 5, 3)
    x[0, 0] = torch.from_numpy(arr.coord)
    x[0, 0, 2] += 6.0  # displace the pulled-only CB
    out = MotifGuidanceEngine(
        MotifGuidanceCfg(rmsd_threshold=1e9, t_threshold_high=1.0), feats
    ).refine(x, t_norm=0.5)
    assert torch.allclose(out[0, 0], torch.from_numpy(arr.coord), atol=1e-3)


def test_worst_group_blocks_arming():
    """anchor_rmsd is a max over groups: one unconverged group must withhold arming.

    Under a mean the sample below would arm on a 0 A / 60 A pair.
    """
    feats, x, _ = _case(perturb=0.0)
    # Stretch group 1's two anchors apart: its internal geometry no longer matches the
    # reference, so no rigid fit can satisfy it (group 0 still fits exactly).
    bond = x[0, 0, 4] - x[0, 0, 5]
    x[0, 0, 4] = x[0, 0, 4] + 16.0 * bond / bond.norm()
    eng = _eng(feats, rmsd_threshold=5.0)
    per_group = eng.potential.anchor_rmsd(x.reshape(-1, L, 3), feats)
    assert 5.0 < per_group.item() < 10.0, per_group  # max is out, mean would be in
    out = eng.refine(x, t_norm=0.5)
    assert eng.armed.tolist() == [False]
    assert out is x


# --- the sampler hook: pin the delta correction's MAGNITUDE, not just its sign ---------
#
# Contract: after refine, the Euler step must move toward x0_refined exactly as if the
# denoiser had predicted it. So a run with a stub that adds a constant v to x0 must match
# a run whose denoiser output is shifted by the same v.

B, I, HOOK_L, D, C = 1, 4, 6, 2, 8
SAMPLER_KW = dict(
    num_timesteps=5,
    min_t=0,
    max_t=1,
    sigma_data=16,
    s_min=4e-4,
    s_max=160,
    p=7,
    gamma_0=0.8,
    gamma_min=1.0,
    noise_scale=1.003,
    step_scale=1.5,
    solver="af3",
)


class _Dummy:
    def __init__(self, bias=0.0):
        self.bias = bias

    def __call__(self, *, X_noisy_L, t, f, S_inputs_I, S_trunk_I, Z_trunk_II, **kw):
        return 0.9 * X_noisy_L + S_inputs_I.mean() + self.bias


class _ConstantShiftStub:
    def __init__(self, v):
        self.v = v

    def refine(self, x0, *, t_norm):
        return x0 + self.v


def _run(module, motif_guidance=None):
    torch.manual_seed(0)
    kw = dict(
        f={"ref_element": torch.zeros(B, HOOK_L, 4)},
        S_inputs_I=torch.zeros(B, I, C),
        S_trunk_I=torch.zeros(B, I, C),
        Z_trunk_II=torch.zeros(B, I, I, C),
        diffusion_module=module,
        diffusion_batch_size=D,
        coord_atom_lvl_to_be_noised=torch.zeros(D, HOOK_L, 3),
        is_padding_L=torch.zeros(B, HOOK_L, dtype=torch.bool),
        is_padding_I=torch.zeros(B, I, dtype=torch.bool),
        is_padding_chiral=torch.zeros(B, 0, dtype=torch.bool),
    )
    if motif_guidance is not None:
        kw["motif_guidance"] = motif_guidance
    return SampleDiffusion(**SAMPLER_KW).sample_diffusion_like_af3(**kw)


def test_refined_x0_is_the_target_the_euler_step_moves_toward():
    v = 0.37
    guided = _run(_Dummy(), motif_guidance=_ConstantShiftStub(v))["X_L"]
    equivalent = _run(_Dummy(bias=v))["X_L"]
    assert torch.allclose(guided, equivalent, atol=1e-4), (
        (guided - equivalent).abs().max().item()
    )


def test_the_correction_is_linear_in_the_refinement():
    a = _run(_Dummy(), motif_guidance=_ConstantShiftStub(0.2))["X_L"]
    b = _run(_Dummy(), motif_guidance=_ConstantShiftStub(0.4))["X_L"]
    base = _run(_Dummy())["X_L"]
    assert torch.allclose(b - base, 2.0 * (a - base), rtol=1e-3, atol=1e-2)
