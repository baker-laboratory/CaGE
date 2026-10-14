"""Sampler-level checks for the CFG noise-level interval (cfg_sigma_min/max).

Hermetic: drives ``SampleDiffusion.sample_diffusion_like_af3`` with a deterministic,
call-counting dummy module. Invariants:

  * an interval covering the whole schedule == no interval (bit-for-bit);
  * an interval covering no step == unconditional baseline (bit-for-bit) + a warning;
  * a partial interval differs from both, and skips the guidance forwards outside it;
  * cfg_sigma_min > cfg_sigma_max raises.
"""

import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion


class _CountingDummy:
    """Deterministic, RNG-free; counts forward calls so skipped steps are observable."""

    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        *,
        X_noisy_L,
        t,
        f,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        is_padding_L,
        is_padding_I,
        is_padding_chiral,
    ):
        self.calls += 1
        return 0.9 * X_noisy_L + S_inputs_I.mean()


def _make_sampler() -> SampleDiffusion:
    return SampleDiffusion(
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


def _inputs(dummy):
    B, I, L, D, c = 1, 4, 6, 2, 8
    return dict(
        f={"ref_element": torch.zeros(B, L, 4)},
        S_inputs_I=torch.zeros(B, I, c),
        S_trunk_I=torch.zeros(B, I, c),
        Z_trunk_II=torch.zeros(B, I, I, c),
        diffusion_module=dummy,
        diffusion_batch_size=D,
        coord_atom_lvl_to_be_noised=torch.zeros(D, L, 3),
        is_padding_L=torch.zeros(B, L, dtype=torch.bool),
        is_padding_I=torch.zeros(B, I, dtype=torch.bool),
        is_padding_chiral=torch.zeros(B, 0, dtype=torch.bool),
    )


def _stream(signal=0.5, scale=-2.0):
    B, I, L, c = 1, 4, 6, 8
    return {
        "f": {"ref_element": torch.zeros(B, L, 4)},
        "S_inputs_I": torch.full((B, I, c), signal),
        "S_trunk_I": torch.full((B, I, c), signal),
        "Z_trunk_II": torch.full((B, I, I, c), signal),
        "scale": scale,
    }


def _run(sigma_min=None, sigma_max=None, guided=True):
    dummy = _CountingDummy()
    sampler = _make_sampler()
    torch.manual_seed(0)
    kwargs = _inputs(dummy)
    if guided:
        kwargs["inference_classifier_free_guidance"] = True
        kwargs["cfg_guidance_streams"] = [_stream()]
    if sigma_min is not None:
        kwargs["cfg_sigma_min"] = sigma_min
    if sigma_max is not None:
        kwargs["cfg_sigma_max"] = sigma_max
    out = sampler.sample_diffusion_like_af3(**kwargs)
    return out["X_L"], [float(t) for t in out["t_hats"]], dummy.calls


def test_full_interval_equals_no_interval():
    """[0, inf] (the default) is identical to not passing an interval."""
    a, _, ca = _run()  # defaults: min=0, max=None(+inf)
    b, _, cb = _run(sigma_min=0.0, sigma_max=float("inf"))
    assert torch.equal(a, b)
    assert ca == cb


def test_empty_interval_equals_baseline_and_warns(caplog):
    """An interval below every t_hat leaves the unconditional trajectory untouched."""
    baseline, t_hats, base_calls = _run(guided=False)
    tiny = min(t_hats) / 2  # below the lowest-noise step
    with caplog.at_level("WARNING"):
        guided, _, calls = _run(sigma_min=0.0, sigma_max=tiny)
    assert torch.equal(guided, baseline)
    assert calls == base_calls  # no guidance forwards happened
    assert "no diffusion step fell in the noise interval" in caplog.text


def test_partial_interval_differs_and_skips_forwards():
    """A window covering some steps differs from both extremes and skips the rest."""
    full, t_hats, full_calls = _run()
    baseline, _, base_calls = _run(guided=False)
    n_steps = len(t_hats)

    # Keep only the single highest-noise step in the window.
    hi = max(t_hats)
    lo_cut = sorted(t_hats)[-2] + 1e-6  # excludes all but the max
    partial, _, calls = _run(sigma_min=lo_cut, sigma_max=hi)

    assert not torch.equal(partial, full)
    assert not torch.equal(partial, baseline)
    # full = n_steps main + n_steps guided; baseline = n_steps main; one guided step:
    assert base_calls == n_steps
    assert full_calls == 2 * n_steps
    assert calls == n_steps + 1


def test_min_above_max_raises():
    with pytest.raises(ValueError, match="must not exceed"):
        _run(sigma_min=100.0, sigma_max=10.0)


def test_interval_inert_without_guidance():
    """Interval args are harmless when CFG is off (e.g. training mini-rollout)."""
    a, _, _ = _run(guided=False)
    b, _, _ = _run(sigma_min=5.0, sigma_max=50.0, guided=False)
    assert torch.equal(a, b)
