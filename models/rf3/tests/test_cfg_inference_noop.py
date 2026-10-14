"""Sampler-level self-checks for composable classifier-free guidance.

Hermetic (no checkpoint / data): drives ``SampleDiffusion.sample_diffusion_like_af3``
with a deterministic dummy diffusion module. The main stream is the unconditional Δ∅;
guidance streams contribute ``scale * (delta_j - delta_∅)``. Invariants checked:

  * a guidance stream at scale 0 is inert (identical to no guidance).
  * a nonzero scale changes the trajectory (guidance is wired in).
  * two streams compose additively (weighted sum of their contributions).
"""

import torch
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion


class _DummyDiffusionModule:
    """Deterministic, RNG-free stand-in. Output depends on the conditioning (via
    ``S_inputs_I``) so different streams give different denoised outputs; being RNG-free
    keeps the sampler runs' random streams aligned for exact comparison."""

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


def _common_inputs():
    B, I, L, D, c = 1, 4, 6, 2, 8
    f = {"ref_element": torch.zeros(B, L, 4)}
    # Main stream = unconditional Δ∅ (all-zero conditioning signal).
    return dict(
        f=f,
        S_inputs_I=torch.zeros(B, I, c),
        S_trunk_I=torch.zeros(B, I, c),
        Z_trunk_II=torch.zeros(B, I, I, c),
        diffusion_module=_DummyDiffusionModule(),
        diffusion_batch_size=D,
        coord_atom_lvl_to_be_noised=torch.zeros(D, L, 3),
        is_padding_L=torch.zeros(B, L, dtype=torch.bool),
        is_padding_I=torch.zeros(B, I, dtype=torch.bool),
        is_padding_chiral=torch.zeros(B, 0, dtype=torch.bool),
    )


def _guidance_stream(signal: float, scale: float):
    B, I, L, c = 1, 4, 6, 8
    return {
        "f": {"ref_element": torch.zeros(B, L, 4)},
        "S_inputs_I": torch.full((B, I, c), signal),
        "S_trunk_I": torch.full((B, I, c), signal),
        "Z_trunk_II": torch.full((B, I, I, c), signal),
        "scale": scale,
    }


def _run(cfg_guidance_streams) -> torch.Tensor:
    sampler = _make_sampler()
    torch.manual_seed(0)
    kwargs = _common_inputs()
    if cfg_guidance_streams is not None:
        kwargs["inference_classifier_free_guidance"] = True
        kwargs["cfg_guidance_streams"] = cfg_guidance_streams
    return sampler.sample_diffusion_like_af3(**kwargs)["X_L"]


def test_scale_zero_is_inert():
    """A guidance stream at scale 0 must reproduce the no-guidance (main Δ∅) output."""
    x_off = _run(None)
    x_zero = _run([_guidance_stream(signal=1.0, scale=0.0)])
    assert torch.equal(x_off, x_zero)


def test_scale_changes_output():
    """A nonzero scale must change the output."""
    x_off = _run(None)
    x_guided = _run([_guidance_stream(signal=1.0, scale=2.0)])
    assert not torch.allclose(x_off, x_guided)


def test_two_streams_compose_additively():
    """Two identical streams at scale 1 equal one stream at scale 2 (Σ_j over streams)."""
    one = _guidance_stream(signal=1.0, scale=1.0)
    x_two_streams = _run([one, _guidance_stream(signal=1.0, scale=1.0)])
    x_double_scale = _run([_guidance_stream(signal=1.0, scale=2.0)])
    assert torch.allclose(x_two_streams, x_double_scale)
    assert not torch.allclose(x_two_streams, _run([one]))
