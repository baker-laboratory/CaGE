"""Sampler-level self-checks for the motif-guidance hook.

Hermetic (no checkpoint / data): drives ``SampleDiffusion.sample_diffusion_like_af3``
with a deterministic dummy diffusion module and a stub guidance engine, so every
assertion is exact. What matters here is not the projection itself (covered in
test_motif_guidance.py) but that it is wired in at the right place:

  * ``motif_guidance=None`` is bitwise inert, so existing callers are unaffected.
  * a stub that returns its argument unchanged is also bitwise inert.
  * a stub that moves x0 changes the trajectory.
  * CFG survives the hook: stream scale still changes the output when guidance is on.
  * guidance works with CFG off, and the guided-x0 trajectory records the refined x0.
  * t_norm is the full-schedule step fraction.
  * partial diffusion and B != 1 are refused, loudly.
"""

import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import (
    SampleDiffusion,
    SamplePartialDiffusion,
)

B, I, L, D, C = 1, 4, 6, 2, 8
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


class _DummyDiffusionModule:
    """Deterministic, RNG-free stand-in; output depends on the conditioning."""

    def __call__(self, *, X_noisy_L, t, f, S_inputs_I, S_trunk_I, Z_trunk_II, **kw):
        return 0.9 * X_noisy_L + S_inputs_I.mean()


class _StubEngine:
    """Records every call; optionally shifts the first atom to perturb x0."""

    def __init__(self, shift: float = 0.0):
        self.shift = shift
        self.t_norms: list[float] = []
        self.returned: list[torch.Tensor] = []

    def refine(self, x0, *, t_norm):
        self.t_norms.append(t_norm)
        if self.shift == 0.0:
            self.returned.append(x0)
            return x0  # identity return: the exact no-op contract
        out = x0.clone()
        out[..., 0, :] = out[..., 0, :] + self.shift
        self.returned.append(out)
        return out


def _inputs(batch: int = B):
    f = {"ref_element": torch.zeros(batch, L, 4)}
    return dict(
        f=f,
        S_inputs_I=torch.zeros(batch, I, C),
        S_trunk_I=torch.zeros(batch, I, C),
        Z_trunk_II=torch.zeros(batch, I, I, C),
        diffusion_module=_DummyDiffusionModule(),
        diffusion_batch_size=D,
        coord_atom_lvl_to_be_noised=torch.zeros(D, L, 3),
        is_padding_L=torch.zeros(batch, L, dtype=torch.bool),
        is_padding_I=torch.zeros(batch, I, dtype=torch.bool),
        is_padding_chiral=torch.zeros(batch, 0, dtype=torch.bool),
    )


def _stream(signal: float, scale: float):
    return {
        "f": {"ref_element": torch.zeros(B, L, 4)},
        "S_inputs_I": torch.full((B, I, C), signal),
        "S_trunk_I": torch.full((B, I, C), signal),
        "Z_trunk_II": torch.full((B, I, I, C), signal),
        "scale": scale,
    }


def _run(*, motif_guidance=None, streams=None, sampler=None, batch: int = B):
    sampler = sampler or SampleDiffusion(**SAMPLER_KW)
    torch.manual_seed(0)
    kwargs = _inputs(batch)
    if streams is not None:
        kwargs["inference_classifier_free_guidance"] = True
        kwargs["cfg_guidance_streams"] = streams
    if motif_guidance is not None:
        kwargs["motif_guidance"] = motif_guidance
    return sampler.sample_diffusion_like_af3(**kwargs)


def test_none_is_bitwise_inert():
    """The default must not perturb any existing caller."""
    assert torch.equal(_run()["X_L"], _run(motif_guidance=None)["X_L"])


def test_identity_stub_is_bitwise_inert():
    """An engine that returns its argument must leave the trajectory untouched."""
    stub = _StubEngine(shift=0.0)
    out = _run(motif_guidance=stub)
    assert torch.equal(out["X_L"], _run()["X_L"])
    # It was called on every step, and never counted as applied.
    assert len(stub.t_norms) == 4


def test_moving_stub_changes_the_trajectory():
    assert not torch.equal(
        _run(motif_guidance=_StubEngine(shift=5.0))["X_L"], _run()["X_L"]
    )


def test_t_norm_is_the_full_schedule_step_fraction():
    stub = _StubEngine()
    _run(motif_guidance=stub)
    assert stub.t_norms == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_cfg_survives_the_hook():
    """Stream scale must still move the output with motif guidance active.

    If the hook replaced delta_L instead of correcting it, the CFG blend would be
    discarded and both scales would give the same trajectory.
    """
    low = _run(motif_guidance=_StubEngine(shift=5.0), streams=[_stream(1.0, 1.0)])[
        "X_L"
    ]
    high = _run(motif_guidance=_StubEngine(shift=5.0), streams=[_stream(1.0, 3.0)])[
        "X_L"
    ]
    assert (low - high).abs().max() > 1e-3


def test_guided_traj_records_the_refined_x0_without_cfg():
    """CFG off: the guided-x0 record must exist and hold what refine returned.

    ``map_trajectories_to_final_frame`` rewrites every recorded frame into the final
    X_L frame, so the stored tensor is a rigid transform of what refine returned, not
    a copy of it. Comparing internal distances checks both that the right content was
    recorded and that only a rigid map was applied to it.
    """
    stub = _StubEngine(shift=5.0)
    out = _run(motif_guidance=stub)
    traj = out["X_denoised_L_guided_traj"]
    assert len(traj) == 4
    for recorded, returned in zip(traj, stub.returned):
        assert recorded.shape == returned.shape
        for d in range(D):
            assert torch.allclose(
                torch.cdist(recorded[0, d], recorded[0, d]),
                torch.cdist(returned[0, d].cpu(), returned[0, d].cpu()),
                atol=1e-2,
            )
    # Without guidance and without CFG the record stays empty.
    assert _run()["X_denoised_L_guided_traj"] == []


class _ContractingEngine:
    """Shrinks x0 toward its own centroid, so its effect is frame-invariant."""

    def refine(self, x0, *, t_norm):
        centre = x0.mean(dim=-2, keepdim=True)
        return centre + 0.5 * (x0 - centre)


def _radius_of_gyration(x_bdl3: torch.Tensor) -> torch.Tensor:
    centred = x_bdl3 - x_bdl3.mean(dim=-2, keepdim=True)
    return centred.pow(2).sum(-1).mean(-1).sqrt()


def test_correction_sign_moves_the_update_toward_the_refined_x0():
    """A contracting engine must shrink the result, not expand it.

    This is the one invariant a sign flip on the delta correction cannot satisfy, and
    it is invisible to every other check here: the engine would still be called, the
    counter would still advance and the trajectory would still be recorded.
    """
    guided = _radius_of_gyration(_run(motif_guidance=_ContractingEngine())["X_L"])
    plain = _radius_of_gyration(_run()["X_L"])
    assert (guided < plain).all(), (guided.tolist(), plain.tolist())


def test_partial_diffusion_refuses_motif_guidance():
    sampler = SamplePartialDiffusion(partial_t=2, **SAMPLER_KW)
    assert sampler.supports_motif_guidance is False
    with pytest.raises(NotImplementedError, match="does not support motif guidance"):
        _run(motif_guidance=_StubEngine(), sampler=sampler)
    # Unguided partial diffusion still works.
    assert (
        _run(sampler=SamplePartialDiffusion(partial_t=2, **SAMPLER_KW))["X_L"]
        is not None
    )


def test_batched_pairformer_input_refuses_motif_guidance():
    with pytest.raises(ValueError, match="requires B == 1"):
        _run(motif_guidance=_StubEngine(), batch=2)
