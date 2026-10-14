"""Sampler-level checks for the recorded diffusion trajectories.

Hermetic (no checkpoint / data), like ``test_cfg_inference_noop.py``. Checked:

  * every recorded frame is mapped into the frame of the final X_L, exactly, despite the
    random rotation + translation each step applies.
  * under CFG, the guided x0 estimate is the composable combination of the per-stream
    estimates, and streams are only recorded on guided steps.
  * the dump writes one file per quantity and sample, with each step re-centred.
"""

import json

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import rf3.diffusion_samplers.inference_sampler as inference_sampler
import torch
from atomworks.enums import ChainType
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion
from rf3.utils.io import dump_diffusion_trajectories

from foundry.utils.alignment import weighted_rigid_align

B, I, L, D, C = 1, 4, 6, 2, 8


class _EquivariantDiffusionModule:
    """RNG-free and rigid-motion equivariant: contracts each sample toward its centroid.

    The stream's conditioning (``S_inputs_I``) sets the contraction, so streams differ.
    """

    def __call__(self, *, X_noisy_L, S_inputs_I, **_):
        centroid = X_noisy_L.mean(dim=-2, keepdim=True)
        return centroid + (0.9 - 0.1 * S_inputs_I.mean()) * (X_noisy_L - centroid)


def _identity_augmentation(X_L, X_exists_L, s_trans, return_rotation=False):
    R = torch.eye(3, dtype=X_L.dtype).expand(*X_L.shape[:2], 3, 3)
    return (X_L, R) if return_rotation else X_L


def _make_sampler(noise_scale: float = 1.003) -> SampleDiffusion:
    return SampleDiffusion(
        num_timesteps=6,
        min_t=0,
        max_t=1,
        sigma_data=16,
        s_min=4e-4,
        s_max=160,
        p=7,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=noise_scale,
        step_scale=1.5,
        solver="af3",
    )


def _stream(signal: float, scale: float, name: str) -> dict:
    return {
        "f": {"ref_element": torch.zeros(B, L, 4)},
        "S_inputs_I": torch.full((B, I, C), signal),
        "S_trunk_I": torch.full((B, I, C), signal),
        "Z_trunk_II": torch.full((B, I, I, C), signal),
        "scale": scale,
        "name": name,
    }


def _run(sampler: SampleDiffusion, streams=None, **cfg_kwargs) -> dict:
    torch.manual_seed(0)
    return sampler.sample_diffusion_like_af3(
        f={"ref_element": torch.zeros(B, L, 4)},
        S_inputs_I=torch.zeros(B, I, C),
        S_trunk_I=torch.zeros(B, I, C),
        Z_trunk_II=torch.zeros(B, I, I, C),
        diffusion_module=_EquivariantDiffusionModule(),
        diffusion_batch_size=D,
        coord_atom_lvl_to_be_noised=torch.zeros(D, L, 3),
        is_padding_L=torch.zeros(B, L, dtype=torch.bool),
        is_padding_I=torch.zeros(B, I, dtype=torch.bool),
        is_padding_chiral=torch.zeros(B, 0, dtype=torch.bool),
        inference_classifier_free_guidance=streams is not None,
        cfg_guidance_streams=streams,
        **cfg_kwargs,
    )


@pytest.fixture
def float64():
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(default)


def _assert_one_rigid_motion(moved_traj: list, fixed_traj: list) -> None:
    """Every step of `moved_traj` equals `fixed_traj` under one rigid motion per sample."""
    # All steps as one point cloud per sample: [D, n_steps * L, 3]
    moved = torch.cat(moved_traj, dim=-2)[0]
    fixed = torch.cat(fixed_traj, dim=-2)[0]
    aligned = weighted_rigid_align(
        X_L=moved, X_gt_L=fixed, w_L=torch.ones(moved.shape[:-1])
    )
    assert torch.allclose(aligned, moved, atol=1e-9 * fixed.abs().max())


def test_frames_share_the_final_frame(float64, monkeypatch):
    """With the dynamics equivariant and noise off, the augmented run's mapped trajectory
    must equal the unaugmented one up to a single rigid motion shared by all steps."""
    sampler = _make_sampler(noise_scale=0.0)
    streams = [_stream(1.0, 2.0, "a")]
    augmented = _run(sampler, streams)
    monkeypatch.setattr(
        inference_sampler, "centre_random_augmentation", _identity_augmentation
    )
    reference = _run(sampler, streams)

    for key in ("X_denoised_L_traj", "X_denoised_L_guided_traj"):
        _assert_one_rigid_motion(augmented[key], reference[key])
    _assert_one_rigid_motion(
        augmented["X_denoised_L_stream_trajs"]["a"],
        reference["X_denoised_L_stream_trajs"]["a"],
    )


def test_last_frame_is_the_final_prediction(float64):
    """The last step's frames reproduce X_L through the sampler's own update rule."""
    sampler = _make_sampler()
    out = _run(sampler)
    schedule = sampler._construct_inference_noise_schedule(torch.device("cpu"))
    t_hat, c_t = out["t_hats"][-1], schedule[-1]
    sigma = sampler.sigma_data
    X_noisy = out["X_noisy_L_traj"][-1] * torch.sqrt(t_hat**2 + sigma**2) / sigma
    delta = (X_noisy - out["X_denoised_L_traj"][-1]) / t_hat
    expected = X_noisy + sampler.step_scale * (c_t - t_hat) * delta
    assert torch.allclose(expected, out["X_L"], atol=1e-9 * out["X_L"].abs().max())


def test_guided_estimate_combines_streams(float64):
    """x0_cfg = x0_uncond + sum_j s_j (x0_j - x0_uncond) on every step."""
    streams = [_stream(1.0, 2.0, "a"), _stream(-0.5, 0.7, "b")]
    out = _run(_make_sampler(), streams)
    for step, guided in enumerate(out["X_denoised_L_guided_traj"]):
        uncond = out["X_denoised_L_traj"][step]
        expected = uncond + sum(
            stream["scale"] * (out["X_denoised_L_stream_trajs"][name][step] - uncond)
            for stream, name in zip(streams, ("a", "b"))
        )
        assert torch.allclose(guided, expected, atol=1e-9)


def test_streams_recorded_only_on_guided_steps(float64):
    """Outside the sigma window streams hold None and the guided estimate is the
    unconditional one."""
    out = _run(_make_sampler(), [_stream(1.0, 2.0, "a")], cfg_sigma_max=100.0)
    guided_steps = [float(t) <= 100.0 for t in out["t_hats"]]
    assert any(guided_steps) and not all(guided_steps)
    stream_traj = out["X_denoised_L_stream_trajs"]["a"]
    assert len(stream_traj) == len(out["t_hats"])
    for step, guided in enumerate(guided_steps):
        assert (stream_traj[step] is not None) == guided
        if not guided:
            assert torch.equal(
                out["X_denoised_L_guided_traj"][step], out["X_denoised_L_traj"][step]
            )


def test_no_cfg_records_no_cfg_trajectories():
    out = _run(_make_sampler())
    assert out["X_denoised_L_guided_traj"] == []
    assert out["X_denoised_L_stream_trajs"] == {}
    assert len(out["X_denoised_L_traj"]) == len(out["t_hats"])


def _read_models(path) -> torch.Tensor:
    return torch.from_numpy(
        pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=None).coord
    )


def test_dump_writes_recentred_frames_per_sample(tmp_path):
    """Files per sample, final step first, streams on guided steps only, and every step
    shifted so the guided x0 sits on the last one without changing within-step geometry."""
    atom_array = sequence_to_annotated_atom_array(
        "GSA", chain_id="A", chain_type=ChainType.POLYPEPTIDE_L
    )
    n_atoms, n_samples = atom_array.array_length(), 2
    # The pipeline's atom_array carries entities; the CIF writer can't derive them on a stack
    atom_array.set_annotation("label_entity_id", np.ones(n_atoms, dtype=np.int8))
    torch.manual_seed(0)
    # Steps in sampler order; the centre of mass wanders by 10 Å per step
    x0 = [torch.randn(1, n_samples, n_atoms, 3) + 10.0 * s for s in range(3)]
    noisy = [x + torch.randn_like(x) for x in x0]
    network_output = {
        "X_noisy_L_traj": noisy,
        "X_denoised_L_traj": [x + 0.5 for x in x0],
        "X_denoised_L_guided_traj": x0,
        "X_denoised_L_stream_trajs": {"a": [None, x0[1] - 1.0, x0[2] - 1.0]},
        "t_hats": [torch.tensor(t) for t in (100.0, 10.0, 1.0)],
    }
    dump_diffusion_trajectories(
        network_output, atom_array, tmp_path, seed=3, file_type="cif"
    )

    for i in range(n_samples):
        traj_dir = tmp_path / f"seed-3_sample-{i}" / "trajectory"
        frames = json.loads((traj_dir / "frames.json").read_text())["t_hat"]
        assert frames["noisy"] == frames["denoised_cfg"] == [1.0, 10.0, 100.0]
        assert frames["denoised_cond-a"] == [1.0, 10.0]

        cfg = _read_models(traj_dir / "denoised_cfg.cif")
        noisy_models = _read_models(traj_dir / "noisy.cif")
        cond = _read_models(traj_dir / "denoised_cond-a.cif")
        target = x0[-1][0, i].mean(dim=0)
        tol = 2e-3  # CIF keeps 3 decimals
        # Model 1 is the last step, unshifted; every step's guided x0 is centred on it
        assert torch.allclose(cfg[0], x0[-1][0, i], atol=tol)
        assert torch.allclose(cfg.mean(dim=1), target.expand(3, 3), atol=tol)
        for model, step in enumerate((2, 1, 0)):
            assert torch.allclose(
                noisy_models[model] - cfg[model],
                noisy[step][0, i] - x0[step][0, i],
                atol=tol,
            )
        assert torch.allclose(cond - cfg[:2], torch.full_like(cond, -1.0), atol=tol)
