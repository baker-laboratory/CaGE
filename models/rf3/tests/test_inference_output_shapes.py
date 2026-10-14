"""Exercise real inference output/export with synthetic trainer results on CPU.

Only model loading, featurization/collation and validation_step are replaced. The
engine's sample loop, confidence compilation (including extra streams), ranking,
B-factor annotation and CIF/JSON writers run normally, without a checkpoint.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import torch
from atomworks.enums import ChainType
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from omegaconf import OmegaConf
from rf3.chemical import NHEAVY
from rf3.inference_engines import rf3 as engine_module
from rf3.utils.inference import InferenceInput
from rf3.utils.model_output import normalize_inference_network_output

N_BINS = 16


def _read_json(base, kind):
    return json.loads(base.with_name(f"{base.name}_{kind}.json").read_text())


def _logits(samples, tokens, offset=0):
    """Give every sample/stream a different, known confidence bin."""
    plddt = torch.full((samples, tokens, NHEAVY, N_BINS), -40.0)
    pair = torch.full((samples, tokens, tokens, N_BINS), -40.0)
    for sample in range(samples):
        plddt[sample, ..., sample + offset + 1] = 40.0
        pair[sample, ..., sample + offset + 1] = 40.0
    return {
        "plddt_logits": plddt.flatten(-2),
        "pae_logits": pair,
        "pde_logits": pair.clone(),
    }


def _engine(monkeypatch, atom_array, network_output, metrics_output):
    """Initialize just the model-independent part of RF3InferenceEngine.run."""
    real_atoms = torch.zeros((2, NHEAVY), dtype=torch.bool)
    for token, residue in enumerate(np.unique(atom_array.res_id)):
        real_atoms[token, : np.count_nonzero(atom_array.res_id == residue)] = True
    pipeline_output = {
        "atom_array": atom_array,
        "ground_truth": {"chain_iid_token_lvl": np.array(["A", "A"])},
        "confidence_feats": {"is_real_atom": real_atoms},
    }
    engine = object.__new__(engine_module.RF3InferenceEngine)
    engine._weights_loaded = True
    engine.initialize = Mock()
    engine.model_seeds = [7]
    engine.compress_outputs = False
    engine.early_stopping_plddt_threshold = 0
    engine._external_distributed_context = True
    engine.cfg = OmegaConf.create(
        {
            "trainer": {
                "loss": {
                    "confidence_loss": {
                        "plddt": {"n_bins": N_BINS, "max_value": 1.0},
                        "pae": {"n_bins": N_BINS, "max_value": 32.0},
                        "pde": {"n_bins": N_BINS, "max_value": 32.0},
                    }
                }
            }
        }
    )
    engine.pipeline = Mock(return_value=pipeline_output)
    engine.trainer = SimpleNamespace(
        fabric=SimpleNamespace(world_size=1, global_rank=0, to_device=lambda x: x),
        validation_step=Mock(
            return_value={
                "network_output": network_output,
                "metrics_output": metrics_output,
            }
        ),
    )
    monkeypatch.setattr(engine_module, "collate_batch", lambda batch, **_: batch[0])
    return engine


@pytest.fixture
def atom_array():
    atoms = sequence_to_annotated_atom_array(
        "AG", chain_id="A", chain_type=ChainType.POLYPEPTIDE_L
    )
    atoms.coord = np.zeros((len(atoms), 3), dtype=np.float32)
    return atoms


@pytest.mark.parametrize("samples", [1, 5])
@pytest.mark.parametrize("metrics_enabled", [False, True])
@pytest.mark.parametrize("head", ["structure", "confidence", "confidence_streams"])
def test_run_exports_every_sample(
    monkeypatch, tmp_path, atom_array, samples, metrics_enabled, head
):
    confidence = head != "structure"
    with_streams = head == "confidence_streams"
    coords = (
        torch.arange(samples * len(atom_array) * 3, dtype=torch.float32).reshape(
            samples, len(atom_array), 3
        )
        / 10
    )
    # The confidence trainer strips B only in its metrics branch. The structure
    # trainer and confidence trainer without metrics return [B=1, D, L, 3].
    trainer_coords = coords if confidence and metrics_enabled else coords.unsqueeze(0)
    raw = {"X_L": trainer_coords}
    if confidence:
        main = _logits(samples, 2)
        raw.update(
            X_pred_rollout_L=trainer_coords,
            plddt=main["plddt_logits"],
            pae=main["pae_logits"],
            pde=main["pde_logits"],
        )
    if with_streams:
        raw["confidence_by_stream"] = {
            "template": _logits(samples, 2, offset=1),
            "merged": _logits(samples, 2, offset=2),
        }
    metrics = (
        {f"ptm.ptm_{i}": 0.2 + 0.01 * i for i in range(samples)}
        if metrics_enabled
        else None
    )
    if with_streams and metrics_enabled:
        metrics["confidence_streams"] = {
            stream: {f"ptm.ptm_{i}": 0.5 + 0.01 * i for i in range(samples)}
            for stream in raw["confidence_by_stream"]
        }
    engine = _engine(monkeypatch, atom_array, raw, metrics)
    input_spec = InferenceInput(
        atom_array=atom_array,
        chain_info={},
        example_id="shape_test",
        template_selection=["A"] if with_streams else None,
    )
    assert (
        engine.run(
            input_spec,
            out_dir=tmp_path,
            annotate_b_factor_with_plddt=confidence,
            inference_classifier_free_guidance=with_streams,
            inference_cfg_scale={"template": 1.0},
            confidence_streams=["main", "conditions", "merged"]
            if with_streams
            else None,
        )
        is None
    )
    engine.trainer.validation_step.assert_called_once()
    assert engine.trainer.validation_step.call_args.kwargs["compute_metrics"] is True
    # Normalization must not alter the trainer's output or drop any sample.
    assert raw["X_L"] is trainer_coords
    example_dir = tmp_path / "shape_test"
    assert len(list(example_dir.glob("seed-*_sample-*"))) == samples
    for sample in range(samples):
        base = (
            example_dir
            / f"seed-7_sample-{sample}"
            / f"shape_test_seed-7_sample-{sample}"
        )
        structure = pdbx.get_structure(
            pdbx.CIFFile.read(f"{base}_model.cif"),
            model=1,
            extra_fields=["b_factor", "occupancy"],
        )
        np.testing.assert_allclose(structure.coord, coords[sample].numpy(), atol=1e-3)
        np.testing.assert_array_equal(structure.occupancy, 1.0)
        summary = _read_json(base, "summary_confidences")
        if metrics_enabled:
            assert summary["ptm"] == pytest.approx(0.2 + 0.01 * sample)
        else:
            assert "ptm" not in summary
        if not confidence:
            assert not base.with_name(base.name + "_confidences.json").exists()
            continue
        for stream, offset in [("main", 0)] + (
            [("template", 1), ("merged", 2)] if with_streams else []
        ):
            suffix = "" if stream == "main" else f"_{stream}"
            conf = _read_json(base, f"confidences{suffix}")
            expected_plddt = (sample + offset + 1.5) / N_BINS
            assert conf["atom_plddts"] == [round(expected_plddt, 2)] * len(atom_array)
            np.testing.assert_allclose(
                conf["pae"], (sample + offset + 1.5) * 32 / N_BINS
            )
            assert len(conf["pae"]) == len(conf["token_chain_ids"]) == 2
            stream_summary = _read_json(base, f"summary_confidences{suffix}")
            # Four-decimal serialization can round either side of a bin midpoint
            # after float32 softmax/averaging.
            assert stream_summary["overall_plddt"] == pytest.approx(
                expected_plddt, abs=5.1e-5
            )
            if stream != "main":
                assert stream_summary["stream"] == stream
                if metrics_enabled:
                    assert stream_summary["ptm"] == pytest.approx(0.5 + 0.01 * sample)
                else:
                    assert "ptm" not in stream_summary
        np.testing.assert_allclose(
            structure.b_factor, (sample + 1.5) * 100 / N_BINS, atol=0.01
        )


def test_normalization_preserves_nested_logits_trajectories_and_other_tensors():
    coords = torch.ones(1, 1, 1, 3)
    raw = {
        "X_L": coords,
        "X_pred_rollout_L": coords[0],
        "plddt": torch.ones(1, 1, NHEAVY * N_BINS),
        "confidence_by_stream": {"merged": _logits(1, 1)},
        "X_noisy_L_traj": [coords],
        "X_denoised_L_stream_trajs": {"main": [coords]},
        "distogram": torch.ones(1, 1, 1, N_BINS),
        "scalar": torch.tensor(1.0),
    }
    out = normalize_inference_network_output(raw)
    assert out is not raw
    assert out["X_L"].shape == out["X_pred_rollout_L"].shape == (1, 1, 3)
    for key in raw.keys() - {"X_L"}:
        assert out[key] is raw[key]
    assert raw["X_L"].shape == (1, 1, 1, 3)


@pytest.mark.parametrize("key", ["X_L", "X_pred_rollout_L"])
@pytest.mark.parametrize("shape", [(4, 3), (2, 1, 4, 3), (1, 4, 2), (1, 1, 1, 4, 3)])
def test_normalization_rejects_ambiguous_coordinate_shapes(key, shape):
    with pytest.raises(ValueError, match=key):
        normalize_inference_network_output({key: torch.zeros(shape)})


def test_early_stop_without_coordinates(monkeypatch, tmp_path, atom_array):
    raw = {"early_stopped": True, "mean_plddt": 0.1, "X_L": None}
    engine = _engine(monkeypatch, atom_array, raw, None)
    engine.early_stopping_plddt_threshold = 0.5
    engine.run(InferenceInput(atom_array, {}, "early"), out_dir=tmp_path)
    assert not list(tmp_path.rglob("*.cif"))
    assert "True" in (tmp_path / "early" / "score.csv").read_text()
