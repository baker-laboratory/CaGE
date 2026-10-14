"""Pin the CFG plumbing across the confidence path.

`RF3WithConfidence` overrides `forward` without calling `super()`, and the
inference engine reaches the model through `validation_step`. If any link stops
forwarding the CFG kwargs, guidance degrades to a silent no-op rather than an
error -- these checks fail loudly instead.

Parsed with `ast` rather than imported, so they run without torch installed.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rf3"

CFG_KWARGS = {
    "inference_classifier_free_guidance",
    "cfg_sigma_min",
    "cfg_sigma_max",
}


def _parse(rel: str) -> ast.Module:
    return ast.parse((SRC / rel).read_text())


def _method(tree: ast.Module, cls: str, fn: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == fn:
                    return item
            pytest.fail(f"{cls}.{fn} not found")
    pytest.fail(f"class {cls} not found")


def _params(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in fn.args.args + fn.args.kwonlyargs}


@pytest.mark.parametrize(
    ("rel", "cls", "fn", "extra"),
    [
        ("trainers/rf3.py", "RF3Trainer", "validation_step", "cfg_guidance_batches"),
        (
            "trainers/rf3.py",
            "RF3TrainerWithConfidence",
            "validation_step",
            "cfg_guidance_batches",
        ),
        ("model/RF3.py", "RF3", "forward", "cfg_guidance_inputs"),
        ("model/RF3.py", "RF3WithConfidence", "forward", "cfg_guidance_inputs"),
        (
            "model/RF3.py",
            "RF3WithConfidence",
            "_run_rollout_no_grad",
            "cfg_guidance_inputs",
        ),
    ],
)
def test_accepts_cfg_kwargs(rel, cls, fn, extra):
    params = _params(_method(_parse(rel), cls, fn))
    missing = (CFG_KWARGS | {extra}) - params
    assert not missing, f"{cls}.{fn} dropped CFG kwargs: {sorted(missing)}"


def test_both_trainers_accept_early_stop_fn():
    """The engine passes should_early_stop_fn unconditionally to either trainer."""
    tree = _parse("trainers/rf3.py")
    for cls in ("RF3Trainer", "RF3TrainerWithConfidence"):
        params = _params(_method(tree, cls, "validation_step"))
        assert "should_early_stop_fn" in params, cls


def test_confidence_forward_threads_cfg_into_rollout():
    """RF3WithConfidence.forward must hand the CFG kwargs to _run_rollout_no_grad."""
    fn = _method(_parse("model/RF3.py"), "RF3WithConfidence", "forward")
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_rollout_no_grad"
        ):
            passed = {kw.arg for kw in node.keywords}
            missing = (CFG_KWARGS | {"cfg_guidance_inputs"}) - passed
            assert not missing, f"_run_rollout_no_grad call dropped {sorted(missing)}"
            return
    pytest.fail("no _run_rollout_no_grad call found in RF3WithConfidence.forward")


def test_confidence_rollout_degrades_like_the_base_path():
    """Guidance with no reference streams must run unguided, not raise.

    The engine legitimately leaves `cfg_guidance_inputs` empty when an input has no
    conditions to split off; there it keeps `input` conditional and warns. The model
    cannot tell that apart from a dropped kwarg, so it must not fail hard.
    """
    fn = _method(_parse("model/RF3.py"), "RF3WithConfidence", "_run_rollout_no_grad")
    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "inference_classifier_free_guidance"
            for n in ast.walk(node.test)
        )
    ]
    assert guards, "no CFG guard found in _run_rollout_no_grad"
    assert any(
        any(
            isinstance(n, ast.Name) and n.id == "cfg_guidance_inputs"
            for n in ast.walk(g.test)
        )
        for g in guards
    ), "CFG guard must also require cfg_guidance_inputs, mirroring RF3.forward"


def test_confidence_streams_arg_is_plumbed_end_to_end():
    """CLI -> engine.run -> validation_step -> model.forward."""
    assert "confidence_streams" in _params(
        _method(_parse("inference_engines/rf3.py"), "RF3InferenceEngine", "run")
    )
    trainers = _parse("trainers/rf3.py")
    for cls in ("RF3Trainer", "RF3TrainerWithConfidence"):
        assert "confidence_streams" in _params(
            _method(trainers, cls, "validation_step")
        )
    assert "confidence_streams" in _params(
        _method(_parse("model/RF3.py"), "RF3WithConfidence", "forward")
    )


def test_guidance_streams_are_returned_for_scoring():
    """_run_rollout_no_grad must hand back the per-condition trunk outputs.

    Re-running the trunk to score them would double the cost of every guided run.
    """
    src = (SRC / "model" / "RF3.py").read_text()
    assert '"cfg_guidance_streams"' in src


def test_guidance_stream_dicts_carry_their_name():
    """Every stream dict needs `name`, or per-condition files fall back to condition_{i}.

    The engine derives the name and the trainer forwards it, but the model rebuilds these
    dicts from scratch -- the one place where the key can silently go missing.
    """
    tree = _parse("model/RF3.py")
    dicts = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cfg_guidance_streams"
        and node.args
        and isinstance(node.args[0], ast.Dict)
    ]
    assert dicts, "no cfg_guidance_streams.append({...}) found"
    for literal in dicts:
        keys = {k.value for k in literal.keys}
        missing = {"name", "scale"} - keys
        assert not missing, f"stream dict at line {literal.lineno} is missing {missing}"


def test_extra_streams_are_dumped_under_their_own_names():
    """One summary (and optional full) confidence file per extra stream."""
    dump = _method(_parse("inference_engines/rf3.py"), "RF3Output", "dump")
    src = ast.unparse(dump)
    assert "extra_confidences" in src, "dump() ignores the extra streams"
    assert "_summary_confidences_" in src
    assert "_confidences_" in src


def test_ranking_goes_through_the_resolved_stream():
    """Ranking and top-ranked selection must not read summary_confidences directly.

    `main` is the unconditional stream under CFG, so ranking by it would order guided
    samples by how good the *unguided* trunk thought they were.
    """
    tree = _parse("inference_engines/rf3.py")
    fields = {
        node.target.id
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "RF3Output"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "ranking_stream" in fields

    for fn_name in ("dump_ranking_scores", "dump_top_ranked_outputs"):
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name
        )
        src = ast.unparse(fn)
        assert "ranking_score" in src
        assert (
            "summary_confidences.get('ranking_score')" not in src
        ), f"{fn_name} bypasses the resolved ranking stream"


def test_every_stream_gets_its_own_metrics():
    """ptm / iptm / ranking_score are re-run per stream, not copied from main."""
    trainers = ast.unparse(
        _method(
            _parse("trainers/rf3.py"), "RF3TrainerWithConfidence", "validation_step"
        )
    )
    assert (
        "confidence_streams" in trainers and "pae_logits" in trainers
    ), "trainer does not re-run metrics per confidence stream"
    engine = _parse("inference_engines/rf3.py")
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "inject_metrics_into_summary"
        for n in engine.body
    )


def test_engine_has_single_validation_step_call():
    """Branching here previously dropped CFG kwargs whenever early stopping was on."""
    tree = _parse("inference_engines/rf3.py")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "validation_step"
    ]
    assert len(calls) == 1, f"expected one validation_step call, found {len(calls)}"
    passed = {kw.arg for kw in calls[0].keywords}
    missing = (CFG_KWARGS | {"cfg_guidance_batches", "should_early_stop_fn"}) - passed
    assert not missing, f"engine call dropped {sorted(missing)}"
