"""Pin the meaning of ``scale: 0`` in a per-condition ``inference_cfg_scale`` dict.

Composable CFG is ``delta_uncond + s * (delta_cond - delta_uncond)``, so ``s = 0`` is the
zero point of the guidance axis: the pure unconditional prediction. The engine used to
`continue` past a zero scale before claiming the condition, which left the main stream
fully conditional -- the *most* conditioned cell in a sweep sitting at the axis origin.

Parsed with `ast` rather than imported, so these run without torch installed.
"""

import ast
from pathlib import Path

import pytest

ENGINE = (
    Path(__file__).resolve().parents[1] / "src" / "rf3" / "inference_engines" / "rf3.py"
)


def _run_method() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(ENGINE.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    pytest.fail("RF3InferenceEngine.run not found")


def _scale_loop() -> ast.For:
    """The `for cond_name, scale in inference_cfg_scale.items()` loop."""
    for node in ast.walk(_run_method()):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and {e.id for e in node.target.elts if isinstance(e, ast.Name)}
            == {"cond_name", "scale"}
        ):
            return node
    pytest.fail("per-condition scale loop not found")


def _stmt_index(body: list[ast.stmt], predicate) -> int:
    for i, stmt in enumerate(body):
        if predicate(stmt):
            return i
    return -1


def _is_unknown_condition_raise(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and any(
        isinstance(n, ast.Raise) for n in ast.walk(stmt)
    )


def _is_zero_scale_skip(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.UnaryOp)
        and isinstance(stmt.test.op, ast.Not)
        and isinstance(stmt.test.operand, ast.Name)
        and stmt.test.operand.id == "scale"
    )


def test_zero_scale_is_recorded_not_silently_dropped():
    """A zero scale appends to `zero_scale_conditions` before skipping the stream."""
    loop = _scale_loop()
    idx = _stmt_index(loop.body, _is_zero_scale_skip)
    assert idx >= 0, "no `if not scale:` guard in the per-condition loop"
    names = {
        n.func.value.id
        for n in ast.walk(loop.body[idx])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name)
    }
    assert "zero_scale_conditions" in names, (
        "a zero scale must claim its condition via zero_scale_conditions, otherwise the "
        "main stream stays conditional and s=0 is not the unconditional prediction"
    )


def test_unknown_condition_raises_even_at_zero_scale():
    """The name check precedes the zero-scale skip, so a typo cannot pass silently."""
    loop = _scale_loop()
    raise_idx = _stmt_index(loop.body, _is_unknown_condition_raise)
    zero_idx = _stmt_index(loop.body, _is_zero_scale_skip)
    assert raise_idx >= 0, "no unknown-condition raise in the per-condition loop"
    assert raise_idx < zero_idx, (
        "`if not scale: continue` runs before the unknown-condition check, so "
        "{bogus_name: 0} would be accepted silently"
    )


def test_unconditional_main_is_built_for_zero_scale_only_runs():
    """`if guidance_specs or zero_scale_conditions:` gates the unconditional main stream.

    With only zero scales `guidance_specs` is empty, so a check on it alone would fall
    through to the "conditioned but unguided" branch.
    """
    guard = None
    for node in ast.walk(_run_method()):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "guidance_specs" in names:
                guard = names
                break
    assert (
        guard is not None
    ), "no `guidance_specs`-based guard on the main-stream rebuild"
    assert "zero_scale_conditions" in guard, (
        "the unconditional main stream must also be built when every requested scale is "
        "0, otherwise s=0 silently runs the fully conditional input"
    )
