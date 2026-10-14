"""Unit tests for the CFG trunk-stream combination rule.

`rf3.model.cfg_streams` is deliberately torch-free, so these run anywhere and exercise
the arithmetic with plain floats.
"""

import pytest
from rf3.model.cfg_streams import extrapolate_trunk_streams, trunk_tensors


def _recycling(s_inputs, s, z):
    """Trunk output spelling."""
    return {"S_inputs_I": s_inputs, "S_I": s, "Z_II": z}


def _stream(s_inputs, s, z, scale, name="c"):
    """Guidance stream spelling."""
    return {
        "S_inputs_I": s_inputs,
        "S_trunk_I": s,
        "Z_trunk_II": z,
        "scale": scale,
        "name": name,
    }


def test_trunk_tensors_reads_both_spellings():
    assert trunk_tensors(_recycling(1, 2, 3)) == (1, 2, 3)
    assert trunk_tensors(_stream(1, 2, 3, scale=1.0)) == (1, 2, 3)


def test_single_stream_is_linear_extrapolation():
    merged = extrapolate_trunk_streams(
        _recycling(0.0, 0.0, 0.0), [_stream(1.0, 2.0, 4.0, scale=3.0)]
    )
    # base + scale * (ref - base) with base = 0
    assert merged == {"S_inputs_I": 3.0, "S_I": 6.0, "Z_II": 12.0}


def test_scale_zero_is_identity():
    base = _recycling(5.0, 6.0, 7.0)
    merged = extrapolate_trunk_streams(base, [_stream(1.0, 1.0, 1.0, scale=0.0)])
    assert merged == {"S_inputs_I": 5.0, "S_I": 6.0, "Z_II": 7.0}


def test_deltas_accumulate_against_the_base_not_the_running_total():
    """Two identical streams at scale 1 must land at base + 2*(ref-base), not ref."""
    merged = extrapolate_trunk_streams(
        _recycling(0.0, 0.0, 0.0),
        [_stream(1.0, 1.0, 1.0, scale=1.0), _stream(1.0, 1.0, 1.0, scale=1.0)],
    )
    assert merged["S_I"] == pytest.approx(2.0)


def test_fractional_scale_interpolates():
    """A scale below 1 lands between the base and the reference."""
    merged = extrapolate_trunk_streams(
        _recycling(0.0, 0.0, 0.0), [_stream(1.0, 1.0, 1.0, scale=0.25)]
    )
    assert merged["S_I"] == pytest.approx(0.25)


def test_no_streams_returns_base_values():
    merged = extrapolate_trunk_streams(_recycling(1.0, 2.0, 3.0), [])
    assert merged == {"S_inputs_I": 1.0, "S_I": 2.0, "Z_II": 3.0}


def test_streams_with_different_scales_compose():
    """Two conditions at different scales each contribute their own delta."""
    merged = extrapolate_trunk_streams(
        _recycling(10.0, 10.0, 10.0),
        [
            _stream(12.0, 12.0, 12.0, scale=1.0),
            _stream(6.0, 6.0, 6.0, scale=0.25, name="d"),
        ],
    )
    # 10 + 1*(12-10) + 0.25*(6-10) = 10 + 2 - 1
    assert merged["S_I"] == pytest.approx(11.0)
