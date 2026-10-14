"""Smoke tests for 2D distance conditioning (C_DIS) featurization.

Covers the crop-time behaviour that motivated porting rfd3's
``remap_2d_annotations``: when ``sample_distance_condition`` writes pair
indices on the *uncropped* atom_array and the pipeline subsequently crops it,
``C_DIS.annotation(...).as_dense_array(...)`` should still produce a
``[L_cropped, L_cropped]`` matrix consistent with the cropped index space.

These tests verify the featurization end-to-end without requiring a model
forward pass. They use the existing ``two_training_examples`` fixture for a
real cropped atom_array, then manually set a tiny known ``C_DIS`` annotation
on it.

Run with::

    pytest models/rf3/tests/test_distance_condition_featurization.py
"""

import numpy as np
import pytest
import torch
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_DIS
from rf3.data.condition_featurization import add_conditions_as_features
from rf3.data.design_tasks.distance_condition import sample_distance_condition

# ``two_training_examples`` is auto-discovered from tests/conftest.py — no import needed.


def _get_atom_array(sample):
    """Extract the atom_array from a pipeline output (dict or dataclass)."""
    if isinstance(sample, dict):
        return sample["atom_array"]
    return sample.atom_array  # dataclass with .atom_array


@pytest.mark.parametrize("experiment", ["quick-rf3"])
def test_distance_features_default_zero(two_training_examples):
    """When no design task wrote C_DIS, featurization yields zero distance + empty mask."""
    for sample in two_training_examples:
        atom_array = _get_atom_array(sample)
        L = atom_array.array_length()

        feats = add_conditions_as_features(atom_array)

        assert "condition_distance_2_atom" in feats
        assert "mask_distance_2_atom" in feats

        dist = feats["condition_distance_2_atom"]
        mask = feats["mask_distance_2_atom"]

        assert dist.shape == (L, L), f"got {tuple(dist.shape)}, expected ({L}, {L})"
        assert mask.shape == (L, L)
        assert dist.dtype == torch.float32
        assert mask.dtype == torch.bool
        assert torch.all(dist == 0.0), "default distance tensor must be all zeros"
        assert not mask.any(), "default mask must be all-False"


@pytest.mark.parametrize("experiment", ["quick-rf3"])
def test_distance_features_round_trip_after_crop(two_training_examples):
    """Set a known C_DIS on the (already cropped) atom_array and verify featurization
    densifies it into the expected ``[L, L]`` tensor at the right indices.
    """
    for sample in two_training_examples:
        atom_array = _get_atom_array(sample)
        L = atom_array.array_length()
        if L < 4:
            pytest.skip("atom_array too small to choose 4 distinct atoms")

        # Pick four valid (non-NaN) atom indices spread across the structure.
        valid_idx = np.where(~np.any(np.isnan(atom_array.coord), axis=1))[0]
        if len(valid_idx) < 4:
            pytest.skip("not enough non-NaN atoms in cropped array")
        anchors = valid_idx[np.linspace(0, len(valid_idx) - 1, 4, dtype=int)]

        # Build a 1D mask "can_have_condition" with those four atoms set.
        can_have = np.zeros(L, dtype=bool)
        can_have[anchors] = True

        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=can_have,
            knockout_p=0.0,  # never drop pairs — we want all C(4,2)=6 pairs preserved
            additive=False,
            allow_overlap=False,
            threshold=None,
        )

        feats = add_conditions_as_features(atom_array)
        dist = feats["condition_distance_2_atom"]
        mask = feats["mask_distance_2_atom"]

        # Mask should be True exactly on the off-diagonal pairs of `anchors`.
        expected_mask = np.zeros((L, L), dtype=bool)
        for i in anchors:
            for j in anchors:
                if i != j:
                    expected_mask[i, j] = True
        assert torch.equal(
            mask, torch.from_numpy(expected_mask)
        ), "mask must mark exactly the conditioned pairs"

        # Distance values at conditioned positions should match Euclidean distance.
        for i in anchors:
            for j in anchors:
                if i == j:
                    continue
                expected = float(
                    np.linalg.norm(atom_array.coord[i] - atom_array.coord[j])
                )
                got = float(dist[i, j].item())
                assert got == pytest.approx(
                    expected, rel=1e-5, abs=1e-5
                ), f"distance mismatch at ({i}, {j}): got {got}, expected {expected}"

        # Non-conditioned positions should be exactly 0.0.
        non_cond = ~torch.from_numpy(expected_mask)
        assert torch.all(
            dist[non_cond] == 0.0
        ), "non-conditioned positions must remain at the default 0.0"


@pytest.mark.parametrize("experiment", ["quick-rf3"])
def test_distance_features_indices_in_bounds(two_training_examples):
    """All ``C_DIS`` pair indices must be < L_cropped after featurization.

    Catches the failure mode where ``sample_distance_condition`` was called
    on the uncropped atom_array (with global indices) and the pipeline did
    not remap them — pair indices would point outside the cropped tensor.
    """
    for sample in two_training_examples:
        atom_array = _get_atom_array(sample)
        L = atom_array.array_length()

        # 2-body annotations require AtomArrayPlus
        atom_array = as_atom_array_plus(atom_array)
        annotation = C_DIS.annotation(atom_array, default="generate")
        if len(annotation.pairs) == 0:
            continue  # no C_DIS to validate

        max_idx = int(annotation.pairs.max())
        assert max_idx < L, (
            f"C_DIS pair index {max_idx} exceeds cropped atom_array length {L} — "
            "looks like crop did not remap pair indices."
        )
