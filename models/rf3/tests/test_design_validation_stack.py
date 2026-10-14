"""Regression tests for the design_validation metrics pipeline.

Covers two failure modes that produced "all-NaN" pairwise_distance_mae /
motif_rmsd in design_validation:

1. ``build_stack_from_atom_array_and_batched_coords`` previously used
   ``biotite.structure.stack``, which silently dropped 2D annotations
   (``C_DIS``) and the custom CCD registry stored on ``AtomArrayPlus``.
   Downstream metrics that read ``C_DIS.mask(...)`` therefore saw an empty
   mask -> every distance MAE bucket was NaN and motif RMSD had no motif
   atoms to subset to. The fix is to use ``stack_any``.

2. ``_calculate_hbond_stats`` passed the *dict* returned by
   ``get_motif_features`` to ``calculate_hbonds``, which expects an atom-level
   bool ``np.ndarray``. The hbplus path always raised, was caught silently,
   and no hbond metric was ever emitted -- not even a warning.

Run with::

    pytest models/rf3/tests/test_design_validation_stack.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlusStack,
    as_atom_array_plus,
)
from atomworks.ml.conditions import C_DIS
from rf3.metrics.design_gt_motif_metrics import MotifComparisonMetric
from rf3.metrics.design_metrics_utils import get_motif_features
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords


def _attach_distance_condition(atom_array, anchors: np.ndarray):
    """Attach a small synthetic C_DIS 2D annotation on ``anchors`` (atom indices)."""
    atom_array = as_atom_array_plus(atom_array)

    # All distinct ordered (i, j) pairs among anchors (excluding self).
    pairs = np.array(
        [(i, j) for i in anchors for j in anchors if i != j],
        dtype=np.int32,
    )
    coords = atom_array.coord
    values = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1).astype(
        float
    )
    C_DIS.set_annotation(atom_array, pairs=pairs, values=values)
    return atom_array


@pytest.fixture(scope="module")
def conditioned_atom_array():
    """A real, small AtomArrayPlus with a synthetic 4-anchor C_DIS condition."""
    cif = Path(__file__).parent / "data" / "5vht_from_file.cif"
    asm = parse(cif)["assemblies"]
    atom_array = asm[sorted(asm)[0]][0]
    valid_idx = np.where(~np.any(np.isnan(atom_array.coord), axis=1))[0]
    assert len(valid_idx) >= 4, "Need >=4 resolved atoms for the test"
    # Spread anchors across the structure.
    anchors = valid_idx[np.linspace(0, len(valid_idx) - 1, 4, dtype=int)]
    return _attach_distance_condition(atom_array, anchors)


def test_build_stack_preserves_2d_condition_and_motif_mask(conditioned_atom_array):
    """Stacking a single AtomArrayPlus across a diffusion batch must preserve C_DIS."""
    n_batch = 3
    coords = np.broadcast_to(
        conditioned_atom_array.coord[None, ...],
        (n_batch, *conditioned_atom_array.coord.shape),
    ).copy()
    # Add a tiny per-batch perturbation so the stack is non-trivial.
    rng = np.random.default_rng(0)
    coords += rng.normal(scale=0.05, size=coords.shape)

    stack = build_stack_from_atom_array_and_batched_coords(
        coords, conditioned_atom_array
    )

    # 1) Result must be an AtomArrayPlusStack so that 2D annotations survive.
    assert isinstance(stack, AtomArrayPlusStack), (
        f"Expected AtomArrayPlusStack to preserve 2D annotations; got {type(stack)}. "
        "If this fails, build_stack_from_atom_array_and_batched_coords likely uses "
        "biotite.structure.stack instead of stack_any."
    )

    # 2) C_DIS 2D mask must be non-empty on every model in the stack.
    for i in range(len(stack)):
        dense_mask = C_DIS.mask(stack[i]).as_dense_array(False).astype(bool)
        assert dense_mask.any(), (
            f"C_DIS mask was empty on stack model {i}; "
            "the 2D distance condition was lost during stacking."
        )

    # 3) get_motif_features (consumed by motif_rmsd) must report motif atoms.
    motif = get_motif_features(stack[0])
    assert motif[
        "is_motif_atom"
    ].any(), "is_motif_atom should be non-empty when C_DIS pairs exist."


def test_pairwise_distance_mae_not_all_nan(conditioned_atom_array):
    """With a non-empty 2D condition, mae_pairs_with_2d_condition must be finite."""
    n_batch = 2
    gt_coords = np.broadcast_to(
        conditioned_atom_array.coord[None, ...],
        (n_batch, *conditioned_atom_array.coord.shape),
    ).copy()
    # Slightly perturbed prediction -> finite, non-zero MAE.
    rng = np.random.default_rng(42)
    pred_coords = gt_coords + rng.normal(scale=0.5, size=gt_coords.shape)

    gt_stack = build_stack_from_atom_array_and_batched_coords(
        gt_coords, conditioned_atom_array
    )
    pred_stack = build_stack_from_atom_array_and_batched_coords(
        pred_coords, conditioned_atom_array
    )

    metric = MotifComparisonMetric(
        metric_type="dist_mae",
        metric_kwargs={"dist_mae": {"extra_distance_types": []}},
    )
    out = metric.compute(
        ground_truth_atom_array_stack=gt_stack,
        predicted_atom_array_stack=pred_stack,
    )

    # Key shape: "mae_<metric>_align__subset_" (no align/subset masks here).
    cond_key = next(k for k in out if k.startswith("mae_pairs_with_2d_condition"))
    all_key = next(k for k in out if k.startswith("mae_all_pairs"))

    cond_mae = float(out[cond_key])
    all_mae = float(out[all_key])

    assert np.isfinite(cond_mae), (
        f"{cond_key} should be finite when C_DIS pairs exist. Got NaN -- "
        "2D conditioning was lost during stacking."
    )
    assert np.isfinite(all_mae), f"{all_key} unexpectedly NaN: got {all_mae}"
    assert (
        cond_mae > 0.0
    ), "Predicted coords were perturbed; expected non-zero MAE on conditioned pairs."


def test_motif_rmsd_finite_with_motif_mask(conditioned_atom_array):
    """motif_rmsd with align_masks/subset_masks=is_motif_atom must produce a finite value."""
    n_batch = 2
    gt_coords = np.broadcast_to(
        conditioned_atom_array.coord[None, ...],
        (n_batch, *conditioned_atom_array.coord.shape),
    ).copy()
    rng = np.random.default_rng(7)
    pred_coords = gt_coords + rng.normal(scale=0.3, size=gt_coords.shape)

    gt_stack = build_stack_from_atom_array_and_batched_coords(
        gt_coords, conditioned_atom_array
    )
    pred_stack = build_stack_from_atom_array_and_batched_coords(
        pred_coords, conditioned_atom_array
    )

    metric = MotifComparisonMetric(
        metric_type="rmsd",
        align_masks=["is_motif_atom"],
        subset_masks=["is_motif_atom"],
    )
    out = metric.compute(
        ground_truth_atom_array_stack=gt_stack,
        predicted_atom_array_stack=pred_stack,
    )

    rmsd_key = next(k for k in out if k.startswith("rmsd_align_is_motif_atom"))
    assert np.isfinite(float(out[rmsd_key])), (
        f"{rmsd_key} should be finite when is_motif_atom resolves to >=1 atom; "
        "all-NaN here usually means C_DIS was lost during stacking and the "
        "motif mask is empty."
    )


def test_calculate_hbond_stats_handles_dict_motif_mask_fix(conditioned_atom_array):
    """Sanity: get_motif_features returns a dict; design_metrics_utils must
    pull out the bool array, not pass the dict to calculate_hbonds.

    This is a structural assertion -- we don't actually invoke hbplus here
    (that requires HBPLUS_PATH and an external binary); we just confirm the
    motif_mask we pass is a 1D bool array of the right length.
    """
    motif_features = get_motif_features(conditioned_atom_array)
    assert isinstance(motif_features, dict)
    motif_mask = motif_features["is_motif_atom"]
    assert isinstance(motif_mask, np.ndarray)
    assert motif_mask.dtype == bool
    assert motif_mask.shape == (conditioned_atom_array.array_length(),)


def test_calculate_hbond_stats_passes_ndarray_motif_mask(
    conditioned_atom_array, monkeypatch
):
    """_calculate_hbond_stats must hand calculate_hbonds a 1D bool ndarray (no hbplus needed)."""
    import rf3.metrics.design_metrics_utils as dmu

    arr = conditioned_atom_array.copy()
    gt = np.zeros(arr.array_length(), dtype=int)
    gt[:4] = 1
    arr.set_annotation("hbond_gt", gt)
    stack = build_stack_from_atom_array_and_batched_coords(arr.coord[None].copy(), arr)
    seen = {}

    def spy(atom_array, motif_mask, **kwargs):
        seen["mask"] = motif_mask
        return None, None, None, True

    monkeypatch.setattr(dmu, "calculate_hbonds", spy)
    dmu._calculate_hbond_stats(stack, stack, 2.2, 3.2)
    assert isinstance(seen["mask"], np.ndarray) and seen["mask"].dtype == bool
    assert seen["mask"].shape == (arr.array_length(),)
