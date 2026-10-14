"""
2D pairwise distance conditioning for design tasks.

Ported from rfd3.transforms.design_task.distance_condition.
"""

from __future__ import annotations

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, as_atom_array_plus
from atomworks.ml.conditions import C_DIS
from biotite.structure import AtomArray
from foundry.common import true_with_prob
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def dropout_symmetric_elements_in_mask(
    mask: np.ndarray,
    min_valid_elements: int = 1,
    low: float = 0.1,
    high: float = 0.9,
) -> np.ndarray:
    """Randomly drop a fraction of upper-triangle entries in a symmetric (L, L) mask.

    Args:
        mask: Symmetric boolean array of shape ``(L, L)``.
        min_valid_elements: Minimum upper-triangle entries to keep.
        low: Lower bound for the fraction of entries to drop.
        high: Upper bound for the fraction of entries to drop.

    Returns:
        A new mask with the same shape and symmetry.
    """
    mask_dropped = mask.copy()
    triu_mask = np.triu(mask_dropped, k=1)
    n_valid = int(np.sum(triu_mask))

    if n_valid > min_valid_elements:
        percent_to_remove = np.random.uniform(low, high)
        n_knockout = min(int(n_valid * percent_to_remove), n_valid - min_valid_elements)

        valid_indices = np.nonzero(triu_mask)
        knockout_indices = np.random.permutation(len(valid_indices[0]))[:n_knockout]
        rows = valid_indices[0][knockout_indices]
        cols = valid_indices[1][knockout_indices]

        mask_dropped[rows, cols] = False
        mask_dropped[cols, rows] = False

    return mask_dropped


def compute_distances(
    atom_array: AtomArray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (pairs, distances) lists implied by a 1D or 2D boolean mask."""
    if mask.ndim == 1:
        assert mask.shape == (
            atom_array.array_length(),
        ), f"mask shape {mask.shape} != ({atom_array.array_length()},)"
        mask_2d = np.outer(mask, mask)
    else:
        assert mask.ndim == 2, f"mask.ndim must be 1 or 2, got {mask.ndim}"
        L = atom_array.array_length()
        assert mask.shape == (L, L), f"mask shape {mask.shape} != ({L}, {L})"
        mask_2d = mask

    mask_2d[np.diag_indices(mask_2d.shape[0])] = False
    pairs = np.array(np.where(mask_2d)).T
    distances = np.linalg.norm(
        atom_array.coord[pairs[:, 0]] - atom_array.coord[pairs[:, 1]], axis=-1
    )
    return pairs, distances


def sample_distance_condition(
    atom_array: AtomArray | AtomArrayPlus,
    can_have_condition: np.ndarray | None = None,
    condition_groups: np.ndarray | None = None,
    knockout_p: float = 0.8,
    dropout_min_fraction: float = 0.1,
    dropout_max_fraction: float = 0.9,
    additive: bool = False,
    allow_overlap: bool = False,
    threshold: float | None = None,
) -> AtomArrayPlus:
    """Sample distance-conditioned index pairs and annotate the atom array with C_DIS.

    Provide exactly one of ``can_have_condition`` or ``condition_groups``.

    - ``can_have_condition``: 1-D boolean mask of length ``L``; pairs ``(i, j)``
      are eligible when both endpoints are True.
    - ``condition_groups``: 1-D integer array of length ``L``; ``0`` means
      non-conditioned. Pairs are restricted to ``(i, j)`` where the two endpoints
      belong to *different* non-zero groups (between-component only).

    Args:
        atom_array: Structure to annotate with distance conditioning.
        can_have_condition: 1-D bool mask of positions eligible for conditioning.
        condition_groups: 1-D group ids per position (``0`` = non-conditioned).
        knockout_p: Probability of applying dropout to the pair mask.
        dropout_min_fraction: Lower bound for dropout fraction when applied.
        dropout_max_fraction: Upper bound for dropout fraction when applied.
        additive: If True, append to existing C_DIS annotation instead of replacing.
        allow_overlap: If True with additive, allow new pairs to overlap existing ones.
        threshold: Optional max distance (Angstroms); pairs above are dropped.

    Returns:
        AtomArrayPlus with C_DIS annotation set.
    """
    if (can_have_condition is None) == (condition_groups is None):
        # Both or neither were provided. Mirror the rfd3 behaviour: prefer
        # ``condition_groups`` and emit a warning.
        if condition_groups is None:
            raise ValueError(
                "Provide exactly one of can_have_condition or condition_groups."
            )
        can_have_condition = condition_groups > 0
        logger.warning(
            "Both can_have_condition and condition_groups are provided. "
            "Using condition_groups to determine can_have_condition."
        )

    atom_array = as_atom_array_plus(atom_array)
    L = atom_array.array_length()

    if condition_groups is not None:
        assert condition_groups.shape == (
            L,
        ), f"condition_groups.shape {condition_groups.shape} != (L={L},)"
        can_have_condition = condition_groups > 0
        different_group = condition_groups[:, None] != condition_groups[None, :]
        is_dist_visible = (
            np.outer(can_have_condition, can_have_condition) & different_group
        )
    else:
        is_dist_visible = np.outer(can_have_condition, can_have_condition)

    is_dist_visible[np.diag_indices(L)] = False

    if additive and not allow_overlap:
        existing_mask = C_DIS.mask(atom_array).as_dense_array(False).any(axis=1)
        assert (can_have_condition & existing_mask).sum() == 0, (
            "can_have_condition must not overlap with existing distance annotations"
        )

    if true_with_prob(knockout_p):
        is_dist_visible = dropout_symmetric_elements_in_mask(
            is_dist_visible,
            min_valid_elements=1,
            low=dropout_min_fraction,
            high=dropout_max_fraction,
        )

    assert (is_dist_visible.T == is_dist_visible).all(), (
        "distance mask must be symmetric"
    )

    pairs, dist_values = compute_distances(atom_array, is_dist_visible)

    if threshold is not None:
        assert threshold > 0, "threshold must be positive"
        keep = dist_values <= threshold
        pairs, dist_values = pairs[keep], dist_values[keep]

    if additive:
        existing_annotation = C_DIS.annotation(atom_array)
        existing_pairs, existing_dist_values = (
            existing_annotation.pairs,
            existing_annotation.values,
        )
        C_DIS.set_annotation(
            atom_array,
            pairs=np.concatenate([existing_pairs, pairs]),
            values=np.concatenate([existing_dist_values, dist_values]),
        )
    else:
        C_DIS.set_annotation(atom_array, pairs=pairs, values=dist_values)

    annot = C_DIS.annotation(atom_array)
    pairs, dist_values = annot.pairs, annot.values
    non_nan = ~np.isnan(dist_values)
    C_DIS.set_annotation(atom_array, pairs=pairs[non_nan], values=dist_values[non_nan])

    return atom_array
