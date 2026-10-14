from typing import Callable

import numpy as np
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack, as_atom_array_plus_stack, stack_any
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from atomworks.ml.conditions import C_DIS

from beartype.typing import Any, List, Literal
from biotite.structure import AtomArray, AtomArrayStack, rmsd, superimpose

from foundry.metrics.metric import Metric

from foundry.utils.ddp import RankedLogger
from rf3.metrics.design_metrics_utils import get_motif_features
global_logger = RankedLogger(__name__, rank_zero_only=False)


def _harmonize_and_stack(arrays: list) -> AtomArrayStack:
    """Stack arrays, dropping annotations not common to all to avoid annotation mismatch errors."""
    try:
        return stack_any(arrays)
    except ValueError:
        common = set(arrays[0].get_annotation_categories())
        for arr in arrays[1:]:
            common &= set(arr.get_annotation_categories())
        harmonized = []
        for arr in arrays:
            arr = arr.copy()
            for annot in set(arr.get_annotation_categories()) - common:
                arr.del_annotation(annot)
            harmonized.append(arr)
        return stack_any(harmonized)

def align_structures_with_masks(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
) -> tuple[AtomArrayStack, AtomArrayStack]:
    """
    Align predicted structure to ground truth using specified masks (boolean arrays).
    Masks are only applied to the groud truth structure and used blindly on the predicted structure

    Args:
        gt_array: Ground truth atom array
        pred_array: Predicted atom array
        masks: List of masks to use for alignment.

    Returns:
        Tuple of (aligned_gt, aligned_pred)
    """

    assert len(gt_array_stack) == len(
        pred_array_stack
    ), "Ground truth and predicted atom array stacks must be the same length"
    assert all(
        len(gt_array_stack[i]) == len(pred_array_stack[i])
        for i in range(len(gt_array_stack))
    ), "Ground truth and predicted atom arrays must be the same length"
    # if len(gt_array_stack) == 1 or len(pred_array_stack) == 1:
    #     pred_array_stack = pred_array_stack[0]
    #     gt_array_stack = gt_array_stack[0]

    # Use the intersection of GT and pred masks so the two atom subsets have
    # the same length. They diverge in practice because pred coords come from
    # the model (no NaN) while GT has unresolved atoms (NaN), and the
    # `~has_nan_coord()` mask appended in `align_structures_by_annotations`
    # then differs between the two stacks.
    align_mask = np.logical_and.reduce(list(gt_masks) + list(pred_masks))
    gt_atoms_to_align = gt_array_stack[:, align_mask]
    pred_atoms_to_align = pred_array_stack[:, align_mask]

    assert len(gt_atoms_to_align) > 0, "No atoms found to align"
    assert len(pred_atoms_to_align) > 0, "No atoms found to align"
    assert gt_atoms_to_align.mask(
        "(~has_nan_coord())"
    ).all(), "Ground truth atoms to align have nan coords"
    assert pred_atoms_to_align.mask(
        "(~has_nan_coord())"
    ).all(), "Predicted atoms to align have nan coords"

    # align
    _, transform = superimpose(gt_atoms_to_align.coord, pred_atoms_to_align.coord)
    pred_array_stack.coord = transform.apply(pred_array_stack.coord)
    # X_L = torch.from_numpy(pred_atoms_to_align.coord)
    # X_gt_L = torch.from_numpy(gt_atoms_to_align.coord)
    # align_func = weighted_rigid_align(X_L, X_gt_L, return_alignment_function=True)
    # pred_array_stack.coord = align_func(torch.from_numpy(pred_array_stack.coord)).numpy()
    return gt_array_stack, pred_array_stack


def align_structures_by_annotations(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    align_to: List[str] | None = None,
) -> tuple[AtomArrayStack, AtomArrayStack]:
    """
    Align predicted structure to ground truth using specified masks.
    Masks are only applied to the groud truth structure and used blindly on the predicted structure
    Note nan coords are always ignored when aligning.

    Args:
        gt_array: Ground truth atom array
        pred_array: Predicted atom array
        align_to: List of mask strings to use for alignment.
            If empty or None, no alignment is performed.
            These annotations must be in the ground truth atom array.

    Returns:
        Tuple of (aligned_gt, aligned_pred)
    """

    if not align_to:
        return gt_array_stack, pred_array_stack

    masks = {"gt": [], "pred": []}
    for array_stack, array_type in zip(
        [gt_array_stack, pred_array_stack], ["gt", "pred"]
    ):
        annotation_categories = get_annotation_categories(array_stack, n_body="all")
        for mask_expr in align_to:
            # Strip leading ~ only for the annotation existence check
            base_name = mask_expr.lstrip("~")
            assert (
                base_name in annotation_categories
            ), f"Align_to mask '{base_name}' is not present in the ground truth atom array annotations"
            # get a list of masks — use the original expression (with any ~) for the actual mask
            masks[array_type].append(array_stack.mask(mask_expr))
        # always ignore nan coords when aligning
        masks[array_type].append(array_stack.mask("(~has_nan_coord())"))
    gt_masks, pred_masks = masks["gt"], masks["pred"]

    return align_structures_with_masks(
        gt_array_stack, pred_array_stack, gt_masks, pred_masks
    )


def compute_rmsd(
    gt_array_stack: AtomArrayStack, pred_array_stack: AtomArrayStack, **kwargs
) -> float:
    """
    Compute RMSD between two atom array stacks. Returns the RMSD for each atom array in the stack.

    Args:
        gt_array_stack: Ground truth atom array stack
        pred_array_stack: Predicted atom array stack

    Returns:
        dict with key "rmsd" and value RMSD values for each atom array in the stack
    """

    assert len(gt_array_stack) == len(
        pred_array_stack
    ), "Ground truth and predicted atom array stacks must be the same length"
    if len(gt_array_stack) == 0 or len(pred_array_stack) == 0:
        return {"rmsd": np.array([np.nan])}

    assert all(
        len(gt_array_stack[i]) == len(pred_array_stack[i])
        for i in range(len(gt_array_stack))
    ), "Ground truth and predicted atom arrays must be the same length"

    rmsds = []
    for i in range(len(gt_array_stack)):
        if len(gt_array_stack[i]) == 0:
            rmsds.append(np.nan)
            continue
        # xyz1 = torch.from_numpy(gt_array_stack[i].coord)
        # xyz2 = torch.from_numpy(pred_array_stack[i].coord)
        # rmsds.append(float(get_rmsd(xyz1, xyz2)))
        rmsds.append(float(rmsd(gt_array_stack[i].coord, pred_array_stack[i].coord)))

    return {"rmsd": np.asarray(rmsds)}


def compute_distance_mae(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    extra_distance_types: List[
        Literal["tip_atom", "inter_chain", "intra_chain", "intra_chain_protein"]
    ] = [],
    aggregation_fn: Literal["max", "min", "mean"] = "mean",
) -> float:
    """
    Compute distance MAE between two atom array stacks. Returns the MAE for each atom array in the stack.
    Will check gt_array_stack for 2D distance annotation and use the mask to mask some distances.
    Args:
        gt_array_stack: Ground truth atom array stack.
        pred_array_stack: Predicted atom array stack.
        extra_distance_types: List of extra distance types to compute. Can be "tip_atom", "inter_chain", "intra_chain", or "intra_chain_protein".
            "tip_atom" will compute the MAE of conditioned distances between protein atoms and non-protein atoms.
        aggregation_fn: Function to aggregate the MAE values for each pair in an atom array.

    Returns:
        dict with keys "mae_all_pairs", "mae_pairs_with_2d_condition", "mae_pairs_no_2d_condition" and values as numpy arrays
    """
    assert len(gt_array_stack) == len(
        pred_array_stack
    ), "Ground truth and predicted atom array stacks must be the same length"

    assert len(gt_array_stack) > 0, "Ground truth atom array stack is empty"

    assert all(
        len(gt_array_stack[i]) == len(pred_array_stack[i])
        for i in range(len(gt_array_stack))
    ), "Ground truth and predicted atom arrays must be the same length"

    if not all(
        distance_type
        in ["tip_atom", "inter_chain", "intra_chain", "intra_chain_protein"]
        for distance_type in extra_distance_types
    ):
        raise ValueError(
            f"One or more distance types are invalid: {extra_distance_types}. Must be one of: tip_atom, inter_chain, intra_chain, intra_chain_protein"
        )

    if aggregation_fn == "max":
        aggregation_fn = np.max
    elif aggregation_fn == "min":
        aggregation_fn = np.min
    elif aggregation_fn == "mean":
        aggregation_fn = np.mean
    else:
        raise ValueError(
            f"Invalid aggregation function: {aggregation_fn}. Must be one of: max, min, mean"
        )

    # ========== Convenience ==========
    def mae(
        pred_dists: np.ndarray,
        gt_dists: np.ndarray,
        mask: np.ndarray,
        aggregation_fn: Callable[[np.ndarray], float] = aggregation_fn,
    ) -> float:
        # Exclude diagonal (self-distances are always 0 and would artificially lower MAE)
        mask = mask.copy().astype(bool)
        mask[np.diag_indices(mask.shape[0])] = False
        if mask.any():
            return float(aggregation_fn(np.abs(pred_dists[mask] - gt_dists[mask])))
        else:
            return np.nan

    # =================================

    metrics = {
        "all_pairs": [],
        "pairs_with_2d_condition": [],
        "pairs_short": [],
        "pairs_medium": [],
        "pairs_long": [],
        "pairs_no_2d_condition": [],
    } | {
        f"{distance_type}_{length_type}": []
        for distance_type in extra_distance_types
        for length_type in ["short", "medium", "long", "all_lengths"]
    }

    for i in range(len(gt_array_stack)):
        # Handle empty arrays
        if len(gt_array_stack[i]) == 0:
            for metric in metrics.keys():
                metrics[metric].append(np.nan)
            continue

        # Compute pairwise distances
        gt_diff = (
            gt_array_stack[i].coord[:, None, :] - gt_array_stack[i].coord[None, :, :]
        )
        gt_dists = np.linalg.norm(gt_diff, axis=2)

        pred_diff = (
            pred_array_stack[i].coord[:, None, :]
            - pred_array_stack[i].coord[None, :, :]
        )
        pred_dists = np.linalg.norm(pred_diff, axis=2)

        dist_conditioning_mask = (
            C_DIS.mask(gt_array_stack[i]).as_dense_array(False).astype(bool)
        )

        metrics["all_pairs"].append(
            mae(pred_dists, gt_dists, np.ones_like(pred_dists, dtype=bool))
        )
        metrics["pairs_with_2d_condition"].append(
            mae(pred_dists, gt_dists, dist_conditioning_mask)
        )
        metrics["pairs_short"].append(
            mae(pred_dists, gt_dists, dist_conditioning_mask & (gt_dists < 4.0))
        )
        metrics["pairs_medium"].append(
            mae(
                pred_dists,
                gt_dists,
                dist_conditioning_mask & (gt_dists >= 4.0) & (gt_dists < 8.0),
            )
        )
        metrics["pairs_long"].append(
            mae(pred_dists, gt_dists, dist_conditioning_mask & (gt_dists >= 8.0))
        )
        metrics["pairs_no_2d_condition"].append(
            mae(pred_dists, gt_dists, ~dist_conditioning_mask)
        )

        for distance_type in extra_distance_types:
            for length_type in ["short", "medium", "long", "all_lengths"]:
                if length_type == "all_lengths":
                    length_mask = np.ones_like(gt_dists, dtype=bool)
                elif length_type == "short":
                    length_mask = gt_dists < 4.0
                elif length_type == "medium":
                    length_mask = (gt_dists >= 4.0) & (gt_dists < 8.0)
                elif length_type == "long":
                    length_mask = gt_dists >= 8.0
                if distance_type == "tip_atom":
                    protein_mask = np.isin(
                        gt_array_stack[i].chain_type,
                        [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L],
                    )
                    tipatom_mask_2d = protein_mask[:, None] & ~protein_mask[None, :]
                    tipatom_mask_2d = tipatom_mask_2d | tipatom_mask_2d.T  # symmetric
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            tipatom_mask_2d & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "inter_chain":
                    inter_chain_mask = (
                        gt_array_stack[i].chain_id[:, None]
                        != gt_array_stack[i].chain_id[None, :]
                    )
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            inter_chain_mask & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "intra_chain":
                    intra_chain_mask = (
                        gt_array_stack[i].chain_id[:, None]
                        == gt_array_stack[i].chain_id[None, :]
                    )
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            intra_chain_mask & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "intra_chain_protein":
                    protein_mask = np.isin(
                        gt_array_stack[i].chain_type,
                        [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L],
                    )
                    protein_mask_2d = protein_mask[:, None] & protein_mask[None, :]
                    intra_chain_mask = (
                        gt_array_stack[i].chain_id[:, None]
                        == gt_array_stack[i].chain_id[None, :]
                    )
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            intra_chain_mask
                            & protein_mask_2d
                            & dist_conditioning_mask
                            & length_mask,
                        )
                    )

    return {f"mae_{metric}": np.asarray(metrics[metric]) for metric in metrics.keys()}


def structure_comparison_metric(
    ground_truth_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus,
    predicted_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus,
    subset_masks: List[str] | None = None,
    align_masks: List[str] | None = None,
    metric_fn: Callable[[AtomArray, AtomArray], float] = compute_rmsd,
    **metric_kwargs: Any,
) -> dict[str, float]:
    """
    Compute structure comparison metrics between ground truth and predicted structures.
    Note that we always ignore nan coords when comparing structures. You do not need to include that in your subset masks.

    Args:
        ground_truth_atom_array_stack: Ground truth atom array or stack
        predicted_atom_array_stack: Predicted atom array or stack (optional if predicted_coords provided)
        subset_masks: List of mask strings to subset the atom arrays
        align_masks: List of mask strings to use for alignment
        metric_type: Callable that computes the metric between two atom arrays. These methods assume everything is subset and aligned properly.
        metric_kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
        Dictionary with comparison results
    """
    # Convert inputs to stacks for uniform handling
    if isinstance(ground_truth_atom_array_stack, list):
        gt_stack = stack_any(ground_truth_atom_array_stack)
    elif isinstance(ground_truth_atom_array_stack, AtomArray):
        gt_stack = stack_any([ground_truth_atom_array_stack])
    else:
        gt_stack = (
            stack_any([ground_truth_atom_array_stack])
            if not isinstance(ground_truth_atom_array_stack, AtomArrayStack)
            else ground_truth_atom_array_stack
        )

    try:
        if isinstance(predicted_atom_array_stack, list):
            pred_stack = _harmonize_and_stack(predicted_atom_array_stack)
        elif isinstance(predicted_atom_array_stack, AtomArray):
            pred_stack = stack_any([predicted_atom_array_stack])
        else:
            pred_stack = (
                stack_any([predicted_atom_array_stack])
                if not isinstance(predicted_atom_array_stack, AtomArrayStack)
                else predicted_atom_array_stack
            )
    except Exception as e:
        global_logger.error(
            f"Error processing predicted_atom_array_stack: {e}."
        )
        return {}

    # ... copy to avoid modifying the original stacks, and ensure AtomArrayPlusStack
    # for 2D annotation support (C_DIS.mask calls get_annotation_2d_categories)
    pred_stack = as_atom_array_plus_stack(pred_stack).copy()
    gt_stack = as_atom_array_plus_stack(gt_stack).copy()

    subset_masks = [] if subset_masks is None else subset_masks
    align_masks = [] if align_masks is None else align_masks
    annotations_to_ensure = [
        annot.strip("~")
        for annot in subset_masks + align_masks
        if annot.strip("~") in ANNOTATOR_REGISTRY
    ]
    ensure_annotations(gt_stack, *annotations_to_ensure)
    ensure_annotations(pred_stack, *annotations_to_ensure)

    # HACK: While we're still supporting both annotation schemes, we have to use a conversion scheme. Once we fully
    # switch over to the Condition API, this will no longer be necessary.
    for aa_stack in [gt_stack, pred_stack]:
        is_motif_atom = get_motif_features(aa_stack[0])["is_motif_atom"]
        aa_stack.set_annotation("is_motif_atom", is_motif_atom)

    # align
    gt_stack, pred_stack = align_structures_by_annotations(
        gt_stack, pred_stack, align_masks
    )

    # subset
    subset_mask = gt_stack.mask(
        f"({' & '.join(subset_masks + ['(~has_nan_coord())'])})"
    )  # always ignore nan coords when comparing structures
    gt_stack = gt_stack[:, subset_mask]
    pred_stack = pred_stack[:, subset_mask]

    # compute metric
    metric_result = metric_fn(gt_stack, pred_stack, **metric_kwargs)
    metric_result = {
        f"{metric}_align_{'_'.join(align_masks)}_subset_{'_'.join(subset_masks)}": np.mean(
            result
        )
        for metric, result in metric_result.items()
    }

    return metric_result


class MotifComparisonMetric(Metric):
    """
    Computes structure comparison metrics between ground truth and predicted structures.

    Supports various comparison types including RMSD and distance MAE, with optional
    subsetting and alignment capabilities.

    Args:
        subset_masks: List of mask strings to subset the atom arrays
        align_masks: List of mask strings to use for alignment
        metric_type: Type of comparison metric ("rmsd" or "dist_mae")
        metric_kwargs: Additional keyword arguments to pass to the metric function. Must be in the format:
            {
                "dist_mae": {  # key must be the name of the metric type ("dist_mae" or "rmsd")
                    "extra_distance_types": ["tip_atom", "inter_chain", "intra_chain"]
                }
            }

    Example:
        >>> metric = StructureComparisonMetric(
        ...     subset_masks=["is_backbone & is_polymer"],
        ...     align_masks=["is_backbone"],
        ...     metric_type="rmsd",
        ... )
        >>> result = metric.compute(
        ...     ground_truth_atom_array_stack=gt, predicted_atom_array_stack=pred
        ... )
        >>> print(result)
        {'rmsd_align_is_backbone_subset_is_backbone_is_polymer': 1.23}
    """

    def __init__(
        self,
        subset_masks: List[str] | None = None,
        align_masks: List[str] | None = None,
        metric_type: Literal["rmsd", "dist_mae"] = "rmsd",
        metric_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset_masks = subset_masks
        self.align_masks = align_masks
        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "predicted_atom_array_stack": "predicted_atom_array_stack",
        }

    def compute(
        self,
        ground_truth_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
        predicted_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    ) -> dict[str, float]:
        """
        Compute structure comparison metrics.

        Args:
            ground_truth_atom_array_stack: Ground truth atom array or stack
            predicted_atom_array_stack: Predicted atom array or stack (optional if predicted_coords provided)
            predicted_coords: Predicted coordinates (optional if predicted_atom_array_stack provided)

        Returns:
            Dictionary with comparison results
        """
        metric_kwargs = self.metric_kwargs.get(self.metric_type, {})
        return structure_comparison_metric(
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            predicted_atom_array_stack=predicted_atom_array_stack,
            subset_masks=self.subset_masks,
            align_masks=self.align_masks,
            metric_fn=STRUCTURE_COMPARISON_METRICS[self.metric_type],
            **metric_kwargs,
        )


STRUCTURE_COMPARISON_METRICS = {
    "rmsd": compute_rmsd,
    "dist_mae": compute_distance_mae,
}
