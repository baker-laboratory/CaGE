"""Transform that realizes expandable segment sentinels in an AtomArrayPlus."""

from typing import Any, ClassVar

import numpy as np

from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.expandable_segment import realize_expandable_segments


class RealizeExpandableSegments(Transform):
    """Realize all expandable segment sentinels in the ``atom_array`` entry of the data dict.

    Each sentinel (``mask_expsegmin_1_residue == True`` and ``res_name != MASKED``)
    is replaced by N coordinate-free ``MASKED`` atom stubs, where N is sampled
    uniformly from ``[seg_min, seg_max]``.

    This transform must be applied **before** any transform that tokenizes or
    indexes atoms (e.g. cropping), since it changes the atom count.

    Args:
        rng: NumPy random generator.  ``None`` uses :py:func:`numpy.random.default_rng`.

    Examples:
        >>> transform = RealizeExpandableSegments(rng=np.random.default_rng(0))
        >>> data = transform({"atom_array": atom_array_with_segments})

    NOTE: This transform should only be used during inference/validation. Many of the incompatible transforms
    are from training, which could cause unusual behavior.
    """

    incompatible_previous_transforms: ClassVar[list[str]] = [
        "UnindexFlaggedTokens",
        "RemoveUnresolvedPNUnits",
        "CropContiguousLikeAF3",  # Cropping needs to be done after realizing segments to avoid OOMs
        "CropSpatialLikeAF3",
        "LoadPolymerMSAs",  # MSAs are not resized by this transform and will therefore be the wrong dimension
        "PairAndMergePolymerMSAs",
        "SampleDesignTask",  # avoid strange behaviors with design tasks
        # All featurization should take place after this transform
        "FeaturizeMSALikeAF3",
    ]

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """Initialize RealizeExpandableSegments.

        Args:
            rng: NumPy random generator.  ``None`` uses :py:func:`numpy.random.default_rng`.
        """
        self._rng = rng if rng is not None else np.random.default_rng()

    def check_input(self, data: dict[str, Any]) -> None:
        if "atom_array" not in data:
            raise ValueError("RealizeExpandableSegments requires 'atom_array' key in data dict.")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = realize_expandable_segments(data["atom_array"], rng=self._rng)
        return data
