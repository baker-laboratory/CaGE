"""Serialization and deserialization helpers for StandardAnnotation objects.

These helpers are used by :py:func:`~atomworks.io.utils.io_utils._to_cif_or_bcif`
(writing) and by :py:func:`~atomworks.io.parser._parse_from_cif` /
:py:func:`~atomworks.io.utils.io_utils.load_any` (reading) to persist all
registered :py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase`
subclasses into CIF extra-category blocks and restore them on load.

**Serialization tiers** (both mask and annotation follow the same rule):

- **Scalar 1-body** annotations (shape ``(n_atoms,)``): written to ``atom_site``
  via biotite's ``extra_fields`` mechanism, like any regular ``AtomArray``
  annotation.
- **Nonscalar 1-body** annotations (shape ``(n_atoms, k)``) and
  **2-body** annotations (``AnnotationList2D``): written as CIF extra-categories
  using a shared ``idx0, [idx1,] val0, val1, …`` format.  For nonscalar 1-body
  ``idx0 = [0, 1, …, n_atoms-1]`` (one index column); for 2-body ``idx0`` and
  ``idx1`` hold the pair atom indices (two index columns). Raises
  :py:exc:`ValueError` for 2-body annotations if the row count exceeds
  ``_MAX_MULTI_BODY_ROWS``
"""

import warnings
from typing import Literal

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
    AtomArrayPlusStack,
    as_atom_array_plus,
)
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase

# Maximum number of rows to save for multi-body AtomArrayPlus annotations (e.g. pairwise distances)
_MAX_MULTI_BODY_ROWS = 10_000

# ---------------------------------------------------------------------------
# Low-level CIF dict helpers
# ---------------------------------------------------------------------------


def _atleast_2d_last(arr: np.ndarray) -> np.ndarray:
    """Same as np.atleast_2d but adds a dimension at the end instead of the beginning."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return np.expand_dims(arr, axis=-1)
    else:
        return arr


def _idxs_values_to_cif_dict(idxs: np.ndarray, values: np.ndarray) -> dict:
    """Convert index and value arrays to a CIF category dict."""
    idxs = _atleast_2d_last(idxs)  # (N, n_body)
    values = _atleast_2d_last(values)  # (N, n_values)
    return {f"idx{i}": idxs[:, i] for i in range(idxs.shape[1])} | {
        f"val{i}": values[:, i] for i in range(values.shape[1])
    }


def _annotation_to_cif_idxs_values(
    annotation: np.ndarray | AnnotationList2D,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(idxs, values)`` from an annotation for CIF serialization.

    For :py:class:`~atomworks.io.utils.atom_array_plus.AnnotationList2D` (2-body):
    returns ``(pairs, values)``; bool values are cast to int.

    For ``np.ndarray`` (nonscalar 1-body): returns ``(arange(n_atoms), annotation)``.
    """
    if isinstance(annotation, AnnotationList2D):
        return annotation.pairs, annotation.values
    else:
        return np.arange(n_atoms), annotation


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_standard_annotation_categories(
    cif_block: pdbx.CIFBlock,
) -> dict[str, tuple[type[StandardAnnotationBase], bool]]:
    """Scan a CIF block and return found StandardAnnotation categories.

    Returns:
        Dict mapping ``{category_name: (annotation_cls, is_mask)}``.
        ``is_mask`` is True when the category represents the mask half.
    """
    found = {}
    valid_masks = STANDARD_ANNOTATIONS.get_valid_mask_names()
    valid_fulls = STANDARD_ANNOTATIONS.get_valid_full_names()
    for key in cif_block:
        if key in valid_masks:
            found[key] = (STANDARD_ANNOTATIONS.from_mask_name(key), True)
        elif key in valid_fulls:
            found[key] = (STANDARD_ANNOTATIONS.from_full_name(key), False)
    return found


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _get_available_sa_masks_and_annotations(
    atom_array: AtomArray,
    n_body: int | Literal["all"] = "all",
) -> tuple[list[type[StandardAnnotationBase]], list[type[StandardAnnotationBase]]]:
    """Get relevant StandardAnnotation objects for the given AtomArray.

    Returns:
        has_mask_list: list of StandardAnnotation objects for which a mask is available on the atom_array.:
        has_annotation_list: list of StandardAnnotation objects for which a non-mask annotation is available on the atom_array.
    """
    annotations = set(get_annotation_categories(atom_array, n_body=n_body))
    has_mask_list = []
    has_annotation_list = []

    for sa_cls in STANDARD_ANNOTATIONS:
        if sa_cls.mask_name in annotations:
            has_mask_list.append(sa_cls)
        if not sa_cls.is_mask and sa_cls.full_name in annotations:
            has_annotation_list.append(sa_cls)

    return has_mask_list, has_annotation_list


def _get_scalar_1body_sa_names(atom_array: AtomArray) -> list[str]:
    """Return names of 1-body StandardAnnotations on ``atom_array`` that should go into ``atom_site``.

    Includes 1-body masks (always scalar) and 1-body non-mask annotations
    whose value array is 1-dimensional.  Nonscalar 1-body annotation
    values are excluded (they go into a CIF extra-category instead).

    Args:
        atom_array: The structure whose annotations are inspected.

    Returns:
        Annotation names suitable for passing as ``extra_fields`` to
        :py:func:`~biotite.structure.io.pdbx.set_structure`.
    """
    names: set[str] = set()
    has_mask_list, has_annotation_list = _get_available_sa_masks_and_annotations(atom_array, n_body=1)

    # All 1-body masks are 1-dimensional
    names.update([sa_cls.mask_name for sa_cls in has_mask_list])

    # Only include 1-body annotations if they are 1-dimensional
    for sa_cls in has_annotation_list:
        annotation = sa_cls.annotation(atom_array)
        if annotation.ndim == 1:
            names.add(sa_cls.full_name)

    # Deduplicate preserving order (mask_name == full_name for is_mask annotations)
    return list(names)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_standard_annotation_categories(atom_array: AtomArray) -> dict[str, dict]:
    """Serialize nonscalar 1-body and 2-body StandardAnnotations to CIF extra-category dicts.

    1-body masks and scalar 1-body annotations are excluded here; they are written
    to ``atom_site`` via :py:func:`_get_scalar_1body_sa_names` instead.

    Args:
        atom_array: The structure to read annotations from.

    Returns:
        Dict suitable for passing as ``extra_categories`` to
        :py:func:`~atomworks.io.utils.io_utils._write_categories_to_block`.

    Raises:
        ValueError: If a 2-body annotation has more than ``MAX_MULTI_BODY_ROWS`` pairs.
    """
    extra_categories: dict[str, dict] = {}
    n_atoms = atom_array.array_length()

    has_mask_list, has_annotation_list = _get_available_sa_masks_and_annotations(atom_array, n_body="all")

    # --- mask ---
    # 1-body masks are handled via atom_site extra_fields; only 2-body masks are saved in extra CIFCategories
    for annotation_cls in has_mask_list:
        mask = annotation_cls.mask(atom_array)

        if annotation_cls.n_body == 1:
            continue
        if len(mask.pairs) > _MAX_MULTI_BODY_ROWS:
            raise ValueError(
                f"StandardAnnotation '{annotation_cls.mask_name}' would write {len(mask.pairs)} rows "
                f"(pairs), exceeding the limit of {_MAX_MULTI_BODY_ROWS}. "
            )

        idxs, values = _annotation_to_cif_idxs_values(mask, n_atoms)
        extra_categories[annotation_cls.mask_name] = _idxs_values_to_cif_dict(idxs, values)

    # --- non-mask annotation ---
    for annotation_cls in has_annotation_list:
        annotation = annotation_cls.annotation(atom_array)

        if annotation_cls.n_body == 1 and annotation.ndim == 1:
            continue  # Scalar 1-body annotations are saved in atom_site extra_fields
        if annotation_cls.n_body == 2 and len(annotation.pairs) > _MAX_MULTI_BODY_ROWS:
            raise ValueError(
                f"StandardAnnotation '{annotation_cls.full_name}' would write {len(annotation.pairs)} rows "
                f"(pairs), exceeding the limit of {_MAX_MULTI_BODY_ROWS}. "
            )

        idxs, values = _annotation_to_cif_idxs_values(annotation, n_atoms)
        extra_categories[annotation_cls.full_name] = _idxs_values_to_cif_dict(idxs, values)

    return extra_categories


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def _deserialize_standard_annotation_cif_category(
    cif_category: pdbx.CIFCategory,
    dtype: np.dtype,
    annotation_cls: type[StandardAnnotationBase],
    is_mask: bool,
    atom_array: AtomArray,
    model_num: int | None = None,
) -> None:
    """Deserialize one CIF extra-category onto ``atom_array`` in-place.

    The body order is inferred from the number of ``idx*`` columns:

    - **1 index column** → 1-body: ``val*`` columns are scattered into a
      zero-filled array of shape ``(n_atoms[, k])`` using ``idx0`` as the
      atom-index selector.
    - **2 index columns** → 2-body: ``idx0`` and ``idx1`` form pair indices;
      an :py:class:`~atomworks.io.utils.atom_array_plus.AnnotationList2D` is
      constructed and set on ``atom_array``.

    When ``model_num`` is provided and a ``model_num`` column exists in the
    category, only rows matching that model are used.
    """
    # Filter rows by model_num if present in the category
    row_mask = None
    if model_num is not None and "model_num" in cif_category:
        model_col = cif_category["model_num"].as_array(int)
        row_mask = model_col == model_num

    idxs = [cif_category[k].as_array(int) for k in sorted(k for k in cif_category if k.startswith("idx"))]
    vals = []
    for key in sorted(k for k in cif_category if k.startswith("val")):
        vals.append(cif_category[key].as_array(dtype))
    if not vals:
        raise ValueError("CIF category has no 'val' columns.")

    if row_mask is not None:
        idxs = [idx[row_mask] for idx in idxs]
        vals = [val[row_mask] for val in vals]

    val_array = np.stack(vals, axis=1) if len(vals) > 1 else vals[0]
    n_atoms = atom_array.array_length()

    if len(idxs) <= 1:
        # 1-body: scatter val columns into a full-length array.
        idx_array = idxs[0] if idxs else np.arange(n_atoms)
        if val_array.ndim == 1:
            full = np.zeros(n_atoms, dtype=dtype)
        else:
            full = np.zeros((n_atoms,) + val_array.shape[1:], dtype=dtype)
        full[idx_array] = val_array.astype(dtype)
        if is_mask:
            annotation_cls.set_mask(atom_array, full.astype(bool))
        else:
            annotation_cls.set_annotation(atom_array, full)
    else:
        # 2-body: reconstruct AnnotationList2D from pair index columns.
        pairs = np.stack(idxs, axis=1)
        value = val_array.astype(dtype)
        if is_mask:
            annotation_cls.set_mask(atom_array, AnnotationList2D(n_atoms=n_atoms, pairs=pairs, values=value))
        else:
            annotation_cls.set_annotation(atom_array, pairs=pairs, values=value)


def _deserialize_standard_annotations(
    atom_array: AtomArray | AtomArrayStack,
    cif_block: pdbx.CIFBlock,
    model_num: int | None = None,
) -> AtomArrayPlus | AtomArrayPlusStack:
    """Reconstruct all StandardAnnotations from a CIF block onto an AtomArray.

    Args:
        atom_array: The AtomArray (or stack) to annotate.
        cif_block: The CIF block containing the serialized annotations.
        model_num: When provided, filters SA category rows to only those
            matching this model number (for variable-length multi-model CIFs).

    Returns:
        The atom array promoted to :py:class:`AtomArrayPlus` with all found
        StandardAnnotations restored.
    """
    if isinstance(atom_array, AtomArray):
        atom_array = as_atom_array_plus(atom_array)
    elif isinstance(atom_array, AtomArrayStack):
        atom_array = AtomArrayPlusStack.from_atom_array_stack(atom_array)
    else:
        raise ValueError(f"AtomArray or AtomArrayStack expected, got {type(atom_array)}")

    for key, (annotation_cls, is_mask) in _detect_standard_annotation_categories(cif_block).items():
        dtype = bool if is_mask else annotation_cls.dtype
        try:
            _deserialize_standard_annotation_cif_category(
                cif_block[key], dtype, annotation_cls, is_mask, atom_array, model_num=model_num
            )
        except (ValueError, KeyError) as e:
            warnings.warn(f"Failed to deserialize StandardAnnotation '{key}': {e}", stacklevel=2)

    return atom_array
