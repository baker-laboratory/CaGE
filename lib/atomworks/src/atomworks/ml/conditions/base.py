"""
Condition system which subclasses StandardAnnotation, providing a separate registry for
StandardAnnotations typically provided as conditions to ML models.
"""

from __future__ import annotations

from typing import Any, ClassVar

from biotite.structure import AtomArray

from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standard_annotations.base import (
    AnnotationRegistryAccessor,
    StandardAnnotationBase,
    StandardAnnotationMeta,
)

__all__ = [
    "CONDITIONS",
    "ConditionBase",
]


class ConditionMeta(StandardAnnotationMeta):
    """Metaclass for model-input conditions.

    Inherits all validation and registration from :py:class:`StandardAnnotationMeta`.
    Additionally registers concrete subclasses into ``_condition_registry``,
    a subset of ``StandardAnnotationMeta._registry``.
    """

    _condition_registry: ClassVar[dict[str, type[ConditionBase]]] = {}

    def __new__(meta, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):  # noqa: N804
        cls = super().__new__(meta, name, bases, namespace, **kwargs)

        annotation_name = namespace.get("name")
        if annotation_name:
            # Also register in condition-specific registry
            ConditionMeta._condition_registry[annotation_name] = cls

        return cls


class ConditionBase(StandardAnnotationBase, metaclass=ConditionMeta):
    """Abstract base class for all model-input conditions.

    A condition is a :py:class:`StandardAnnotationBase` that is semantically
    a model input (e.g., sequence, coordinates, chain pairing).

    Attributes:
        (required)
        name: The name of the condition.
        n_body: The number of bodies involved in the condition.
        level: The level at which the condition applies.

        (optional)
        is_symmetric: Whether the condition is symmetric. Only
            applies if ``n_body > 1``. Otherwise always ``True``.
        is_mask: Whether the condition itself is a mask. In this case the
            ``full_name`` is the same as the ``mask_name``.

        (derived)
        full_name: The full, systematic name of the condition of the form
            ``condition_{name}_{n_body}_{level}``.
        mask_name: The mask, systematic name of the condition of the form
            ``mask_{name}_{n_body}_{level}``.
    """

    _prefix: ClassVar[str] = "condition"


class ConditionAccessor(AnnotationRegistryAccessor):
    """Provides dynamic, attribute-based access to all registered conditions.

    Iterates ``ConditionMeta._condition_registry``, which contains only
    :py:class:`ConditionBase` subclasses (not other standard annotations).
    """

    _entry_kind: ClassVar[str] = "condition"

    @property
    def _registry(self) -> dict[str, type[ConditionBase]]:
        return ConditionMeta._condition_registry

    def fill_missing_conditions_with_defaults(self, atom_array: AtomArray) -> AtomArray:
        """Fill all missing condition annotations with their defaults.

        Iterates all registered conditions and generates default mask and
        annotation values for any that are not already present on the AtomArray.

        Args:
            atom_array: The AtomArray to fill. Modified in-place and returned.

        Returns:
            The atom array with all missing condition defaults applied.
        """
        all_annotation_categories = set(get_annotation_categories(atom_array, n_body="all"))
        for condition_cls in self:
            is_mask_annotated = condition_cls.mask_name in all_annotation_categories
            is_condition_annotated = condition_cls.full_name in all_annotation_categories

            if is_mask_annotated and (condition_cls.is_mask or is_condition_annotated):
                continue

            if not is_mask_annotated:
                default_mask = condition_cls.mask(atom_array, default="generate")
                condition_cls.set_mask(atom_array, default_mask)

            if not is_condition_annotated:
                default_annotation = condition_cls.annotation(atom_array, default="generate")
                condition_cls.set_annotation(atom_array, default_annotation)

        return atom_array

    def __repr__(self) -> str:
        return f"Conditions({self.list()})"


# Singleton instances for easy, clean access
CONDITIONS = ConditionAccessor()
