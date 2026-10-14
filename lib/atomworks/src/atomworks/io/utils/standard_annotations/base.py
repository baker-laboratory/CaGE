"""
StandardAnnotation system with:
 - a ``StandardAnnotationBase`` abstract base class for typed, defaulted, registry-driven
   annotations on ``AtomArray`` objects
 - metaclasses ``StandardAnnotationMeta`` that handles auto-registration and attribute pre-computation
 - accessor singleton ``STANDARD_ANNOTATIONS``
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar, Literal

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils import scatter
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.selection import get_annotation, get_annotation_categories, get_residue_starts

__all__ = [
    "STANDARD_ANNOTATIONS",
    "AnnotationRegistryAccessor",
    "StandardAnnotationBase",
]


class Level(StrEnum):
    """
    Level-based hierarchy for describing information in a molecular structure.

    Used for example to specify the level at which a `StandardAnnotation` applies.
    """

    ATOM = "atom"
    TOKEN = "token"
    RESIDUE = "residue"
    CHAIN = "chain"
    MOLECULE = "molecule"
    SYSTEM = "system"

    def _get_segment_or_group(self, atom_array: AtomArray) -> tuple[str, np.ndarray]:
        if self == Level.ATOM:
            raise ValueError("Apply and spread does not make sense for atom level.")
        if self == Level.TOKEN:
            try:
                from atomworks.ml.utils.token import get_token_starts
            except ImportError:
                raise ImportError(
                    "`atomworks.ml` module not found. All token-level functionality requires `atomworks.ml`, "
                    "so `Level.TOKEN` is not supported when it is unavailable."
                ) from None
            return "segment", get_token_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.RESIDUE:
            return "segment", get_residue_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.CHAIN:
            chain_key = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
            return "group", atom_array.get_annotation(chain_key)
        elif self == Level.MOLECULE:
            return "group", atom_array.get_annotation("molecule_id")
        elif self == Level.SYSTEM:
            return "group", np.ones(atom_array.array_length(), dtype=np.int32)
        else:
            raise ValueError(f"Invalid level: {self}")

    def apply(self, atom_array: AtomArray, data: np.ndarray, func: Callable[[AtomArray], Any]) -> Any:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def spread(self, atom_array: AtomArray, data: np.ndarray) -> np.ndarray:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.spread_segment_wise(grouping, data)
        elif strategy == "group":
            return scatter.spread_group_wise(grouping, data)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def apply_and_spread(
        self,
        atom_array: AtomArray,
        data: np.ndarray,
        func: Callable[[np.ndarray], Any],
    ) -> np.ndarray:
        """
        Apply a function to the data and spread the result back to the original positions.
        """

        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_and_spread_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_and_spread_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


class StandardAnnotationMeta(ABCMeta):
    """Metaclass for all standard annotations.

    Registers concrete subclasses into a shared registry and pre-computes
    derived class attributes (``full_name``, ``mask_name``).

    The registry ``StandardAnnotationMeta._registry`` is shared across all
    subclasses. This enforces name uniqueness across ALL standard annotations.
    """

    _registry: ClassVar[dict[str, type[StandardAnnotationBase]]] = {}

    def __new__(meta, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):  # noqa: N804
        # Create the class as usual
        cls = super().__new__(meta, name, bases, namespace, **kwargs)

        annotation_name = namespace.get("name")
        if annotation_name:
            # Validate required attributes
            required_attrs = ["name", "n_body", "level", "dtype"]
            for attr in required_attrs:
                assert attr in namespace, f"StandardAnnotations must define the {attr} attribute."

            if "is_symmetric" not in namespace:
                assert cls.n_body == 1, "Multi-body StandardAnnotations must define the is_symmetric attribute."
                cls.is_symmetric = True

            assert (
                "_" not in annotation_name
            ), "StandardAnnotations must not contain underscores in their names (reserved for internal use)."

            # Enforce global name uniqueness across all StandardAnnotations
            if annotation_name in StandardAnnotationMeta._registry:
                raise ValueError(
                    f"StandardAnnotation with name '{annotation_name}' is already registered. "
                    "Names must be unique across all StandardAnnotations"
                )

            # Resolve is_mask
            if "is_mask" in namespace:
                cls.is_mask = namespace["is_mask"]
            else:
                cls.is_mask = np.issubdtype(cls.dtype, bool)

            # Validate alias: only masks may have aliases
            alias = namespace.get("alias")
            if alias is not None and not cls.is_mask:
                raise ValueError(
                    f"StandardAnnotation '{annotation_name}' has alias='{alias}' but is_mask=False. "
                    "Only mask annotations may have aliases, since for non-masks the alias "
                    "would be ambiguous between the annotation and its mask."
                )

            cls.level = Level(cls.level)

            # Eagerly compute and cache full_name / mask_name
            cls.full_name = cls.get_full_name()
            cls.mask_name = cls.get_mask_name()

            # Register in shared registry
            StandardAnnotationMeta._registry[annotation_name] = cls

        return cls

    def __repr__(self) -> str:  # noqa: N804
        try:
            if self.n_body > 1:
                return (
                    f"{self.__name__}(name={self.name}, n_body={self.n_body}, "
                    f"level={self.level}, is_symmetric={self.is_symmetric})"
                )
            else:
                return f"{self.__name__}(name={self.name}, n_body={self.n_body}, level={self.level})"
        except AttributeError:
            # NOTE: This is a fallback for the abstract base class to ensure that methods like `__mro__` continue to work
            return super().__repr__()


class StandardAnnotationBase(ABC, metaclass=StandardAnnotationMeta):
    """Abstract base class for all standard annotations.

    A standard annotation is a typed, defaulted annotation with mask semantics
    that can be stored on an :py:class:`~biotite.structure.AtomArray` and
    serialized/deserialized from CIF files.

    Attributes:
        (required on concrete subclasses)
        name: Registry name. No underscores allowed.
        n_body: Number of bodies (1 for per-atom/residue, 2 for pairwise).
        level: Level at which the annotation applies.
        dtype: NumPy dtype for the annotation values.

        (optional)
        alias: If set (masks only), completely overrides ``full_name`` and
            ``mask_name``. The alias becomes the name used everywhere.
        is_symmetric: Whether pairwise annotation is symmetric. Required if ``n_body > 1``.
        is_mask: Whether this annotation is itself a mask (boolean). If not set,
            inferred from ``dtype``.

        (computed by metaclass)
        full_name: Systematic full name, or alias if set.
        mask_name: Systematic mask name, or alias if set.
    """

    _prefix: ClassVar[str] = "annotation"

    # --- Attributes to be defined by concrete subclasses ---
    name: ClassVar[str]
    n_body: ClassVar[int]
    level: ClassVar[Level]
    dtype: ClassVar[np.dtype | float | int | str | bool]
    alias: ClassVar[str | None] = None
    is_symmetric: ClassVar[bool]  # only needs to be set if n_body > 1

    # --- Pre-computed attributes (set by metaclass) ---
    full_name: ClassVar[str]
    mask_name: ClassVar[str]
    is_mask: ClassVar[bool]

    def __init__(self):
        """DO NOT INSTANTIATE THIS CLASS. Use its class methods directly."""
        raise RuntimeError(
            f"{self.__class__.__name__} is a definitional class and should not be instantiated. "
            "Use its class methods directly."
        )

    @classmethod
    def _resolve_level(cls, level: Level | str | None) -> Level:
        """Resolves the level to a Level enum."""
        return Level(level if level is not None else cls.level)

    @classmethod
    def _resolve_n_body(cls, n_body: int | None) -> int:
        """Resolves the number of bodies to an integer."""
        return n_body if n_body is not None else cls.n_body

    # --- Name generation methods ---
    @classmethod
    def get_full_name(cls) -> str:
        """Returns the full name of the StandardAnnotation at a given body order and level."""
        if cls.alias is not None:
            return cls.alias
        if cls.is_mask:
            # ... re-route to the mask name generation for masks
            return cls.get_mask_name()

        return f"{cls._prefix}_{cls.name}_{cls.n_body}_{cls.level}"

    @classmethod
    def get_mask_name(
        cls,
        n_body: int | None = None,
        level: Level | str | None = None,
    ) -> str:
        """Returns the mask name of the StandardAnnotation at a given body order and level.

        By default (i.e. if `n_body` and `level` are not provided), n_body and level
            are resolved to the default values for the StandardAnnotation.

        Args:
            n_body: The number of bodies involved in the StandardAnnotation.
            level: The level at which the StandardAnnotation applies.

        Returns:
            The mask name of the StandardAnnotation.

        Example:
            For a 2-body distance annotation which applies at the `ATOM` level, the
            default mask is a 2-body atom-level mask `mask_distance_2_atom`. In the future,
            we may implement automatic derivation of lower-body masks from higher-body masks.
            I.e. the 1-body token-level mask `mask_distance_1_token` could be derived from the default
        """
        if cls.alias is not None:
            return cls.alias

        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        return f"mask_{cls.name}_{n_body}_{level}"

    @classmethod
    def get_feature_name(
        cls,
        n_body: int | None = None,
        level: Level | str | None = None,
        suffix: str = "",
    ) -> str:
        """Returns the feature name at a given body order and level.

        Args:
            n_body: The body order to get the feature name for.
            level: The level to get the feature name for.
            suffix: An optional suffix to add to the feature name to allow
             flexibility for adding multiple different features for a single StandardAnnotation.

        Returns:
            The feature name of the StandardAnnotation. This will be of the form:
            `feature-<suffix>_<standard_annotation_name>_<n_body>_<level>`
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        suffix_str = f"-{suffix}" if suffix and not suffix.startswith("-") else suffix
        return f"feature{suffix_str}_{cls.name}_{n_body}_{level}"

    @classmethod
    def has_mask(cls, atom_array: AtomArray) -> bool:
        """Check if the mask for this StandardAnnotation is present on the AtomArray."""
        return cls.mask_name in get_annotation_categories(atom_array, n_body=cls.n_body)

    @classmethod
    def has_annotation(cls, atom_array: AtomArray) -> bool:
        """Check if the value annotation for this StandardAnnotation is present on the AtomArray."""
        return cls.full_name in get_annotation_categories(atom_array, n_body=cls.n_body)

    # --- Abstract Default Generation Methods ---
    @classmethod
    @abstractmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Generates the default mask for the annotation."""
        raise NotImplementedError(
            f"StandardAnnotation `{cls.name}` (class `{cls.__name__}`) does not have a default mask."
        )

    @classmethod
    @abstractmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.is_mask:
            return cls.default_mask(atom_array)
        raise NotImplementedError(
            f"StandardAnnotation `{cls.name}` (class `{cls.__name__}`) does not have a default annotation."
        )

    # --- Core Functionality ---
    @classmethod
    def mask(
        cls,
        atom_array: AtomArray,
        n_body: int | None = None,
        level: Level | str | None = None,
        *,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray:
        """Gets a mask from an AtomArray, falling back to a generated default.

        Args:
            atom_array: The AtomArray to get the mask from.
            n_body: The number of bodies involved in the annotation.
            level: The level at which the annotation applies.
            default: The default value to return if the mask is not found.
                If ``"generate"``, the default mask is generated and returned.
                If ``"raise"``, a ValueError is raised if the mask is not found.
                If any other value, that value is returned.

        Returns:
            The mask for this StandardAnnotation.

        Raises:
            ValueError: If the mask is not found and ``default`` is ``"raise"``.
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        mask_name = cls.get_mask_name(n_body, level)

        mask = get_annotation(atom_array, mask_name, n_body=n_body)

        if mask is None:
            # TODO: Enable default generation for levels & n_body other than the default
            default = "raise" if (level != cls.level) | (n_body != cls.n_body) else default

            if default == "generate":
                mask = cls.default_mask(atom_array)
            elif default == "raise":
                raise ValueError(f"AtomArray is missing {n_body}-body annotation `{mask_name}`.")
            else:
                mask = default

        return mask

    @classmethod
    def annotation(
        cls,
        atom_array: AtomArray,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray:
        """Gets an annotation from an AtomArray, falling back to a generated default.

        Args:
            atom_array: The AtomArray to get the annotation from.
            default: The default value to return if the annotation is not found.
                If ``"generate"``, the default annotation is generated and returned.
                If ``"raise"``, a ValueError is raised if the annotation is not found.
                If any other value, that value is returned.

        Returns:
            The annotation for this StandardAnnotation.

        Raises:
            ValueError: If the annotation is not found and ``default`` is ``"raise"``.
        """
        annotation = get_annotation(atom_array, cls.full_name, n_body=cls.n_body)

        if annotation is None:
            if default == "generate":
                annotation = cls.default_annotation(atom_array)
            elif default == "raise":
                raise ValueError(f"AtomArray is missing {cls.n_body}-body annotation `{cls.full_name}`.")
            else:
                annotation = default

        return annotation

    @classmethod
    def set_mask(
        cls,
        atom_array: AtomArray,
        mask: np.ndarray | AnnotationList2D,
        n_body: int | None = None,
        level: Level | None = None,
    ) -> None:
        """Sets the mask for the annotation on the AtomArray."""
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        mask_name = cls.get_mask_name(n_body, level)
        if n_body == 1:
            if not np.issubdtype(mask.dtype, bool):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must be a boolean array. Got {mask.dtype}.")
            atom_array.set_annotation(mask_name, mask)
        elif n_body == 2:
            if cls.is_symmetric:
                mask = mask.symmetrized()
            n_atoms = mask.n_atoms
            pairs = mask.pairs
            values = mask.values
            if n_atoms != atom_array.array_length():
                raise ValueError(
                    f"Mask `{mask_name}` for `{cls.name}` must have {atom_array.array_length()} atoms. Got {n_atoms}."
                )
            if len(pairs) and ((pairs.max() >= n_atoms) or (pairs.min() < 0)):
                raise ValueError(
                    f"Mask `{mask_name}` for `{cls.name}` must have pairs within the range [0, {n_atoms}). "
                    f"Got min={min(pairs)}, max={max(pairs)}."
                )
            if not np.issubdtype(pairs.dtype, np.integer):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must have pairs as integers. Got {pairs.dtype}.")
            if not np.issubdtype(values.dtype, np.bool_):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must be a boolean array. Got {mask.dtype}.")
            atom_array.set_annotation_2d(mask_name, pairs, values)
        else:
            raise NotImplementedError(
                f"Currently only 1-body and 2-body annotations are supported. Got {n_body}-body annotation."
            )

    @classmethod
    def set_annotation(cls, atom_array: AtomArray, *args, **annotation_kwargs) -> None:
        """Sets the annotation on the AtomArray. Dynamically dispatches to the correct body order."""
        if cls.n_body == 1:
            # ... handle correct argument logic
            if len(args) == 0:
                array = annotation_kwargs.pop("array")
            elif len(args) == 1:
                array = args[0]
            else:
                raise ValueError(f"Only one argument is allowed for 1-body annotations. Got {len(args)} arguments.")
            assert len(annotation_kwargs) == 0, "No keyword arguments are allowed for 1-body annotations."

            # ... set the annotation
            atom_array.set_annotation(cls.full_name, array)

        elif cls.n_body == 2:
            if not hasattr(atom_array, "_annot_2d"):
                raise ValueError("AtomArray must have 2D annotations to set 2D annotation.")

            # ... handle correct argument logic
            if len(args) == 1:
                assert isinstance(args[0], AnnotationList2D), "Only AnnotationList2D is allowed for 2-body annotations."
                annot = args[0]
                pairs, values = annot.pairs, annot.values
                annotation_kwargs = {"pairs": pairs, "values": values}
            elif len(args) == 0:
                pass
            else:
                raise ValueError(f"Only one argument is allowed for 2-body annotations. Got {len(args)} arguments.")

            # ... set the annotation
            atom_array.set_annotation_2d(cls.full_name, **annotation_kwargs)
        else:
            raise NotImplementedError("Currently only 1-body and 2-body annotations are supported.")

    @classmethod
    def is_valid(cls, atom_array: AtomArray) -> bool:
        """
        Check if the annotation (and mask, if not is_mask) is consistent at the relevant level.
        (i.e. do all atoms/tokens/residues/chains/molecules/systems have the same value for the StandardAnnotation?)

        Returns:
            bool: True if valid, False otherwise.
        """
        if cls.level == Level.ATOM:
            return True

        if cls.n_body == 1:
            is_same = lambda x: np.all(x == x[0]) if len(x) > 0 else True  # noqa: E731
            is_annotation_valid = cls.level.apply(atom_array, cls.annotation(atom_array), is_same)
            if not cls.is_mask:
                is_mask_valid = cls.level.apply(atom_array, cls.mask(atom_array), is_same)
                is_annotation_valid &= is_mask_valid
            return np.all(is_annotation_valid)

        # TODO: Implement n-body aggregation
        return True


class AnnotationRegistryAccessor:
    """Base class for annotation registry accessor singletons.

    Provides dynamic attribute-based and subscript access to a registry dict,
    plus utilities for querying field names by body order.

    Subclasses must override :py:attr:`_registry` to return the appropriate
    :py:class:`StandardAnnotationMeta` registry dict.
    """

    _entry_kind: ClassVar[str] = "standard annotation"

    @property
    def _registry(self) -> dict[str, type[StandardAnnotationBase]]:
        raise NotImplementedError

    def __getattr__(self, name: str) -> type[StandardAnnotationBase]:
        """Dynamically retrieves an entry from the registry."""
        name = name.replace("_", "-")
        return self.__getitem__(name)

    def __getitem__(self, name: str) -> type[StandardAnnotationBase]:
        """Retrieves an entry by its registry name."""
        try:
            return self._registry[name]
        except KeyError:
            raise AttributeError(
                f"No {self._entry_kind} named '{name}' is registered. " f"Available {self._entry_kind}s: {self.list()}"
            ) from None

    def list(self) -> list[str]:
        """Returns a list of registry names of all registered entries."""
        return list(self._registry.keys())

    def __iter__(self):
        """Iterate over all registered classes."""
        for name in self.list():
            yield self[name]

    def get_valid_full_names(self) -> frozenset[str]:
        """Returns the set of full names of all registered entries."""
        return frozenset(cls.full_name for cls in self)

    def get_valid_mask_names(self) -> frozenset[str]:
        """Returns the set of mask names of all registered entries."""
        return frozenset(cls.mask_name for cls in self)

    def get(self, name: str) -> type[StandardAnnotationBase]:
        """Retrieves an entry by its registry name."""
        return self.__getitem__(name)

    def from_full_name(self, full_name: str) -> type[StandardAnnotationBase]:
        """Resolves a full name to a registered class."""
        for cls in self:
            if cls.full_name == full_name:
                return cls
        raise KeyError(f"No {self._entry_kind} with full_name '{full_name}' found.")

    def from_mask_name(self, mask_name: str) -> type[StandardAnnotationBase]:
        """Resolves a mask name to a registered class."""
        for cls in self:
            if cls.mask_name == mask_name:
                return cls
        raise KeyError(f"No {self._entry_kind} with mask_name '{mask_name}' found.")

    def get_field_names(self, n_body: int | Literal["all"] = "all") -> list[str]:
        """Return all mask and annotation field names for the given body order.

        For each registered entry matching ``n_body``, includes its
        :py:attr:`~StandardAnnotationBase.mask_name` and, for non-mask
        annotations, its :py:attr:`~StandardAnnotationBase.full_name`.

        Args:
            n_body: Body order to filter by, or ``"all"`` for no filtering.
                Defaults to ``"all"``.

        Returns:
            Deduplicated list of field names.
        """
        names: set[str] = set()
        for cls in self:
            if n_body != "all" and cls.n_body != n_body:
                continue
            names.add(cls.mask_name)
            if not cls.is_mask:
                names.add(cls.full_name)
        return list(names)


class StandardAnnotationAccessor(AnnotationRegistryAccessor):
    """Provides dynamic, attribute-based access to all registered standard annotations.

    Iterates ``StandardAnnotationMeta._registry``, which also includes subclasses of
    :py:class:`StandardAnnotationBase`.
    """

    @property
    def _registry(self) -> dict[str, type[StandardAnnotationBase]]:
        return StandardAnnotationMeta._registry

    def __repr__(self) -> str:
        return f"StandardAnnotations({self.list()})"


STANDARD_ANNOTATIONS = StandardAnnotationAccessor()
