"""Non-condition standard annotations.

Defines standard annotations that benefit from the registry-driven save/load
and default-generation pattern, but are NOT model-input conditions.
"""

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils.standard_annotations.base import Level, StandardAnnotationBase


class ExpandableSegmentMinLength(StandardAnnotationBase):
    """Per-residue minimum length for expandable segment sentinels.

    A value of ``-1`` indicates the residue is not an expandable segment sentinel.
    Values ``>= 0`` indicate the sentinel's minimum realized segment length.
    """

    name = "expsegmin"
    n_body = 1
    level = Level.RESIDUE
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return cls.annotation(atom_array, default="raise") != -1
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), -1, dtype=int)


class ExpandableSegmentMaxLength(StandardAnnotationBase):
    """Per-residue maximum length for expandable segment sentinels.

    A value of ``-1`` indicates the residue is not an expandable segment sentinel.
    Values ``>= 0`` indicate the sentinel's maximum realized segment length.
    """

    name = "expsegmax"
    n_body = 1
    level = Level.RESIDUE
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return cls.annotation(atom_array, default="raise") != -1
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), -1, dtype=int)


class Atomize(StandardAnnotationBase):
    """Per-atom flag indicating whether a residue should be treated atomistically.

    Unlike standard polymer residues (which are tokenized at the residue level),
    atomized residues are represented at the individual atom level. This includes
    non-polymers and non-canonical polymer residues by default.

    The name ``"atomize"`` is preserved via an alias for backward compatibility:
    ``atom_array.atomize`` continues to work.
    """

    name = "atomize"
    alias = "atomize"  # total override: full_name = mask_name = "atomize"
    n_body = 1
    level = Level.ATOM
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: True for non-polymers and non-canonical polymer residues."""
        # Lazy to avoid a module-load cycle with `annotator`
        from atomworks.io.utils.annotator import ensure_annotations

        ensure_annotations(atom_array, "is_polymer", "is_standard_aa", "is_standard_rna", "is_standard_dna")
        is_standard_polymer = atom_array.is_polymer & (
            atom_array.is_standard_aa
            | atom_array.is_standard_rna
            | atom_array.is_standard_dna
            | ExpandableSegmentMinLength.mask(atom_array, default="generate")
        )
        return ~is_standard_polymer

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return cls.default_mask(atom_array)
