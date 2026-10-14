from enum import IntEnum

import numpy as np
from atomworks.ml.conditions import CONDITIONS, ConditionBase, Level
from biotite.structure import AtomArray


class Hotspot(ConditionBase):
    """Hotspot annotation for protein design."""

    name = "hotspot"
    n_body = 1
    level = Level.ATOM
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """
        If there exists a hotspot annotation, we derive the mask by setting the mask to true for all atoms in
        any chain that has a hotspot annotation.
        """
        if cls.full_name in atom_array.get_annotation_categories():
            return atom_array.get_annotation(cls.full_name).astype(bool)

        return np.zeros(atom_array.array_length(), dtype=bool)


class RasaEnum(IntEnum):
    UNSPECIFIED = 0
    BURIED = 1
    INTERMEDIATE = 2
    EXPOSED = 3


class Rasa(ConditionBase):
    """Conditioning that provides binned RASA values to the model. Zero denotes no conditioning."""

    name = "rasa"
    n_body = 1
    level = Level.ATOM
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            binned_rasa_annotation = atom_array.get_annotation(cls.full_name)
            return binned_rasa_annotation != RasaEnum.UNSPECIFIED
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), RasaEnum.UNSPECIFIED, dtype=int)



class HBondEnum(IntEnum):
    NONE = 0
    DONOR = 1
    ACCEPTOR = 2
    BOTH = 3


class HydrogenBond(ConditionBase):
    """Conditioning that provides hydrogen bond information (donor/acceptor) to the model."""

    name = "hbond"
    n_body = 1
    level = Level.ATOM
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            annotation = cls.annotation(atom_array, default="raise")
            return annotation != HBondEnum.NONE
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), HBondEnum.NONE, dtype=int)


class ItypeEnum(IntEnum):
    NONE = 0
    PPI = 1
    PLI = 2
    PNA = 3  # covers both DNA and RNA


class InterfaceType(ConditionBase):
    """Interface type annotation for hotspot atoms (PPI/PLI/PNA)."""

    name = "itype"
    n_body = 1
    level = Level.ATOM
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            annotation = atom_array.get_annotation(cls.full_name)
            return annotation != ItypeEnum.NONE
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), ItypeEnum.NONE, dtype=int)


# Aliases for convenience
C_HOT = CONDITIONS.hotspot
C_HB = CONDITIONS.hbond
C_RAS = CONDITIONS.rasa
C_ITYPE = CONDITIONS.itype
