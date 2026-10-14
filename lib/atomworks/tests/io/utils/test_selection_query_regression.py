"""Regression tests for the selection-query fixes CaGE depends on."""

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.io.utils.selection import AtomSelectionStack


@pytest.fixture
def three_residues() -> struc.AtomArray:
    return struc.array(
        [
            struc.Atom(np.array([0.0, 0, 0]), chain_id="A", res_id=0, res_name="ALA", atom_name="CA"),
            struc.Atom(np.array([1.0, 0, 0]), chain_id="A", res_id=1, res_name="GLY", atom_name="CA"),
            struc.Atom(np.array([2.0, 0, 0]), chain_id="A", res_id=113, res_name="SER", atom_name="CA"),
        ]
    )


@pytest.mark.parametrize("query", ["A/*/28", "A/*/113", "A/*/48"])
def test_multi_digit_scalar_res_id_parses(query: str) -> None:
    # Regression: the old range regex r"-?\d+-?\d+" matched bare multi-digit ids
    AtomSelectionStack.from_query(query)


def test_res_id_zero_is_not_a_wildcard(three_residues: struc.AtomArray) -> None:
    mask = np.asarray(AtomSelectionStack.from_query("A/*/0").get_mask(three_residues))
    assert mask.tolist() == [True, False, False]


def test_explicit_range_still_works(three_residues: struc.AtomArray) -> None:
    mask = np.asarray(AtomSelectionStack.from_query("A/*/0-1").get_mask(three_residues))
    assert mask.tolist() == [True, True, False]
