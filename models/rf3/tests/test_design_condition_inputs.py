"""Hotspot, interface-type and hbond conditions: inference selection -> annotation -> features."""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from rf3.conditions import C_HB, C_HOT, C_ITYPE, HBondEnum, ItypeEnum
from rf3.data.condition_featurization import add_conditions_as_features
from rf3.utils.inference import (
    apply_hbond_selection,
    apply_hotspot_selection,
    apply_interface_type,
)

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def dimer():
    """5vht chains A/B (protein-protein), first assembly model."""
    asm = parse(TEST_DATA_DIR / "5vht_from_file.cif")["assemblies"]
    return as_atom_array_plus(asm[sorted(asm)[0]][0])


def _n_atoms(aa, chain, res_ids):
    return int(((aa.chain_id == chain) & np.isin(aa.res_id, res_ids)).sum())


def test_hotspot_multi_digit_scalar_and_range(dimer):
    """'A/*/28' is a scalar (not a range); 'A/*/20-25' is an inclusive range."""
    assert (
        apply_hotspot_selection(dimer.copy(), ["A/*/28"]).sum()
        == _n_atoms(dimer, "A", [28])
        > 0
    )
    assert apply_hotspot_selection(dimer.copy(), ["A/*/20-25"]).sum() == _n_atoms(
        dimer, "A", range(20, 26)
    )


def test_hotspot_and_itype_features(dimer):
    aa = dimer.copy()
    mask = apply_hotspot_selection(aa, ["A/*/20-25"])
    apply_interface_type(aa, "PPI", mask)
    feats = add_conditions_as_features(aa)
    hot = feats[C_HOT.get_full_name()].numpy().astype(bool).reshape(-1)
    np.testing.assert_array_equal(hot, mask)
    itype = feats[C_ITYPE.get_full_name()].numpy()
    assert itype.shape == (len(aa), len(ItypeEnum) - 1)  # NONE channel dropped
    assert itype[mask].sum() == mask.sum() and itype[~mask].sum() == 0


def test_itype_without_hotspot_is_skipped(dimer, caplog):
    aa = dimer.copy()
    apply_interface_type(aa, "PLI", None)
    assert "without hotspot_selection" in caplog.text
    assert C_ITYPE.full_name not in aa.get_annotation_categories()


def test_hbond_explicit_labels(dimer):
    aa = dimer.copy()
    # A/29 is ARG, so NE exists; a selection naming an atom the residue does not have
    # (e.g. "A/*/29/OG") now raises instead of being skipped.
    apply_hbond_selection(
        aa, {"A/*/28/N": "DONOR", "A/*/28/O": "ACCEPTOR", "A/*/29/NE": "BOTH"}
    )
    feat = add_conditions_as_features(aa)[C_HB.get_full_name()].numpy()
    labels = C_HB.annotation(aa)
    for col, enum in enumerate((HBondEnum.DONOR, HBondEnum.ACCEPTOR, HBondEnum.BOTH)):
        np.testing.assert_array_equal(feat[:, col], labels == enum)
    assert feat.sum(0).tolist() == [1, 1, int((labels == HBondEnum.BOTH).sum())]


def test_hbond_list_form_is_rejected(dimer):
    with pytest.raises(ValueError):
        apply_hbond_selection(dimer.copy(), ["A/*/28"])


def test_unresolvable_selection_raises(dimer):
    """A selection that matches nothing must fail loudly, not be skipped."""
    with pytest.raises(ValueError, match="Could not resolve selection"):
        apply_hotspot_selection(dimer.copy(), ["A/*/29/OG"])
