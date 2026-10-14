"""The motif_guidance_selection grammar and the feats it resolves to.

Hermetic: a synthetic AtomArray plus the real AtomSelectionStack resolver, so the
selector syntax checked here is the same one every other condition uses.
"""

import numpy as np
import pytest
from biotite.structure import AtomArray
from rf3.guidance import build_motif_guidance_feats, normalize_motif_guidance_groups

# chain A: HIS 10 (N, CA), GLU 11 (N, CA);  chain X: LIG 99 (C1, C2)
CHAIN = ["A", "A", "A", "A", "X", "X"]
RES_ID = [10, 10, 11, 11, 99, 99]
RES_NAME = ["HIS", "HIS", "GLU", "GLU", "LIG", "LIG"]
ATOM = ["N", "CA", "N", "CA", "C1", "C2"]


def _array(shift: float = 0.0) -> AtomArray:
    n = len(CHAIN)
    arr = AtomArray(n)
    arr.chain_id = np.array(CHAIN)
    arr.res_id = np.array(RES_ID)
    arr.res_name = np.array(RES_NAME)
    arr.atom_name = np.array(ATOM)
    arr.element = np.array(["N", "C", "N", "C", "C", "C"])
    arr.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3) + shift
    return arr


def test_empty_selection_resolves_to_nothing():
    assert normalize_motif_guidance_groups(None) == []
    assert normalize_motif_guidance_groups([]) == []
    assert build_motif_guidance_feats(_array(), None) is None


def test_flat_list_of_single_atom_selections_is_one_group_all_anchors():
    feats = build_motif_guidance_feats(_array(), ["A/HIS/10/N", "A/HIS/10/CA"])
    assert feats["motif_idx"].tolist() == [0, 1]
    assert feats["anchor_idx_in_motif"].tolist() == [0, 1]
    assert feats["group_starts"].tolist() == [0, 2]
    assert feats["group_names"] == ["default"]


def test_flat_list_applies_the_arity_rule_to_a_mixed_selection():
    """A bare chain selector is pulled but must not become an anchor."""
    feats = build_motif_guidance_feats(_array(), ["A/HIS/10/CA", "X"])
    assert feats["motif_idx"].tolist() == [1, 4, 5]
    # Only the 4-field selection anchors; the chain's two atoms are pulled along.
    assert feats["anchor_idx_in_motif"].tolist() == [0]


def test_bare_chain_only_group_anchors_on_everything():
    """With no single-atom selection there is nothing to fall back on but the group."""
    feats = build_motif_guidance_feats(_array(), ["X"])
    assert feats["motif_idx"].tolist() == [4, 5]
    assert feats["anchor_idx_in_motif"].tolist() == [0, 1]


def test_named_mapping_makes_one_group_per_key():
    feats = build_motif_guidance_feats(
        _array(), {"protein": ["A/HIS/10/CA", "A/GLU/11/CA"], "ligand": ["X"]}
    )
    assert feats["group_names"] == ["protein", "ligand"]
    assert feats["group_starts"].tolist() == [0, 2, 4]
    assert feats["motif_idx"].tolist() == [1, 3, 4, 5]


def test_explicit_anchor_is_a_subset_of_atoms():
    feats = build_motif_guidance_feats(
        _array(), [{"name": "m", "atoms": ["A"], "anchor": ["A/HIS/10/CA"]}]
    )
    assert feats["motif_idx"].tolist() == [0, 1, 2, 3]
    assert feats["anchor_idx_in_motif"].tolist() == [1]


def test_anchor_outside_atoms_raises():
    with pytest.raises(ValueError, match="outside 'atoms'"):
        build_motif_guidance_feats(
            _array(), [{"atoms": ["A/HIS/10/CA"], "anchor": ["X"]}]
        )


def test_unknown_group_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        build_motif_guidance_feats(_array(), [{"atoms": ["A"], "distance": 3.0}])


def test_missing_atoms_key_raises():
    with pytest.raises(ValueError, match="'atoms' is required"):
        build_motif_guidance_feats(_array(), [{"name": "m", "anchor": ["A"]}])


def test_selection_matching_nothing_raises():
    """The shared resolver already refuses an unmatched selection, so we inherit that."""
    with pytest.raises(ValueError, match="Could not resolve selection"):
        build_motif_guidance_feats(_array(), ["B/XXX/1/CA"])


def test_all_blank_selections_raise():
    """Blank strings are skipped by the resolver, which would otherwise yield an empty
    group and a degenerate Kabsch fit rather than an error."""
    with pytest.raises(ValueError, match="matched no atoms"):
        build_motif_guidance_feats(_array(), [""])


def test_an_atom_selected_twice_raises():
    """index_add_ sums per-atom pulls, so a shared atom would be over-corrected."""
    with pytest.raises(ValueError, match="more than once"):
        build_motif_guidance_feats(
            _array(), {"g0": ["A/HIS/10/CA"], "g1": ["A/HIS/10/CA"]}
        )


def test_targets_default_to_the_inputs_own_coordinates():
    arr = _array()
    feats = build_motif_guidance_feats(arr, ["A/HIS/10/N", "A/HIS/10/CA"])
    assert np.allclose(feats["target"].numpy(), arr.coord[[0, 1]])


def test_reference_array_supplies_the_targets():
    feats = build_motif_guidance_feats(
        _array(),
        ["A/HIS/10/N", "A/HIS/10/CA"],
        reference_atom_array=_array(shift=100.0),
    )
    assert np.allclose(feats["target"].numpy(), _array().coord[[0, 1]] + 100.0)


def test_pulled_atom_absent_from_the_reference_is_dropped():
    reference = _array(shift=100.0)[:5]  # drops chain X atom C2
    feats = build_motif_guidance_feats(
        _array(), ["A/HIS/10/CA", "X"], reference_atom_array=reference
    )
    # C2 is pulled-only, so it is dropped rather than fatal.
    assert feats["motif_idx"].tolist() == [1, 4]
    assert feats["anchor_idx_in_motif"].tolist() == [0]


def test_named_anchor_absent_from_the_reference_raises():
    """Narrowing an anchor the caller named by hand would silently change the fit."""
    reference = _array(shift=100.0)[:4]  # drops all of chain X
    with pytest.raises(ValueError, match="absent from motif_guidance_reference"):
        build_motif_guidance_feats(
            _array(),
            {"g": {"atoms": ["A", "X"], "anchor": ["X/LIG/99/C1"]}},
            reference_atom_array=reference,
        )


def test_default_anchor_absent_from_the_reference_is_dropped_not_raised():
    """A group that named no anchor gets "all of its atoms" -- all of the usable ones.

    A reference file routinely lacks parser-added hydrogens, and raising on those would
    make the common whole-chain selection unusable.
    """
    reference = _array(shift=100.0)[:5]  # keeps chain A and only C1 of chain X
    feats = build_motif_guidance_feats(_array(), ["X"], reference_atom_array=reference)
    assert feats["motif_idx"].tolist() == [4]  # C2 dropped with its missing counterpart
    assert feats["anchor_idx_in_motif"].tolist() == [0]


def test_a_group_with_no_counterpart_at_all_still_raises():
    reference = _array(shift=100.0)[:4]  # drops all of chain X
    with pytest.raises(ValueError, match="no selected atom was found in the reference"):
        build_motif_guidance_feats(_array(), ["X"], reference_atom_array=reference)


def test_duplicate_reference_atom_raises():
    reference = _array(shift=100.0)
    reference.res_id = np.array([10, 10, 10, 10, 99, 99])
    reference.atom_name = np.array(["N", "CA", "N", "CA", "C1", "C2"])
    with pytest.raises(ValueError, match="duplicate atom"):
        build_motif_guidance_feats(
            _array(), ["A/HIS/10/CA"], reference_atom_array=reference
        )
