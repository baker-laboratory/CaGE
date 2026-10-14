"""Charge correction with mixed explicit and missing hydrogen information."""

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure.io import save_structure

from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from atomworks.io.template import get_empty_ccd_template
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.utils.bonds import correct_formal_charges_for_specified_atoms


@pytest.fixture
def mixed_hydrogens() -> struc.AtomArray:
    """A peptide N without its H, alongside an explicit ammonium ion."""
    atoms = struc.AtomArray(8)
    atoms.chain_id = ["A"] * 3 + ["B"] * 5
    atoms.res_id = [1, 2, 2] + [2] * 5
    atoms.res_name = ["GLY"] * 3 + ["NH4"] * 5
    atoms.atom_name = ["C", "N", "CA", "N", "H1", "H2", "H3", "H4"]
    atoms.element = ["C", "N", "C", "N", "H", "H", "H", "H"]
    atoms.set_annotation("charge", np.zeros(len(atoms), dtype=int))
    atoms.bonds = struc.BondList(
        len(atoms),
        np.array([(0, 1, 1), (1, 2, 1), (3, 4, 1), (3, 5, 1), (3, 6, 1), (3, 7, 1)]),
    )
    return atoms


@pytest.mark.parametrize("hydrogen", HYDROGEN_LIKE_SYMBOLS)
@pytest.mark.parametrize("same_chain", [False, True])
def test_other_residues_hydrogens_do_not_change_peptide_charge(mixed_hydrogens, hydrogen, same_chain):
    atoms = mixed_hydrogens
    atoms.element[4:] = hydrogen
    if same_chain:
        atoms.chain_id[3:] = "A"
        atoms.res_id[3:] = 3
    mask = atoms.element == "N"
    original_mask = mask.copy()

    result = correct_formal_charges_for_specified_atoms(atoms, mask)

    assert result is atoms
    assert atoms.charge[1] == 0
    assert atoms.charge[3] == 1
    np.testing.assert_array_equal(mask, original_mask)


def test_only_selected_atoms_are_corrected(mixed_hydrogens):
    atoms = mixed_hydrogens
    mask = np.zeros(len(atoms), dtype=bool)
    mask[1] = True

    correct_formal_charges_for_specified_atoms(atoms, mask)

    np.testing.assert_array_equal(atoms.charge, 0)
    np.testing.assert_array_equal(np.flatnonzero(mask), [1])


def test_no_hydrogen_information_preserves_existing_charges(mixed_hydrogens, caplog):
    atoms = mixed_hydrogens[:3]
    atoms.charge[1] = 1
    original = atoms.charge.copy()

    correct_formal_charges_for_specified_atoms(atoms, atoms.element == "N")

    np.testing.assert_array_equal(atoms.charge, original)
    assert "Neither hydrogens nor nhyd annotation present" in caplog.text


@pytest.mark.parametrize("nhyd, expected_charge", [(0, -1), (1, 0)])
def test_implicit_hydrogen_counts_still_determine_charge(mixed_hydrogens, nhyd, expected_charge):
    atoms = mixed_hydrogens[:3]
    atoms.set_annotation("nhyd", np.array([0, nhyd, 0]))
    atoms.charge[1] = 1

    correct_formal_charges_for_specified_atoms(atoms, atoms.element == "N")

    assert atoms.charge[1] == expected_charge


def test_deprotonated_oxygen_in_hydrogenated_residue_is_still_corrected():
    atoms = get_empty_ccd_template("ALA", res_id=1, hydrogen_policy="keep")
    atoms = atoms[atoms.atom_name != "HXT"]
    mask = atoms.atom_name == "OXT"
    assert mask.sum() == 1
    assert atoms.charge[mask].item() == 0

    correct_formal_charges_for_specified_atoms(atoms, mask)

    assert atoms.charge[mask].item() == -1


def test_empty_update_mask_leaves_charges_unchanged(mixed_hydrogens):
    atoms = mixed_hydrogens
    original = atoms.charge.copy()

    correct_formal_charges_for_specified_atoms(atoms, np.zeros(len(atoms), dtype=bool))

    np.testing.assert_array_equal(atoms.charge, original)


@pytest.mark.parametrize("from_file", [False, True])
def test_components_input_preserves_protein_and_ligand_charges(cleanup_registry, tmp_path, from_file):
    protein = {"seq": "GAKL", "chain_id": "A"}
    reference = components_to_atom_array([protein])
    if from_file:
        # PDB input omits hydrogens and the nhyd annotation.
        reference.coord[:] = np.arange(len(reference))[:, None] * 0.1
        reference.b_factor[:] = 0
        path = tmp_path / "protein.pdb"
        save_structure(path, reference)
        protein = {"path": str(path)}
    mixed = components_to_atom_array([protein, {"ccd_code": "NH4", "chain_id": "B"}])
    protein_atoms = mixed[mixed.chain_id == "A"]

    assert (protein_atoms.atom_name == "N").sum() == 4
    np.testing.assert_array_equal(protein_atoms.atom_name, reference.atom_name)
    np.testing.assert_array_equal(protein_atoms.res_id, reference.res_id)
    np.testing.assert_array_equal(protein_atoms.charge, reference.charge)
    np.testing.assert_array_equal(protein_atoms.charge[protein_atoms.atom_name == "N"], 0)
    ligand_n = (mixed.chain_id == "B") & (mixed.element == "N")
    assert ligand_n.sum() == 1
    assert mixed.charge[ligand_n].item() == 1
