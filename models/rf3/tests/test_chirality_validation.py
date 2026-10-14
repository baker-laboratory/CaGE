"""Tests for ValidateProteinChirality transform."""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from rf3.data.extra_xforms import ValidateProteinChirality

_PDB_DIR = Path(__file__).parent / "data" / "pdb"


def cached_parse(pdb_id: str, **kwargs) -> dict:
    """Parse a bundled test structure so the test runs without a PDB mirror."""
    data = parse(_PDB_DIR / f"{pdb_id}.cif.gz", config=ParseConfig(**kwargs))
    if "atom_array" not in data:
        data["atom_array"] = next(iter(data["assemblies"].values()))[0]
    return data


@pytest.mark.parametrize("pdb_id", ["1mch", "1y0y"])
def test_validate_chirality(pdb_id: str):
    """Test that ValidateProteinChirality correctly identifies and handles chirality issues."""
    data = cached_parse(pdb_id, hydrogen_policy="remove")
    data["example_id"] = pdb_id
    aa_before = data["atom_array"].copy()
    transform = ValidateProteinChirality(
        swap_pdb_codes=["1mch"]
    )  # Only swap for 1mch, unoccupy for 2z51
    result = transform(data)
    aa_after = result["atom_array"]

    if pdb_id == "1mch":
        # Ensure that C3PHE and C5PRO have been swapped to the D residues
        c3f_mask = (aa_after.res_id == 3) & (aa_after.chain_id == "C")
        assert np.all(
            aa_after.res_name[c3f_mask] == "DPN"
        ), "Expected residue 3 in chain C to have been changed PHE -> DPN"
        assert np.all(
            aa_after.occupancy == aa_before.occupancy
        ), "Expected occupancies to be unchanged for whitelisted pdb 1mch"
        c5p_mask = (aa_after.res_id == 5) & (aa_after.chain_id == "C")
        assert np.all(
            aa_after.res_name[c5p_mask] == "DPR"
        ), "Expected residue 5 in chain C to have been changed PRO -> DPR"
        assert np.all(
            aa_after.occupancy == aa_before.occupancy
        ), "Expected occupancies to be unchanged for whitelisted pdb 1mch"

    elif pdb_id == "1y0y":
        # Ensure B3VAL has occupancy zeroed but res_name unchanged
        b3v_mask = (aa_after.res_id == 3) & (aa_after.chain_id == "B")
        assert np.all(
            aa_after.res_name == aa_before.res_name
        ), "Expected residue names to be unchanged for non-whitelisted pdb 2z51"
        assert np.all(
            aa_after.occupancy[b3v_mask] == 0
        ), "Expected occupancy to be set to 0 for residue 3 in chain B of 2z51 due to chirality issue"
        assert np.all(
            aa_after.occupancy[~b3v_mask] == aa_before.occupancy[~b3v_mask]
        ), "Expected occupancies to be unchanged for residues other than residue 3 in chain B of 2z51"
