# Vendored AtomWorks

CaGE includes a patched AtomWorks snapshot under `lib/atomworks`. When installing
with pip, install this copy before CaGE:

```bash
pip install -e "./lib/atomworks[ml]"
pip install -e .
```

With uv, the root `pyproject.toml` selects this directory automatically.

| Item | Value |
| --- | --- |
| Upstream commit | `15f4cabab64d928db306168ab68b25f01087cc21` |
| Package version | `2.1.1` |
| Public upstream mirror | <https://github.com/RosettaCommons/atomworks> |
| License | BSD 3-Clause, Copyright (c) 2025 Institute for Protein Design, University of Washington; see [LICENSE.md](LICENSE.md). |

See [README.md](README.md) for the AtomWorks citation and
[CHANGELOG.md](CHANGELOG.md) for its release history.

## Dependencies on this snapshot

CaGE uses the following interfaces from this copy of AtomWorks:

- `atomworks.io.config.PrepareConfig`
- `atomworks.ml.conditions.Level`
- `atomworks.ml.msa.finding.get_msa_dirs`
- `register_custom_residues_from_atom_array`
- `AtomSelectionStack` and the selection syntax used by the bundled examples

The PyPI package has different interfaces and ligand featurization. The version
string `2.1.1` also appears on PyPI, so use the local path above to select this copy.

## Local patches

### Selection parsing

`src/atomworks/io/utils/selection.py` includes two changes:

- `_extract_field_options` requires a separator dash in residue ranges. A
  multi-digit residue ID such as `28` is parsed as a scalar; `5-10` is a range.
- `get_mask_from_atom_selection` distinguishes residue ID `0` from the wildcard
  `"*"`, so selecting residue 0 no longer selects every residue.

The regression cases are in `tests/io/utils/test_selection_query_regression.py`.

### Formal charges

In `src/atomworks/io/utils/bonds.py`,
`correct_formal_charges_for_specified_atoms` handles structures that combine
residues with and without explicit hydrogens. Without an `nhyd` annotation, it
updates selected atoms only in residues containing explicit hydrogens. Other
residues retain their existing charges. This prevents peptide backbone nitrogens
from receiving a charge of -1 when a protein is supplied alongside a hydrogenated
ligand.

`tests/io/utils/test_formal_charge_regression.py` covers mixed hydrogen
information, hydrogen isotopes, update masks, implicit hydrogen counts,
deprotonated oxygen and protein-ligand inputs.

### Taxonomy lookup

`_get_taxonomy_id_lookup_df` in `src/atomworks/ml/utils/misc.py` reads its default
CSV path from `PDB_CHAIN_TAXONOMY_CSV`. Set that variable or pass
`taxonomy_id_csv_path` explicitly when using taxonomy lookup.

## Included files

This copy includes the AtomWorks library, CLI and the `tests/io` subset.
Upstream documentation, the PyMOL plugin, CI, development tooling, internal
configuration and tests requiring external datasets are omitted.

Run the included tests from this directory:

```bash
python -m pytest tests/io -q
```

The pinned commit identifies the internal upstream snapshot. The public mirror's
release history is recorded in [CHANGELOG.md](CHANGELOG.md).
