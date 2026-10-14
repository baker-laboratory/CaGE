# This directory is a vendored, patched snapshot of AtomWorks

**Do not `pip install atomworks` for this repository.** Install this directory instead:

```bash
pip install -e "./lib/atomworks[ml]"   # first
pip install -e .                       # then the repository itself
```

| Item | Value |
| --- | --- |
| Upstream commit pinned | `15f4cabab64d928db306168ab68b25f01087cc21` |
| Local modifications | `src/atomworks/io/utils/selection.py` (selection parsing), `src/atomworks/io/utils/bonds.py` (formal charges with mixed hydrogen information), and `src/atomworks/ml/utils/misc.py` (one hardcoded default path removed) |
| Package version declared here | `2.1.1` (see "Version collision" below) |
| Public mirror of upstream | <https://github.com/RosettaCommons/atomworks> |
| License | BSD 3-Clause, Copyright (c) 2025 Institute for Protein Design, University of Washington — see `LICENSE.md` in this directory |

Copyright of this code belongs to IPD/UW; its contributors are listed in the AtomWorks/RF3
preprint (see the Citation section of `README.md` in this directory) and its release history is
in `CHANGELOG.md`.

## Why vendored instead of a PyPI dependency

The public release on PyPI (v2.2.1 at the time of export) does **not** expose several symbols
the RF3 design-condition code imports. Measured when the public release was substituted for
this directory:

- `atomworks.io.config.PrepareConfig` — missing. Imported by `rf3/utils/inference.py`; this is
  the entry point that turns a design-condition JSON into an `AtomArray`.
- `atomworks.ml.conditions.Level` — not re-exported. Imported by `rf3/conditions.py:4`, which
  is where C_HOT / C_ITYPE / C_HB / C_RAS are defined.
- `atomworks.ml.msa.finding.get_msa_dirs` — missing.
- `register_custom_residues_from_atom_array` — missing; used by the RF3 inference engine.

Result: `ImportError` on importing the `rf3` modules, i.e. the repository does not run at all
against the public release. Beyond import failures, the chemistry does not match either —
ligand featurization on the same input gave a different number of chiral centers
(489 with this snapshot vs. 465 with the public release), so results would not be comparable
even where imports happened to succeed.

**Exit condition.** When upstream publishes a release that both exposes these symbols and
contains the `selection.py` and `bonds.py` fixes below, this directory can be deleted and replaced by an
ordinary pinned dependency in the root `pyproject.toml`.

**Do not re-pin to a newer upstream commit without re-testing.** Later upstream commits move
`AtomSelectionStack` to `atomworks.io.utils.query` and change the bare range syntax, which
would break the example design-condition JSON files shipped in
`docs/releases/rf3/examples/design_conditions/`.

## The `selection.py` patch

Every design condition parses its atom selections through `AtomSelectionStack.from_query`
(hotspot, RASA, h-bond, interface type, distance template and relative template all go
through it), so a selection-parsing bug silently disables conditions. Two fixes, both in
`src/atomworks/io/utils/selection.py`; `git log -p` on this directory has the diffs.

**1. Multi-digit scalar `res_id` was misparsed as a range** (`_extract_field_options`). The
pattern `-?\d+-?\d+` matched a bare `"28"`, which then crashed on `split("-")`; the caller
caught the failure and skipped the selection with a warning, so `'A/*/28'` matched **0
atoms** instead of residue 28 and any condition label attached to it was dropped without
error. The separator dash is now required, so a single index falls through to the scalar
branch and `'5-10'` still expands to a range.

**2. `res_id = 0` acted as a wildcard** (`get_mask_from_atom_selection`). `res_id` is
normalized to `int` or `"*"` in `__init__`, so a truthiness test let the perfectly valid
residue id `0` skip the comparison and select every residue. The test is now against `"*"`.

`lib/atomworks/tests/io/utils/test_selection_query_regression.py`, added with this
snapshot, covers both: 5 passed with the patch, 5 failed without it, and
`models/rf3/tests/test_design_condition_inputs.py` goes red without it too. That test
cannot run against the public AtomWorks release at all — its `conftest` fixtures do not
exist there, which is independent confirmation that the two trees are not interchangeable.

## The `bonds.py` patch

`correct_formal_charges_for_specified_atoms` previously treated explicit hydrogens anywhere
in a structure as evidence that every residue had complete hydrogen information. In a
`components` input containing a protein without explicit hydrogens and a hydrogenated
ligand, this assigned `charge = -1` to peptide backbone nitrogens. The incorrect charges
propagated into model reference features (`ref_charge`) and exported structures.

Without an `nhyd` annotation, charge correction now only updates selected atoms in residues
that contain explicit hydrogens. Other residues retain their existing charges. The `nhyd`
path and charge correction within hydrogenated residues are unchanged. This carries over
the local development-tree fix without changing the pinned upstream commit.

`tests/io/utils/test_formal_charge_regression.py` covers mixed hydrogen information,
hydrogen isotopes, the update mask, implicit hydrogen counts, deprotonated oxygen, and a
protein-plus-ligand `components` input. Charges and model inputs for affected structures
change as a result of this fix.

## The `misc.py` patch

`_get_taxonomy_id_lookup_df` in `src/atomworks/ml/utils/misc.py` defaulted
`taxonomy_id_csv_path` to an absolute path on the authors' cluster, which is meaningless
outside it. The default is now `None`, resolved from `$PDB_CHAIN_TAXONOMY_CSV` and raising
with the SIFTS download link when unset.

Neither this function nor its caller `get_taxonomy_id` is reachable from any CaGE code path
(no call sites outside `misc.py` itself), so this patch changes no behaviour here — it only
keeps a real internal path out of a public tree. Upstream users who relied on the old
default need to set the environment variable or pass the path.

## What was removed from the upstream snapshot

Removed before export, because they carry internal infrastructure configuration, are large, or
are irrelevant to a released snapshot:

| Removed | Why |
| --- | --- |
| `.env` (a symlink into `.ipd/`) and the whole `.ipd/` directory | the target file held real internal configuration: cluster MSA and database directories, internal host paths and container bind mounts. `import foundry` auto-loads a `.env` found in the source tree, so this had to go. |
| `docs/` | upstream Sphinx documentation; read it upstream instead |
| `src/atomworks/pymol_plugin/` | unused by RF3 |
| `.github/` | upstream CI, tied to upstream infrastructure |
| `CLAUDE.md`, `Makefile`, `.project-root`, `.gitignore`, `.releaserc.js` | upstream development tooling and release automation |
| most of `tests/` | upstream's full suite needs data that is not distributed |

`tests/` was reduced to the `io` subset that runs without external data — the files present
are the authoritative list:

```
tests/__init__.py
tests/conftest.py
tests/io/__init__.py
tests/io/conftest.py
tests/io/utils/test_atom_array_plus.py
tests/io/utils/test_formal_charge_regression.py
tests/io/utils/test_selection_query_regression.py
tests/io/utils/test_selection_utils.py
```

`pytest tests/io` from inside this directory passes against this snapshot. Note that
`tests/conftest.py` and `src/atomworks/ml/` still contain upstream prose mentioning the
cluster the package was developed on; those are comments and docstrings only, and are the
expected hits of the sanitization scan in `PROVENANCE.md` §5.

## Version collision

`pyproject.toml` here declares `version = "2.1.1"`, the same version string as a public PyPI
release whose contents differ. Anything resolving dependencies by name and version can
therefore mistake one for the other.
`<TODO: decide whether to give this directory a PEP 440 local version
(e.g. 2.1.1+cage.15f4caba) and pin the root pyproject to match>`

## Upstream relationship

Development happens in an internal AtomWorks repository; `RosettaCommons/atomworks` is its
public mirror. This snapshot is a point-in-time copy of the internal tree, so individual
commits here cannot be looked up in the mirror by hash — use the pinned commit above as the
reference point, and `CHANGELOG.md` to locate the corresponding public release.
