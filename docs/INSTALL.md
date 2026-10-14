# Installation

CaGE is the RF3 inference and training stack with the five design conditions and
classifier-free guidance. It is one Python package (`cage-rf3`, importable as `foundry`
and `rf3`) plus a **vendored copy of AtomWorks** under `lib/atomworks`.

> [!IMPORTANT]
> The pip/uv commands below were not exercised end to end while preparing this release
> (the verification environment had no network access), so treat them as the intended
> route rather than a tested transcript. Everything *after* installation — imports, the
> test suites, featurization of every bundled example, and a full inference run — was
> verified in a CPU-only container built from this tree.

## Requirements

- Linux (x86-64) for GPU inference; macOS/Linux CPU works for parsing, featurization and
  the test suites.
- Python >= 3.11 (3.12 is what this release was developed and tested against).
- An NVIDIA GPU with recent CUDA drivers for any real prediction. A ~400-residue complex
  is comfortable on 24 GB; the atom-pair-distance condition atomizes the residues it
  selects, which increases the token count and the memory footprint.
- Git, if you want a meaningful package version: the version is derived from git tags
  (`hatch-vcs`), and a download without `.git` reports `0.0.0`.

## Install

### With `uv` (recommended)

`pyproject.toml` pins AtomWorks to the vendored copy for `uv`
(`[tool.uv.sources] atomworks = { path = "lib/atomworks", editable = true }`), so a single
command is enough:

```bash
git clone <TODO: public repository URL> cage && cd cage
uv python install 3.12
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[rf3]"
```

The `rf3` extra adds the Linux-only cuEquivariance CUDA op packages. Drop it
(`uv pip install -e .`) on CPU or macOS.

### With plain `pip`

`pip` does not read `[tool.uv.sources]`, and `cage-rf3` depends on `atomworks[ml]`.
**Install the vendored AtomWorks first**, otherwise pip resolves `atomworks` from PyPI and
you get a different copy than the one this release was verified against:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./lib/atomworks[ml]"
pip install -e .            # or: pip install -e ".[rf3]" on a Linux GPU box
```

### Check the installation

```bash
python -c "import foundry, rf3, atomworks; print('ok')"
pytest tests models/rf3/tests                  # 214 passed
(cd lib/atomworks && pytest tests/io)          # 82 passed
```

These are the two steps `.github/workflows/tests.yaml` runs on a plain `ubuntu-latest`
runner, which is the authoritative install recipe if the counts above have drifted.

The vendored AtomWorks ships only its `io` test subset, and it has its own
`conftest.py` / rootdir, so run it from `lib/atomworks`: `pytest` at the repository root
deliberately does not recurse into `lib/`.

## Model weights

```bash
foundry install run14_confhead      # the default: conditions + confidence head
foundry install run14               # structure-only weights (ckpt_path=run14)
```

Weights land in `~/.foundry/checkpoints` by default. See [WEIGHTS.md](WEIGHTS.md) for what
each checkpoint contains, which one to use, and the download record
(`<TODO: Zenodo DOI / record id, filled when the weights are deposited>`).

## Environment variables

`import foundry` loads a `.env` file from the working directory automatically (via
`environs`), so you can either export these or copy `.env.sample` to `.env` and fill it
in. Nothing below is required for a basic prediction with downloaded weights.

| Variable | Needed for | Notes |
|---|---|---|
| `FOUNDRY_CHECKPOINT_DIRS` | finding weights outside the default directory | Colon-separated list, searched before `~/.foundry/checkpoints`. The older name `FOUNDRY_CHECKPOINTS_DIR` is still read as a fallback. |
| `CCD_MIRROR_PATH` | optional | Local RCSB CCD mirror (`H/HEM/HEM.cif` layout); also the place to drop a custom CCD entry for a non-standard ligand. Without it the CCD bundled with biotite is used. |
| `PDB_MIRROR_PATH` | optional | Local PDB mirror (`a2/1a2b.cif.gz` layout), for pipelines that fetch structures by ID. Build either mirror with `atomworks pdb sync` / `atomworks ccd sync`. |
| `PROTEIN_MSA_DIRS` | optional | Directories searched for precomputed MSAs. MSAs can also be given per chain in the input JSON. |
| `HBPLUS_PATH` | **training and h-bond metrics only** | Path to the `hbplus` binary, a third-party program not distributed here; obtain it from its authors. Inference with `hbond_selection` does **not** need it. Unset, the condition is skipped at training time and `HbondMetrics` reports nothing, each with a one-shot warning. |
| `X3DNA` | **training / design validation only** | Path to the X3DNA installation, a third-party program not distributed here, used by the DNA-padding transform. Inference skips that transform entirely and logs that it did. |

## GPU notes

- **cuEquivariance kernels are disabled in this release.** `SHOULD_USE_CUEQUIVARIANCE` in
  `src/foundry/__init__.py` is initialized to `False` and never assigned anywhere else, so
  the standard PyTorch paths are used even when the CUDA packages are installed and even
  though the import succeeds (the log line claiming the kernels "will be used" is left over
  from the upstream tree). This costs some speed and changes nothing about the results. On
  CPU the kernels are unavailable in any case.
- CPU-only runs work but are slow; they are how the bundled examples were checked for
  featurization correctness.

## First prediction

```bash
rf3 fold inputs='models/rf3/tests/data/5vht_from_json.json'
```

Then try a design condition:

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json'
```

Run `rf3 fold` **from the repository root**: the bundled example JSONs reference their
component files with repository-relative paths, resolved against the working directory.

Next: [DESIGN_CONDITIONS.md](DESIGN_CONDITIONS.md) for the conditions and guidance,
[the example index](releases/rf3/examples/design_conditions/README.md) for runnable
inputs, and [models/rf3/README.md](../models/rf3/README.md) for the full inference API.
