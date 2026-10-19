# Installation

CaGE is the RF3 inference and training stack with the five design conditions and
classifier-free guidance. It is one Python package (`cage-rf3`, importable as `foundry`
and `rf3`) plus a **vendored copy of AtomWorks** under `lib/atomworks`.

## Requirements

- Linux (x86-64) for GPU inference; macOS/Linux CPU works for parsing, featurization and
  the test suites.
- Python >= 3.11; the commands below use Python 3.12.
- An NVIDIA GPU with recent CUDA drivers for any real prediction. A ~400-residue complex
  is comfortable on 24 GB; the atom-pair-distance condition atomizes the residues it
  selects, which increases the token count and the memory footprint.
- Git for package versioning: the version is derived from git tags
  (`hatch-vcs`), and a download without `.git` reports `0.0.0`.

## Install

### With `uv` (recommended)

`pyproject.toml` pins AtomWorks to the vendored copy for `uv`
(`[tool.uv.sources] atomworks = { path = "lib/atomworks", editable = true }`), so a single
command is enough:

```bash
git clone https://github.com/baker-laboratory/CaGE.git cage && cd cage
uv python install 3.12
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[rf3]"
```

The `rf3` extra adds the Linux-only cuEquivariance CUDA op packages. Drop it
(`uv pip install -e .`) on CPU or macOS.

### With plain `pip`

`pip` does not read `[tool.uv.sources]`. Install the vendored AtomWorks before
CaGE so that its `atomworks[ml]` dependency uses the bundled copy:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./lib/atomworks[ml]"
pip install -e .            # or: pip install -e ".[rf3]" on a Linux GPU box
```

### Check the installation

After installing the package, install the minimal `test` extra in the same
environment. It includes `pytest` and `pytest-dotenv`, which supports the vendored
AtomWorks `env_files` configuration; the full `dev` extra is not required.

```bash
uv pip install -e ".[test]"
# With plain pip, after installing vendored AtomWorks as above:
# python -m pip install -e ".[test]"
```

For a new GPU installation with tests, combine the extras as `.[rf3,test]`.
Then run both suites using that environment's Python:

```bash
python -c "import foundry, rf3, atomworks; print('ok')"
python -m pytest tests models/rf3/tests -q
(cd lib/atomworks && python -m pytest tests/io -q)
```

These are the two suites `.github/workflows/tests.yaml` runs on a plain
`ubuntu-latest` runner after installing `.[test]`.

The vendored AtomWorks ships only its `io` test subset, and it has its own
`conftest.py` / rootdir, so run it from `lib/atomworks`: `python -m pytest` at the
repository root deliberately does not recurse into `lib/`.

## Model weights

Download the checkpoints from [Zenodo](https://doi.org/10.5281/zenodo.23460926)
using the CLI:

```bash
foundry install run14_confhead      # the default: conditions + confidence head
foundry install run14               # structure-only weights (ckpt_path=run14)
```

Downloaded weights are saved in `~/.foundry/checkpoints` by default. To use a
checkpoint stored elsewhere, pass `ckpt_path=/path/to/checkpoint.ckpt`.
See [WEIGHTS.md](WEIGHTS.md) for direct download links and checksums.

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

- Inference uses the standard PyTorch kernels. cuEquivariance kernels are disabled
  by `SHOULD_USE_CUEQUIVARIANCE = False` in `src/foundry/__init__.py`.
- CPU inference is supported but slower than GPU inference.

## First prediction

```bash
rf3 fold inputs='models/rf3/tests/data/5vht_from_json.json' out_dir=./predictions
```

Then try a design condition:

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json' out_dir=./predictions
```

Run `rf3 fold` **from the repository root**: the bundled example JSONs reference their
component files with repository-relative paths, resolved against the working directory.

Next: [DESIGN_CONDITIONS.md](DESIGN_CONDITIONS.md) for the conditions and guidance,
[the example index](releases/rf3/examples/design_conditions/README.md) for runnable
inputs, and [models/rf3/README.md](../models/rf3/README.md) for the full inference API.
