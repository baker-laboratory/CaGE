# Model weights

Download the CaGE checkpoints from [Zenodo](https://doi.org/10.5281/zenodo.23460926)
or install them with the `foundry` CLI:

```bash
foundry install run14_confhead
```

`run14_confhead` is the default checkpoint for inference. Use `run14` for structure
prediction without confidence scores.

| Checkpoint | Contents | Trained by |
| --- | --- | --- |
| `run14` | Structure model with all five design conditions. | Frank DiMaio |
| `run14_confhead` | Structure model with an AF3-style confidence head for per-stream confidence and ranking. | Han Altae-Tran (confidence head), on Frank DiMaio's structure weights |

## Downloads and checksums

Both checkpoints share DOI [10.5281/zenodo.23460926](https://doi.org/10.5281/zenodo.23460926).

| File | Size | SHA256 |
| --- | --- | --- |
| [run14.ckpt](https://zenodo.org/records/23460926/files/run14.ckpt?download=1) | 1,423,455,749 B (1.33 GiB) | `8aa680d8ebdb553f3d996ad4dd10292a371e0ef1c5b3c0a4acb760d6a9a267cf` |
| [run14_confhead.ckpt](https://zenodo.org/records/23460926/files/run14_confhead.ckpt?download=1) | 1,474,868,299 B (1.37 GiB) | `3c02431345f65580fbc60e10e5d4dae2c8d63988cd605e33db6392c9f8b88772` |

For manual downloads, save [SHA256SUMS](https://zenodo.org/records/23460926/files/SHA256SUMS?download=1)
beside both checkpoint files, then check them with:

```bash
sha256sum -c SHA256SUMS
```

## Installing and selecting a checkpoint

```bash
# List the registered checkpoints
foundry list-available

# Install the default checkpoint into ~/.foundry/checkpoints
foundry install run14_confhead

# Install both checkpoints into a directory of your choice
foundry install run14 run14_confhead --checkpoint-dir /path/to/checkpoints
```

Select a registered checkpoint by name:

```bash
rf3 fold inputs='models/rf3/tests/data/5vht_from_json.json' out_dir=./predictions ckpt_path=run14
rf3 fold inputs='models/rf3/tests/data/5vht_from_json.json' out_dir=./predictions ckpt_path=run14_confhead
```

You can also pass a local file with `ckpt_path=/path/to/run14_confhead.ckpt`.
Name-based lookup searches the colon-separated directories in
`FOUNDRY_CHECKPOINT_DIRS`, followed by `~/.foundry/checkpoints`.
`FOUNDRY_CHECKPOINTS_DIR` is accepted as a legacy environment-variable name.

## Checkpoint contents

The checkpoints contain inference weights and the model configuration. Both use
402-dimensional atom features and support the five CaGE design conditions.
`run14_confhead` also supports `confidence_streams` and `ranking_stream` for
per-stream scoring.

The [export script](../models/rf3/scripts/export_release_checkpoint.py) removes
optimizer and scheduler state, replaces internal paths in the configuration, and
retains the EMA weights used for inference. It checks parameter names and shapes
against the model before writing the checkpoint.

To inspect the configuration:

```python
import torch

ckpt = torch.load("run14.ckpt", map_location="cpu", weights_only=False)
print(ckpt["train_cfg"].model.net)
```

`weights_only=False` is needed to load the embedded OmegaConf configuration.

## Training configuration

| Checkpoint | Configuration |
| --- | --- |
| `run14` | [`rf3_run14_invfold.yaml`](../models/rf3/configs/experiment/rf3_run14_invfold.yaml), derived from [`rf3_run14.yaml`](../models/rf3/configs/experiment/rf3_run14.yaml). |
| `run14_confhead` | The checkpoint's embedded `train_cfg`. |

The structure model was trained with the five design conditions. The confidence
head was finetuned with those conditions disabled (`p_unconditional: 0.99`).

## Citation

Cite the [CaGE manuscript](../CITATION.cff) and the weight DOI
[10.5281/zenodo.23460926](https://doi.org/10.5281/zenodo.23460926), and specify
which checkpoint you used.
