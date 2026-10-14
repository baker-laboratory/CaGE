# Released weights

CaGE releases two checkpoints. They are **not** part of this repository and are **not**
covered by `LICENSE.md`; they are hosted on Zenodo and registered by name in
`src/foundry/inference_engines/checkpoint_registry.py`.

| Registered name | What it is | Trained by |
| --- | --- | --- |
| `run14` | Structure model. 402-d atom-1d input including the four 1-body design-condition features (`mask_hotspot_1_atom`, `condition_itype_1_atom`, `condition_hbond_1_atom`, `condition_rasa_1_atom`) and the 2-body `Conditioning2DProcessor` (C_DIS). | Frank DiMaio |
| `run14_confhead` | The same structure model plus the AF3-style confidence head, used for per-stream confidence and stream ranking. **The default `ckpt_path`** of the `rf3` inference engine, and a superset of `run14`. | Han Altae-Tran (confidence head), on Frank DiMaio's structure weights |

Both are acknowledged in `README.md` under "References and sources".

## Identifiers — to be filled at upload time

The two released files are **inference-only exports** of the training checkpoints, produced by
[`models/rf3/scripts/export_release_checkpoint.py`](../models/rf3/scripts/export_release_checkpoint.py):

| Released as | Exported from | Export |
| --- | --- | --- |
| `run14.ckpt` | `run14_ep649_dc3fff46.ckpt` (epoch 649 / step 63115) | 5.30 GiB -> 1.33 GiB; 103 internal paths scrubbed |
| `run14_confhead.ckpt` | `run14_confhead_ft260803_ep0020.ckpt` (confidence-head finetune, epoch 20 / step 10000) | 2.80 GiB -> 1.37 GiB; 57 internal paths scrubbed |

Verified on the source files: both declare `c_atom_1d_features: 402`, so all five design
conditions are consumable at inference. The confidence-head file adds the 362 tensors of
`confidence_head`, 28 of `conditioning_2d_processor` and 4 of `masked_token_recovery_head` on
top of the structure weights it was finetuned from.

| Field | `run14` | `run14_confhead` |
| --- | --- | --- |
| Zenodo DOI | `<TODO: Zenodo DOI>` | `<TODO: Zenodo DOI>` |
| Zenodo record / file URL | `<TODO: Zenodo record id>` | `<TODO: Zenodo record id>` |
| sha256 | `8aa680d8ebdb553f3d996ad4dd10292a371e0ef1c5b3c0a4acb760d6a9a267cf` | `3c02431345f65580fbc60e10e5d4dae2c8d63988cd605e33db6392c9f8b88772` |
| Size, as released | 1,423,455,749 B (1.33 GiB) | 1,474,868,299 B (1.37 GiB) |
| File name | `run14.ckpt` | `run14_confhead.ckpt` |

> The sha256 and size rows describe the exported files as the script above produces them, and
> are already written into `REGISTERED_CHECKPOINTS`; **re-exporting with different flags
> changes both**. The DOI and URL rows are filled by the user after uploading, and the same
> values must replace the `<RECORD_ID>` placeholders in
> `src/foundry/inference_engines/checkpoint_registry.py`.

## Installing and using them

```bash
# list what is registered
foundry list-available

# download into ~/.foundry/checkpoints (or a directory you choose)
foundry install run14_confhead
foundry install run14 run14_confhead --checkpoint-dir /path/to/checkpoints
```

A registered **name** is accepted wherever a checkpoint path is: the default inference config
already uses one (`ckpt_path: run14_confhead` in `models/rf3/configs/inference_engine/rf3.yaml`), so

```bash
rf3 fold ... ckpt_path=run14                 # structure only
rf3 fold ... ckpt_path=run14_confhead        # with the confidence head
```

both resolve through the registry. Resolution searches `~/.foundry/checkpoints` plus any
colon-separated directories in `FOUNDRY_CHECKPOINT_DIRS` (legacy: `FOUNDRY_CHECKPOINTS_DIR`).
The string is matched against the registry first; a string that contains no `.` is also
assumed to be a registered name and fails loudly if it is not one
(`src/foundry/inference_engines/base.py:99`).

## How the released files differ from the original training checkpoints

`models/rf3/scripts/export_release_checkpoint.py` makes an inference-only export in three
steps, and the numbers above are what they produce:

1. **Training state is dropped** — only `model`, `train_cfg`, `current_epoch` and
   `global_step` survive. For `run14` the optimizer state alone was half the file.
2. **The embedded `train_cfg` is scrubbed** — every absolute path under a cluster-internal
   root becomes `<redacted-training-path>` (103 strings in `run14`, 57 in
   `run14_confhead`): dataset caches, MSA directories, output and logger directories, and
   the parent-checkpoint paths. Paths are replaced by value and no key is removed, because
   inference rebuilds the network from `train_cfg.model.net` and the transform pipeline
   from the first `train_cfg.datasets.val` entry
   (`src/foundry/inference_engines/base.py:184`, `:340`). Removing `model.net.**` — the
   402-d `atom_1d_features` list, the `conditioning_2d_processor`, the confidence-loss
   configuration — would make the checkpoint unloadable, and that feature list is also the
   evidence that these weights were trained with the five conditions.
3. **The EMA wrapper is collapsed** — the training checkpoints carry an online copy and an
   EMA shadow of every tensor, and `EMA.forward` dispatches to the shadow in eval, so the
   shadow is the weight set inference has always used. It is promoted to the top level
   (`shadow.X` → `X`), the online half is dropped, and `train_cfg.model.ema` is set to
   null. `--keep-ema` keeps the training layout instead.

Before writing, the export rebuilds the network from the scrubbed config and refuses unless
every model parameter has a same-shaped counterpart in the checkpoint, with nothing left
over. This is not a formality: `load_weights_with_policies` answers a missing or mis-shaped
parameter by falling back to `REINIT` — a freshly randomized tensor — behind one WARNING
that this repository's logging configuration suppresses, so a renaming mistake in step 3
would otherwise ship as a quietly half-random model that still loads and still predicts.

Re-running the export on the same source file reproduces the same bytes and the same
sha256, so the hashes above can be re-derived rather than taken on trust.

### Reading a released checkpoint

The file carries one pickled `omegaconf` config beside the tensors, so it has to be loaded
with `weights_only=False` — verify the sha256 first. Unpickling needs only `torch` and
`omegaconf`; no module from this repository is referenced.

```python
import torch
ckpt = torch.load("run14.ckpt", map_location="cpu", weights_only=False)
print(ckpt.keys())                 # model, train_cfg, current_epoch, global_step
print(ckpt["train_cfg"].model.net) # what the weights were trained with
```

## Training-recipe mapping

| Weight | Recipe in this repository |
| --- | --- |
| `run14` | `models/rf3/configs/experiment/rf3_run14_invfold.yaml` (its parent recipe is `models/rf3/configs/experiment/rf3_run14.yaml`) |
| `run14_confhead` | **not in this repository** — see the warning below |

`rf3_run14_invfold.yaml` is `rf3_run14` plus an inverse-folding training subtask and
validation set; it is the recipe whose settings match the embedded `train_cfg` of the released
structure weights (`provide_diffusion_ref_copy: true`, the `ref_pos_diff` atom-1d feature set
including the four condition features, and `alpha_hotspot_lddt` / `alpha_hotspot_diffusion` /
`alpha_hotspot_distogram` all overridden to `0.0`).

**The recipe alone will not reproduce the released weights from a random initialization.** The
released structure model was obtained by **continuing training from an earlier run14 checkpoint
(epoch 0584)**, on **8 nodes × 12 GPUs with `grad_accum: 2`**. Reproducing the paper's weights
therefore requires that starting checkpoint and that hardware configuration, neither of which
is derivable from the configuration files. Treat the recipe as documentation of *how* the model
was trained, not as a one-command reproduction.

> Note also that RF3 inference is **not bit-reproducible** even with a fixed seed. Any
> regression check against these weights must use a tolerance (e.g. RMSD), never byte
> equality.

## Two declarations about `run14_confhead`

1. **The confidence head's training recipe is not in this repository.** The configuration
   group that trained it (`conf_*` experiment recipes and their dataset/trainer closure) was
   excluded from the public snapshot because it carries internal cluster paths. The only
   authoritative record of that configuration is the sanitized `train_cfg` embedded in the
   released checkpoint itself — which is also where the 402-d condition-feature list must be
   read from.
   `<TODO: decide whether to archive the sanitized confidence-head recipe alongside the weights>`

2. **The confidence head was finetuned with all five design conditions switched off.** The
   embedded `train_cfg` of `run14_confhead` contains **no `design_tasks` block at all**
   (no `design_tasks`, `p_add_rasa_condition`, `p_add_hbond_condition` or
   `annotate_interface_type` keys), and its training-side `p_unconditional` is `0.99`. The
   structure weights it builds on were trained with the conditions; the confidence head on top
   of them was not. If you report confidence or ranking numbers for *conditioned* runs, say so.

## Citing the weights

Cite the CaGE manuscript (`CITATION.cff`) and the Zenodo DOI of the specific checkpoint you
used. Record provenance of the code side in `PROVENANCE.md`.
