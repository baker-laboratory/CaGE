# Design Conditions and Classifier-Free Guidance

CaGE extends RF3 with **five atom-level design conditions** that are injected during
training and can be requested at inference time, plus **per-axis classifier-free guidance
(CFG)** over those conditions.

This page is the reference for all of them: the JSON field, the CLI form, the legal
values, the feature the model actually sees, how often the condition was seen in
training, the metric that scores it, a runnable example, and the semantics that are easy
to get wrong.

All commands on this page are run **from the repository root** — the bundled example
JSONs reference their component files with repository-relative paths (`docs/releases/rf3/...`),
and those paths are resolved against the current working directory, not against the JSON.

---

## 1. The five CaGE design conditions

| Condition | JSON / CLI field | Value | Feature seen by the model | Training rate |
|---|---|---|---|---|
| Interface hotspots | `hotspot_selection` | selection, or list of selections | `mask_hotspot_1_atom`, `Bool[L]` | 15 % |
| Interface type | `interface_type` | `PPI` / `PLI` / `PNA`, or `{selection: label}` | `condition_itype_1_atom`, `Int[L, 3]` | 15 % |
| Relative solvent accessibility | `rasa_selection` | selection list (auto-binned), or `{selection: label}` | `condition_rasa_1_atom`, `Int[L, 3]` | 2.4 % |
| H-bond donor / acceptor identity | `hbond_selection` | `{selection: label}` only | `condition_hbond_1_atom`, `Int[L, 3]` | 0.4 % |
| Atom-pair distances | `distance_template_selection` | flat list, or groups with `distance` | `condition_distance_2_atom`, `Float[L, L]` | 4 % |

Field names, shapes and dtypes are those declared in
`models/rf3/src/rf3/data/schema/rf3.py:45-66`. The three enum conditions are one-hot
encoded with the "unset" level dropped (hence `Int[L, 3]`), and hotspots are a single
boolean channel; all four 1-body conditions are concatenated into the per-atom input of
the released weights. The 2-body distance condition is injected by `Conditioning2DProcessor`
in **two** places — the trunk pair representation
(`models/rf3/src/rf3/model/layers/embedders/feature_init.py:267`) and the diffusion
atom-attention encoder (`models/rf3/src/rf3/model/layers/diffusion/encoder.py:147`).

### Training rates: do not read them off the config surface

The rates above are the effective per-example rates of the released recipe
(`models/rf3/configs/design_tasks/all.yaml`, whose enabled tasks sum to 100, so each
weight reads directly as a percentage):

- `ppi_hotspot` has `frequency: 15.0` and `annotate_interface_type: true`, so **hotspot
  and interface type both occur at 15 %**, and interface type never occurs without
  hotspots — it is written only onto the sampled hotspot atoms.
- The four atom-pair-distance tasks have `frequency: 1.0` each, so **C_DIS is 4 %**.
- RASA and h-bond are *modifiers* of those distance tasks:
  `0.6 x 4 % = 2.4 %` for RASA and `(0.0 + 0.0 + 0.2 + 0.2) x 1 % = 0.4 %` for h-bond.

> [!WARNING]
> `ppi_hotspot` also carries `rasa_kwargs` / `hbond_kwargs` in its
> `global_design_task_kwargs`, but **`PPIHotspot.forward` never calls
> `apply_design_task_modifiers`** — the only reader of those kwargs. They are inert, kept
> only so the config still matches the recipe the released weights were trained with.
> Counting them (e.g. as `0.6 x (15 + 4) %`) overestimates the RASA rate by ~5x.

### Conditions, not constraints

Every one of the five is a *condition*, not a constraint: the model was trained to
respect it, and nothing at inference time enforces it. The released weights also apply
**no hotspot-weighted loss** — the `alpha_hotspot_*` terms are overridden to `0.0` in the
released recipe (`models/rf3/configs/experiment/rf3_run14_invfold.yaml`), so hotspots act
purely as an input condition. Use the metric listed with each condition below to check adherence.

---

## 2. Interface hotspots (`hotspot_selection`)

Marks atoms that the designed partner is expected to contact.

- **JSON**: `"hotspot_selection": ["A/*/88", "A/*/90", "A/*/164", "A/*/167"]`
- **CLI**: `hotspot_selection="[A/*/88, A/*/90]"`
- **Value**: one selection string or a list of them, in AtomWorks `AtomSelection` syntax
  (`CHAIN/RES_NAME/RES_ID/ATOM_NAME/TRANSFORM_ID`, see
  [models/rf3/README.md](../models/rf3/README.md#design-conditions-cage)).
- **Feature**: `mask_hotspot_1_atom`, `Bool[L]` — a boolean mask condition (`is_mask = True`
  on `Hotspot` in `models/rf3/src/rf3/conditions.py`): the mask marks which atoms carry
  the condition. The number of selected atoms is reported as "Selected N hotspot atoms",
  but only under `rf3 fold -v` (see §6).
- **Training**: `ppi_hotspot` (15 %), `PPIHotspot` with `max_hotspots_frac_to_provide: 0.1`
  and `hotspot_max_distance: 4.5` Å. Hotspots are sampled on the **target** side of an
  interface, i.e. on the chain that is given, while the partner is the one being
  designed. Put the hotspots on the chain you template, not on the chain you want
  generated.
- **Metric**: `PPIMetrics` → `fraction_hotspots_contacted` (4.5 Å cutoff), in
  `models/rf3/src/rf3/metrics/design_metrics.py`. `InterfaceHitsByType`
  (`models/rf3/src/rf3/metrics/interface_hits.py`) additionally reports pocket/interface
  hits; note its `type` field is the *dataset's* interface classification, not the
  `C_ITYPE` condition.
- **Example**: [`docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json`](releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json)
  — 63 hotspot atoms on the templated antigen chain A of 7XLI.

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json'
```

---

## 3. Interface type (`interface_type`)

Labels *what kind* of interface the hotspots belong to.

- **JSON, scalar**: `"interface_type": "PLI"`
- **JSON, dict**: `"interface_type": {"A/*/38": "PLI", "A/*/45": "PLI"}`
- **CLI**: `interface_type=PLI`, or `'interface_type={A/*/38: PLI}'`
- **Legal values**: `PPI`, `PLI`, `PNA` — **case-sensitive and upper-case**. `PNA` covers
  both DNA and RNA (`ItypeEnum` in `models/rf3/src/rf3/conditions.py:84-88`). An unknown
  label raises `ValueError` listing the legal labels.
- **Feature**: `condition_itype_1_atom`, `Int[L, 3]` (the `NONE` level is dropped).
- **Training**: written **only** by `PPIHotspot`, **only** on the sampled hotspot atoms
  (`annotate_interface_type: true`). It therefore has exactly the hotspot rate (15 %) and
  **never appears in training without hotspots** — it is a property of a hotspot, not an
  independent condition.
- **Metric**: none specific. Score it through the hotspot metrics.
- **Example**: [`.../interface_type/pli_interface_type_9dfn.json`](releases/rf3/examples/design_conditions/interface_type/pli_interface_type_9dfn.json)
  (scalar, 78 labelled atoms) and
  [`.../interface_type/pli_interface_type_dict_9dfn.json`](releases/rf3/examples/design_conditions/interface_type/pli_interface_type_dict_9dfn.json)
  (dict form over the same five residues).

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/interface_type/pli_interface_type_9dfn.json'
```

### Two pitfalls

1. **The scalar form needs `hotspot_selection`.** A bare string is written onto the atoms
   picked by `hotspot_selection`; with no hotspots there is nothing to write on, and
   `apply_interface_type` logs a warning and returns without applying the condition
   (`apply_interface_type` in `models/rf3/src/rf3/utils/inference.py`). The dict form
   writes onto its own selections and does not need hotspots.
2. **Under per-condition CFG this axis keeps its prerequisite.** Isolating an axis
   normally drops every other condition field, which for a scalar `interface_type` would
   leave no hotspot mask to attach the label to — a stream bit-identical to the
   unconditional one, i.e. guidance of exactly zero with nothing to show it. So
   `interface_type` is declared a dependent axis (`CFG_CONDITION_PREREQUISITES` in
   `models/rf3/src/rf3/utils/inference.py`) and its stream keeps `hotspot_selection` as
   well. If you guide a scalar `interface_type` without supplying `hotspot_selection`,
   the run raises rather than quietly guiding nothing; the dict form stands alone and
   needs no prerequisite.

---

## 4. Relative solvent accessibility (`rasa_selection`)

Asks for atoms to end up buried, intermediate, or exposed.

- **JSON, auto-binned**: `"rasa_selection": ["C/60C/101"]` — per-atom relative SASA is
  computed over the selection and binned.
- **JSON, explicit**: `"rasa_selection": {"C/60C/101": "BURIED"}`
- **CLI**: `rasa_selection="[C/60C/101]"`, or `'rasa_selection={C/60C/101: BURIED}'`
- **Legal labels**: `BURIED`, `INTERMEDIATE`, `EXPOSED` (and `UNSPECIFIED` = no
  condition), **upper-case** — `RasaEnum` is looked up by name, so `buried` raises
  `ValueError`.
- **Bins**: relative SASA `[0, 0.1) → BURIED`, `[0.1, 0.2) → INTERMEDIATE`,
  `>= 0.2 → EXPOSED`. Computed with biotite Shrake–Rupley, `probe_radius=1.4`,
  `atom_radii="ProtOr"`, `point_number=100` — identical parameters in training
  (`add_rasa_condition`), inference (`apply_rasa_selection`) and the metric
  (`get_rasa_metrics`). No external dependency.
- **Feature**: `condition_rasa_1_atom`, `Int[L, 3]`.
- **Training**: 2.4 % (see section 1). **Only ligand atoms were ever conditioned**:
  `p_include_non_ligand = 0.0` in `models/rf3/src/rf3/data/design_tasks/base.py`, with no
  override anywhere in the released configs or in the checkpoints' embedded train config.
  Applying `rasa_selection` to protein atoms is therefore out of distribution.
- **Metric**: `RasaAdherenceMetric` →
  `buried_conditioning_adherence` / `intermediate_conditioning_adherence` /
  `exposed_conditioning_adherence` (fraction of conditioned atoms whose predicted bin
  matches). Exactly `1.0` on ground-truth coordinates.
- **Example**: [`.../rasa/ligand_rasa_5hkn.json`](releases/rf3/examples/design_conditions/rasa/ligand_rasa_5hkn.json)
  — two entries, auto and explicit, over the 5HKN ligand (120 conditioned atoms each).

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/rasa/ligand_rasa_5hkn.json'
```

### Two pitfalls

1. **The label is accessibility *within the selected set*, not burial in the complex.**
   SASA is computed after slicing to the selected atoms (that is what AtomWorks'
   `atom_filter` does), in training, at inference and in the metric alike. A ligand atom
   that is buried by the protein but exposed within the isolated ligand is labelled
   `EXPOSED`.
2. **Do not select ions or metals.** `ProtOr` radii do not cover them, their SASA comes
   out `NaN`, and `np.digitize(NaN)` falls into the top bin — silently labelling them
   `EXPOSED`. The behaviour is kept deliberately (training did the same), but it is not
   what you want to condition on.

---

## 5. Hydrogen-bond donor / acceptor identity (`hbond_selection`)

Declares which atoms act as h-bond donors and which as acceptors.

- **JSON**: `"hbond_selection": {"C/SAM/402": "ACCEPTOR", "A/*/110": "DONOR"}`
- **CLI**: `'hbond_selection={C/SAM/402: ACCEPTOR}'`
- **Legal labels**: `DONOR`, `ACCEPTOR`, `BOTH`, `NONE` (`HBondEnum`), upper-case.
- **Only the explicit dict form is accepted.** A list or a string raises `ValueError`:
  deriving donors/acceptors automatically requires the third-party HBPLUS program, which
  is intentionally not wired into the inference path.
- **Feature**: `condition_hbond_1_atom`, `Int[L, 3]`.
- **Training**: 0.4 % — by far the least-seen of the five. Labels come from HBPLUS via
  `HBPLUS_PATH`; without it the condition is skipped (a one-shot warning is logged).
- **Metric**: `HbondMetrics` → `mean_correct_donors_percent`,
  `mean_correct_acceptors_percent`, `mean_num_hbonds`. **Needs `HBPLUS_PATH`**; without it
  the metric is skipped with a warning.
- **Example**: [`.../hbond/explicit_hbond_9dfn.json`](releases/rf3/examples/design_conditions/hbond/explicit_hbond_9dfn.json)
  — 63 labelled atoms around the SAM cofactor of 9DFN.

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hbond/explicit_hbond_9dfn.json'
```

### One pitfall

**`BOTH` was never produced in training.** `calculate_hbonds` assigns donor and
acceptor per h-bond, overwriting as it goes, and the two partners of a bond are always
different atoms; an atom that is both only keeps the label of the last bond processed.
Passing `BOTH` at inference is out of distribution.

## 6. Checking that a selection matched

A selection that cannot be parsed raises `ValueError` naming the offending string (a
residue range matches only if every residue in it is present) — conditions are never
dropped silently through a parse failure. A *well-formed* selection that happens to match
no atom, however, simply writes nothing and does not raise.

Three things make that easy to trip over, so verify your selections against the **parsed**
structure, not the raw file:

- AtomWorks assigns chain IDs during parsing, splitting non-polymer entities into their
  own chains. The 5HKN ligand sits in chain `B` of the input file but is selected as
  `C/60C/101`; the 9DFN ligands likewise become chains of their own.
- **Residue numbers change as well.** Selections run against the parsed structure, whose
  numbering follows the file's `label_*` fields rather than the author numbering a PDB
  entry or a paper quotes. In the bundled `9dfn.cif` the author numbering runs 29-860
  while the parsed numbering runs 22-304, so author residue 38 is `ASN` but *selected*
  residue 38 is `CYS`. Copying author numbers into a selection therefore picks different
  residues, and silently so whenever the selection omits the `RES_NAME` field: `A/*/38`
  takes whatever sits at parsed index 38, while `A/CYS/38/SG` at least matches nothing
  rather than the wrong thing. The `_model.cif` written by a run carries the parsed
  numbering, so one unconditioned run tells you which numbers to select on.
- **The counts are printed only with `rf3 fold -v`.** `hotspot_selection` and the two
  template selections report theirs ("Selected 63 hotspot atoms", "Selected 1830 atoms for
  token-level templating with 1 syntaxes"), and a default run shows neither: those three
  call sites log on the root logger, and `configure_minimal_inference_logging` drops the
  root logger -- and `rf3.utils.inference` by name -- to ERROR. The other three conditions
  report nothing at any verbosity; for them the quickest check is the metric for that
  condition, which only reports on atoms that actually carry it.

---

## 7. Atom-pair distances (`distance_template_selection`)

Conditions the pairwise distances *within* each group of atoms — an active site, a
metal-coordination shell, two motifs held at a fixed separation.

- **Feature**: `condition_distance_2_atom`, `Float[L, L]` (2-body, symmetric; defined
  upstream in AtomWorks, `lib/atomworks/src/atomworks/ml/conditions/definitions.py:92`).
- **Training**: 4 %, from the four `*_distance` tasks in
  `models/rf3/configs/design_tasks/all.yaml`
  (`tip_atom_partial_ligand_distance`, `tip_atom_distance`, `partial_ligand_distance`,
  `motif_scaffold_distance`).
- **Metric**: `compute_distance_mae` /`MotifComparisonMetric` in
  `models/rf3/src/rf3/metrics/design_gt_motif_metrics.py` →
  `mae_all_pairs`, `mae_pairs_with_2d_condition`, `mae_pairs_no_2d_condition`.

### Three ways to specify it

```json
"distance_template_selection": ["A/HIS/111/NE2", "A/HIS/45/ND1", "A/LYS/126/NZ"]
```
One group; all `N(N-1)/2` distances taken from the input coordinates.

```json
"distance_template_selection": [
    {"atoms": ["A/HIS/111/NE2", "A/HIS/45/ND1", "A/LYS/126/NZ"]},
    {"atoms": ["A/HIS/111/NE2", "B/GLU/20/OE1"], "distance": 3.0}
]
```
Independent groups; only *within*-group pairs are conditioned. A scalar `distance` applies
to every pair of that group, so it is most natural for a two-atom group.

```json
"distance_template_selection": [
    {"atoms": ["A/*/48/CB", "A/*/48/CG", "A/*/52/OD1"],
     "distance": [[0.0, 1.5, 5.8], [1.5, 0.0, 4.6], [5.8, 4.6, 0.0]]}
]
```
An `N x N` matrix sets each pair individually; row/column order follows `atoms` and the
diagonal is ignored. In matrix form **each selection must match exactly one atom**, or the
run raises.

- **CLI** (flat-list form only):
  `distance_template_selection="[A/HIS/111/NE2, A/HIS/45/ND1]"`
- **Example**: [`.../2D_distance/distval_matrix_9dfn.json`](releases/rf3/examples/design_conditions/2D_distance/distval_matrix_9dfn.json)
  (matrix form, 4 atoms = 6 conditioned pairs) and
  [`.../motif/motif_template_9dfn.json`](releases/rf3/examples/design_conditions/motif/motif_template_9dfn.json)
  (from-coordinates form, 3192 conditioned pairs, combined with `template_selection`).

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/2D_distance/distval_matrix_9dfn.json'
```

### Pitfalls

1. **Selected residues are automatically atomized, whole-residue, and that changes the
   tokenization of the entire run.** Without it the trunk averages atom-pair distances
   token-wise and erases exactly the side-chain-level information the condition carries.
   Atomization happens at construction time in **all three** input paths — JSON, CIF and
   `AtomArray` (`_atomize_distance_template_residues`, called from `from_cif_path`,
   `from_json_dict` and `from_atom_array` in
   `models/rf3/src/rf3/utils/inference.py`) — so a CLI run and a JSON
   run see the same topology. It has to stay there: if the streams of a CFG run were
   tokenized differently their deltas would not be comparable. Expect more tokens, more
   memory, and a different token layout in the outputs than the same input without the
   condition (the 9DFN motif example atomizes 103 atoms).

   The *set of residues* atomized is the same across the three paths, but the absolute
   atom count is not: the CIF path parses with `hydrogen_policy="remove"` while the JSON
   and `AtomArray` paths keep hydrogens, so the same selection atomizes fewer atoms when
   read straight from a `.cif`. The published examples and the paper runs go through the
   JSON path; use it when reproducing reported numbers.
2. **The condition is chain-agnostic.** Unlike `template_selection` /
   `relative_template_selection`, intra- and inter-chain pairs are both allowed with no
   filtering; a cross-chain distance is just two atoms from different chains in the same
   group.
3. A selection matching several atoms (e.g. a whole residue `A/HIS/45`) puts all of them
   in the group, which grows the pair count quadratically.

---

## 8. The three RF3-inherited axes

These are **not** CaGE design conditions. They are gated by `p_unconditional` in the
dataset configs, not by the `design_tasks` frequencies, and they predate this work. They
are listed here because they are guidable CFG axes alongside the five.

| Axis | Field | Status |
|---|---|---|
| `template` | `template_selection` | RF3's token-level ground-truth distogram template. |
| `ground_truth_conformer` | `ground_truth_conformer_selection` | RF3's reference-conformer track; part of the per-atom input. |
| `relative_template` | `relative_template_selection` | **A new inference interface onto a condition RF3 already trained**, not a new condition. |

`relative_template_selection` deserves the explicit caveat: it exposes RF3's trained
inter-molecule distance conditioning (`p_provide_inter_molecule_distances`) through a
selection API, and CaGE adds the API and the CFG axis — not the training signal. The
`is_relative_template_selected__<group>` annotations it writes did not exist during
training, so their training-time contribution is identically zero. It is written as
`{group_name: [selections]}` (a flat list becomes a single `__all__` group), and it
conditions distances *between* the components inside a group:

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/relative_template/relative_template_7xli.json'
```

The inverse-folding behaviour of the released weights likewise rests on these inherited
axes (ground-truth conformers plus sequence masking), not on the five design conditions.

---

## 9. Classifier-free guidance

Guidance is off by default; with it off a run is byte-for-byte an ordinary conditional
run. The guidable axes are exactly the eight keys of `CFG_CONDITION_FIELDS`
(`models/rf3/src/rf3/utils/inference.py:50-59`): the five design conditions plus the three
inherited axes.

```bash
# Guide one axis in isolation at scale 3
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/2D_distance/distval_matrix_9dfn.json' \
  inference_classifier_free_guidance=true \
  'inference_cfg_scale={distance_template: 3.0}'

# Guide everything the input selects, as one stream, at scale 2
rf3 fold inputs='...' inference_classifier_free_guidance=true inference_cfg_scale=2.0
```

**Float vs dict.** A float builds **one** guidance stream carrying every condition the
input selects, at that scale. A dict `{axis: scale}` builds **one stream per named axis**,
each at its own scale, and makes the main stream unconditional — this is how a single
condition is swept in isolation.

**What the scale means.** Guidance composes as
`delta_uncond + s * (delta_named - delta_uncond)`, so:

- `s = 0` is that axis' **zero point** — no condition field is set anywhere. It is *not*
  the same input as a structure that carries no conditions at all: whatever a selection
  already wrote onto the atom array survives, and `distance_template_selection` in
  particular atomizes its residues at parse time, which changes tokenization.
- `s = 1` is the named-conditions-only prediction with **no extrapolation**. It equals the
  plain conditional prediction only when the dict names everything the input selects.
- `s > 1` extrapolates: the usual "guide harder" regime.

**The dict drops what it does not name.** It removes every condition from the main stream
and re-enters only the named ones, so a condition the input selects but the dict omits is
absent from the run entirely. Naming an axis the input does not select is ignored with a
warning; naming an axis that does not exist raises at any scale.

**Noise-interval gating.** `inference_cfg_sigma_min` / `inference_cfg_sigma_max` restrict
guidance to diffusion steps whose noise level `t_hat` lies in `[sigma_min, sigma_max]`.
The defaults (`0.0` / `null`) guide every step. Under the default schedule `t_hat` runs
from about 2560 at the first, highest-noise step down to about 0.006 at the last, so
`inference_cfg_sigma_min=1 inference_cfg_sigma_max=300` guides only the middle band.

**Per-stream confidence.** `confidence_streams` chooses which trunk streams the confidence
head scores:

| Stream | Meaning | Output file |
|---|---|---|
| `main` | the stream that is rolled out — the *unconditional* input whenever guidance streams were built | `{sample}_summary_confidences.json` |
| `conditions` | one per guidance stream, named after the axis (`all` for a float scale) | `{sample}_summary_confidences_condition-{name}.json` |
| `merged` | trunk features extrapolated the way CFG combines scores, `S_main + sum_i scale_i (S_i - S_main)` | `{sample}_summary_confidences_merged.json` |

Anything beyond `main` requires a checkpoint with a confidence head (see
[WEIGHTS.md](WEIGHTS.md)); asking for it without one raises. Each stream gets its own
`ptm` / `iptm` / `has_clash` / `ranking_score`, re-run on that stream's PAE, and names
itself in a `stream` field — so the files are directly comparable. `merged` is the closest
thing to "the guided sample", but the head never saw extrapolated features in training;
treat it as indicative. Cost is one extra confidence-head pass per stream per diffusion
sample, with no extra trunk passes.

`ranking_stream` (default `merged`) chooses which scored stream orders
`{example_id}_ranking_scores.csv` and picks the model copied to the top level, falling
back to `main` for any sample where the requested stream was not produced. The stream
actually used is recorded per row in the CSV.

```bash
rf3 fold inputs='...' \
  inference_classifier_free_guidance=true 'inference_cfg_scale={hotspot: 2.0}' \
  'confidence_streams=[main, conditions, merged]' ranking_stream=merged
```

---

## 10. Motif guidance (`motif_guidance_selection`)

Motif guidance is **not** one of the five design conditions and **not** a CFG axis. Where a
design condition is a trained signal that enters the trunk as a feature, motif guidance is
an untrained projection applied by the sampler: every diffusion step it Kabsch-fits a
reference motif onto the current prediction's anchor atoms and interpolates the motif atoms
toward it. It never reaches the denoiser, is never featurized, never atomizes anything, and
so it cannot be named in `inference_cfg_scale` and does not change tokenization. It acts on
`x0` *after* the CFG blend, so it refines the clean estimate the Euler step actually moves
toward rather than the unconditional one.

It is off by default, and a selection alone does not switch it on: an input that carries one
still folds unguided and stays comparable with its own baseline. Enabling it without a
selection raises.

### The eight keys

| Key | Default | Meaning |
|---|---|---|
| `motif_guidance_selection` | `null` | Groups of atoms to pull, each rigid-aligned independently. See the grammar below. |
| `motif_guidance_reference` | `null` | Where the target coordinates come from. `null` means the input's own coordinates. A path, or a list of `{path, chain_id}`, pulls toward a different structure — the only way to decouple guidance from whatever `template_selection` already conditions on. |
| `inference_motif_guidance` | `false` | The explicit opt-in. |
| `inference_motif_guidance_mu` | `1.0` | Fraction of the motif-atom → aligned-target gap closed per inner step. Dimensionless: no noise-level weighting, no step-size schedule, no clipping. **The default is an exact hard snap, not a soft pull.** Lower it to leave the denoiser room to relax neighbouring bonds. `mu=0` is off, not weak — the gate still evaluates and still logs arming. |
| `inference_motif_guidance_inner_steps` | `1` | Projection steps per diffusion step. The aligned target is refit each one, so the residual gap goes as `(1 - mu)^inner_steps`. `0` is an exact no-op. |
| `inference_motif_guidance_rmsd_threshold` | `5.0` | A diffusion sample arms once its own anchor RMSD (max over groups) falls below this, in Å, and stays armed. Arming is **per sample**, not per batch: the engine rolls out several samples per trunk pass, each with its own noise draw and its own random frame. |
| `inference_motif_guidance_t_high` | `0.5` | Upper edge of the active window on the step fraction `t_norm`, which runs `1.0` (first step, high noise) → `~0.0` (last). The default leaves the noisier first half unguided. |
| `inference_motif_guidance_t_low` | `0.0` | Lower edge. Raise it to leave the final steps to the denoiser. Both edges are inclusive, and leaving the window re-silences guidance even for a sample that had already armed. |

### Selection grammar

A group is `{name, atoms, anchor}`. `atoms` is the set that gets pulled; `anchor` is the
subset that drives that group's Kabsch fit and **defaults to all of the group's atoms**.
Selections are ordinary `AtomSelectionStack` queries — the same
`CHAIN/RES_NAME/RES_ID/ATOM_NAME` syntax every other condition uses — so `A/HIS/47/NE2` is
one atom, `A/*/38` a whole residue and `B` a whole chain. Use this explicit form:

```bash
rf3 fold inputs='docs/releases/rf3/examples/9dfn.cif' \
  inference_motif_guidance=true \
  'motif_guidance_selection=[{name: catalytic_site, atoms: [A/*/38, A/*/45, A/*/47, A/*/110], anchor: [A/CYS/38/SG, A/CYS/45/SG, A/HIS/47/NE2, A/ASN/110/OD1]}, {name: cofactor, atoms: [B]}]'
```

Hydra lexes each override as one token, so a selection has to stay **on a single line**: a
newline inside the quotes is an `OverrideParseException`, not a continuation. Backslash
continuations between separate overrides, as above, are fine. For a selection this long the
readable alternative is a task file, which takes the same two groups as indented JSON —
[`motif_guidance/catalytic_site_9dfn.json`](releases/rf3/examples/design_conditions/motif_guidance/catalytic_site_9dfn.json).
A task file supplies `motif_guidance_selection` and `motif_guidance_reference`;
`inference_motif_guidance=true` is an engine argument, not a per-example field, so it still
has to be given on the command line.

Each group is fitted on its own, so **inter-group relative pose is not constrained** — above,
the catalytic side chains and the SF4 cluster are each held rigid internally while their
relative pose can drift. Put atoms in one group when you want their relative pose preserved.

A bare string or a flat list of selections is also accepted, as one group whose anchor is its
4-field (single-atom) selections, or all of its atoms if it has none. It exists for
compatibility with task files written against the earlier standalone implementation, and it
is easy to get wrong — `[A/HIS/47/NE2, A/ASN/110/OD1, B]` anchors a whole chain on two atoms,
which is exactly the failure below. Prefer the explicit form.

### Give every group at least 3 non-collinear anchors

This is the one thing that will silently ruin a run. The sampler re-randomizes the global
frame every step, independently per diffusion sample, while the reference coordinates are
fixed — so a group's Kabsch fit is the only thing that maps the reference into the current
frame. Three non-collinear anchors pin the rotation. Fewer, or a collinear set, leaves a free
rotation, the fit cannot absorb the random frame, and the motif is placed in an orientation
that is **wrong and that changes between seeds**. That is worse than guidance not firing.

Measured on a 12-atom motif, worst equivariance error over three random frames:

| Anchors | Equivariance error |
|---|---|
| ≥3, general position | **0.0000 Å** |
| 2 | 4–16 Å |
| 1 | 32–43 Å |
| 3 or 4, collinear | 10–29 Å |

Resolution cannot repair this, so it warns — look for `will NOT steer this group as
intended` in the log. The check fires only below a second-singular-value ratio of `1e-4`,
which real motifs sit far above, so it does not cry wolf.

### Other things worth knowing

- **Atoms with no coordinates are dropped.** A whole-residue or whole-chain selection picks
  up parser-added hydrogens, which carry none; keeping them would put `NaN` into every
  refined `x0`. They are dropped with a warning, and an anchor that exists only because the
  group named none is dropped with them. An anchor you named explicitly still raises.
- **One example per trunk pass.** Guidance refuses `B != 1`: the motif indices address a
  single example's atom axis.
- **Not available under partial diffusion.** `SamplePartialDiffusion` truncates the noise
  schedule, which would renormalize `t_norm` onto a different noise range, so it refuses.
- **Watch for the inert run.** If no step falls inside the window, or no sample ever arms,
  the sampler warns that guidance was attached but never applied — the output is then
  bit-for-bit an unguided run. With the default `rmsd_threshold=5.0` a motif that never
  converges that far simply never arms.

---

## 11. What you can and cannot reproduce from this repository

- **Featurization and inference** for all five conditions, plus CFG, run on the structures
  bundled in `docs/releases/rf3/examples/` — see
  [the example index](releases/rf3/examples/design_conditions/README.md).
- **RASA and atom-pair-distance adherence** can be scored with no external tool.
- **H-bond metrics and h-bond training** need third-party HBPLUS (`HBPLUS_PATH`).
- **Hotspot / interface-type validation metrics** have no ready-to-run path here: the
  datasets they were computed on are not distributed. Point the metrics at your own
  interface structures.
- Training-data preparation for DNA-containing inputs additionally needs X3DNA; inference
  skips it entirely.
