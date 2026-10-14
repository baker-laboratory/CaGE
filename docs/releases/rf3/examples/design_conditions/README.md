# Design-condition examples

Runnable inputs for the five CaGE design conditions, the relative-template inference
interface, per-axis classifier-free guidance, and motif guidance. Every example uses a
structure that ships with this repository, so all of them run as-is — except motif
guidance, which also needs `inference_motif_guidance=true` on the command line (see
[Motif guidance](#motif-guidance) below).

Run them **from the repository root** — the `components` paths inside each JSON are
repository-relative and are resolved against the working directory:

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json'
```

Field semantics, legal values, training rates and the pitfalls of each condition are in
[docs/DESIGN_CONDITIONS.md](../../../../DESIGN_CONDITIONS.md).

## Index

| Condition | Input | Structure | What it conditions | Verified featurization |
|---|---|---|---|---|
| Interface hotspots (+ interface type) | [`hotspot/ppi_hotspot_7xli.json`](hotspot/ppi_hotspot_7xli.json) | 7XLI nanobody–antigen, chains A and B from `../templates/` | 4 antigen residues as hotspots, `interface_type: PPI`, antigen chain A templated | 63 hotspot atoms, 63 interface-type atoms, 1830 templated atoms |
| Interface type, scalar | [`interface_type/pli_interface_type_9dfn.json`](interface_type/pli_interface_type_9dfn.json) | 9DFN (protein + SAM / CTP / SF4 ligands) | 5 pocket residues as hotspots, all labelled `PLI` | 78 hotspot atoms, 78 interface-type atoms |
| Interface type, dict | [`interface_type/pli_interface_type_dict_9dfn.json`](interface_type/pli_interface_type_dict_9dfn.json) | 9DFN | same 5 residues, labelled per selection — stands alone, no hotspot prerequisite | labels written per selection rather than via the hotspot mask |
| Relative solvent accessibility | [`rasa/ligand_rasa_5hkn.json`](rasa/ligand_rasa_5hkn.json) | 5HKN (ligand `60C`) | two entries: auto-binned RASA, and an explicit `BURIED` label, on the ligand | 120 conditioned atoms in each entry |
| H-bond donor / acceptor | [`hbond/explicit_hbond_9dfn.json`](hbond/explicit_hbond_9dfn.json) | 9DFN | SAM cofactor as `ACCEPTOR`, one protein residue as `DONOR` | 63 labelled atoms |
| Atom-pair distances, matrix form | [`2D_distance/distval_matrix_9dfn.json`](2D_distance/distval_matrix_9dfn.json) | 9DFN | an explicit 4x4 distance matrix over 3 CA atoms and the SAM `SD` atom | 6 conditioned pairs (4 atoms) |
| Atom-pair distances, from coordinates | [`motif/motif_template_9dfn.json`](motif/motif_template_9dfn.json) | 9DFN | 4 motif residues plus the whole ligand chain as one group, distances taken from the input, combined with `template_selection` | 3192 conditioned pairs, 103 atoms auto-atomized, 54 templated atoms |
| Relative template (RF3-inherited axis) | [`relative_template/relative_template_7xli.json`](relative_template/relative_template_7xli.json) | 7XLI chains A and B | an `epitope` group spanning both chains, conditioning inter-chain distances | 113 atoms in `is_relative_template_selected__epitope` |
| Motif guidance (**not** a design condition, **not** a CFG axis) | [`motif_guidance/catalytic_site_9dfn.json`](motif_guidance/catalytic_site_9dfn.json) | 9DFN | two rigid groups — 4 catalytic residues anchored on 4 non-collinear side-chain atoms, and the SF4 cluster anchoring itself — projected onto `x0` every diffusion step | 38 motif atoms, 12 anchors in 2 groups; 24 coordinate-less hydrogens dropped with a warning |

Counts are from a CPU featurization pass over these exact files; they are a quick way to
confirm your environment writes the same conditions. Only the hotspot and template counts
are observable from a run, and only with `rf3 fold -v` -- the hotspot example prints
"Selected 63 hotspot atoms" and "Selected 1830 atoms for token-level templating".

The glycosidase set in [`2D_distance/`](2D_distance/) is a four-way comparison rather than a
single example; it has its own section [below](#the-glycosidase-comparison-set).

## Expected outputs

Every run creates `<name>/` under `out_dir`, named after the `name` field of the entry; the
layout is in [models/rf3/README.md](../../../../../models/rf3/README.md). Two things are
specific to these examples: `rasa/ligand_rasa_5hkn.json` holds **two** entries and so
produces two directories, `ligand_rasa_5hkn_auto/` and `ligand_rasa_5hkn_explicit/`; and
the distance examples atomize the residues they select, so their outputs carry a different
token layout than the same structure folded without the condition — expected, and required
for the condition to reach the model at atom resolution.

## The glycosidase comparison set

[`2D_distance/`](2D_distance/) also holds four files that are meant to be run *against each
other*: the same three de novo glycosidase designs under four condition settings, so the
distance axis can be compared with the template axis and with no condition at all.

| File | Conditions on the entry |
|---|---|
| [`glycosidase_baseline.json`](2D_distance/glycosidase_baseline.json) | none — the control |
| [`glycosidase_template_motif.json`](2D_distance/glycosidase_template_motif.json) | `template_selection` over the 6 motif residues **and** the ligand chain |
| [`glycosidase_distance_motif.json`](2D_distance/glycosidase_distance_motif.json) | `template_selection` over the motif, plus one `distance_template_selection` group spanning the motif and the ligand chain |
| [`glycosidase_distance_explicit.json`](2D_distance/glycosidase_distance_explicit.json) | the same motif group, plus a second group carrying an explicit 3x3 matrix over one ligand atom and two motif `CB` atoms |

Each file holds three entries, one per design — `chunk75` (145 residues), `chunk11` (143)
and `chunk66` (138), each with a 22-atom aryl-glycoside ligand as chain `X`. All three
designs carry the **same six-residue motif** (Tyr, Glu, Tyr, Arg, Gln, Glu) on different
scaffolds and at different residue numbers, which is what makes the four-way comparison
meaningful across them.

Entry names are suffixed with the condition (`glycosidase_chunk75_baseline`,
`..._template_motif`, ...) so that all four files can be run into one `out_dir` without
overwriting each other.

Two things about this set are worth knowing before you read results off it:

- **The explicit matrix in `glycosidase_distance_explicit.json` reproduces the input
  coordinates** to the two decimals it is written with. It demonstrates the matrix *syntax*;
  it is not a designed target geometry, and it asks the model for the distances the input
  already has. Change the numbers to condition on something the input does not show.
- **The ligand is supplied as `.pdb`, so its bond orders are lost.** A component `path` may
  be `.pdb` (the extension set is in `atomworks.io.utils.io_utils.CIF_LIKE_EXTENSIONS`), and
  connectivity does survive — the parser reads `CONECT` records, and the `L:1` residue name
  is AtomWorks' own convention for a custom ligand, deliberately outside the CCD. But every
  `CONECT` bond is flattened to a single bond, and RDKit re-perceives aromaticity from that
  graph when it builds the reference conformer. For this ligand, which has a conjugated aryl
  group, supplying the ligand as `.sdf` instead would preserve the bond orders.

## Motif guidance

Motif guidance is the one example that needs an argument on the command line. A task file
supplies `motif_guidance_selection` and `motif_guidance_reference`, but the switch that arms
the projection is an engine argument rather than a per-example field, so the JSON on its own
parses the selection and then never fires — silently, with no warning. The
`"inference_motif_guidance": true` inside the example file documents intent; it is not read:

```bash
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/motif_guidance/catalytic_site_9dfn.json' \
  inference_motif_guidance=true
```

Motif guidance is neither a design condition nor a CFG axis — it never reaches the denoiser
and never changes tokenization — so it cannot be named in `inference_cfg_scale`. The anchor
rules (at least 3 non-collinear anchors per group) are in
[docs/DESIGN_CONDITIONS.md](../../../../DESIGN_CONDITIONS.md).

## Classifier-free guidance

Guidance is off unless asked for. A dict scale guides one axis in isolation; the axis
names are `hotspot`, `interface_type`, `rasa`, `hbond`, `distance_template`, `template`,
`relative_template`, `ground_truth_conformer`.

```bash
# Hotspots, guided at scale 2
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/hotspot/ppi_hotspot_7xli.json' \
  inference_classifier_free_guidance=true \
  'inference_cfg_scale={hotspot: 2.0}'

# Atom-pair distances, guided at scale 3, with per-stream confidence
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/2D_distance/distval_matrix_9dfn.json' \
  inference_classifier_free_guidance=true \
  'inference_cfg_scale={distance_template: 3.0}' \
  'confidence_streams=[main, conditions, merged]' ranking_stream=merged

# RASA, guided only over the mid-noise part of the schedule
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/rasa/ligand_rasa_5hkn.json' \
  inference_classifier_free_guidance=true \
  'inference_cfg_scale={rasa: 2.0}' \
  inference_cfg_sigma_min=1.0 inference_cfg_sigma_max=300.0

# Everything the input selects, as a single guidance stream
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/motif/motif_template_9dfn.json' \
  inference_classifier_free_guidance=true inference_cfg_scale=2.0

# The glycosidase set: the distance axis against the template axis, into one out_dir
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/2D_distance/glycosidase_distance_motif.json' \
  inference_classifier_free_guidance=true 'inference_cfg_scale={distance_template: 1.0}'
rf3 fold inputs='docs/releases/rf3/examples/design_conditions/2D_distance/glycosidase_template_motif.json' \
  inference_classifier_free_guidance=true 'inference_cfg_scale={template: 1.0}'
```

A dict scale drops every axis the dict does not name, and `interface_type` is a dependent
axis that keeps `hotspot_selection` alongside its label — both are specified in
[docs/DESIGN_CONDITIONS.md](../../../../DESIGN_CONDITIONS.md), and both bite here, which is
why `interface_type/pli_interface_type_dict_9dfn.json` exists as a stand-alone form.

`confidence_streams` beyond `main` needs a checkpoint with a confidence head
(`run14_confhead`); see [docs/WEIGHTS.md](../../../../WEIGHTS.md).
