"""Resolve a ``motif_guidance_selection`` into the tensors the projection needs.

Grammar
-------
A selection is a list of groups, each aligned independently by Kabsch, so inter-group
relative pose is NOT constrained. One group is::

    {"atoms": <selection(s)>, "anchor": <selection(s)>, "name": <str>}

``atoms`` (required) is the set that gets pulled. ``anchor`` (optional) is the subset
that drives the Kabsch fit; it must be a subset of ``atoms`` and defaults to all of
them. ``name`` (optional) labels the group in logs and errors.

Selections are ordinary ``AtomSelectionStack`` queries -- the same
``CHAIN/RES_NAME/RES_ID/ATOM_NAME`` syntax every other condition uses, with ``*``
wildcards and trailing fields defaulting to ``*``. So ``"A/HIS/112/NE2"`` is one atom,
``"X"`` is a whole chain and ``"A/*/5-10"`` is a residue range.

Three shorthands are accepted and normalized:

* a single group dict, detected by its ``atoms`` key;
* a mapping ``{name: group-or-selections}``, one group per key;
* a flat list or bare string, which becomes one group whose anchor is its 4-field
  (single-atom) selections, or all of its atoms if it has none.

Only that last form infers anchorhood from selector arity, because it is the shape the
upstream task files use. Everywhere else anchorhood is explicit.

Reference coordinates
---------------------
Targets come from ``reference_atom_array`` when one is given, matched by
``(chain_id, res_id, atom_name)``, and otherwise from ``atom_array``'s own coordinates
-- which is what the upstream implementation effectively did, since the file it read
its reference from was also the input structure's source.

Indices resolve against the post-pipeline AtomArray, whose order is the sampler's L
axis. Resolve against any other array and the projection pulls the wrong atoms.
"""

from __future__ import annotations

import logging
from os import PathLike

import numpy as np
import torch
from atomworks.common import as_list
from biotite.structure import AtomArray

from rf3.utils.inference import apply_atom_selection_mask

logger = logging.getLogger(__name__)

_GROUP_KEYS = frozenset({"atoms", "anchor", "name"})


def load_motif_guidance_reference(
    motif_guidance_reference: list[dict] | dict | str | None,
) -> AtomArray | None:
    """Load the reference structure a motif-guidance selection is pulled toward.

    Accepts a path, a single ``{"path": ..., "chain_id": ...}`` entry, or a list of them.
    Only path-carrying entries are allowed: a sequence-only component has no coordinates
    to offer, and passing one through the shared component parser would fail far from
    here with an unrelated message.
    """
    if not motif_guidance_reference:
        return None
    from atomworks.io.tools.inference import components_to_atom_array

    entries = motif_guidance_reference
    if isinstance(entries, (str, PathLike)):
        entries = [{"path": str(entries)}]
    elif isinstance(entries, dict):
        entries = [entries]

    components = []
    for i, entry in enumerate(entries):
        if isinstance(entry, (str, PathLike)):
            entry = {"path": str(entry)}
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(
                f"motif_guidance_reference entry {i} has no 'path'; the reference must "
                "point at structure files, not sequences"
            )
        component = {"path": str(entry["path"])}
        if entry.get("chain_id") is not None:
            component["chain_id"] = entry["chain_id"]
        components.append(component)

    return components_to_atom_array(components, bonds=None)


def normalize_motif_guidance_groups(
    motif_guidance_selection: dict | list | str | None,
) -> list[dict]:
    """Normalize a motif-guidance selection to a list of ``{name, atoms, anchor}``.

    ``anchor`` stays None when the user gave none, so the resolver can default it to the
    group's own resolved mask instead of to its selector strings: a group like
    ``{"ligand": ["X"]}`` has no single-atom selection to fall back on.

    Unknown group keys raise. Unlike ``_normalize_distance_groups`` this grammar has an
    optional key whose absence changes behaviour, so a typo must not pass silently.
    """
    value = motif_guidance_selection
    if not value:
        return []

    if isinstance(value, dict) and "atoms" in value:
        raw = [dict(value)]
    elif isinstance(value, dict):
        raw = []
        for name, group in value.items():
            group = dict(group) if isinstance(group, dict) else {"atoms": group}
            group["name"] = str(name)
            raw.append(group)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        raw = [dict(g) for g in value]
    else:
        atoms = as_list(value)
        single = [s for s in atoms if str(s).strip().strip("/").count("/") == 3]
        raw = [{"name": "default", "atoms": atoms, "anchor": single or None}]

    groups = []
    for i, group in enumerate(raw):
        unknown = set(group) - _GROUP_KEYS
        if unknown:
            raise ValueError(
                f"motif_guidance_selection group {i}: unknown key(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(_GROUP_KEYS)}"
            )
        if not group.get("atoms"):
            raise ValueError(f"motif_guidance_selection group {i}: 'atoms' is required")
        anchor = group.get("anchor")
        groups.append(
            {
                "name": str(group.get("name", f"g{i}")),
                "atoms": as_list(group["atoms"]),
                "anchor": as_list(anchor) if anchor else None,
            }
        )
    return groups


def _reference_index(reference: AtomArray) -> dict[tuple[str, int, str], int]:
    index: dict[tuple[str, int, str], int] = {}
    for j in range(len(reference)):
        key = (
            str(reference.chain_id[j]),
            int(reference.res_id[j]),
            str(reference.atom_name[j]),
        )
        if key in index:
            raise ValueError(f"motif_guidance_reference has duplicate atom {key}")
        index[key] = j
    return index


def _resolve_group(
    atom_array: AtomArray,
    group: dict,
    reference: AtomArray | None,
    ref_index: dict[tuple[str, int, str], int] | None,
) -> tuple[list[int], np.ndarray, list[int]]:
    """Resolve one group to (atom indices, target coords, anchor positions within it)."""
    name = group["name"]
    idx = np.where(apply_atom_selection_mask(atom_array, group["atoms"]))[0]
    if idx.size == 0:
        raise ValueError(
            f"motif_guidance group {name!r}: selection {group['atoms']} matched no atoms"
        )

    # A default anchor means "all of this group's atoms", which can only mean all of its
    # *usable* ones: a whole-residue or whole-chain selection picks up parser-added
    # hydrogens that carry no coordinates, and those are dropped below. An anchor the
    # caller named explicitly is never silently narrowed -- it raises instead.
    anchor_is_default = group["anchor"] is None
    if anchor_is_default:
        anchor_local = set(range(idx.size))
    else:
        anchor_idx = np.where(apply_atom_selection_mask(atom_array, group["anchor"]))[0]
        if anchor_idx.size == 0:
            raise ValueError(
                f"motif_guidance group {name!r}: anchor {group['anchor']} matched no atoms"
            )
        outside = set(anchor_idx.tolist()) - set(idx.tolist())
        if outside:
            raise ValueError(
                f"motif_guidance group {name!r}: anchor selects {len(outside)} atom(s) "
                "outside 'atoms'; the anchor must be a subset of the pulled set"
            )
        position = {int(a): local for local, a in enumerate(idx)}
        anchor_local = {position[int(a)] for a in anchor_idx}

    if reference is None:
        # Drop pulled atoms whose own coordinates are non-finite, for the same reason the
        # reference branch below drops unmatched ones: a whole-residue or whole-chain
        # selection picks up parser-added hydrogens, which carry no coordinates. Keeping
        # them would put NaN in `target`, and from there into every refined x0 -- the
        # engine's finite check gates arming on x, not on the target. Anchors are exempt:
        # a non-finite anchor still raises below, since it would undefine the fit.
        all_coords = np.asarray(atom_array.coord[idx], dtype=np.float32)
        finite = np.isfinite(all_coords).all(axis=-1)
        if anchor_is_default:
            anchor_local = {local for local in anchor_local if finite[local]}
        keep = [local for local in range(idx.size) if finite[local] or local in anchor_local]
        dropped = int(idx.size - len(keep))
        if dropped:
            logger.warning(
                "motif_guidance group %r: dropped %d pulled atom(s) with no coordinates "
                "in the input (typically parser-added hydrogens)",
                name,
                dropped,
            )
        coords = all_coords[keep]
    else:
        # An anchor atom the caller named but the reference lacks raises: it would
        # silently change the fit. A pulled non-anchor atom may be dropped -- reference
        # files routinely lack parser-added hydrogens -- and so may an atom that is only
        # an anchor by default, for the same reason.
        keep, rows, missing = [], [], 0
        for local, a in enumerate(idx):
            key = (
                str(atom_array.chain_id[a]),
                int(atom_array.res_id[a]),
                str(atom_array.atom_name[a]),
            )
            j = ref_index.get(key)
            if j is None:
                if local in anchor_local and not anchor_is_default:
                    raise ValueError(
                        f"motif_guidance group {name!r}: anchor atom {key} is absent "
                        "from motif_guidance_reference"
                    )
                missing += 1
                continue
            keep.append(local)
            rows.append(reference.coord[j])
        if not keep:
            raise ValueError(
                f"motif_guidance group {name!r}: no selected atom was found in the reference"
            )
        if missing:
            logger.warning(
                "motif_guidance group %r: dropped %d pulled atom(s) absent from the "
                "reference (typically parser-added hydrogens)",
                name,
                missing,
            )
        coords = np.asarray(np.stack(rows), dtype=np.float32)

    keep_pos = {local: n for n, local in enumerate(keep)}
    kept_anchor = [local for local in keep if local in anchor_local]
    if anchor_is_default:
        # Same rule once more, now against the target coordinates: an atom that is only
        # an anchor because the group named none cannot anchor a fit it has no finite
        # coordinate for. An explicitly named one still raises just below.
        kept_anchor = [
            local
            for local in kept_anchor
            if np.isfinite(coords[keep_pos[local]]).all()
        ]
    if not kept_anchor:
        raise ValueError(
            f"motif_guidance group {name!r}: no anchor atom survived resolution"
        )
    anchor_rows = [keep_pos[local] for local in kept_anchor]
    if not np.isfinite(coords[anchor_rows]).all():
        raise ValueError(
            f"motif_guidance group {name!r}: an anchor reference coordinate is non-finite, "
            "so its Kabsch fit would be undefined"
        )
    _warn_if_anchor_degenerate(name, coords[anchor_rows])

    return [int(idx[local]) for local in keep], coords, anchor_rows


# Smallest s1/s0 of a centred anchor set that still pins the rotation. Chosen from
# measurement, not taste: equivariance error is 0.004 A at 1e-4 and 0.024 A at 2.7e-5,
# but 2.6 A at 2.7e-7 and 29 A at exact collinearity. Real coordinates sit far above it
# -- over 931 windows of consecutive CA atoms in 9dfn and 5vht (extended strands
# included) the minimum was 0.157, and the ligands of 9dfn/NAG ranged 0.38-0.97 -- so
# this flags genuine degeneracy without crying wolf on a real motif.
_ANCHOR_COLLINEARITY_RATIO = 1e-4


def _warn_if_anchor_degenerate(name: str, anchor_coords: np.ndarray) -> None:
    """Warn when a group's anchors cannot pin its Kabsch rotation.

    The sampler re-randomizes the global frame every step (``centre_random_augmentation``,
    independently per diffusion sample) while the reference coordinates are fixed, so the
    per-group fit is the only thing that maps the reference into the current frame. Three
    non-collinear anchors are enough -- a rank-2 centred configuration fixes R once the
    reflection is corrected -- but fewer, or a collinear set, leaves a free rotation. The
    fit then cannot absorb the random frame, and guidance snaps the motif to an
    orientation that is wrong and that changes from seed to seed. That is worse than
    guidance not firing at all, so say so rather than calling it "partly unconstrained".
    """
    n = len(anchor_coords)
    reason = None
    if n < 3:
        reason = f"only {n} anchor atom(s)"
    else:
        centred = np.asarray(anchor_coords, dtype=np.float64)
        centred = centred - centred.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        ratio = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0
        if ratio < _ANCHOR_COLLINEARITY_RATIO:
            reason = (
                f"{n} anchor atoms that are collinear "
                f"(second singular value ratio {ratio:.2e} < "
                f"{_ANCHOR_COLLINEARITY_RATIO:g})"
            )
    if reason is None:
        return
    logger.warning(
        "motif_guidance group %r has %s, which leaves the Kabsch rotation "
        "underdetermined. Motif guidance will NOT steer this group as intended: it "
        "cannot recover the sampler's per-step random frame, so the motif is placed in "
        "an arbitrary orientation that varies between seeds. Give this group at least 3 "
        "non-collinear anchor atoms (an explicit `anchor` is a subset of `atoms`).",
        name,
        reason,
    )


def build_motif_guidance_feats(
    atom_array: AtomArray,
    motif_guidance_selection: dict | list | str | None,
    reference_atom_array: AtomArray | None = None,
) -> dict[str, torch.Tensor] | None:
    """Resolve a selection to indices plus reference coordinates, per rigid group.

    ``atom_array`` must be the post-pipeline array whose order is the sampler's L axis.
    Returns None for an empty selection; otherwise CPU tensors:

        motif_idx            LongTensor[M]     indices into the L axis, in group order
        target               FloatTensor[M,3]  reference coords, aligned to motif_idx
        anchor_idx_in_motif  LongTensor[A]     positions in [0, M) driving each Kabsch fit
        group_starts         LongTensor[G+1]   CSR bounds; group g is [starts[g], starts[g+1])
        group_names          list[str]
    """
    groups = normalize_motif_guidance_groups(motif_guidance_selection)
    if not groups:
        return None

    ref_index = (
        None if reference_atom_array is None else _reference_index(reference_atom_array)
    )

    motif_idx: list[int] = []
    target: list[np.ndarray] = []
    anchor_global: list[int] = []
    group_starts: list[int] = [0]
    group_names: list[str] = []

    for group in groups:
        idx, coords, anchor_rows = _resolve_group(
            atom_array, group, reference_atom_array, ref_index
        )
        offset = len(motif_idx)
        motif_idx.extend(idx)
        target.extend(coords)
        anchor_global.extend(offset + row for row in anchor_rows)
        group_starts.append(len(motif_idx))
        group_names.append(group["name"])

    repeated = len(motif_idx) - len(set(motif_idx))
    if repeated:
        raise ValueError(
            f"motif_guidance selects {repeated} atom(s) more than once; the projection "
            "sums per-atom pulls, so a shared atom would be over-corrected"
        )

    return {
        "motif_idx": torch.tensor(motif_idx, dtype=torch.long),
        "target": torch.tensor(np.stack(target), dtype=torch.float32),
        "anchor_idx_in_motif": torch.tensor(anchor_global, dtype=torch.long),
        "group_starts": torch.tensor(group_starts, dtype=torch.long),
        "group_names": group_names,
    }
