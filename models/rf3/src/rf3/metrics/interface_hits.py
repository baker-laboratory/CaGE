"""Per-interface pocket / hotspot hit metrics for PPI and PLI.

Models on :class:`rf3.metrics.lddt.InterfaceLDDTByType`. Reads
``interfaces_to_score`` from ``extra_info``, filters to
``{"protein-ligand", "protein-protein"}``, picks a ``pocket_chain`` per
interface (protein side for PLI, hotspot-bearing chain for PPI with a
seeded-random fallback), and emits ``pocket_hit``, ``pocket_iou``, and
``hotspot_hit`` at a configurable distance cutoff (default 4.5A).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from atomworks.enums import ChainTypeInfo
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, AtomArrayStack
from rf3.conditions import C_HOT
from scipy.spatial.distance import cdist

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


_SUPPORTED_INTERFACE_TYPES = frozenset({"protein-ligand", "protein-protein"})


def _pocket_res_ids(
    atom_array: AtomArray,
    pocket_pn_unit_iid: str,
    partner_pn_unit_iid: str,
    cutoff: float,
) -> set[int]:
    """Return ``res_id``s on ``pocket_pn_unit_iid`` with any heavy atom within
    ``cutoff`` of any heavy atom on ``partner_pn_unit_iid``.

    """
    pocket_mask = (atom_array.pn_unit_iid == pocket_pn_unit_iid) & (atom_array.element != "H")
    partner_mask = (atom_array.pn_unit_iid == partner_pn_unit_iid) & (
        atom_array.element != "H"
    )

    if pocket_mask.sum() == 0 or partner_mask.sum() == 0:
        return set()

    pocket_coords = atom_array.coord[pocket_mask]
    partner_coords = atom_array.coord[partner_mask]

    dists = cdist(pocket_coords, partner_coords)
    min_dists = dists.min(axis=1)

    pocket_res_ids = atom_array.res_id[pocket_mask]
    return set(pocket_res_ids[min_dists < cutoff].tolist())


def _is_protein_pn_unit(atom_array: AtomArray, pn_unit_iid: str) -> bool | None:
    """``True`` if ``chain_id``'s ``ChainType`` is in
    :attr:`ChainTypeInfo.PROTEINS`, ``False`` if not, ``None`` if unknown.

    Returns ``None`` when ``chain_type`` is missing from the atom array or
    no atoms match ``pn_unit_iid`` (caller should treat as "cannot decide").
    """
    if "chain_type" not in atom_array.get_annotation_categories():
        return None
    pn_unit_atoms = atom_array.pn_unit_iid == pn_unit_iid
    if not pn_unit_atoms.any():
        return None
    return bool(atom_array.chain_type[pn_unit_atoms][0] in ChainTypeInfo.PROTEINS)


def _has_hotspot_atoms_on_chain(
    atom_array: AtomArray,
    hotspot_mask: np.ndarray | None,
    pn_unit_iid: str,
) -> bool:
    """``True`` if any atom on ``pn_unit_iid`` is C_HOT-marked."""
    if hotspot_mask is None:
        return False
    pn_unit_atoms = atom_array.pn_unit_iid == pn_unit_iid
    return bool((hotspot_mask & pn_unit_atoms).any())


def _select_pocket_pn_unit(
    gt_array: AtomArray,
    pn_unit_i: str,
    pn_unit_j: str,
    interface_type: str,
    hotspot_mask: np.ndarray | None,
    example_id: str | None,
) -> str | None:
    """Pick which side of the interface carries the residue set we score.

    PLI: the chain whose ``ChainType`` is in :attr:`ChainTypeInfo.PROTEINS`;
    returns ``None`` if neither (or both) qualify so the caller can skip+warn.

    PPI: the chain carrying any C_HOT atom; if exactly one carries hotspots
    that's the pocket. Otherwise (no annotation, or hotspots on both/neither)
    a seeded-random choice keyed off ``(example_id, chain_i, chain_j)`` so
    the same example deterministically picks the same side across runs.
    """
    if interface_type == "protein-ligand":
        i_protein = _is_protein_pn_unit(gt_array, pn_unit_i)
        j_protein = _is_protein_pn_unit(gt_array, pn_unit_j)
        # Need exactly one of the two to be a protein chain.
        if i_protein is True and j_protein is not True:
            return pn_unit_i
        if j_protein is True and i_protein is not True:
            return pn_unit_j
        return None

    if interface_type == "protein-protein":
        i_has_hot = _has_hotspot_atoms_on_chain(gt_array, hotspot_mask, pn_unit_i)
        j_has_hot = _has_hotspot_atoms_on_chain(gt_array, hotspot_mask, pn_unit_j)
        if i_has_hot and not j_has_hot:
            return pn_unit_i
        if j_has_hot and not i_has_hot:
            return pn_unit_j
        # Either no hotspots, or hotspots on both sides -> seeded-random.
        # Use a string seed so random.Random hashes it stably across processes
        # (Python's built-in hash() is randomized via PYTHONHASHSEED).
        seed = f"{example_id}|{pn_unit_i}|{pn_unit_j}"
        return random.Random(seed).choice([pn_unit_i, pn_unit_j])

    return None


def _max_ignore_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmax(arr))


class InterfaceHitsByType(Metric):
    """Per-interface pocket / hotspot hit metrics, grouped by interface type.

    Mirrors :class:`rf3.metrics.lddt.InterfaceLDDTByType` in shape: reads
    ``interfaces_to_score`` from ``extra_info`` and returns a list of
    per-interface dicts, each carrying ``pn_units`` and ``type`` so the
    callback can group by interface type. Emits set-overlap metrics instead
    of LDDT:

    - ``pocket_hit_<cutoff>``: ``1.0`` if any GT pocket residue appears in
      the predicted pocket, else ``0.0``.
    - ``pocket_iou_<cutoff>``: ``|gt ∩ pred| / |gt ∪ pred|`` over pocket
      residue sets.
    - ``hotspot_hit_<cutoff>``: ``1.0`` if any C_HOT-marked residue appears
      in the predicted pocket, else ``0.0``. ``NaN`` when no hotspot
      annotation is present for this interface.

    Only ``protein-ligand`` and ``protein-protein`` interfaces are scored;
    others are skipped silently. The "pocket" side is the protein chain for
    PLI, and the hotspot-bearing chain for PPI with a seeded-random fallback
    (see :func:`_select_pocket_pn_unit`).

    Assumes that the predicted and ground truth atom arrays are the same length
        and symmetry resolved 
    """

    def __init__(self, cutoff: float = 4.5, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "interfaces_to_score": ("extra_info", "interfaces_to_score"),
            "example_id": ("extra_info", "example_id"),
        }

    @property
    def optional_kwargs(self) -> set[str]:
        return {"interfaces_to_score", "example_id"}

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        interfaces_to_score: list | None = None,
        example_id: str | list[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if not interfaces_to_score:
            return []

        # Legacy parquet path: still-stringified list-of-tuples.
        if isinstance(interfaces_to_score, str):
            interfaces_to_score = (
                eval(interfaces_to_score) if interfaces_to_score else []
            )

        # Top-level wrapper is the batch dimension. We only support batch=1.
        if len(interfaces_to_score) > 1:
            raise NotImplementedError(
                "InterfaceHitsByType currently only supports batch size 1."
            )
        interfaces_to_score = interfaces_to_score[0] if interfaces_to_score else None

        if not interfaces_to_score:
            return []

        if isinstance(interfaces_to_score, str):
            interfaces_to_score = (
                eval(interfaces_to_score) if interfaces_to_score else []
            )

        # Drop any leftover None placeholders the collator may insert.
        interfaces_to_score = [x for x in interfaces_to_score if x is not None]
        if not interfaces_to_score:
            return []

        # Unwrap example_id batch dim and coerce to a stable string seed.
        if isinstance(example_id, list) and len(example_id) >= 1:
            example_id = example_id[0]
        if example_id is not None and not isinstance(example_id, str):
            example_id = str(example_id)

        predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
        ground_truth_atom_array_stack = ensure_atom_array_stack(
            ground_truth_atom_array_stack
        )
        gt_array = ground_truth_atom_array_stack[0]
        D = predicted_atom_array_stack.stack_depth()

        # Read hotspot atom-level mask once per example (None if no annotation
        # or all-zero).
        hotspot_mask: np.ndarray | None = None
        if C_HOT.full_name in gt_array.get_annotation_categories():
            hs = np.asarray(C_HOT.annotation(gt_array)).astype(bool)
            if hs.any():
                hotspot_mask = hs

        cutoff_tag = f"{self.cutoff:g}"
        pocket_hit_key = f"pocket_hit_{cutoff_tag}"
        pocket_iou_key = f"pocket_iou_{cutoff_tag}"
        hotspot_hit_key = f"hotspot_hit_{cutoff_tag}"

        results: list[dict[str, Any]] = []

        for tup in interfaces_to_score:
            try:
                pn_unit_i, pn_unit_j, interface_type = tup
            except (TypeError, ValueError):
                ranked_logger.warning(f"Skipping malformed interface tuple: {tup!r}")
                continue

            if interface_type not in _SUPPORTED_INTERFACE_TYPES:
                continue

            pocket_pn_unit = _select_pocket_pn_unit(
                gt_array,
                pn_unit_i,
                pn_unit_j,
                interface_type,
                hotspot_mask,
                example_id,
            )
            if pocket_pn_unit is None:
                ranked_logger.warning(
                    f"Cannot determine pocket side for {interface_type} "
                    f"interface {pn_unit_i}-{pn_unit_j} (no protein chain "
                    f"found or chain_type annotation missing); skipping."
                )
                continue
            partner_pn_unit = pn_unit_j if pocket_pn_unit == pn_unit_i else pn_unit_i

            gt_pocket = _pocket_res_ids(
                gt_array, pocket_pn_unit, partner_pn_unit, self.cutoff
            )
            if not gt_pocket:
                ranked_logger.warning(
                    f"No GT pocket residues for {pocket_pn_unit}-{partner_pn_unit} "
                    f"(cutoff={self.cutoff}A); skipping interface "
                    f"{pn_unit_i}-{pn_unit_j}."
                )
                continue

            # Hotspot residues live on the pocket side by construction
            # (PLI: hotspots mark protein-side atoms; PPI: we picked the
            # hotspot-bearing chain as pocket_chain).
            hotspot_res_ids: set[int] = set()
            if hotspot_mask is not None:
                pocket_atoms = gt_array.pn_unit_iid == pocket_pn_unit
                hotspot_res_ids = set(
                    gt_array.res_id[hotspot_mask & pocket_atoms].tolist()
                )

            pocket_hits: list[float] = []
            pocket_ious: list[float] = []
            hotspot_hits: list[float] = []

            for d in range(D):
                pred = predicted_atom_array_stack[d]
                pred_pocket = _pocket_res_ids(
                    pred, pocket_pn_unit, partner_pn_unit, self.cutoff
                )
                inter = len(gt_pocket & pred_pocket)
                union = len(gt_pocket | pred_pocket)

                pocket_hits.append(1.0 if inter >= 1 else 0.0)
                pocket_ious.append(inter / union if union > 0 else 0.0)
                if hotspot_res_ids:
                    hotspot_hits.append(
                        1.0 if len(hotspot_res_ids & pred_pocket) >= 1 else 0.0
                    )
                else:
                    hotspot_hits.append(float("nan"))

            results.append(
                {
                    "pn_units": [pn_unit_i, pn_unit_j],
                    "type": interface_type,
                    "pocket_pn_unit": pocket_pn_unit,
                    "n_gt_pocket_residues": len(gt_pocket),
                    "n_hotspot_residues": len(hotspot_res_ids),
                    f"best_of_1_{pocket_hit_key}": pocket_hits[0],
                    f"best_of_{D}_{pocket_hit_key}": _max_ignore_nan(pocket_hits),
                    f"best_of_1_{pocket_iou_key}": pocket_ious[0],
                    f"best_of_{D}_{pocket_iou_key}": _max_ignore_nan(pocket_ious),
                    f"best_of_1_{hotspot_hit_key}": hotspot_hits[0],
                    f"best_of_{D}_{hotspot_hit_key}": _max_ignore_nan(hotspot_hits),
                }
            )

        return results
