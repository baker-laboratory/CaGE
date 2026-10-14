"""Design quality metrics, ported from RFdiffusion3 (original authors: see README.md).

Compared to the original, this port:
- Operates on the ``predicted_atom_array_stack`` rather than re-deriving features
  from ``network_input["f"]``. Backbone / CA / motif masks are read directly from
  ``atom_name`` and the C_CRD ``mask_coordinate_1_atom`` annotation.
"""

import os
from typing import Any
from collections import defaultdict

import numpy as np
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArrayStack

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from rf3.conditions import C_RAS, RasaEnum
from rf3.metrics.design_metrics_utils import (
    BACKBONE_ATOM_NAMES,
    get_hotspot_contacts,
    get_rasa_metrics,
    _calculate_hbond_stats,
)

logger = RankedLogger(__name__, rank_zero_only=True)

CLASH_THRESHOLD = 1.2
FLOAT_THRESHOLD = 3.0  # max nearest-neighbor distance before considered floating
CHAINBREAK_DEVIATION_THRESHOLD = 0.75


class RasaAdherenceMetric(Metric):
    """Adherence of the predicted structure to the RASA condition (C_RAS annotation)."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array_stack": ("predicted_atom_array_stack",)}

    def compute(self, atom_array_stack: AtomArrayStack) -> dict[str, float]:
        o = defaultdict(list)
        for atom_array in atom_array_stack:
            if C_RAS.full_name not in atom_array.get_annotation_categories():
                continue
            condition = atom_array.get_annotation(C_RAS.full_name).astype(int)
            if not (condition != RasaEnum.UNSPECIFIED).any():
                continue
            for k, v in get_rasa_metrics(atom_array, condition).items():
                o[k].append(v)
        return {
            k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
            for k, v in o.items()
        }


class PPIMetrics(Metric):
    """Hotspot contact metric for PPI / enzyme tasks."""

    def __init__(self, distance_cutoff: float = 4.5):
        super().__init__()
        self.distance_cutoff = distance_cutoff

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, atom_array_stack: AtomArrayStack) -> dict[str, float]:
        contacted: list[float] = []
        for atom_array in atom_array_stack:
            cats = atom_array.get_annotation_categories()
            if "mask_hotspot_1_atom" in cats:
                hotspot_mask = atom_array.get_annotation(
                    "mask_hotspot_1_atom"
                ).astype(bool)
            elif "hotspot" in cats:
                hotspot_mask = atom_array.get_annotation("hotspot").astype(bool)
            else:
                continue

            if not hotspot_mask.any():
                continue

            # Inter-chain contacts need pn_unit identities; without them the metric is
            # undefined (and get_hotspot_contacts would raise).
            if "pn_unit_iid" not in cats:
                logger.warning(
                    "PPIMetrics: no 'pn_unit_iid' annotation on the predicted structure; "
                    "skipping the hotspot contact metric."
                )
                continue

            contacted.append(
                get_hotspot_contacts(
                    atom_array,
                    hotspot_mask=hotspot_mask,
                    distance_cutoff=self.distance_cutoff,
                )
            )

        if not contacted:
            return {}
        return {"fraction_hotspots_contacted": float(np.mean(contacted))}



class HbondMetrics(Metric):
    def __init__(
        self,
        cutoff_HA_dist: float = 2.2,
        cutoff_DA_distance: float = 3.2,
        enabled: bool = True,
    ):
        super().__init__()
        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_distance = cutoff_DA_distance
        self.enabled = enabled
        # One-shot warnings: avoid spamming logs every batch.
        self._warned_disabled = False
        self._warned_no_hbplus = False
        self._warned_failed = False

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "ground_truth_atom_array_stack": ("ground_truth_atom_array_stack",),
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(
        self,
        *,
        ground_truth_atom_array_stack: AtomArrayStack,
        predicted_atom_array_stack: AtomArrayStack,
    ) -> dict[str, float]:
        if not self.enabled:
            if not self._warned_disabled:
                logger.warning("HbondMetrics is disabled (enabled=False); skipping.")
                self._warned_disabled = True
            return {}

        if not os.environ.get("HBPLUS_PATH"):
            if not self._warned_no_hbplus:
                logger.warning(
                    "HbondMetrics: HBPLUS_PATH is not set; hbond metric will be skipped. "
                    "Set HBPLUS_PATH to your hbplus executable to enable."
                )
                self._warned_no_hbplus = True
            return {}

        try:
            d_pct, a_pct, n_hbonds, n_valid = _calculate_hbond_stats(
                ground_truth_atom_array_stack,
                predicted_atom_array_stack,
                cutoff_HA_dist=self.cutoff_HA_dist,
                cutoff_DA_distance=self.cutoff_DA_distance,
            )
        except Exception as e:
            if not self._warned_failed:
                logger.warning(
                    f"HbondMetrics: _calculate_hbond_stats raised {type(e).__name__}: {e}. "
                    "Subsequent failures will be silenced."
                )
                self._warned_failed = True
            return {}

        if n_valid == 0:
            return {}

        return {
            "mean_correct_donors_percent": float(d_pct),
            "mean_correct_acceptors_percent": float(a_pct),
            "mean_num_hbonds": float(n_hbonds),
        }
