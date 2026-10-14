"""Motif guidance: a sampler-side motif constraint applied during inference.

Distinct from the trained design conditions (see ``rf3.conditions`` and
``distance_template_selection``): nothing here is featurized or trained on, it never
reaches the denoiser, and it is deliberately absent from ``CFG_CONDITION_FIELDS``.
"""

from rf3.guidance.engine import MotifGuidanceCfg, MotifGuidanceEngine
from rf3.guidance.feats import (
    build_motif_guidance_feats,
    load_motif_guidance_reference,
    normalize_motif_guidance_groups,
)
from rf3.guidance.potential import MotifAnchorPotential

__all__ = [
    "MotifAnchorPotential",
    "MotifGuidanceCfg",
    "MotifGuidanceEngine",
    "build_motif_guidance_feats",
    "load_motif_guidance_reference",
    "normalize_motif_guidance_groups",
]
