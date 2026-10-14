"""On-disk cache for confidence-head *validation* rollouts.

The trunk recycling + full diffusion rollout is by far the most expensive part of confidence-head
validation. Because the trunk and diffusion module are frozen during confidence-head training,
their rollout outputs for a given validation example do **not** depend on the confidence-head
weights being trained -- so rollouts can be shared across confidence checkpoints that were trained
from the *same* frozen base. Validation also runs uncropped (``crop_size=null``), so each example's
token set is deterministic across epochs.

**Base-checkpoint dependence (important):** the rollout DOES depend on the frozen base (trunk +
diffusion) weights. Two different base models (e.g. run11conf vs run14) produce completely different
rollouts for the same ``example_id``. To make it impossible to reuse a rollout computed with the
wrong base, the cache is namespaced by a fingerprint of the frozen (non-confidence) weights:
``<cache_dir>/<base_fingerprint>/<example_id>.pt``. Same base -> same fingerprint -> reuse;
different base -> different fingerprint -> automatically separate files (a cache *miss*, recomputed).

Cached per example (all tensors): recycling reps ``S_inputs_I``, ``S_I``, ``Z_II``; the rollout
coordinates ``X_L`` (== ``X_pred_rollout_L``); the trunk ``distogram_pred``. The per-step diffusion
trajectories are NOT cached (large; only used for optional structure dumping) and are returned as
empty lists on a hit.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

# All cached values are single tensors.
_RECYCLING_KEYS = ("S_inputs_I", "S_I", "Z_II")
_DIFFUSION_TENSOR_KEYS = ("X_L",)
# Trajectory keys returned as empty lists on a cache hit (not cached; not used by metrics).
_EMPTY_DIFFUSION_KEYS = ("X_noisy_L_traj", "X_denoised_L_traj", "t_hats")

# Number of tensors sampled + elements per tensor for the base fingerprint (cheap but distinctive).
_FP_N_TENSORS = 6
_FP_ELEMS_PER_TENSOR = 4096


def compute_base_fingerprint(model: torch.nn.Module) -> str:
    """Fingerprint the frozen base (trunk + diffusion) weights.

    Hashes a slice of a few deterministically-chosen **non-confidence** float parameters, so the
    fingerprint changes iff the frozen base weights change. Cheap (hashes a few KB), stable across
    calls for a given loaded checkpoint, and shared across confidence checkpoints from the same base.
    """
    return compute_base_fingerprint_from_state_dict(model.state_dict())


def compute_base_fingerprint_from_state_dict(sd: dict[str, torch.Tensor]) -> str:
    """Same as :func:`compute_base_fingerprint`, but from an already-loaded state dict.

    Lets callers fingerprint a checkpoint straight off disk (e.g. to verify it matches
    a model's current weights) without constructing/loading it into an ``nn.Module`` first.
    """
    keys = sorted(
        k
        for k, v in sd.items()
        if "confidence_head" not in k and torch.is_tensor(v) and v.is_floating_point() and v.numel() > 0
    )
    if not keys:
        return "nobase"
    n = len(keys)
    # Spread the sampled tensors across the sorted key list (feature_init -> recycler -> diffusion).
    idxs = sorted({round(i * (n - 1) / max(1, _FP_N_TENSORS - 1)) for i in range(_FP_N_TENSORS)})
    h = hashlib.sha256()
    for i in idxs:
        k = keys[i]
        flat = sd[k].detach().to("cpu", dtype=torch.float32).contiguous().flatten()
        h.update(k.encode())
        h.update(flat[:_FP_ELEMS_PER_TENSOR].numpy().tobytes())
    return h.hexdigest()[:16]


def _cache_path(cache_dir: str | os.PathLike, namespace: str, key: str) -> Path:
    """Return ``<cache_dir>/<namespace>/<sanitized key>.pt``."""
    safe_key = str(key).replace("/", "__").replace(os.sep, "__")
    safe_ns = str(namespace).replace("/", "__").replace(os.sep, "__")
    return Path(cache_dir) / safe_ns / f"{safe_key}.pt"


def try_load_rollout_cache(
    cache_dir: str | os.PathLike, namespace: str, key: str, device: torch.device | str
) -> dict[str, Any] | None:
    """Load a cached rollout for ``(namespace, key)`` onto ``device``, or ``None`` on a miss.

    ``namespace`` is the base fingerprint (see :func:`compute_base_fingerprint`); a mismatch simply
    yields a different path (miss), so rollouts from a different base are never reused.
    """
    path = _cache_path(cache_dir, namespace, key)
    if not path.exists():
        return None
    blob = torch.load(path, map_location=device)
    sample_diffusion_outs: dict[str, Any] = {k: blob[k] for k in _DIFFUSION_TENSOR_KEYS}
    for k in _EMPTY_DIFFUSION_KEYS:
        sample_diffusion_outs[k] = []
    return {
        "recycling_outputs": {k: blob[k] for k in _RECYCLING_KEYS},
        "sample_diffusion_outs": sample_diffusion_outs,
        "distogram_pred": blob["distogram_pred"],
    }


def save_rollout_cache(
    cache_dir: str | os.PathLike, namespace: str, key: str, rollout: dict[str, Any]
) -> None:
    """Atomically write the essential rollout tensors to ``<cache_dir>/<namespace>/<key>.pt``.

    Tensors are detached and moved to CPU so the cache is portable and does not pin GPU memory.
    Writes to a temp file then ``os.replace`` so a crash mid-write cannot leave a partial cache.
    """
    path = _cache_path(cache_dir, namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # Concurrent SLURM array tasks race to create the same namespace dir; on NFS,
        # exist_ok=True's own is_dir() recheck can still lose this race (stale attribute
        # cache across nodes) even though the directory now genuinely exists.
        pass
    blob: dict[str, torch.Tensor] = {}
    for k in _RECYCLING_KEYS:
        blob[k] = rollout["recycling_outputs"][k].detach().to("cpu")
    for k in _DIFFUSION_TENSOR_KEYS:
        blob[k] = rollout["sample_diffusion_outs"][k].detach().to("cpu")
    blob["distogram_pred"] = rollout["distogram_pred"].detach().to("cpu")

    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp_path)
    os.replace(tmp_path, path)
