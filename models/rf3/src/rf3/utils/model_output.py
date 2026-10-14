"""Shape conventions at the trainer-to-inference boundary."""

from typing import Any

import torch


def normalize_inference_network_output(network_output: dict[str, Any]) -> dict[str, Any]:
    """Remove only the structure batch axis from final predicted coordinates.

    The sampler returns coordinates as ``[B, D, L, 3]`` (structure batch,
    diffusion samples, atoms, xyz). Inference processes one structure at a time,
    so B must be 1. The confidence trainer may already remove B when computing
    metrics, leaving ``[D, L, 3]``; retain that shape even when D is 1.

    Confidence logits, including ``confidence_by_stream``, already start with D.
    Trajectory frames retain B for their writer. Neither should be squeezed.
    Return a shallow copy so the trainer's output dictionary is not modified.
    """
    normalized = dict(network_output)
    for key in ("X_L", "X_pred_rollout_L"):
        coords = network_output.get(key)
        if coords is None:  # Also permits early-stopped outputs without coordinates.
            continue
        if not isinstance(coords, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(coords).__name__}")
        if coords.ndim == 4 and coords.shape[0] == 1:
            coords = coords[0]
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"{key} must have shape [D, L, 3] or [1, D, L, 3] during "
                f"inference, got {tuple(network_output[key].shape)}"
            )
        normalized[key] = coords
    return normalized
