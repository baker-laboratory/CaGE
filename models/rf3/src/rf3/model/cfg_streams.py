"""Trunk-stream helpers for classifier-free guidance.

Pure arithmetic on whatever the caller passes (tensors in practice), so this module
stays importable without torch and the combination rule can be unit-tested directly.
"""

#: A trunk stream spells its tensors differently depending on where it came from:
#: ``trunk_forward_with_recycling`` output vs. a CFG guidance stream.
_TRUNK_KEYS = (
    ("S_inputs_I", "S_inputs_I"),
    ("S_I", "S_trunk_I"),
    ("Z_II", "Z_trunk_II"),
)


def trunk_tensors(stream: dict) -> tuple:
    """Pull (S_inputs_I, S_trunk_I, Z_trunk_II) out of either spelling."""
    return tuple(
        stream[recycling_key] if recycling_key in stream else stream[stream_key]
        for recycling_key, stream_key in _TRUNK_KEYS
    )


def extrapolate_trunk_streams(recycling_outputs: dict, guidance_streams: list) -> dict:
    """Combine trunk features the way CFG combines scores: main + Σ scale·(ref − main).

    Approximates the representation the guided sample corresponds to. The confidence head
    only ever saw real trunk features in training, so the result is indicative, not
    calibrated.
    """
    base = trunk_tensors(recycling_outputs)
    merged = list(base)
    for stream in guidance_streams:
        scale = float(stream["scale"])
        for i, ref in enumerate(trunk_tensors(stream)):
            # Every delta is measured against the unconditional base, matching the
            # sampler; accumulating against the running total would compound them.
            merged[i] = merged[i] + scale * (ref - base[i])
    return {"S_inputs_I": merged[0], "S_I": merged[1], "Z_II": merged[2]}
