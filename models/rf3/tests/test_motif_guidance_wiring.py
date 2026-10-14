"""The motif_guidance kwarg must reach the sampler from every inference entry point.

A half-threaded kwarg is silent: the engine is built, the config says guidance is on,
and the run completes unguided with nothing in the log. These checks pin the thread at
each hop so a dropped parameter fails at collection time rather than in a result table.
"""

import inspect

from rf3.diffusion_samplers.inference_sampler import SampleDiffusion
from rf3.model.RF3 import RF3, RF3WithConfidence
from rf3.trainers.rf3 import RF3Trainer, RF3TrainerWithConfidence

HOPS = [
    (SampleDiffusion.sample_diffusion_like_af3, "sampler"),
    (RF3.forward, "RF3.forward"),
    (
        RF3WithConfidence._run_rollout_no_grad,
        "RF3WithConfidence._run_rollout_no_grad",
    ),
    (RF3WithConfidence.forward, "RF3WithConfidence.forward"),
    (RF3Trainer.validation_step, "RF3Trainer.validation_step"),
    (
        RF3TrainerWithConfidence.validation_step,
        "RF3TrainerWithConfidence.validation_step",
    ),
]


def test_every_hop_accepts_motif_guidance_defaulting_to_none():
    for fn, label in HOPS:
        params = inspect.signature(fn).parameters
        assert "motif_guidance" in params, f"{label} drops the motif_guidance kwarg"
        assert (
            params["motif_guidance"].default is None
        ), f"{label} must default motif_guidance to None so existing callers are inert"


def test_every_hop_forwards_it_onward():
    """Accepting the kwarg is not enough -- each hop must pass it to the next."""
    for fn, label in HOPS:
        if fn is SampleDiffusion.sample_diffusion_like_af3:
            continue  # terminal hop: it consumes the engine rather than forwarding it
        src = inspect.getsource(fn)
        assert (
            "motif_guidance=motif_guidance" in src
        ), f"{label} accepts motif_guidance but never forwards it"


def test_rollout_cache_is_disabled_under_motif_guidance():
    """The cache key covers (base weights, example_id) only, so a guided run must not
    read or write it -- otherwise guidance is silently replaced by a cached unguided
    rollout and a mu sweep returns identical results."""
    src = inspect.getsource(RF3WithConfidence.forward)
    assert "and motif_guidance is None" in src


def test_training_mini_rollout_refuses_guidance():
    src = inspect.getsource(RF3WithConfidence._run_rollout_no_grad)
    assert "motif_guidance is inference-only" in src
    # The engine must not be folded into the kwargs the mini-rollout shares.
    base_kwargs_block = src.split("sample_diffusion_kwargs = dict(")[1].split(")")[0]
    assert "motif_guidance" not in base_kwargs_block
