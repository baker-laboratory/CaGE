"""Per-stream padding masks across the CFG plumbing.

Each guidance stream is featurized and collated on its own, so a feature whose length
is example-dependent is padded to its own length. `num_chiral_centers` is one: the
chirals feature is read off a ligand reference conformer that RDKit generates from the
global RNG, so two streams of the same input can disagree on how many chiral centers
there are. Indexing one stream's chirals with another's padding mask is what
`calc_chiral_grads_flat_impl` raises `IndexError` on.

Hermetic: the helper takes plain mask dicts, and the sampler checks drive
`SampleDiffusion.sample_diffusion_like_af3` with dummy diffusion modules. No GPU,
checkpoint, or structure mirror.
"""

import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion
from rf3.model.layers.utils.chirals import calc_chiral_grads_flat_impl
from rf3.model.RF3 import _guidance_stream_padding_masks

N_TOKENS, N_ATOMS, N_DIFF, C = 4, 6, 2, 8


# --------------------------------------------------------------------------- #
# _guidance_stream_padding_masks
# --------------------------------------------------------------------------- #
def _masks(n_tokens=N_TOKENS, n_atoms=N_ATOMS, n_chiral=3) -> dict:
    """A collated padding-mask dict, in the spelling `collate_batch` produces."""
    return {
        "I": torch.zeros(1, n_tokens, dtype=torch.bool),
        "L": torch.zeros(1, n_atoms, dtype=torch.bool),
        "num_chiral_centers": torch.zeros(1, n_chiral, dtype=torch.bool),
    }


def test_stream_with_own_masks_keeps_them():
    """A differing chiral length is the point of the fix, not an error."""
    main, own = _masks(n_chiral=3), _masks(n_chiral=7)
    out = _guidance_stream_padding_masks({"padding_masks": own}, main)
    assert out is own
    assert out["num_chiral_centers"].shape == (1, 7)


def test_stream_without_own_masks_falls_back_to_main():
    main = _masks()
    assert _guidance_stream_padding_masks({"padding_masks": None}, main) is main
    assert _guidance_stream_padding_masks({}, main) is main


@pytest.mark.parametrize(
    ("key", "differing"),
    [("I", {"n_tokens": N_TOKENS + 1}), ("L", {"n_atoms": N_ATOMS + 3})],
)
def test_token_or_atom_mismatch_raises_naming_both_shapes(key, differing):
    """The streams share one set of coordinates, so I/L disagreeing is a bug."""
    main = _masks()
    ref = {"name": "hotspot_selection", "padding_masks": _masks(**differing)}
    with pytest.raises(ValueError) as excinfo:
        _guidance_stream_padding_masks(ref, main)
    message = str(excinfo.value)
    assert "hotspot_selection" in message
    assert key in message
    assert str(tuple(ref["padding_masks"][key].shape)) in message
    assert str(tuple(main[key].shape)) in message


def test_unnamed_stream_is_still_identifiable():
    """The structure path carries no stream name; the message must not read 'None'."""
    with pytest.raises(ValueError, match="<unnamed>"):
        _guidance_stream_padding_masks(
            {"padding_masks": _masks(n_tokens=N_TOKENS + 1)}, _masks()
        )


def test_partial_mask_dict_is_not_a_keyerror():
    """A hand-built stream dict may carry only the keys its caller cares about."""
    main = _masks()
    only_chiral = {"num_chiral_centers": torch.zeros(1, 7, dtype=torch.bool)}
    assert _guidance_stream_padding_masks({"padding_masks": only_chiral}, main) is (
        only_chiral
    )


# --------------------------------------------------------------------------- #
# sampler: is_padding_chiral is taken per stream
# --------------------------------------------------------------------------- #
def _sampler() -> SampleDiffusion:
    return SampleDiffusion(
        num_timesteps=2,
        min_t=0,
        max_t=1,
        sigma_data=16,
        s_min=4e-4,
        s_max=160,
        p=7,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        solver="af3",
    )


def _trunk(signal: float = 0.0, n_chiral: int | None = None) -> dict:
    """Trunk tensors plus features. `signal` labels the stream in recorded calls."""
    f = {"ref_element": torch.zeros(1, N_ATOMS, 4)}
    if n_chiral is not None:
        f["chiral_centers"] = torch.randint(0, N_ATOMS, (1, n_chiral, 4))
        f["chiral_center_dihedral_angles"] = torch.zeros(1, n_chiral)
    return {
        "f": f,
        "S_inputs_I": torch.full((1, N_TOKENS, C), signal),
        "S_trunk_I": torch.full((1, N_TOKENS, C), signal),
        "Z_trunk_II": torch.full((1, N_TOKENS, N_TOKENS, C), signal),
    }


def _stream(signal: float, n_chiral: int | None, carry_mask: bool = True) -> dict:
    stream = {**_trunk(signal, n_chiral), "scale": 1.0}
    if carry_mask and n_chiral is not None:
        stream["is_padding_chiral"] = torch.zeros(1, n_chiral, dtype=torch.bool)
    return stream


def _run(dummy, streams: list[dict], main_chiral: int, main_f_chirals: bool = False):
    torch.manual_seed(0)
    return _sampler().sample_diffusion_like_af3(
        **_trunk(0.0, main_chiral if main_f_chirals else None),
        diffusion_module=dummy,
        diffusion_batch_size=N_DIFF,
        coord_atom_lvl_to_be_noised=torch.zeros(N_DIFF, N_ATOMS, 3),
        is_padding_L=torch.zeros(1, N_ATOMS, dtype=torch.bool),
        is_padding_I=torch.zeros(1, N_TOKENS, dtype=torch.bool),
        is_padding_chiral=torch.zeros(1, main_chiral, dtype=torch.bool),
        inference_classifier_free_guidance=True,
        cfg_guidance_streams=streams,
    )


class _MaskRecordingDummy:
    """Records (stream label, chiral mask width) for every forward."""

    def __init__(self):
        self.calls: list[tuple[float, int]] = []

    def __call__(
        self,
        *,
        X_noisy_L,
        t,
        f,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        is_padding_L,
        is_padding_I,
        is_padding_chiral,
    ):
        self.calls.append(
            (round(float(S_inputs_I.mean()), 3), int(is_padding_chiral.shape[1]))
        )
        return 0.9 * X_noisy_L


def test_sampler_hands_each_stream_its_own_chiral_padding():
    dummy = _MaskRecordingDummy()
    _run(dummy, [_stream(signal=0.5, n_chiral=7)], main_chiral=3)
    assert (0.0, 3) in dummy.calls, "the main forward keeps the main mask"
    assert (0.5, 7) in dummy.calls, "the guidance forward gets the stream's mask"
    assert (0.5, 3) not in dummy.calls, "the stream must never see the main mask"


def test_sampler_falls_back_when_a_stream_carries_no_chiral_padding():
    """Streams built by older callers (and by the other CFG tests) still work."""
    dummy = _MaskRecordingDummy()
    _run(dummy, [_stream(signal=0.5, n_chiral=None)], main_chiral=3)
    assert {width for _, width in dummy.calls} == {3}


def test_each_stream_gets_its_own_mask_with_several_streams():
    dummy = _MaskRecordingDummy()
    _run(
        dummy,
        [_stream(signal=0.5, n_chiral=7), _stream(signal=0.25, n_chiral=2)],
        main_chiral=3,
    )
    assert (0.5, 7) in dummy.calls
    assert (0.25, 2) in dummy.calls


# --------------------------------------------------------------------------- #
# end to end: the IndexError this fix removes
# --------------------------------------------------------------------------- #
class _ChiralGradDummy:
    """Reproduces the real module's coupling between `f` and `is_padding_chiral`.

    The chirals feature comes from this stream's `f`; its padding comes from
    `is_padding_chiral`. `calc_chiral_grads_flat_impl` indexes the one with the other,
    which is where a length disagreement surfaces.
    """

    def __call__(
        self,
        *,
        X_noisy_L,
        t,
        f,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        is_padding_L,
        is_padding_I,
        is_padding_chiral,
    ):
        calc_chiral_grads_flat_impl(
            X_noisy_L.detach(),
            f["chiral_centers"],
            f["chiral_center_dihedral_angles"],
            False,
            is_padding_chiral,
            is_padding_L,
        )
        return 0.9 * X_noisy_L


def test_streams_with_different_chiral_lengths_complete_without_indexerror():
    """Two streams, 3 vs 5 chiral centers -- the shape the conformer RNG produces."""
    out = _run(
        _ChiralGradDummy(),
        [_stream(signal=0.5, n_chiral=5)],
        main_chiral=3,
        main_f_chirals=True,
    )
    assert torch.isfinite(out["X_L"]).all()


def test_reusing_the_main_mask_still_raises_so_the_check_above_is_live():
    """Guard the guard: the failure mode is real, not retired by something else."""
    with pytest.raises(IndexError, match="shape of the mask"):
        calc_chiral_grads_flat_impl(
            torch.zeros(1, N_DIFF, N_ATOMS, 3),
            torch.randint(0, N_ATOMS, (1, 5, 4)),
            torch.zeros(1, 5),
            False,
            torch.zeros(1, 3, dtype=torch.bool),  # the main stream's mask
            torch.zeros(1, N_ATOMS, dtype=torch.bool),
        )
