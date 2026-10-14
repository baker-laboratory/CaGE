"""Motif guidance must stay out of the trained-condition machinery.

It targets the same goal as ``distance_template`` -- hold a motif in place -- by a
completely different mechanism: an untrained projection on x0 at sampling time, versus
a trained pairwise-distance condition featurized before the trunk. Letting the two
share a registry, a field or a code path would be actively harmful, so each boundary
gets an assertion.
"""

import dataclasses
import inspect

from rf3.utils.inference import (
    CFG_CONDITION_FIELDS,
    CONDITION_FIELDS,
    InferenceInput,
    InputOptions,
)

MOTIF_FIELDS = ("motif_guidance_selection", "motif_guidance_reference")


def test_declared_on_both_dataclasses():
    """Declared fields, not ad-hoc attributes: dataclasses.replace drops the latter."""
    for cls in (InputOptions, InferenceInput):
        names = {f.name for f in dataclasses.fields(cls)}
        for field in MOTIF_FIELDS:
            assert field in names, f"{cls.__name__} is missing {field}"


def test_not_a_cfg_guidable_axis():
    """A CFG stream keeps only its own axis and nulls the rest. Motif guidance is not
    featurized, so its stream would be identical to the unconditional one -- zero
    guidance at double the compute, with nothing to show it."""
    assert "motif_guidance" not in CFG_CONDITION_FIELDS
    for field in MOTIF_FIELDS:
        assert field not in CFG_CONDITION_FIELDS.values()
        assert field not in CONDITION_FIELDS


def test_survives_the_condition_nulling_used_to_build_cfg_streams():
    spec = InferenceInput(
        atom_array=None,
        chain_info={},
        example_id="x",
        distance_template_selection=["A/HIS/10/CA"],
        motif_guidance_selection=["A/HIS/10/CA"],
        motif_guidance_reference="ref.cif",
    )
    stream_spec = dataclasses.replace(spec, **{f: None for f in CONDITION_FIELDS})
    assert stream_spec.distance_template_selection is None
    # The projection runs once per step on the combined x0, not per stream, so every
    # stream's spec must keep the motif fields.
    assert stream_spec.motif_guidance_selection == ["A/HIS/10/CA"]
    assert stream_spec.motif_guidance_reference == "ref.cif"


def test_never_enters_the_pipeline_input():
    """to_pipeline_input is where conditions become annotations and where
    distance_template atomizes whole residues. Motif guidance resolves later, against
    the post-pipeline atom array, so it must not appear here at all."""
    src = inspect.getsource(InferenceInput.to_pipeline_input)
    assert "motif_guidance" not in src


def test_never_triggers_atomization():
    src = inspect.getsource(InferenceInput.from_json_dict)
    atomize_calls = [
        line
        for line in src.splitlines()
        if "_atomize_distance_template_residues" in line
    ]
    assert (
        atomize_calls
    ), "expected the distance-template atomization call to still exist"
    for line in atomize_calls:
        assert "motif_guidance" not in line
