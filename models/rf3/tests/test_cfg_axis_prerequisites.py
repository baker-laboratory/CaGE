"""Pin which condition fields a per-axis CFG guidance stream keeps.

Isolating one axis drops every other condition field. For `interface_type` that is not
enough: a bare label is written onto the atoms `hotspot_selection` picks, so dropping the
hotspots leaves a stream bit-identical to the unconditional one -- guidance of exactly
zero, with nothing in the output to show it. The axis therefore declares `hotspot` as a
prerequisite and keeps it.

The dict form carries its own selections, so it must *not* drag the prerequisite in:
doing so would make a stream nominally guiding interface type also guide hotspots.
"""

import pytest
from rf3.utils.inference import (
    CFG_CONDITION_FIELDS,
    CFG_CONDITION_PREREQUISITES,
    InputOptions,
    cfg_axis_fields_to_keep,
)


def test_only_interface_type_declares_a_prerequisite():
    assert dict(CFG_CONDITION_PREREQUISITES) == {"interface_type": ("hotspot",)}


@pytest.mark.parametrize(
    "axis",
    [a for a in CFG_CONDITION_FIELDS if a not in CFG_CONDITION_PREREQUISITES],
)
def test_independent_axes_keep_only_themselves(axis):
    """Every axis without a prerequisite stays strictly isolated."""
    field = CFG_CONDITION_FIELDS[axis]
    # Select every axis, so a leak would show up as extra kept fields.
    options = InputOptions(
        **{
            f: {"A/*/1": "x"} if f == "interface_type" else ["A"]
            for f in CFG_CONDITION_FIELDS.values()
        }
    )
    assert cfg_axis_fields_to_keep(options, axis, field) == {field}


def test_scalar_interface_type_keeps_its_hotspot_selection():
    options = InputOptions(interface_type="PPI", hotspot_selection=["A/*/10"])
    assert cfg_axis_fields_to_keep(options, "interface_type", "interface_type") == {
        "interface_type",
        "hotspot_selection",
    }


def test_scalar_interface_type_without_hotspots_raises():
    """Better a hard error than a guidance stream that silently carries no condition."""
    options = InputOptions(interface_type="PPI")
    with pytest.raises(ValueError, match="hotspot_selection"):
        cfg_axis_fields_to_keep(options, "interface_type", "interface_type")


def test_dict_interface_type_stands_alone():
    """The per-selection form needs no prerequisite -- and must not pull one in."""
    options = InputOptions(interface_type={"A/*/10": "PPI"})
    assert cfg_axis_fields_to_keep(options, "interface_type", "interface_type") == {
        "interface_type"
    }


def test_dict_interface_type_ignores_a_present_hotspot_selection():
    """Regression: keeping hotspots here would make this stream guide hotspots too."""
    options = InputOptions(
        interface_type={"A/*/10": "PPI"}, hotspot_selection=["A/*/20"]
    )
    assert cfg_axis_fields_to_keep(options, "interface_type", "interface_type") == {
        "interface_type"
    }
