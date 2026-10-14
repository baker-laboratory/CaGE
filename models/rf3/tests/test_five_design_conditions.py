"""The five design conditions, end to end on a bundled structure.

Covers, for interface hotspots (C_HOT), interface type (C_ITYPE), RASA (C_RAS),
hydrogen-bond identities (C_HB) and atom-pair distances (C_DIS):

* every enum level is reachable through ``apply_design_conditions`` -- the function the
  inference path calls -- and survives into the feature tensors, so none of the five
  ships with a dead label or a silently ignored value;
* the five can be requested together without clobbering each other;
* the labels the inference entry points write are bit-identical to the ones the
  training-time task writes (checked on hotspot + interface type, the pair the
  training task emits together).

CPU only, no PDB mirror, no checkpoint, no hbplus.
"""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_DIS  # rf3.conditions does not re-export C_DIS
from rf3.conditions import (
    C_HB,
    C_HOT,
    C_ITYPE,
    C_RAS,
    HBondEnum,
    ItypeEnum,
    RasaEnum,
)
from rf3.data.condition_featurization import add_conditions_as_features
from rf3.data.design_tasks.ppi import PPIHotspot
from rf3.utils.inference import (
    apply_design_conditions,
    apply_hotspot_selection,
    apply_interface_type,
)

TEST_DATA_DIR = Path(__file__).parent / "data"
CIF = TEST_DATA_DIR / "5vht_from_file.cif"

HOTSPOT_SEL = ["A/*/20-25"]


@pytest.fixture(scope="module")
def dimer():
    """5vht chains A/B (protein-protein), first assembly model."""
    parsed = parse(CIF, hydrogen_policy="remove", remove_waters=True)
    asm = parsed["assemblies"]
    return as_atom_array_plus(asm[sorted(asm)[0]][0])


def _feature_sum(atom_array, condition) -> int:
    return int(
        add_conditions_as_features(atom_array)[condition.get_full_name()].numpy().sum()
    )


def test_hotspot_condition_reaches_features(dimer):
    out = apply_design_conditions(dimer.copy(), hotspot_selection=HOTSPOT_SEL)
    mask = C_HOT.annotation(out).astype(bool)
    assert mask.sum() > 0
    np.testing.assert_array_equal(
        add_conditions_as_features(out)[C_HOT.get_full_name()]
        .numpy()
        .astype(bool)
        .reshape(-1),
        mask,
    )


@pytest.mark.parametrize(
    "label", [e.name for e in ItypeEnum if e is not ItypeEnum.NONE]
)
def test_every_interface_type_level_is_live(dimer, label):
    out = apply_design_conditions(
        dimer.copy(), hotspot_selection=HOTSPOT_SEL, interface_type=label
    )
    hotspot = C_HOT.annotation(out).astype(bool)
    np.testing.assert_array_equal(
        C_ITYPE.annotation(out) == ItypeEnum[label].value, hotspot
    )
    assert _feature_sum(out, C_ITYPE) == int(hotspot.sum())


@pytest.mark.parametrize(
    "label", [e.name for e in RasaEnum if e is not RasaEnum.UNSPECIFIED]
)
def test_every_rasa_level_is_live(dimer, label):
    out = apply_design_conditions(dimer.copy(), rasa_selection={"A/*/20-25": label})
    assert (C_RAS.annotation(out) == RasaEnum[label].value).sum() > 0
    assert _feature_sum(out, C_RAS) > 0


def test_rasa_list_form_bins_from_coordinates(dimer):
    """The list form carries no label: RASA is computed from the input coordinates."""
    out = apply_design_conditions(dimer.copy(), rasa_selection=["A/*/20-25"])
    labels = C_RAS.annotation(out)
    assert (labels != RasaEnum.UNSPECIFIED).sum() > 0
    assert set(np.unique(labels[labels != RasaEnum.UNSPECIFIED]).tolist()) <= {
        RasaEnum.BURIED.value,
        RasaEnum.INTERMEDIATE.value,
        RasaEnum.EXPOSED.value,
    }


@pytest.mark.parametrize(
    "label", [e.name for e in HBondEnum if e is not HBondEnum.NONE]
)
def test_every_hbond_level_is_live(dimer, label):
    out = apply_design_conditions(dimer.copy(), hbond_selection={"A/*/28/N": label})
    assert (C_HB.annotation(out) == HBondEnum[label].value).sum() == 1
    assert _feature_sum(out, C_HB) == 1


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(["A/*/28", "A/*/30"], id="flat_list_from_coordinates"),
        pytest.param(
            [{"atoms": ["A/*/28/CA", "A/*/30/CA"], "distance": 7.5}], id="scalar"
        ),
        pytest.param(
            {
                "atoms": ["A/*/28/CA", "A/*/30/CA", "A/*/32/CA"],
                "distance": [[0.0, 5.0, 8.0], [5.0, 0.0, 6.0], [8.0, 6.0, 0.0]],
            },
            id="n_by_n_matrix",
        ),
    ],
)
def test_every_distance_semantics_writes_pairs(dimer, selection):
    out = apply_design_conditions(dimer.copy(), distance_template_selection=selection)
    feats = add_conditions_as_features(out)
    mask = feats[C_DIS.get_mask_name()].numpy().astype(bool)
    values = feats[C_DIS.get_full_name()].numpy()
    assert mask.sum() > 0
    assert np.isfinite(values[mask]).all()
    assert (values[mask] > 0).all()


def test_all_five_conditions_coexist(dimer):
    """Requesting all five at once must leave all five readable (they share one AtomArray)."""
    out = apply_design_conditions(
        dimer.copy(),
        hotspot_selection=HOTSPOT_SEL,
        interface_type="PPI",
        rasa_selection={"A/*/40": "EXPOSED"},
        hbond_selection={"A/*/28/N": "DONOR"},
        distance_template_selection=[
            {"atoms": ["A/*/28/CA", "A/*/30/CA"], "distance": 7.5}
        ],
    )
    feats = add_conditions_as_features(out)
    assert feats[C_HOT.get_full_name()].numpy().sum() > 0
    assert feats[C_ITYPE.get_full_name()].numpy().sum() > 0
    assert feats[C_RAS.get_full_name()].numpy().sum() > 0
    assert feats[C_HB.get_full_name()].numpy().sum() == 1
    assert feats[C_DIS.get_mask_name()].numpy().astype(bool).sum() == 2


def test_inference_selections_reproduce_the_training_task_labels(dimer):
    """What ``PPIHotspot`` writes during training, the inference selections write too.

    The training loader carries a unique per-atom ``atom_id`` (``sample_hotspots`` grows
    each seed to a single atom through it); the bundled assembly parse does not, so the
    fixture adds it before driving the task.
    """
    train = dimer.copy()
    train.set_annotation("atom_id", np.arange(train.array_length()))
    task = PPIHotspot(annotate_interface_type=True)
    data = task.annotate_for_task_selection(
        {
            "example_id": "5vht",
            "atom_array": train,
            "query_pn_unit_iids": ["A_1", "B_1"],
        }
    )
    assert task.can_apply(data)
    np.random.seed(0)
    train = task.forward(data)["atom_array"]

    train_hotspot = C_HOT.annotation(train).astype(bool)
    train_itype = C_ITYPE.annotation(train)
    assert train_hotspot.sum() > 0
    # 5vht is a protein-protein pair, so the task must label the interface PPI.
    assert set(np.unique(train_itype[train_hotspot]).tolist()) == {ItypeEnum.PPI.value}

    selection = [
        f"{chain}/*/{int(res_id)}/{atom_name}"
        for chain, res_id, atom_name in zip(
            train.chain_id[train_hotspot],
            train.res_id[train_hotspot],
            train.atom_name[train_hotspot],
        )
    ]
    inference = dimer.copy()
    hotspot_mask = apply_hotspot_selection(inference, selection)
    apply_interface_type(inference, "PPI", hotspot_mask)

    np.testing.assert_array_equal(
        C_HOT.annotation(inference).astype(bool), train_hotspot
    )
    np.testing.assert_array_equal(C_ITYPE.annotation(inference), train_itype)
    for condition in (C_HOT, C_ITYPE):
        np.testing.assert_array_equal(
            add_conditions_as_features(inference)[condition.get_full_name()].numpy(),
            add_conditions_as_features(train)[condition.get_full_name()].numpy(),
        )
