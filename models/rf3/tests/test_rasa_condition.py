"""RASA condition: inference selection, training modifier, featurization and adherence metric."""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.ml.conditions.annotator import ensure_annotations
from biotite.structure import stack
from rf3.conditions import C_RAS, RasaEnum
from rf3.data.condition_featurization import add_conditions_as_features
from rf3.data.design_tasks.design_task_modifiers import add_rasa_condition
from rf3.metrics.design_metrics import RasaAdherenceMetric
from rf3.utils.inference import apply_rasa_selection

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def complex_with_ligand():
    """8vkf protein + ligands (HEM, LVK, ions), waters removed; returns the ligand selections."""
    aa = parse(
        TEST_DATA_DIR / "8vkf_from_file.cif",
        hydrogen_policy="remove",
        remove_waters=True,
    )
    aa = aa["assemblies"]["1"][0]
    ensure_annotations(aa, "is_ligand")
    assert aa.is_ligand.any()
    lig = sorted(
        {
            (c, r, i)
            for c, r, i in zip(
                aa.chain_id[aa.is_ligand],
                aa.res_name[aa.is_ligand],
                aa.res_id[aa.is_ligand],
            )
        }
    )
    return aa, [f"{c}/{r}/{i}" for c, r, i in lig]


def test_inference_selection_matches_training_modifier(complex_with_ligand):
    """apply_rasa_selection on the ligand reproduces add_rasa_condition's ligand-only labels."""
    aa, sel = complex_with_ligand
    inf = aa.copy()
    apply_rasa_selection(inf, sel)
    train = add_rasa_condition({"atom_array": aa.copy()}, p_include_non_ligand=0.0)[
        "atom_array"
    ]
    np.testing.assert_array_equal(C_RAS.annotation(inf), C_RAS.annotation(train))
    assert (C_RAS.annotation(inf)[~aa.is_ligand] == RasaEnum.UNSPECIFIED).all()


def test_explicit_labels_and_bad_label(complex_with_ligand):
    aa, sel = complex_with_ligand
    arr = aa.copy()
    apply_rasa_selection(arr, {q: "EXPOSED" for q in sel})
    assert (C_RAS.annotation(arr)[arr.is_ligand] == RasaEnum.EXPOSED).all()
    with pytest.raises((KeyError, ValueError)):
        apply_rasa_selection(aa.copy(), {sel[0]: "buried"})


def test_feature_one_hot_layout(complex_with_ligand):
    """Columns are BURIED/INTERMEDIATE/EXPOSED; unconditioned atoms are all-zero."""
    aa, sel = complex_with_ligand
    arr = aa.copy()
    apply_rasa_selection(arr, sel)
    feat = add_conditions_as_features(arr)[C_RAS.get_full_name()].numpy()
    labels = C_RAS.annotation(arr)
    for col, enum in enumerate(
        (RasaEnum.BURIED, RasaEnum.INTERMEDIATE, RasaEnum.EXPOSED)
    ):
        np.testing.assert_array_equal(feat[:, col], labels == enum)
    assert feat[labels == RasaEnum.UNSPECIFIED].sum() == 0


def test_adherence_is_one_on_ground_truth(complex_with_ligand):
    """Scoring the structure the labels came from must give adherence 1.0."""
    aa, _ = complex_with_ligand
    arr = aa.copy()
    apply_rasa_selection(
        arr, ["*/*/*"]
    )  # auto-bin every atom -> all three classes present
    out = RasaAdherenceMetric().compute(atom_array_stack=stack([arr, arr]))
    labels = C_RAS.annotation(arr)
    for enum in (RasaEnum.BURIED, RasaEnum.INTERMEDIATE, RasaEnum.EXPOSED):
        key = f"{enum.name.lower()}_conditioning_adherence"
        if (labels == enum).any():
            assert out[key] == pytest.approx(1.0)


def test_adherence_skips_unconditioned(complex_with_ligand):
    aa, _ = complex_with_ligand
    assert RasaAdherenceMetric().compute(atom_array_stack=stack([aa])) == {}
