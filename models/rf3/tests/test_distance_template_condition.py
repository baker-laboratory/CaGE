"""2D distance condition (C_DIS) from inference selections, and the automatic residue atomization."""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.ml.conditions import C_DIS  # rf3.conditions does not re-export C_DIS
from rf3.data.condition_featurization import add_conditions_as_features
from rf3.utils.inference import (
    InferenceInput,
    InputOptions,
    apply_distance_template_selection,
    atomize_selection_on_atom_array,
)

TEST_DATA_DIR = Path(__file__).parent / "data"
CIF = TEST_DATA_DIR / "5vht_from_file.cif"


@pytest.fixture(scope="module")
def dimer():
    asm = parse(CIF)["assemblies"]
    return asm[sorted(asm)[0]][0]


def _idx(aa, chain, res_id, atom_name):
    (i,) = np.flatnonzero(
        (aa.chain_id == chain) & (aa.res_id == res_id) & (aa.atom_name == atom_name)
    )
    return int(i)


def _res_mask(aa, chain, res_id):
    return (aa.chain_id == chain) & (aa.res_id == res_id)


def test_flat_selection_uses_input_coordinates(dimer):
    out = apply_distance_template_selection(dimer.copy(), ["A/*/28", "A/*/30"])
    annot = C_DIS.annotation(out)
    sel = np.flatnonzero(_res_mask(out, "A", 28) | _res_mask(out, "A", 30))
    assert annot.pairs.shape[1] == 2 and len(annot.pairs) > 0
    assert set(np.unique(annot.pairs)) <= set(sel.tolist())
    pairs = {tuple(p) for p in annot.pairs.tolist()}
    assert all(
        (b, a) in pairs for a, b in pairs
    )  # both orientations, as in compute_distances
    expected = np.linalg.norm(
        out.coord[annot.pairs[:, 0]] - out.coord[annot.pairs[:, 1]], axis=-1
    )
    np.testing.assert_allclose(annot.values, expected, atol=1e-4)
    assert not np.isnan(annot.values).any()


def test_scalar_and_matrix_distances(dimer):
    out = apply_distance_template_selection(
        dimer.copy(), [{"atoms": ["A/*/28/CA", "A/*/30/CA"], "distance": 7.5}]
    )
    np.testing.assert_allclose(C_DIS.annotation(out).values, 7.5)

    atoms = ["A/*/28/CA", "A/*/30/CA", "A/*/32/CA"]
    mat = [[0.0, 4.0, 5.0], [4.0, 0.0, 6.0], [5.0, 6.0, 0.0]]
    out = apply_distance_template_selection(
        dimer.copy(), {"atoms": atoms, "distance": mat}
    )
    annot = C_DIS.annotation(out)
    idx = [_idx(out, "A", r, "CA") for r in (28, 30, 32)]
    got = {tuple(p): v for p, v in zip(annot.pairs.tolist(), annot.values.tolist())}
    assert len(got) == 6  # off-diagonal, both orientations
    for a in range(3):
        for b in range(3):
            if a != b:
                assert got[(idx[a], idx[b])] == pytest.approx(mat[a][b])


def test_matrix_requires_single_atom_selections(dimer):
    with pytest.raises(ValueError, match="exactly one atom"):
        apply_distance_template_selection(
            dimer.copy(),
            {"atoms": ["A/*/28", "A/*/30/CA"], "distance": [[0, 5], [5, 0]]},
        )
    with pytest.raises(ValueError, match="shape"):
        apply_distance_template_selection(
            dimer.copy(),
            {
                "atoms": ["A/*/28/CA", "A/*/30/CA"],
                "distance": [[0, 5, 1], [5, 0, 1], [1, 1, 0]],
            },
        )


def test_c_dis_reaches_features(dimer):
    out = apply_distance_template_selection(
        dimer.copy(), [{"atoms": ["A/*/28/CA", "A/*/30/CA"], "distance": 7.5}]
    )
    feats = add_conditions_as_features(out)
    mask = feats[C_DIS.get_mask_name()].numpy().astype(bool)
    dis = feats[C_DIS.get_full_name()].numpy()
    assert int(mask.sum()) == 2
    np.testing.assert_allclose(dis[mask], 7.5)


def test_atomize_whole_residue_from_side_chain_atom(dimer):
    out = atomize_selection_on_atom_array(dimer.copy(), ["A/*/28/CB"])
    atomize = out.get_annotation("atomize").astype(bool)
    np.testing.assert_array_equal(atomize, _res_mask(out, "A", 28))


@pytest.mark.parametrize("source", ["json", "cif", "atom_array"])
def test_inference_input_atomizes_distance_residues(source, dimer):
    """distance_template_selection always atomizes its residues, whatever the entry point.

    Without this the trunk averages the atom-pair distances token-wise and the
    side-chain resolution the condition carries is erased, silently.
    """
    sel = ["A/*/28/CB", "A/*/30/CA"]
    options = InputOptions(distance_template_selection=sel)
    if source == "json":
        spec = InferenceInput.from_json_dict(
            {
                "name": "5vht",
                "components": [{"path": str(CIF)}],
                "distance_template_selection": sel,
            }
        )
    elif source == "cif":
        spec = InferenceInput.from_cif_path(CIF, input_options=options)
    else:
        spec = InferenceInput.from_atom_array(dimer.copy(), input_options=options)

    aa = spec.atom_array
    assert "atomize" in aa.get_annotation_categories()
    expected = _res_mask(aa, "A", 28) | _res_mask(aa, "A", 30)
    assert expected.sum() > 0
    np.testing.assert_array_equal(aa.get_annotation("atomize").astype(bool), expected)
