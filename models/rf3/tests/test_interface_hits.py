"""Tests for InterfaceHitsByType on a real protein-ligand fixture.

Verifies the three core metrics (`pocket_hit_4.5`, `pocket_iou_4.5`,
`hotspot_hit_4.5`) under three perturbations on the MELK kinase + inhibitor
complex (PDB ``5ocm``):

- ``displacement = 0 A`` (pred == GT)        -> all metrics 1.0
- ``displacement = 1 A`` (small ligand shift) -> hits still 1.0, IoU > 0.5
- ``displacement = 100 A`` (ligand far away)  -> all metrics 0.0

Two parametrized tests cover the with- and without-hotspot variants.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from atomworks.enums import ChainTypeInfo
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from biotite.structure import AtomArray
from rf3.conditions import C_HOT
from rf3.metrics.interface_hits import InterfaceHitsByType
from scipy.spatial.distance import cdist

_PDB_DIR = Path(__file__).parent / "data" / "pdb"


def cached_parse(pdb_id: str, **kwargs) -> dict:
    """Parse a bundled test structure so the test runs without a PDB mirror."""
    data = parse(_PDB_DIR / f"{pdb_id}.cif.gz", config=ParseConfig(**kwargs))
    if "atom_array" not in data:
        data["atom_array"] = next(iter(data["assemblies"].values()))[0]
    return data


# Kinase-inhibitor complex; same PDB used by `test_chiral_metrics.py`,
# so it's already vetted in the rf3 test data mirror.
PDB_ID = "5ocm"
# Number of pocket residues to mark as hotspots in the GT. The metric
# de-duplicates hotspot atoms to unique residue IDs, so we mark whole
# residues (every atom of the chosen residues) to keep the test
# assertion `n_hotspot_residues == N_HOTSPOT_RESIDUES` exact.
N_HOTSPOT_RESIDUES = 5


def _identify_pli_chains(atom_array: AtomArray) -> tuple[str, str]:
    """Return the ``(protein_chain_id, ligand_chain_id)`` pair that actually
    binds in the asymmetric unit.

    Reads the per-atom ``chain_type`` annotation to enumerate candidate
    protein chains (``ChainTypeInfo.PROTEINS``) and candidate non-water
    non-polymer chains, then picks the (protein, ligand) pair with the
    smallest minimum heavy-atom-to-heavy-atom distance — i.e. the chain
    pair that is genuinely in contact. Some PDBs (``5ocm`` is one) have
    multiple protein chains in the asymmetric unit but only one of them
    contacts the ligand, so a "first-match" heuristic is not sufficient.
    """
    protein_chains: list[str] = []
    ligand_chains: list[str] = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_atoms = atom_array.chain_id == chain_id
        chain_type = atom_array.chain_type[chain_atoms][0]
        if chain_type in ChainTypeInfo.PROTEINS:
            protein_chains.append(chain_id)
        elif chain_type in ChainTypeInfo.NON_POLYMERS:
            res_names = set(atom_array.res_name[chain_atoms].tolist())
            if not res_names <= {"HOH", "WAT"}:
                ligand_chains.append(chain_id)
    assert protein_chains, f"no protein chain found in {PDB_ID}"
    assert ligand_chains, f"no ligand chain found in {PDB_ID}"

    heavy = atom_array.element != "H"
    finite = ~np.isnan(atom_array.coord).any(axis=1)
    best_pair: tuple[str, str] | None = None
    best_dist = np.inf
    for p in protein_chains:
        p_mask = (atom_array.chain_id == p) & heavy & finite
        if not p_mask.any():
            continue
        for lig in ligand_chains:
            l_mask = (atom_array.chain_id == lig) & heavy & finite
            if not l_mask.any():
                continue
            d = cdist(atom_array.coord[p_mask], atom_array.coord[l_mask]).min()
            if d < best_dist:
                best_dist = d
                best_pair = (p, lig)
    assert (
        best_pair is not None
    ), f"no protein/ligand chain pair with finite heavy-atom coords in {PDB_ID}"
    assert best_dist < 4.5, (
        f"closest protein/ligand chain pair in {PDB_ID} is {best_dist:.2f} A apart, "
        "no actual interface to test"
    )
    return best_pair


def _mark_pocket_residues_as_hotspots(
    gt: AtomArray,
    protein_chain: str,
    ligand_chain: str,
    n: int = N_HOTSPOT_RESIDUES,
) -> None:
    """Mark every atom of the ``n`` protein residues whose closest heavy
    atom is closest to the ligand with the C_HOT annotation.

    Marking whole residues (rather than individual atoms) keeps the
    metric's per-residue de-duplication aligned with our test
    expectation. Picking the *closest* residues also ensures hotspots
    stay inside the 4.5A pocket window after a 1A ligand perturbation,
    so the ``hotspot_hit`` assertion at displacement=1A is robust.
    """
    heavy = gt.element != "H"
    finite = ~np.isnan(gt.coord).any(axis=1)
    protein_mask = (gt.chain_id == protein_chain) & heavy & finite
    ligand_mask = (gt.chain_id == ligand_chain) & heavy & finite

    # Distance from each in-scope protein atom to its closest ligand atom.
    atom_min_dists = cdist(gt.coord[protein_mask], gt.coord[ligand_mask]).min(axis=1)
    atom_res_ids = gt.res_id[protein_mask]

    # Reduce atom-level distances to residue-level (min over atoms of each residue).
    res_id_to_min_dist: dict[int, float] = {}
    for res_id, dist in zip(atom_res_ids, atom_min_dists):
        prev = res_id_to_min_dist.get(int(res_id))
        if prev is None or dist < prev:
            res_id_to_min_dist[int(res_id)] = float(dist)

    # Pick the `n` residues with the smallest min-distance to the ligand.
    closest_res_ids = sorted(res_id_to_min_dist, key=res_id_to_min_dist.get)[:n]

    chain_mask = gt.chain_id == protein_chain
    hotspot_mask = chain_mask & np.isin(gt.res_id, list(closest_res_ids))
    gt.set_annotation(C_HOT.full_name, hotspot_mask)


def _build_inputs(
    displacement_A: float, with_hotspots: bool
) -> tuple[AtomArray, AtomArray, list[list[tuple[str, str, str]]]]:
    """Return ``(pred, gt, interfaces_to_score)`` ready to feed `compute()`."""
    gt = cached_parse(PDB_ID, hydrogen_policy="remove")["atom_array"]
    protein_chain, ligand_chain = _identify_pli_chains(gt)
    if with_hotspots:
        _mark_pocket_residues_as_hotspots(gt, protein_chain, ligand_chain)

    pred = deepcopy(gt)
    if displacement_A != 0:
        ligand_mask = pred.chain_id == ligand_chain
        pred.coord[ligand_mask] += np.array(
            [displacement_A, 0.0, 0.0], dtype=pred.coord.dtype
        )

    iface = (f"{protein_chain}_1", f"{ligand_chain}_1", "protein-ligand")
    # Outer wrapper = batch dim of 1 (see `InterfaceHitsByType.compute`
    # batch unwrap).
    return pred, gt, [[iface]]


@pytest.mark.parametrize(
    "displacement_A, expected_pocket_hit, pocket_iou_lower_bound",
    [
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.5),
        (100.0, 0.0, 0.0),
    ],
)
def test_pocket_metrics_perfect_small_far(
    displacement_A: float,
    expected_pocket_hit: float,
    pocket_iou_lower_bound: float,
):
    pred, gt, ifs = _build_inputs(displacement_A, with_hotspots=False)

    out = InterfaceHitsByType(cutoff=4.5).compute(
        predicted_atom_array_stack=pred,
        ground_truth_atom_array_stack=gt,
        interfaces_to_score=ifs,
        example_id=["test_5ocm"],
    )

    assert len(out) == 1
    r = out[0]
    assert r["type"] == "protein-ligand"
    assert r["best_of_1_pocket_hit_4.5"] == expected_pocket_hit
    if expected_pocket_hit == 1.0:
        assert r["best_of_1_pocket_iou_4.5"] >= pocket_iou_lower_bound
    else:
        assert r["best_of_1_pocket_iou_4.5"] == 0.0
    # No C_HOT annotation set on this GT -> hotspot_hit must be NaN.
    assert math.isnan(r["best_of_1_hotspot_hit_4.5"])


@pytest.mark.parametrize(
    "displacement_A, expected_pocket_hit, pocket_iou_lower_bound, expected_hotspot_hit",
    [
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 0.5, 1.0),
        (100.0, 0.0, 0.0, 0.0),
    ],
)
def test_hotspot_metrics_perfect_small_far(
    displacement_A: float,
    expected_pocket_hit: float,
    pocket_iou_lower_bound: float,
    expected_hotspot_hit: float,
):
    pred, gt, ifs = _build_inputs(displacement_A, with_hotspots=True)

    out = InterfaceHitsByType(cutoff=4.5).compute(
        predicted_atom_array_stack=pred,
        ground_truth_atom_array_stack=gt,
        interfaces_to_score=ifs,
        example_id=["test_5ocm"],
    )

    assert len(out) == 1
    r = out[0]
    assert r["type"] == "protein-ligand"
    assert r["n_hotspot_residues"] == N_HOTSPOT_RESIDUES
    assert r["best_of_1_pocket_hit_4.5"] == expected_pocket_hit
    if expected_pocket_hit == 1.0:
        assert r["best_of_1_pocket_iou_4.5"] >= pocket_iou_lower_bound
    else:
        assert r["best_of_1_pocket_iou_4.5"] == 0.0
    assert r["best_of_1_hotspot_hit_4.5"] == expected_hotspot_hit
