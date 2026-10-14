"""Regression tests: missing (NaN) token centers must not become valid template conditions.

A whole-chain ``template_selection=["A"]`` sets ``is_input_file_templated`` on every
token, which the featurizer treats as a forced template. Previously this forced mask
bypassed the resolved/finite check, so a missing CA (occupancy 0, NaN coordinate)
produced a NaN distance that ``torch.bucketize`` silently sent to the last distance bin
and ``has_distogram_condition`` flagged as a real condition.

These tests build a tiny protein AtomArray directly (no model forward) and check the
missing residue is excluded while fully-resolved residues are still conditioned.

Run with::

    pytest models/rf3/tests/test_template_missing_ca_condition.py
"""

import numpy as np
import torch
from atomworks.enums import ChainType
from biotite.structure import AtomArray
from rf3.data.ground_truth_template import (
    featurize_noised_ground_truth_as_template_distogram,
)
from rf3.utils.inference import InferenceInput


def _build_ca_only_protein(
    n_res: int,
    missing_idx: int | None = None,
    *,
    missing_occupancy: float = 0.0,
) -> AtomArray:
    """Build an ``n_res``-residue protein with one CA per residue (each CA is a token center).

    If ``missing_idx`` is given, that residue's CA is marked unresolved (occupancy 0) with
    NaN coordinates, mimicking a missing center atom in a selected chain.
    """
    aa = AtomArray(n_res)
    aa.coord = np.stack(
        [np.array([i * 4.0, 0.0, 0.0], dtype=np.float32) for i in range(n_res)]
    )
    aa.chain_id = np.array(["A"] * n_res)
    aa.res_id = np.arange(1, n_res + 1)
    aa.res_name = np.array(["ALA"] * n_res)
    aa.atom_name = np.array(["CA"] * n_res)
    aa.element = np.array(["C"] * n_res)

    aa.set_annotation("atomize", np.zeros(n_res, dtype=bool))
    aa.set_annotation("chain_type", np.full(n_res, int(ChainType.POLYPEPTIDE_L)))
    aa.set_annotation("molecule_id", np.zeros(n_res, dtype=int))
    aa.set_annotation("occupancy", np.ones(n_res, dtype=np.float32))
    # Whole-chain selection -> forced template on every token.
    aa.set_annotation("is_input_file_templated", np.ones(n_res, dtype=bool))

    if missing_idx is not None:
        aa.coord[missing_idx] = np.nan
        occ = aa.occupancy.copy()
        occ[missing_idx] = missing_occupancy
        aa.set_annotation("occupancy", occ)

    return aa


def _featurize(atom_array: AtomArray) -> dict[str, torch.Tensor]:
    # noise_scale=0 makes conditioned distances exactly the true distances (deterministic
    # bin assertions); the NaN-exclusion path is independent of the noise magnitude.
    return featurize_noised_ground_truth_as_template_distogram(
        atom_array,
        noise_scale=torch.tensor(0.0),
        distogram_bins=torch.cat(
            (torch.arange(1.0, 4.0, 0.1), torch.arange(4.0, 20.5, 0.5))
        ),
        allowed_chain_types=[ChainType.POLYPEPTIDE_L],
        is_unconditional=False,
    )


def test_missing_ca_excluded_from_condition():
    """A forced-template token with a missing CA must be excluded; output has no NaNs."""
    missing = 1
    aa = _build_ca_only_protein(n_res=3, missing_idx=missing)

    out = _featurize(aa)
    has_cond = out["has_distogram_condition"]

    # No NaNs leak into any output tensor.
    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"NaN/Inf found in {key}"

    # Every pair touching the missing residue must be unconditioned.
    assert not has_cond[missing, :].any(), "missing-CA row is conditioned"
    assert not has_cond[:, missing].any(), "missing-CA column is conditioned"

    # The two fully-resolved residues (0, 2) are still conditioned against each other.
    assert has_cond[0, 2] and has_cond[2, 0], "resolved pair should remain conditioned"

    # And that condition encodes the true distance (bin of ||coord0 - coord2||).
    bins = torch.cat((torch.arange(1.0, 4.0, 0.1), torch.arange(4.0, 20.5, 0.5)))
    expected_dist = float(np.linalg.norm(aa.coord[0] - aa.coord[2]))
    expected_bin = int(torch.bucketize(torch.tensor(expected_dist), bins))
    assert int(out["distogram_condition"][0, 2].argmax()) == expected_bin


def test_full_template_behavior_unchanged():
    """With all CAs resolved, every off-diagonal pair is conditioned and no NaNs appear."""
    n_res = 3
    aa = _build_ca_only_protein(n_res=n_res, missing_idx=None)

    out = _featurize(aa)
    has_cond = out["has_distogram_condition"]

    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"NaN/Inf found in {key}"

    # Every token pair (incl. self-pairs on the diagonal) is conditioned when all CAs
    # are resolved — this is the pre-existing full-template behavior, kept unchanged.
    assert has_cond.all(), "all pairs of a fully-resolved template must be conditioned"


def test_to_pipeline_input_preserves_unresolved_state_before_nan_clobber():
    """The inference NaN sentinel must not turn a missing forced CA into a condition."""

    missing = 1
    aa = _build_ca_only_protein(
        n_res=3,
        missing_idx=missing,
        # Sequence-built inputs can carry occupancy=1 while the coordinate itself is
        # still unresolved (NaN).
        missing_occupancy=1.0,
    )
    assert aa.occupancy[missing] == 1.0
    assert np.isnan(aa.coord[missing]).all()

    spec = InferenceInput(
        atom_array=aa,
        chain_info={},
        example_id="missing_ca_whole_chain",
        template_selection=["A"],
    )
    pipeline_atom_array = spec.to_pipeline_input()["atom_array"]

    # Downstream code receives finite sentinel coordinates, but occupancy retains
    # the semantically important unresolved mask.
    assert np.array_equal(
        pipeline_atom_array.coord[missing],
        np.array([-1.0, -1.0, -1.0]),
    )
    assert pipeline_atom_array.occupancy[missing] == 0.0

    out = _featurize(pipeline_atom_array)
    has_cond = out["has_distogram_condition"]
    assert not has_cond[missing, :].any()
    assert not has_cond[:, missing].any()
    assert has_cond[0, 2] and has_cond[2, 0]


def test_to_pipeline_input_keeps_unconditioned_sequence_atoms_present():
    """NaN coordinates in a sequence-only query must not erase its topology."""

    aa = _build_ca_only_protein(n_res=3)
    aa.coord[:] = np.nan
    aa.set_annotation("is_input_file_templated", np.zeros(3, dtype=bool))

    spec = InferenceInput(
        atom_array=aa,
        chain_info={},
        example_id="unconditioned_sequence",
        template_selection=None,
    )
    pipeline_atom_array = spec.to_pipeline_input()["atom_array"]

    assert np.all(pipeline_atom_array.coord == -1.0)
    assert np.all(pipeline_atom_array.occupancy == 1.0)
