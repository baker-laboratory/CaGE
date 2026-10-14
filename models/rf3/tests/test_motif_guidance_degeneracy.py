"""Degenerate anchor sets, and the coordinates that never made it into the input.

Two silent failures, both reachable from documented usage:

1. The sampler re-randomizes the global frame every step while the reference coordinates
   are fixed, so a group's Kabsch fit is the only thing mapping the reference into the
   current frame. Three non-collinear anchors pin the rotation; fewer, or a collinear
   set, leaves it free, and the motif is then snapped to an orientation that is wrong and
   that moves with the seed. Measured on a 12-atom motif, worst over three random frames:
   >=3 non-collinear anchors give 0.0000 A of equivariance error, 2 anchors give 4-16 A,
   1 anchor gives 32-43 A, and 3 or 4 collinear anchors give 10-29 A. Resolution cannot
   fix this, so it warns.

2. A whole-residue or whole-chain selection picks up parser-added hydrogens, which carry
   no coordinates. Keeping them would put NaN in `target`, and from there into every
   refined x0 -- the engine's finite check gates arming on x, not on the target. They are
   dropped instead, and an anchor that exists only because the group named none is
   dropped with them; an explicitly named one still raises.

A warning only counts if it reaches the user, so the last test here pins that these
survive the logging configuration real inference runs under.
"""

import logging
import math

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from rf3.guidance import (
    MotifGuidanceCfg,
    MotifGuidanceEngine,
    build_motif_guidance_feats,
)


def _rot(axis, deg):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    th = math.radians(deg)
    K = torch.tensor(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64
    )
    return (
        torch.eye(3, dtype=torch.float64)
        + math.sin(th) * K
        + (1 - math.cos(th)) * (K @ K)
    ).float()


def _equivariance_error(n_anchor: int, collinear: bool = False) -> float:
    """Worst ``refine(R x + t) - (R refine(x) + t)`` over three rigid frames, in A."""
    torch.manual_seed(0)
    n_atoms, n_motif = 40, 12
    base = torch.randn(n_atoms, 3) * 5.0
    motif_idx = torch.arange(10, 10 + n_motif)
    x = base.clone()
    if collinear:
        for k in range(n_anchor):
            x[motif_idx[k]] = torch.tensor([float(k) * 3.0, 0.0, 0.0])
    target = x[motif_idx].clone() + torch.tensor([3.0, -2.0, 1.0])
    feats = {
        "motif_idx": motif_idx,
        "target": target,
        "anchor_idx_in_motif": torch.arange(n_anchor),
    }
    eng = MotifGuidanceEngine(
        MotifGuidanceCfg(rmsd_threshold=1e9, t_threshold_high=1.0), feats
    )
    eng.reset()
    plain = eng.refine(x.unsqueeze(0), t_norm=0.5)[0]
    worst = 0.0
    for deg, axis, shift in (
        (37.0, (0.3, -0.7, 0.5), 11.0),
        (-118.0, (1.0, 0.2, -0.4), -6.0),
        (73.0, (0.2, 0.9, -0.3), 30.0),
    ):
        R = _rot(axis, deg)
        t = torch.full((3,), shift)
        eng.reset()
        moved = eng.refine((x @ R.T + t).unsqueeze(0), t_norm=0.5)[0]
        err = (moved - (plain @ R.T + t))[motif_idx].norm(dim=-1).max().item()
        worst = max(worst, err)
    return worst


@pytest.mark.parametrize("n_anchor", [3, 4, 12])
def test_three_non_collinear_anchors_are_exactly_equivariant(n_anchor):
    assert _equivariance_error(n_anchor) < 1e-3


@pytest.mark.parametrize("n_anchor", [1, 2])
def test_too_few_anchors_break_equivariance(n_anchor):
    """Not "guidance is weaker": the motif lands in a frame-dependent orientation."""
    assert _equivariance_error(n_anchor) > 1.0


@pytest.mark.parametrize("n_anchor", [3, 4])
def test_collinear_anchors_break_equivariance_even_at_three(n_anchor):
    assert _equivariance_error(n_anchor, collinear=True) > 1.0


def _array(coords: np.ndarray, atom_names: list[str]) -> AtomArray:
    arr = AtomArray(len(atom_names))
    arr.chain_id = np.array(["A"] * len(atom_names))
    arr.res_id = np.array([10] * len(atom_names))
    arr.res_name = np.array(["GLY"] * len(atom_names))
    arr.atom_name = np.array(atom_names)
    arr.element = np.array([n[0] for n in atom_names])
    arr.coord = coords.astype(np.float32)
    return arr


_TRIANGLE = np.array(
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 1.0, 2.0]]
)
_LINE = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
_NAMES = ["N", "CA", "C", "CB"]


def test_well_conditioned_anchors_warn_about_nothing(caplog):
    arr = _array(_TRIANGLE, _NAMES)
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        build_motif_guidance_feats(
            arr,
            {
                "g": {
                    "atoms": ["A"],
                    "anchor": ["A/GLY/10/N", "A/GLY/10/CA", "A/GLY/10/C"],
                }
            },
        )
    assert "underdetermined" not in caplog.text


def test_fewer_than_three_anchors_warns_that_guidance_will_not_steer(caplog):
    arr = _array(_TRIANGLE, _NAMES)
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        build_motif_guidance_feats(
            arr, {"g": {"atoms": ["A"], "anchor": ["A/GLY/10/N", "A/GLY/10/CA"]}}
        )
    assert "only 2 anchor atom(s)" in caplog.text
    assert "underdetermined" in caplog.text
    # The user has to be told the consequence, not just the condition.
    assert "will NOT steer" in caplog.text


def test_collinear_anchors_warn_even_though_there_are_three(caplog):
    arr = _array(_LINE, _NAMES)
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        build_motif_guidance_feats(
            arr,
            {
                "g": {
                    "atoms": ["A"],
                    "anchor": ["A/GLY/10/N", "A/GLY/10/CA", "A/GLY/10/C"],
                }
            },
        )
    assert "collinear" in caplog.text
    assert "will NOT steer" in caplog.text


def test_the_flat_list_shorthand_can_under_anchor_a_whole_group(caplog):
    """`[atom, atom, chain]` anchors the chain on two atoms -- warn, do not stay quiet."""
    arr = _array(_TRIANGLE, _NAMES)
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        feats = build_motif_guidance_feats(arr, ["A/GLY/10/N", "A/GLY/10/CA", "A"])
    assert int(feats["anchor_idx_in_motif"].numel()) == 2
    assert "underdetermined" in caplog.text


# --- coordinates missing from the input ------------------------------------------------


def _with_missing() -> AtomArray:
    coords = np.vstack([_TRIANGLE, [[np.nan, np.nan, np.nan]]])
    return _array(coords, _NAMES + ["H"])


def test_atoms_without_coordinates_are_dropped_not_propagated_as_nan(caplog):
    arr = _with_missing()
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        feats = build_motif_guidance_feats(arr, {"g": ["A"]})
    assert int(feats["motif_idx"].numel()) == 4  # the H is gone
    assert torch.isfinite(feats["target"]).all()
    assert "no coordinates" in caplog.text

    # and nothing non-finite reaches a refined x0
    x = torch.zeros(1, 5, 3)
    x[0] = torch.nan_to_num(torch.from_numpy(arr.coord), nan=0.0)
    out = MotifGuidanceEngine(
        MotifGuidanceCfg(rmsd_threshold=1e9, t_threshold_high=1.0), feats
    ).refine(x, t_norm=0.5)
    assert torch.isfinite(out).all()


def test_a_default_anchor_skips_the_atom_it_has_no_coordinate_for():
    """ "anchor defaults to all of its atoms" can only mean all of its usable ones."""
    feats = build_motif_guidance_feats(_with_missing(), {"g": ["A"]})
    assert int(feats["anchor_idx_in_motif"].numel()) == 4


def test_an_explicitly_named_anchor_without_a_coordinate_still_raises():
    """Narrowing what the caller asked for by name would silently change the fit."""
    with pytest.raises(ValueError, match="non-finite"):
        build_motif_guidance_feats(
            _with_missing(),
            {
                "g": {
                    "atoms": ["A"],
                    "anchor": ["A/GLY/10/N", "A/GLY/10/CA", "A/GLY/10/H"],
                }
            },
        )


# --- the warnings have to survive the logging real inference runs under -----------------


@pytest.fixture
def restore_log_levels():
    """configure_minimal_inference_logging mutates global logger levels."""
    names = ["", "root", "rf3.guidance", "rf3.diffusion_samplers", "foundry.metrics"]
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def test_degeneracy_warning_survives_inference_logging(restore_log_levels, caplog):
    """`rf3 fold` quiets almost everything; this warning must not be quieted with it.

    `configure_minimal_inference_logging` puts "root" in its ERROR list, and
    `logging.getLogger("root")` *is* the root logger -- so every logger without an
    explicit exception drops to ERROR. Both `rf3.cli` and `BaseInferenceEngine` call it,
    so without the exception the only signal that guidance cannot do what was asked is
    swallowed on exactly the path users take.
    """
    from foundry.utils.logging import configure_minimal_inference_logging

    configure_minimal_inference_logging()
    assert logging.getLogger("rf3.guidance.feats").isEnabledFor(logging.WARNING)
    assert logging.getLogger("rf3.diffusion_samplers.inference_sampler").isEnabledFor(
        logging.WARNING
    )

    arr = _array(_LINE, _NAMES)
    with caplog.at_level(logging.WARNING, logger="rf3.guidance.feats"):
        build_motif_guidance_feats(
            arr,
            {
                "g": {
                    "atoms": ["A"],
                    "anchor": ["A/GLY/10/N", "A/GLY/10/CA", "A/GLY/10/C"],
                }
            },
        )
    assert "will NOT steer" in caplog.text
