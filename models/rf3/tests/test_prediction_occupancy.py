"""Regression: a prediction must not export the input's resolved/unresolved mask.

``occupancy == 0`` on the pipeline's atom_array means "unresolved in the input file":
atomworks inserts missing residues that way, and
``MaskPolymerResiduesWithUnresolvedFrameAtoms`` (unconditional, inference included)
spreads it over the whole residue. The output CIF is built from that same atom_array
plus predicted coordinates, so without an explicit reset the marker lands in
``_atom_site.occupancy`` -- and occupancy filters downstream (MPNN's
``MaskResiduesWithSpecificUnresolvedAtoms`` -> ``RemoveUnresolvedTokens``) then drop a
residue the model actually predicted, shortening the sequence.

``RF3InferenceEngine.run`` needs a checkpoint, so the wiring is pinned with ``ast``
(same approach as ``test_cfg_signature_parity.py``) and the behaviour is checked
against the real stack builder and CIF writer.

Run with::

    pytest models/rf3/tests/test_prediction_occupancy.py
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from atomworks.enums import ChainType
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from atomworks.io.utils.io_utils import to_cif_file
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords

ENGINE_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "rf3" / "inference_engines" / "rf3.py"
)


def _run_method() -> ast.FunctionDef:
    tree = ast.parse(ENGINE_SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RF3InferenceEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run":
                    return item
    pytest.fail("RF3InferenceEngine.run not found")


def test_engine_resets_occupancy_between_stack_build_and_output():
    """The reset must sit after the stack is built and before RF3Output is constructed.

    Both dump paths (``RF3Output.dump`` and ``dump_top_ranked_outputs``) share that
    atom_array, so a reset in that window covers every predicted CIF.
    """
    run = _run_method()

    build_lines, reset_lines, output_lines = [], [], []
    for node in ast.walk(run):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id == "build_stack_from_atom_array_and_batched_coords"
        ):
            build_lines.append(node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "set_annotation"
            and isinstance(func.value, ast.Name)
            and func.value.id == "atom_array_stack"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "occupancy"
        ):
            reset_lines.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id == "RF3Output":
            output_lines.append(node.lineno)

    assert build_lines, "run() no longer builds a predicted AtomArrayStack"
    assert reset_lines, (
        "run() no longer resets `occupancy` on the predicted stack; the input's "
        "unresolved-atom marker will leak into the output CIF"
    )
    assert output_lines, "run() no longer constructs RF3Output"
    assert min(reset_lines) > min(build_lines), "reset runs before the stack exists"
    assert max(reset_lines) < min(output_lines), "reset runs after RF3Output is built"


@pytest.fixture
def atom_array_with_unresolved_residue():
    """A 5-residue chain whose ARG (11 heavy atoms) is unresolved, as a parsed gap would be."""
    atom_array = sequence_to_annotated_atom_array(
        "ARGSA", chain_id="A", chain_type=ChainType.POLYPEPTIDE_L
    )
    occupancy = np.ones(atom_array.array_length(), dtype=np.float32)
    occupancy[atom_array.res_id == 2] = 0.0
    atom_array.set_annotation("occupancy", occupancy)
    # Stand in for the parser's NaN coordinates on the unresolved residue.
    atom_array.coord = np.zeros((atom_array.array_length(), 3), dtype=np.float32)
    atom_array.coord[atom_array.res_id == 2] = np.nan
    return atom_array


def _occupancies_in_cif(path: Path) -> list[str]:
    text = path.read_text()
    columns = [
        line.strip() for line in text.splitlines() if line.startswith("_atom_site.")
    ]
    index = columns.index("_atom_site.occupancy")
    return [
        line.split()[index]
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def _dump_like_rf3_output(atom_array, path: Path) -> Path:
    """Write one predicted model with the same kwargs as ``RF3Output.dump``."""
    to_cif_file(
        atom_array,
        str(path),
        file_type="cif",
        include_entity_poly=True,
        include_nan_coords=False,
    )
    return path.with_suffix(".cif")


def test_input_occupancy_would_leak_without_the_reset(
    atom_array_with_unresolved_residue, tmp_path
):
    """Pin the mechanism the reset exists for: annotations ride along into the CIF."""
    atom_array = atom_array_with_unresolved_residue
    n_atoms = atom_array.array_length()
    coords = np.zeros((1, n_atoms, 3), dtype=np.float32)

    stack = build_stack_from_atom_array_and_batched_coords(coords, atom_array)
    written = _dump_like_rf3_output(stack[0], tmp_path / "leaked")

    assert "0.0" in _occupancies_in_cif(written)


def test_reset_stack_writes_full_occupancy(
    atom_array_with_unresolved_residue, tmp_path
):
    """After the engine's reset every predicted atom is written at occupancy 1.0."""
    atom_array = atom_array_with_unresolved_residue
    n_atoms = atom_array.array_length()
    # The model places every atom, so predicted coordinates are finite everywhere.
    coords = np.arange(2 * n_atoms * 3, dtype=np.float32).reshape(2, n_atoms, 3)

    stack = build_stack_from_atom_array_and_batched_coords(coords, atom_array)
    stack.set_annotation("occupancy", np.ones(stack.array_length(), dtype=np.float32))

    for sample_idx in range(len(stack)):
        written = _dump_like_rf3_output(
            stack[sample_idx], tmp_path / f"model_{sample_idx}"
        )
        occupancies = _occupancies_in_cif(written)
        assert len(occupancies) == n_atoms
        assert set(occupancies) == {"1.0"}


def test_reset_does_not_mutate_the_source_atom_array(
    atom_array_with_unresolved_residue,
):
    """Confidence outputs and metrics read the pipeline's atom_array, not the stack."""
    atom_array = atom_array_with_unresolved_residue
    n_atoms = atom_array.array_length()
    unresolved = atom_array.res_id == 2

    stack = build_stack_from_atom_array_and_batched_coords(
        np.zeros((2, n_atoms, 3), dtype=np.float32), atom_array
    )
    stack.set_annotation("occupancy", np.ones(stack.array_length(), dtype=np.float32))

    assert np.all(atom_array.occupancy[unresolved] == 0.0)
    assert np.all(atom_array.occupancy[~unresolved] == 1.0)
