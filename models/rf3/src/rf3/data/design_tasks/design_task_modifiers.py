import ast
import os
import shutil
import string
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_pdb_string
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import AtomArray

from foundry.utils.ddp import RankedLogger

from rf3.conditions import (
    C_HB,
    C_RAS,
    HBondEnum,
    RasaEnum,
)

logger = RankedLogger(__name__, rank_zero_only=True)

# hbplus is an optional third-party dependency; warn once per process rather than
# dropping the hbond condition in complete silence.
_WARNED_NO_HBPLUS = False


def _warn_no_hbplus_once() -> None:
    global _WARNED_NO_HBPLUS
    if not _WARNED_NO_HBPLUS:
        logger.warning(
            "HBPLUS_PATH is not set; the hydrogen-bond design condition will be skipped "
            "for every example in this process. hbplus is third-party software and is "
            "not distributed with this repository."
        )
        _WARNED_NO_HBPLUS = True


def add_rasa_condition(
    data: dict,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "ProtOr",
    point_number: int = 100,
    p_include_non_ligand: bool = 0.1,
    rasa_max: float = 0.2,
) -> dict:
    """
    probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to
        1.4 (for water).
    atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to
        "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions
        or certain atoms with charges
    point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for
        calculating SASA. Defaults to 100.
    p_include_non_ligand (float, optional): Probability of providing RASA conditioning for
        non-ligand atoms. Defaults to 0.1.
    rasa_max (float, optional): Maximum RASA value for binning. Defaults to 0.2.
    """
    atom_array = as_atom_array_plus(data["atom_array"])

    ensure_annotations(atom_array, "is_ligand")

    # Determine valid residues for rasa conditioning
    if np.random.rand() < p_include_non_ligand:
        # With RF3, everything is "sequence-conditioned" so we could give rasa to anything in theory
        inclusion_mask = np.ones(len(atom_array), dtype=bool)
    else:
        inclusion_mask = atom_array.is_ligand

    rasa = calculate_atomwise_rasa(atom_array, probe_radius, atom_radii, point_number, atom_filter=inclusion_mask)

    # NOTE: Bin count is hard-coded at 3 to match the RasaEnum
    bin_edges = np.linspace(0, rasa_max, 3)  # e.g., [0.0, 0.1, 0.2]
    bins = np.digitize(rasa, bin_edges, right=False)  # bins in [0, 3] -- however, bin 0 would mean negative rasa!
    if (bins == 0).any():
        raise ValueError("Negative RASA value was detected. This indicates an error.")

    # Convert to RasaEnum
    enum_rasa_values = C_RAS.default_annotation(atom_array)
    enum_rasa_values[bins == 1] = RasaEnum.BURIED
    enum_rasa_values[bins == 2] = RasaEnum.INTERMEDIATE
    enum_rasa_values[bins == 3] = RasaEnum.EXPOSED
    enum_rasa_values[~inclusion_mask] = RasaEnum.UNSPECIFIED

    C_RAS.set_annotation(atom_array, enum_rasa_values.astype(int))

    data["atom_array"] = atom_array
    return data


@contextmanager
def temporary_pdb_directory(atom_array: AtomArray, raise_on_failure: bool = True) -> Any:
    """Context manager that creates a temporary directory with a PDB file.

    Creates a temporary directory and writes the PDB file there. This ensures
    hbplus output files are created in a known location and can be easily cleaned up.

    Yields:
        tuple: (temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, failed)
               or (None, None, None, None, None, True) on failure
    """

    def _handle_nan_coords(atom_array: AtomArray, noise_level: float = 1e-3) -> tuple:
        coords = atom_array.coord
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(-noise_level, noise_level, size=nan_mask.sum())
        atom_array.coord = coords
        return atom_array, nan_mask

    # To avoid modifying the original atom_array with this preprocessing
    atom_array = atom_array.copy()
    atom_array, nan_mask = _handle_nan_coords(atom_array)

    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        if raise_on_failure:
            raise ValueError("Too many chain_iids, cannot convert to PDB")
        yield None, None, None, None, None, True
        return

    # Build chain ID mapping (PDB format only supports single-character chain IDs)
    all_possible_chain_ids = string.ascii_letters
    chain_map = {}
    for chain_iid in chain_iids:
        if len(chain_iid) == 1:
            chain_map[chain_iid] = chain_iid
            all_possible_chain_ids = all_possible_chain_ids.replace(chain_iid, "")
    for chain_iid in chain_iids:
        if len(chain_iid) > 1:
            chain_map[chain_iid] = all_possible_chain_ids[0]
            all_possible_chain_ids = all_possible_chain_ids.replace(chain_map[chain_iid], "")

    atom_array.chain_id = [chain_map[iid] for iid in atom_array.chain_iid]
    inverted_chain_map = {v: k for k, v in chain_map.items()}
    atom_array.b_factor = np.zeros(len(atom_array))

    # Create temporary directory and PDB file
    temp_dir = tempfile.mkdtemp(prefix="hbplus_")
    pdb_filename = "structure.pdb"
    pdb_path = os.path.join(temp_dir, pdb_filename)

    try:
        try:
            pdb_str = to_pdb_string(atom_array)
        except Exception:
            # Honor `raise_on_failure=False` for any biotite/atomworks error here
            # (e.g., res_name >3 chars, hybrid36 overflow). Mirrors the existing
            # `n_chains > 52` fallback above; without this guard the exception
            # escapes through the calling design-task pipeline and crashes
            # validation, even though `add_hbond_condition` calls us with
            # `raise_on_failure=False` precisely to avoid that.
            if raise_on_failure:
                raise
            yield None, None, None, None, None, True
            return

        with open(pdb_path, "w") as f:
            f.write(pdb_str)

        yield temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, False
    finally:
        # Clean up entire temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def check_atom_array_has_hydrogen(data: dict[str, Any]) -> None:
    if not np.any(data["atom_array"].element == "H"):
        raise ValueError("Key `atom_array` in data has no hydrogens.")


def _parse_hbplus_line(line: str, chain_map: dict[str, str]) -> dict[str, Any]:
    """Parse a single line from hbplus output (.hb2 file).

    HBplus output format (fixed-width columns):
    - Columns 0: donor chain
    - Columns 1-5: donor residue ID (with insertion code at position 5)
    - Columns 6-9: donor residue name
    - Columns 9-13: donor atom name
    - Columns 14: acceptor chain
    - Columns 15-19: acceptor residue ID (with insertion code at position 19)
    - Columns 20-23: acceptor residue name
    - Columns 23-27: acceptor atom name
    - Columns 27-32: distance
    """
    return {
        "d_chain": chain_map[line[0]],
        "d_resi": str(int(line[1:5].strip())),
        "d_resn": line[6:9].strip(),
        "d_ins": line[5].replace("-", " "),
        "d_atom": line[9:13].strip(),
        "a_chain": chain_map[line[14]],
        "a_resi": str(int(line[15:19].strip())),
        "a_resn": line[20:23].strip(),
        "a_ins": line[19].replace("-", " "),
        "a_atom": line[23:27].strip(),
        "dist": float(line[27:32].strip()),
    }


def _parse_hbplus_output(hb2_path: str, chain_map: dict[str, str]) -> list[dict[str, Any]]:
    """Parse hbplus output file and return list of hydrogen bonds.

    Args:
        hb2_path: Path to the .hb2 output file
        chain_map: Mapping from PDB chain IDs to original chain IIDs

    Returns:
        List of hydrogen bond dictionaries
    """
    if not os.path.exists(hb2_path):
        raise FileNotFoundError(f"HBplus output file not found: {hb2_path}")

    hbonds = []
    with open(hb2_path) as f:
        lines = f.readlines()
        # Skip header (first 8 lines)
        for line in lines[8:]:
            # Skip empty lines or lines that are too short
            if len(line.strip()) < 32:
                continue
            try:
                hbonds.append(_parse_hbplus_line(line, chain_map))
            except (ValueError, KeyError, IndexError):
                # Skip malformed lines but continue processing
                continue
    return hbonds


def _identify_atom(atom_array: AtomArray, chain_iid: str, res_id: str, atom_name: str) -> np.ndarray:
    """Create a mask identifying a specific atom in the array."""
    return (
        (atom_array.chain_iid == chain_iid) & (atom_array.res_id == float(res_id)) & (atom_array.atom_name == atom_name)
    )


def calculate_hbonds(
    atom_array: AtomArray,
    motif_mask: np.ndarray,
    cutoff_HA_dist: float = 2.2,
    cutoff_DA_distance: float = 3.2,
    raise_on_failure: bool = True,
) -> tuple[AtomArray | None, list | None, int | None, bool]:
    """Calculate hydrogen bonds using the hbplus external tool.

    Args:
        atom_array: The atomic structure
        motif_mask: Boolean mask indicating which atoms are part of the motif
        cutoff_HA_dist: Hydrogen-acceptor distance cutoff in Angstroms
        cutoff_DA_distance: Donor-acceptor distance cutoff in Angstroms
        raise_on_failure: Whether to raise exceptions on failure

    Returns:
        tuple: (atom_array, motif_hbonds, num_motif_hbonds, failed)
            - atom_array: Modified atom array with hbond annotations (or None if failed)
            - motif_hbonds: List of hydrogen bond dicts (or None if failed)
            - num_motif_hbonds: Number of motif hbonds (or None if failed)
            - failed: Boolean indicating if calculation failed
    """
    with temporary_pdb_directory(atom_array, raise_on_failure=raise_on_failure) as result:
        if result[0] is None:
            return None, None, None, True

        temp_dir, pdb_filename, atom_array, _nan_mask, inverted_chain_map, _failed = result
        _pdb_path = os.path.join(temp_dir, pdb_filename)

        # HBplus writes output to the current working directory
        # We need to run it from the temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)

            hbplus_exe = os.environ.get("HBPLUS_PATH")
            if not hbplus_exe:
                _warn_no_hbplus_once()
                if raise_on_failure:
                    raise OSError(
                        "Environment variable HBPLUS_PATH is not set: point it at your hbplus "
                        "executable (third-party software, not distributed with this "
                        "repository; see .env.sample)."
                    )
                return None, None, None, True

            # Run hbplus (output will be created in temp_dir)
            subprocess.run(
                [
                    hbplus_exe,
                    "-h",
                    str(cutoff_HA_dist),
                    "-d",
                    str(cutoff_DA_distance),
                    pdb_filename,  # Use just the filename since we're in the temp dir
                    pdb_filename,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,  # hbplus may return non-zero even on success
            )

            # HBplus creates output with .hb2 extension in current directory
            hb2_filename = pdb_filename.replace(".pdb", ".hb2")
            hb2_path = os.path.join(temp_dir, hb2_filename)

            # Parse hbplus output
            try:
                hbonds = _parse_hbplus_output(hb2_path, inverted_chain_map)
            except FileNotFoundError:
                if raise_on_failure:
                    raise
                return None, None, None, True

        finally:
            # Restore original working directory
            os.chdir(original_dir)

    # Initialize arrays for tracking hydrogen bonds using enum
    hbond_all = np.full(len(atom_array), HBondEnum.NONE, dtype=int)

    motif_hbonds = []
    for hbond in hbonds:
        # Identify donor and acceptor atoms
        current_donor_mask = _identify_atom(atom_array, hbond["d_chain"], hbond["d_resi"], hbond["d_atom"])
        current_acceptor_mask = _identify_atom(atom_array, hbond["a_chain"], hbond["a_resi"], hbond["a_atom"])

        # Ensure that we can uniquely identify the donor and acceptor atoms
        if current_donor_mask.sum() != 1:
            if raise_on_failure:
                raise ValueError(
                    f"Unable to uniquely identify a donor atom with chain_iid={hbond['d_chain']}, "
                    f"res_id={hbond['d_resi']}, atom_name={hbond['d_atom']}."
                )
            return None, None, None, True
        if current_acceptor_mask.sum() != 1:
            if raise_on_failure:
                raise ValueError(
                    f"Unable to uniquely identify an acceptor atom with chain_iid={hbond['a_chain']}, "
                    f"res_id={hbond['a_resi']}, atom_name={hbond['a_atom']}."
                )
            return None, None, None, True

        current_donor_is_motif = motif_mask[current_donor_mask][0]
        current_acceptor_is_motif = motif_mask[current_acceptor_mask][0]

        # Record this hbond as a motif hbond if either end touches the motif.
        # `motif_hbonds` is recorded by HbondMetrics as `n_motif_hbonds` 
        if current_donor_is_motif or current_acceptor_is_motif:
            motif_hbonds.append(hbond)

        # Record all ground-truth hbonds
        donor_only = current_donor_mask & ~current_acceptor_mask
        acceptor_only = current_acceptor_mask & ~current_donor_mask
        both = current_donor_mask & current_acceptor_mask

        hbond_all[donor_only] = HBondEnum.DONOR
        hbond_all[acceptor_only] = HBondEnum.ACCEPTOR
        hbond_all[both] = HBondEnum.BOTH

    # Set annotations on the atom array
    atom_array.set_annotation("hbond_gt", hbond_all)

    return atom_array, motif_hbonds, len(motif_hbonds), False


def add_hbond_condition(
    data: dict,
    cutoff_HA_dist: float = 2.2,
    cutoff_DA_distance: float = 3.2,
    p_hbond_subsample: float = 0.5,
) -> dict:
    # Keep a reference to the ORIGINAL atom_array 
    atom_array_orig: AtomArray = as_atom_array_plus(data["atom_array"])

    # Get mask for valid atoms to receive hbond conditioning. In RF3 this is everything because everything is "sequence-conditioned"
    motif_mask = np.ones(len(atom_array_orig), dtype=bool)

    atom_array_with_hbond, _hbonds, _, hbond_calculation_failed = calculate_hbonds(
        atom_array_orig,
        motif_mask=motif_mask,
        cutoff_HA_dist=cutoff_HA_dist,
        cutoff_DA_distance=cutoff_DA_distance,
        raise_on_failure=False,
    )

    data.setdefault("log_dict", {})
    log_dict = data["log_dict"]

    # NOTE: If the calculation failed, log it but then behave as if we had skipped this modifier.
    # AddIsXFeats will add the default feature (zeros).
    if hbond_calculation_failed:
        log_dict["hbond_calculation_failed"] = True
        return data

    log_dict["hbond_calculation_failed"] = False

    # Pull the hbond_gt annotation from the (mangled) returned copy.
    hbond_gt_full = atom_array_with_hbond.hbond_gt.copy()

    # Persist the FULL `hbond_gt` annotation on the original atom_array. 
    atom_array_orig.set_annotation("hbond_gt", hbond_gt_full)

    # Count total hydrogen bonds
    n_hbonds = np.sum(hbond_gt_full != HBondEnum.NONE)
    log_dict["hbond_total_count"] = n_hbonds

    # Subsample hydrogen bonds for model conditioning (separate from `hbond_gt`).
    hbond_gt_for_condition = hbond_gt_full.copy()
    should_subsample = np.random.rand() < p_hbond_subsample
    if should_subsample and n_hbonds > 3:
        base_fraction = 0.01
        max_fraction = 0.2
        max_hbonds = 20

        fraction = max_fraction - (max_fraction - base_fraction) * min(n_hbonds / max_hbonds, 1.0)
        hbond_gt_for_condition = subsample_hbond_enum(hbond_gt_for_condition, fraction)

    # Set the (subsampled) hbond condition on the ORIGINAL atom_array
    # (chain_id intact). This becomes `condition_hbond_1_atom` for the model.
    C_HB.set_annotation(atom_array_orig, hbond_gt_for_condition)
    log_dict["hbond_subsample_atoms"] = np.sum(hbond_gt_for_condition != HBondEnum.NONE)

    data["log_dict"] = log_dict
    data["atom_array"] = atom_array_orig

    return data


def subsample_hbond_enum(hbond_array: np.ndarray, fraction: float) -> np.ndarray:
    """Subsample hydrogen bond annotations by randomly keeping a fraction of them.

    Args:
        hbond_array: Array with HBondEnum values
        fraction: Fraction of hydrogen bonds to keep (0 < fraction <= 1)

    Returns:
        New array with subsampled hydrogen bonds
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    hbond_array = hbond_array.copy()

    # Find all atoms that have hydrogen bonds (donor or acceptor)
    hbond_indices = np.where(hbond_array != HBondEnum.NONE)[0]
    num_hbonds = len(hbond_indices)
    keep_count = int(num_hbonds * fraction)

    # Randomly select which hydrogen bonds to keep
    np.random.shuffle(hbond_indices)
    remove_indices = hbond_indices[keep_count:]

    # Set removed hydrogen bonds to NONE
    hbond_array[remove_indices] = HBondEnum.NONE

    return hbond_array
