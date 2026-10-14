import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from atomworks.common import as_list
from atomworks.constants import CRYSTALLIZATION_AIDS
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io import parse
from atomworks.io.config import PrepareConfig
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.selection import AtomSelectionStack
from atomworks.ml.conditions import C_DIS
from atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from biotite.structure import AtomArray
from rf3.conditions import (
    C_HB,
    C_HOT,
    C_ITYPE,
    C_RAS,
    HBondEnum,
    ItypeEnum,
    RasaEnum,
)
from rf3.utils.io import (
    CIF_LIKE_EXTENSIONS,
    DICTIONARY_LIKE_EXTENSIONS,
    get_sharded_output_path,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# CFG condition name -> InferenceInput field. Single source of truth for the guidable
# axes: the inference engine builds its guidance streams from this.
CFG_CONDITION_FIELDS = {
    "template": "template_selection",
    "relative_template": "relative_template_selection",
    "ground_truth_conformer": "ground_truth_conformer_selection",
    "hotspot": "hotspot_selection",
    "interface_type": "interface_type",
    "rasa": "rasa_selection",
    "hbond": "hbond_selection",
    "distance_template": "distance_template_selection",
}
CONDITION_FIELDS = tuple(CFG_CONDITION_FIELDS.values())

# Axes that cannot be featurized on their own. A guidance stream keeps only the axis it
# guides and drops every other condition, so an axis whose annotation is written onto
# *another* condition's atoms would otherwise produce a stream identical to the
# unconditional one -- guidance of exactly zero, with nothing to show it. Isolating such
# an axis keeps its prerequisites as well. `interface_type` in its single-label form is
# written onto the hotspot atoms (see `apply_interface_type`); its {selection: label}
# form stands alone and needs no prerequisite.
CFG_CONDITION_PREREQUISITES = {"interface_type": ("hotspot",)}


def cfg_axis_fields_to_keep(input_spec, cond_name: str, field: str) -> set[str]:
    """Condition fields a guidance stream for ``cond_name`` must keep besides its own.

    Raises if a prerequisite is needed but absent, since the stream would silently carry
    no condition at all.
    """
    keep = {field}
    value = getattr(input_spec, field, None)
    # The per-selection dict form carries its own selections, so it needs no prerequisite
    # and must not drag one in: that would make its stream guide the prerequisite too.
    if isinstance(value, dict):
        return keep
    for prereq in CFG_CONDITION_PREREQUISITES.get(str(cond_name), ()):
        prereq_field = CFG_CONDITION_FIELDS[prereq]
        if getattr(input_spec, prereq_field, None):
            keep.add(prereq_field)
        else:
            raise ValueError(
                f"CFG condition '{cond_name}' is written onto the atoms selected by "
                f"'{prereq}', so guiding it requires '{prereq_field}' in the input, or "
                f"the per-selection dict form of '{field}'."
            )
    return keep


def _resolve_override(override_value, source_value, param_name: str, example_id: str):
    """Resolve CLI override vs source value with warning."""
    if override_value is not None and source_value:
        logger.warning(f"CLI {param_name} overriding source value for {example_id}")
        return override_value
    return override_value if override_value is not None else source_value


def _normalize_relative_template_selection(
    relative_template_selection: dict[str, list[str] | str] | list[str] | str | None,
) -> dict[str, list[str]] | list[str] | None:
    """Normalize relative template selections into lists (grouped dict or flat list)."""
    if relative_template_selection is None:
        return None
    if isinstance(relative_template_selection, dict):
        return {
            str(group): as_list(selection)
            for group, selection in relative_template_selection.items()
        }
    return as_list(relative_template_selection)


def extract_example_id_from_path(path: Path) -> str:
    """Extract example ID from file path."""
    path_str = str(path.name)
    # Check for known extensions (longer matches first to handle .cif.gz before .gz)
    for ext in sorted(CIF_LIKE_EXTENSIONS | {".json"}, key=len, reverse=True):
        if path_str.endswith(ext):
            return path_str[: -len(ext)]
    # Fallback to simple stem
    return path.stem


def extract_example_ids_from_json(path: Path) -> list[str]:
    """Extract example IDs from a JSON file containing one or more examples."""
    with open(path, "r") as f:
        data = json.load(f)
    return [ex["name"] for ex in data]


def _atomize_distance_template_residues(
    atom_array: AtomArray, distance_template_selection: dict | list | str | None
) -> AtomArray:
    """Atomize every residue touched by a ``distance_template_selection``, whole-residue.

    Without this the trunk averages atom-pair distances token-wise, which erases the
    side-chain-level distance information the condition is meant to carry. Every
    ``InferenceInput`` constructor calls it so the CIF/AtomArray entry points behave
    like the JSON one; it has to stay at construction time, otherwise the guided and
    unconditional CFG streams would be tokenized differently and their deltas would
    not be comparable.
    """
    atoms = [
        atom
        for group in _normalize_distance_groups(distance_template_selection)
        for atom in as_list(group["atoms"])
    ]
    return atomize_selection_on_atom_array(atom_array, atoms)


@dataclass
class InputOptions:
    """Options controlling how inference inputs are parsed and selected."""

    template_selection: list[str] | str | None = None
    relative_template_selection: dict[str, list[str]] | list[str] | str | None = None
    ground_truth_conformer_selection: list[str] | str | None = None
    # Design-task conditions (written as AtomArray annotations, featurized by
    # AddConditionsAsFeatures, guidable as independent CFG axes).
    hotspot_selection: list[str] | str | None = None
    interface_type: dict[str, str] | str | None = None
    rasa_selection: dict[str, str] | list[str] | str | None = None
    hbond_selection: dict[str, str] | list[str] | str | None = None
    distance_template_selection: dict | list[str] | str | None = None
    # Motif guidance: a sampler-side constraint, NOT a trained condition. Deliberately
    # absent from CFG_CONDITION_FIELDS, never featurized, and never atomized, so it is
    # neither guidable as a CFG axis nor able to change tokenization.
    motif_guidance_selection: dict | list[str] | str | None = None
    motif_guidance_reference: list[dict] | dict | str | None = None
    keep_all: bool = True


@dataclass
class InferenceInput:
    """Input specification for RF3 inference."""

    atom_array: AtomArray
    chain_info: dict
    example_id: str
    template_selection: list[str] | None = None
    relative_template_selection: dict[str, list[str]] | list[str] | None = None
    ground_truth_conformer_selection: list[str] | None = None
    hotspot_selection: list[str] | str | None = None
    interface_type: dict[str, str] | str | None = None
    rasa_selection: dict[str, str] | list[str] | str | None = None
    hbond_selection: dict[str, str] | list[str] | str | None = None
    distance_template_selection: dict | list[str] | str | None = None
    # Declared fields, not ad-hoc attributes: `dataclasses.replace` is used to null the
    # condition fields when building CFG streams, and it drops anything undeclared.
    motif_guidance_selection: dict | list[str] | str | None = None
    motif_guidance_reference: list[dict] | dict | str | None = None
    cyclic_chains: list[str] | None = None

    @classmethod
    def from_cif_path(
        cls,
        path: PathLike,
        example_id: str | None = None,
        input_options: "InputOptions | None" = None,
    ) -> "InferenceInput":
        """Load from CIF/PDB file.

        Args:
          path: Path to CIF/PDB file.
          example_id: Example ID. Defaults to filename stem.
          input_options: Selection and parsing overrides. If None, uses defaults.

        Returns:
          InferenceInput object.
        """
        cfg = input_options or InputOptions()

        parsed = parse(
            path,
            hydrogen_policy="remove",
            keep_cif_block=True,
            remove_ccds=[] if cfg.keep_all else None,
            remove_waters=not cfg.keep_all,
        )

        atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )

        example_id = example_id or extract_example_id_from_path(Path(path))

        atom_array = _atomize_distance_template_residues(
            atom_array, cfg.distance_template_selection
        )

        # Extract from CIF
        cif_template_sel = None
        cif_conformer_sel = None
        if "cif_block" in parsed:
            template_dict = category_to_dict(parsed["cif_block"], "template_selection")
            if template_dict:
                cif_template_sel = list(template_dict.get("template_selection", []))

            conformer_dict = category_to_dict(
                parsed["cif_block"], "ground_truth_conformer_selection"
            )
            if conformer_dict:
                cif_conformer_sel = list(
                    conformer_dict.get("ground_truth_conformer_selection", [])
                )

        # Resolve overrides (CLI priority)
        final_template_sel = _resolve_override(
            cfg.template_selection, cif_template_sel, "template_selection", example_id
        )
        final_conformer_sel = _resolve_override(
            cfg.ground_truth_conformer_selection,
            cif_conformer_sel,
            "ground_truth_conformer_selection",
            example_id,
        )

        return cls(
            atom_array=atom_array,
            chain_info=parsed["chain_info"],
            example_id=example_id,
            template_selection=final_template_sel,
            relative_template_selection=_normalize_relative_template_selection(
                cfg.relative_template_selection
            ),
            ground_truth_conformer_selection=final_conformer_sel,
            hotspot_selection=as_list(cfg.hotspot_selection)
            if cfg.hotspot_selection
            else None,
            interface_type=cfg.interface_type,
            rasa_selection=cfg.rasa_selection,
            hbond_selection=cfg.hbond_selection,
            distance_template_selection=cfg.distance_template_selection,
            motif_guidance_selection=cfg.motif_guidance_selection,
            motif_guidance_reference=cfg.motif_guidance_reference,
        )

    @classmethod
    def from_json_dict(
        cls,
        data: dict,
        input_options: "InputOptions | None" = None,
    ) -> "InferenceInput":
        """Create from JSON dict with components.

        CLI args override JSON metadata.

        Args:
          data: JSON dictionary with components.
          input_options: Selection and parsing overrides. If None, uses defaults.

        Returns:
          InferenceInput object.
        """
        cfg = input_options or InputOptions()

        # Build atom_array from components
        atom_array, component_list = components_to_atom_array(
            data["components"],
            bonds=data.get("bonds"),
            return_components=True,
        )

        parsed = parse_atom_array(
            atom_array,
            config=PrepareConfig(
                hydrogen_policy="keep",
                remove_ccds=() if cfg.keep_all else tuple(CRYSTALLIZATION_AIDS),
                remove_waters=not cfg.keep_all,
            ),
        )

        chain_info = parsed.get("chain_info", {})
        atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )

        # Merge MSA paths into chain_info
        msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(
            component_list
        )
        if data.get("msa_paths") and isinstance(data.get("msa_paths"), dict):
            msa_paths_by_chain_id.update(data.get("msa_paths"))

        for chain_id, msa_path in msa_paths_by_chain_id.items():
            if chain_id in chain_info:
                chain_info[chain_id]["msa_path"] = msa_path

        # Resolve overrides (CLI priority)
        final_template_sel = _resolve_override(
            cfg.template_selection,
            data.get("template_selection"),
            "template_selection",
            data["name"],
        )
        final_relative_template_sel = _resolve_override(
            cfg.relative_template_selection,
            data.get("relative_template_selection"),
            "relative_template_selection",
            data["name"],
        )
        final_conformer_sel = _resolve_override(
            cfg.ground_truth_conformer_selection,
            data.get("ground_truth_conformer_selection"),
            "ground_truth_conformer_selection",
            data["name"],
        )
        final_hotspot_sel = _resolve_override(
            cfg.hotspot_selection,
            data.get("hotspot_selection"),
            "hotspot_selection",
            data["name"],
        )
        final_interface_type = _resolve_override(
            cfg.interface_type,
            data.get("interface_type"),
            "interface_type",
            data["name"],
        )
        final_rasa_sel = _resolve_override(
            cfg.rasa_selection,
            data.get("rasa_selection"),
            "rasa_selection",
            data["name"],
        )
        final_hbond_sel = _resolve_override(
            cfg.hbond_selection,
            data.get("hbond_selection"),
            "hbond_selection",
            data["name"],
        )
        final_distance_template_sel = _resolve_override(
            cfg.distance_template_selection,
            data.get("distance_template_selection"),
            "distance_template_selection",
            data["name"],
        )
        atom_array = _atomize_distance_template_residues(
            atom_array, final_distance_template_sel
        )
        # No atomization pass for motif guidance: it acts on the sampler's atom axis,
        # where every atom exists regardless of tokenization, so it has no reason to
        # change it -- unlike distance_template above.
        final_motif_guidance_sel = _resolve_override(
            cfg.motif_guidance_selection,
            data.get("motif_guidance_selection"),
            "motif_guidance_selection",
            data["name"],
        )
        final_motif_guidance_ref = _resolve_override(
            cfg.motif_guidance_reference,
            data.get("motif_guidance_reference"),
            "motif_guidance_reference",
            data["name"],
        )
        return cls(
            atom_array=atom_array,
            chain_info=chain_info,
            example_id=data["name"],
            template_selection=final_template_sel,
            relative_template_selection=_normalize_relative_template_selection(
                final_relative_template_sel
            ),
            ground_truth_conformer_selection=final_conformer_sel,
            hotspot_selection=as_list(final_hotspot_sel) if final_hotspot_sel else None,
            interface_type=final_interface_type,
            rasa_selection=final_rasa_sel,
            hbond_selection=final_hbond_sel,
            distance_template_selection=final_distance_template_sel,
            motif_guidance_selection=final_motif_guidance_sel,
            motif_guidance_reference=final_motif_guidance_ref,
        )

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        chain_info: dict | None = None,
        example_id: str | None = None,
        input_options: "InputOptions | None" = None,
    ) -> "InferenceInput":
        """Create from AtomArray.

        Args:
          atom_array: Input AtomArray.
          chain_info: Chain info dict. Defaults to extracted from atom_array.
          example_id: Example ID. Defaults to generated ID.
          input_options: Selection and parsing overrides. If None, uses defaults.

        Returns:
          InferenceInput object.
        """
        cfg = input_options or InputOptions()

        # Use parse_atom_array
        parsed = parse_atom_array(
            atom_array,
            config=PrepareConfig(
                hydrogen_policy="keep",
                remove_ccds=() if cfg.keep_all else tuple(CRYSTALLIZATION_AIDS),
                remove_waters=not cfg.keep_all,
            ),
        )

        extracted_chain_info = parsed.get("chain_info", {})

        # Merge with provided chain_info (provided takes priority)
        if chain_info is not None:
            for chain_id, chain_data in chain_info.items():
                if chain_id in extracted_chain_info:
                    extracted_chain_info[chain_id].update(chain_data)
                else:
                    extracted_chain_info[chain_id] = chain_data

        final_atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )
        final_atom_array = _atomize_distance_template_residues(
            final_atom_array, cfg.distance_template_selection
        )

        return cls(
            atom_array=final_atom_array,
            chain_info=extracted_chain_info,
            example_id=example_id or f"inference_{id(atom_array)}",
            template_selection=cfg.template_selection,
            relative_template_selection=_normalize_relative_template_selection(
                cfg.relative_template_selection
            ),
            ground_truth_conformer_selection=cfg.ground_truth_conformer_selection,
            hotspot_selection=as_list(cfg.hotspot_selection)
            if cfg.hotspot_selection
            else None,
            interface_type=cfg.interface_type,
            rasa_selection=cfg.rasa_selection,
            hbond_selection=cfg.hbond_selection,
            distance_template_selection=cfg.distance_template_selection,
            motif_guidance_selection=cfg.motif_guidance_selection,
            motif_guidance_reference=cfg.motif_guidance_reference,
        )

    def to_pipeline_input(self) -> dict:
        """Apply transformations and return input for Transform pipeline.

        Returns:
          Pipeline input dict with example_id, atom_array, and chain_info.
        """
        atom_array = self.atom_array.copy()

        # Apply template and conformer selections (NaN clobber relocated below so the
        # design conditions can read true coordinates).
        atom_array = apply_conformer_and_template_selections(
            atom_array,
            template_selection=self.template_selection,
            relative_template_selection=self.relative_template_selection,
            ground_truth_conformer_selection=self.ground_truth_conformer_selection,
        )

        # Design-task conditions read coordinates (distance/rasa), so run BEFORE the
        # NaN->-1 clobber. May return an AtomArrayPlus (keeps the 2-body C_DIS alive).
        atom_array = apply_design_conditions(
            atom_array,
            hotspot_selection=self.hotspot_selection,
            interface_type=self.interface_type,
            rasa_selection=self.rasa_selection,
            hbond_selection=self.hbond_selection,
            distance_template_selection=self.distance_template_selection,
        )

        # Safety: avoid unexpected behavior downstream while preserving the fact that
        # unresolved atoms selected for token-level templating are not real conditions.
        # Do not zero occupancy for every non-finite input coordinate: sequence-only
        # query atoms intentionally start with NaN coordinates and still need to survive
        # the topology pipeline.
        unresolved_coord_mask = ~np.isfinite(atom_array.coord).all(axis=-1)
        templated_mask = atom_array.get_annotation("is_input_file_templated").astype(
            bool
        )
        atom_array.occupancy[unresolved_coord_mask & templated_mask] = 0.0
        atom_array.coord[unresolved_coord_mask] = -1.0

        if self.cyclic_chains:
            atom_array = cyclize_atom_array(atom_array, self.cyclic_chains)

        return {
            "example_id": self.example_id,
            "atom_array": atom_array,
            "chain_info": self.chain_info,
        }


def _process_single_path(
    path: Path,
    existing_outputs_dir: Path | None,
    sharding_pattern: str | None,
    input_options: "InputOptions | None" = None,
) -> list[InferenceInput]:
    """Worker function to process a single input file path.

    This function is defined at module level to be picklable for multiprocessing.

    Args:
      path: Path to a single input file.
      existing_outputs_dir: If set, skip examples with existing outputs.
      sharding_pattern: Sharding pattern for output paths.
      input_options: Selection and parsing overrides. If None, uses defaults.

    Returns:
      List of InferenceInput objects (may be empty if file is skipped).
    """

    def example_exists(example_id: str) -> bool:
        """Check if example already has predictions (sharding-aware)."""
        if not existing_outputs_dir:
            return False
        example_dir = get_sharded_output_path(
            example_id, existing_outputs_dir, sharding_pattern
        )
        return (example_dir / f"{example_id}_metrics.csv").exists()

    inference_inputs = []

    if path.suffix == ".json":
        # Load JSON and convert each entry
        with open(path, "r") as f:
            data = json.load(f)

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        for item in data:
            example_id = item["name"]
            if not example_exists(example_id):
                inference_inputs.append(
                    InferenceInput.from_json_dict(
                        item,
                        input_options=input_options,
                    )
                )

    elif any(path.name.endswith(ext) for ext in CIF_LIKE_EXTENSIONS):
        # CIF/PDB file
        example_id = extract_example_id_from_path(path)
        if not example_exists(example_id):
            inference_inputs.append(
                InferenceInput.from_cif_path(
                    path,
                    example_id=example_id,
                    input_options=input_options,
                )
            )
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix} (path: {path}). "
            f"Supported: {CIF_LIKE_EXTENSIONS | DICTIONARY_LIKE_EXTENSIONS}"
        )

    return inference_inputs


def prepare_inference_inputs_from_paths(
    inputs: PathLike | list[PathLike],
    existing_outputs_dir: PathLike | None = None,
    sharding_pattern: str | None = None,
    disable_multiprocessing: bool = False,
    input_options: "InputOptions | None" = None,
) -> list[InferenceInput]:
    """Load InferenceInput objects from file paths.

    Handles CIF, PDB, and JSON files. Filters out existing outputs if requested.
    Uses multiprocessing to parallelize file loading across all available CPUs,
    unless disabled (e.g., when already in a distributed context).

    Args:
      inputs: File path(s) or directory path(s).
      existing_outputs_dir: If set, skip examples with existing outputs.
      sharding_pattern: Sharding pattern for output paths.
      disable_multiprocessing: If True, process files serially. Useful within
          distributed contexts where forking is unsafe.
      input_options: Selection and parsing overrides. If None, uses defaults.

    Returns:
      List of InferenceInput objects.
    """
    input_paths = as_list(inputs)

    # Collect all raw input files (reusing logic from build_file_paths_for_prediction)
    paths_to_raw_input_files = []
    for _path in input_paths:
        if Path(_path).is_dir():
            # Scan directory for supported file types (JSON + CIF-like)
            for file_type in CIF_LIKE_EXTENSIONS | DICTIONARY_LIKE_EXTENSIONS:
                paths_to_raw_input_files.extend(Path(_path).rglob(f"*{file_type}"))
        else:
            paths_to_raw_input_files.append(Path(_path))

    # Convert existing_outputs_dir to Path if needed
    existing_outputs_dir_path = (
        Path(existing_outputs_dir) if existing_outputs_dir else None
    )

    inference_inputs = []
    if disable_multiprocessing:
        logger.info(f"Processing {len(paths_to_raw_input_files)} files serially")
        for path in paths_to_raw_input_files:
            result = _process_single_path(
                path,
                existing_outputs_dir_path,
                sharding_pattern,
                input_options,
            )
            inference_inputs.extend(result)
    else:
        num_cpus = min(os.cpu_count() or 1, len(paths_to_raw_input_files))
        logger.info(
            f"Processing {len(paths_to_raw_input_files)} files using {num_cpus} CPUs"
        )
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(
                    _process_single_path,
                    path,
                    existing_outputs_dir_path,
                    sharding_pattern,
                    input_options,
                )
                for path in paths_to_raw_input_files
            ]
            for future in futures:
                inference_inputs.extend(future.result())

    logger.info(f"Loaded {len(inference_inputs)} inference inputs")
    return inference_inputs


def apply_atom_selection_mask(
    atom_array: AtomArray, selection_list: Iterable[str]
) -> np.ndarray:
    """Return a combined boolean mask for a list of AtomSelectionStack queries.

    Args:
        atom_array: AtomArray to select from.
        selection_list: Iterable of AtomSelectionStack queries (e.g., "*/LIG", "A1-10").

    Returns:
        A boolean numpy array of shape (num_atoms,) where True indicates a selected atom.
    """
    selection_mask = np.zeros(len(atom_array), dtype=bool)
    for selection in selection_list:
        if not selection:
            continue
        try:
            selector = AtomSelectionStack.from_query(selection)
            mask = selector.get_mask(atom_array)
        except Exception as exc:
            # Skipping would silently drop the condition (or a whole CFG stream) and look
            # like a run that simply was not guided, so fail loudly instead.
            raise ValueError(
                f"Could not resolve selection '{selection}': {exc}. Note that a residue "
                "range matches only if every residue in it is present."
            ) from exc
        selection_mask = selection_mask | mask
    return selection_mask


def apply_template_selection(
    atom_array: AtomArray, template_selection: list[str] | str | None
) -> AtomArray:
    """Apply token-level template selection to `atom_array` with OR semantics.

    If the `is_input_file_templated` annotation already exists, this function ORs
    the new selection with the existing annotation. Otherwise, it creates it.

    Args:
        atom_array: AtomArray to annotate.
        template_selection: Selection string(s). Single strings are converted to lists. If None/empty, no-op.

    Returns:
        The same AtomArray with `is_input_file_templated` updated.
    """
    # Convert to list if needed
    template_selection_list = as_list(template_selection) if template_selection else []

    if not template_selection_list:
        # Ensure the annotation exists even if no selection provided
        if "is_input_file_templated" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "is_input_file_templated", np.zeros(len(atom_array), dtype=bool)
            )
        return atom_array

    # Build new mask
    selection_mask = apply_atom_selection_mask(atom_array, template_selection_list)
    logging.info(
        "Selected %d atoms for token-level templating with %d syntaxes",
        int(np.sum(selection_mask)),
        len([s for s in template_selection_list if s]),
    )

    # OR with existing annotation if present
    if "is_input_file_templated" in atom_array.get_annotation_categories():
        existing = atom_array.get_annotation("is_input_file_templated").astype(bool)
        selection_mask = existing | selection_mask
    atom_array.set_annotation("is_input_file_templated", selection_mask)
    return atom_array


def apply_relative_template_selection_mask(
    atom_array: AtomArray,
    relative_template_selection: dict[str, list[str]] | list[str] | str | None,
) -> dict[str, np.ndarray]:
    """Return per-group relative-template masks from selection syntax.

    A flat list is treated as a single ``__all__`` group.
    """
    normalized = _normalize_relative_template_selection(relative_template_selection)
    if not normalized:
        return {}
    if isinstance(normalized, dict):
        return {
            str(group): apply_atom_selection_mask(atom_array, selection_list)
            for group, selection_list in normalized.items()
            if selection_list
        }
    return {"__all__": apply_atom_selection_mask(atom_array, normalized)}


def apply_relative_template_selection(
    atom_array: AtomArray,
    relative_template_selection: dict[str, list[str]] | list[str] | str | None,
) -> AtomArray:
    """Apply inter-component (relative) template selection with OR semantics.

    Writes a boolean ``is_relative_template_selected__<group>`` annotation per group;
    the relative-template featurizer reads these to add inter-chain conditioning.
    """
    group_masks = apply_relative_template_selection_mask(
        atom_array, relative_template_selection
    )
    for group, selection_mask in group_masks.items():
        annotation_name = f"is_relative_template_selected__{group}"
        if annotation_name in atom_array.get_annotation_categories():
            existing = atom_array.get_annotation(annotation_name).astype(bool)
            selection_mask = existing | selection_mask
        logging.info(
            "Selected %d atoms for inter-component templating group '%s'",
            int(np.sum(selection_mask)),
            group,
        )
        atom_array.set_annotation(annotation_name, selection_mask)
    return atom_array


def apply_ground_truth_conformer_selection(
    atom_array: AtomArray,
    ground_truth_conformer_selection: list[str] | str | None,
) -> AtomArray:
    """Apply ground-truth conformer selection with union semantics.

    If a ``ground_truth_conformer_policy`` annotation already exists on
    *atom_array* it is preserved: selected atoms are upgraded to ADD but
    existing stronger policies (e.g. REPLACE) are never downgraded.
    When no annotation exists yet one is created initialised to IGNORE.

    Args:
        atom_array: AtomArray to annotate.
        ground_truth_conformer_selection: Selection string(s).

    Returns:
        The same AtomArray with ``ground_truth_conformer_policy`` updated.
    """
    # Preserve existing annotation if present; otherwise start at IGNORE
    if "ground_truth_conformer_policy" in atom_array.get_annotation_categories():
        annotation = atom_array.get_annotation("ground_truth_conformer_policy").astype(
            np.int8
        )
    else:
        annotation = np.full(
            len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8
        )

    # Upgrade selected atoms to ADD (lower value = stronger policy)
    selection_list = (
        as_list(ground_truth_conformer_selection)
        if ground_truth_conformer_selection
        else []
    )
    if selection_list:
        mask = apply_atom_selection_mask(atom_array, selection_list)
        annotation[mask] = np.minimum(annotation[mask], GroundTruthConformerPolicy.ADD)
        logging.info(
            "Applied ground-truth conformer ADD for %d atom(s) from %d selection(s)",
            int(np.sum(mask)),
            len(selection_list),
        )

    atom_array.set_annotation("ground_truth_conformer_policy", annotation)
    return atom_array


def apply_conformer_and_template_selections(
    atom_array: AtomArray,
    template_selection: list[str] | str | None = None,
    relative_template_selection: dict[str, list[str]] | list[str] | str | None = None,
    ground_truth_conformer_selection: list[str] | str | None = None,
) -> AtomArray:
    """Apply template, relative-template, and conformer selections.

    Applies `apply_template_selection`, `apply_relative_template_selection`, then
    `apply_ground_truth_conformer_selection`. The NaN-coordinate clobber is done by
    the caller (`to_pipeline_input`) after design conditions, which read true coords.

    Args:
        atom_array: AtomArray to prepare.
        template_selection: Template selection string(s). Single strings are converted to lists.
        relative_template_selection: Inter-component (relative) template selection; grouped dict or flat list.
        ground_truth_conformer_selection: Ground-truth conformer selection string(s). Single strings are converted to lists.

    Returns:
        The same AtomArray with `is_input_file_templated` and `ground_truth_conformer_policy` updated.
    """
    atom_array = apply_template_selection(atom_array, template_selection)
    atom_array = apply_relative_template_selection(
        atom_array, relative_template_selection
    )
    atom_array = apply_ground_truth_conformer_selection(
        atom_array, ground_truth_conformer_selection
    )
    return atom_array


def _enum_value(enum_cls, label, param_name: str) -> int:
    """Look up a condition label in its IntEnum, reporting the legal labels on failure."""
    try:
        return int(enum_cls[label])
    except KeyError as exc:
        raise ValueError(
            f"{param_name}: unknown label {label!r}; expected one of "
            f"{[member.name for member in enum_cls]}."
        ) from exc


def apply_hotspot_selection(
    atom_array: AtomArray, hotspot_selection: list[str] | str | None
) -> np.ndarray | None:
    """Write the hotspot condition (C_HOT bool mask) from an atom selection.

    Returns the hotspot atom mask (consumed by interface-type) or None if no selection.
    """
    if not hotspot_selection:
        return None
    mask = apply_atom_selection_mask(atom_array, as_list(hotspot_selection))
    C_HOT.set_annotation(atom_array=atom_array, array=mask)
    C_HOT.set_mask(atom_array, mask, 1, "atom")
    logging.info("Selected %d hotspot atoms", int(mask.sum()))
    return mask


def apply_interface_type(
    atom_array: AtomArray,
    interface_type: dict[str, str] | str | None,
    hotspot_mask: np.ndarray | None,
) -> None:
    """Write the interface-type condition (C_ITYPE enum: PPI/PLI/PNA), meaningful only on
    hotspot atoms. A single label applies to all hotspot atoms; a {selection: label} dict
    writes per selection.
    """
    if not interface_type:
        return
    itype_array = np.full(len(atom_array), ItypeEnum.NONE, dtype=int)
    if isinstance(interface_type, dict):
        for sel, label in interface_type.items():
            itype_array[apply_atom_selection_mask(atom_array, [sel])] = _enum_value(
                ItypeEnum, label, "interface_type"
            )
    elif hotspot_mask is None:
        logging.warning(
            "interface_type given without hotspot_selection; interface type is only "
            "meaningful on hotspot atoms and will not be applied."
        )
        return
    else:
        itype_array[hotspot_mask] = _enum_value(
            ItypeEnum, interface_type, "interface_type"
        )
    C_ITYPE.set_annotation(atom_array=atom_array, array=itype_array)


def apply_rasa_selection(
    atom_array: AtomArray, rasa_selection: dict[str, str] | list[str] | str | None
) -> None:
    """Write the RASA condition (C_RAS enum).

    A {selection: label} dict sets explicit RasaEnum labels; a list/str auto-computes
    per-atom relative SASA over the selected atoms and bins into BURIED/INTERMEDIATE/EXPOSED
    (matching training's `add_rasa_condition`). SASA is biotite Shrake-Rupley (no external dep).
    """
    if not rasa_selection:
        return
    enum_values = C_RAS.default_annotation(atom_array)  # all UNSPECIFIED
    if isinstance(rasa_selection, dict):
        for sel, label in rasa_selection.items():
            enum_values[apply_atom_selection_mask(atom_array, [sel])] = _enum_value(
                RasaEnum, label, "rasa_selection"
            )
    else:
        inclusion = apply_atom_selection_mask(atom_array, as_list(rasa_selection))
        rasa = calculate_atomwise_rasa(
            atom_array,
            probe_radius=1.4,
            atom_radii="ProtOr",
            point_number=100,
            atom_filter=inclusion,
        )
        # 3 bins match RasaEnum: [0,0.1)->BURIED, [0.1,0.2)->INTERMEDIATE, >=0.2 & NaN->EXPOSED
        bins = np.digitize(rasa, np.linspace(0, 0.2, 3), right=False)
        if (bins[inclusion] == 0).any():
            raise ValueError("Negative RASA detected while binning rasa_selection.")
        enum_values[bins == 1] = RasaEnum.BURIED
        enum_values[bins == 2] = RasaEnum.INTERMEDIATE
        enum_values[bins == 3] = RasaEnum.EXPOSED
        enum_values[~inclusion] = RasaEnum.UNSPECIFIED
    C_RAS.set_annotation(atom_array, enum_values.astype(int))


def apply_hbond_selection(
    atom_array: AtomArray, hbond_selection: dict[str, str] | list[str] | str | None
) -> None:
    """Write the hydrogen-bond condition (C_HB enum) from explicit donor/acceptor labels.

    Only the explicit {selection: label} form (DONOR/ACCEPTOR/BOTH/NONE) is supported at
    inference; auto hbplus computation is intentionally not wired here (external dependency).
    """
    if not hbond_selection:
        return
    if not isinstance(hbond_selection, dict):
        raise ValueError(
            "hbond_selection must be a {selection: label} dict at inference "
            "(auto hbplus computation is not supported); got a list/str."
        )
    values = C_HB.default_annotation(atom_array)  # all NONE
    for sel, label in hbond_selection.items():
        values[apply_atom_selection_mask(atom_array, [sel])] = _enum_value(
            HBondEnum, label, "hbond_selection"
        )
    C_HB.set_annotation(atom_array, values.astype(int))


def _normalize_distance_groups(
    distance_template_selection: dict | list | str | None,
) -> list[dict]:
    """Normalize a ``distance_template_selection`` to a list of ``{"atoms": [...], "distance": ...}`` groups.

    Accepts a single group dict, a list of group dicts, or a flat list/str of selections
    (treated as one auto-computed group). Returns ``[]`` for an empty selection.
    """
    if not distance_template_selection:
        return []
    if isinstance(distance_template_selection, dict):
        return [distance_template_selection]
    if (
        isinstance(distance_template_selection, list)
        and distance_template_selection
        and isinstance(distance_template_selection[0], dict)
    ):
        return distance_template_selection
    return [{"atoms": as_list(distance_template_selection)}]


def _resolve_single_atom_index(atom_array: AtomArray, selection: str) -> int:
    """Resolve a selection that must match exactly one atom, returning its index."""
    idx = np.where(apply_atom_selection_mask(atom_array, [selection]))[0]
    if len(idx) != 1:
        raise ValueError(
            f"distance matrix requires each atom selection to match exactly one atom; "
            f"'{selection}' matched {len(idx)}."
        )
    return int(idx[0])


def apply_distance_template_selection(
    atom_array: AtomArray, distance_template_selection: dict | list | str | None
) -> AtomArray:
    """Write the 2D distance condition (C_DIS) from groups of anchor atoms.

    Accepts a flat list/str of selections (one auto-computed group) or a list of group
    dicts ``{"atoms": [<selections>], "distance": <scalar | N×N matrix, optional>}``. All
    pairwise distances WITHIN each group are templated. ``distance`` can be:
      - omitted → each pair computed from the input coordinates;
      - a scalar → every pair in the group set to that value;
      - an N×N matrix (row/col ordered like ``atoms``) → pair ``(atoms[a], atoms[b])`` set
        to ``matrix[a][b]``; the diagonal is ignored. Each selection must then match
        exactly one atom.


    Returns an AtomArrayPlus (carries the 2-body annotation through collation).
    """
    if not distance_template_selection:
        return atom_array
    from rf3.data.design_tasks.distance_condition import compute_distances

    atom_array = as_atom_array_plus(atom_array)

    groups = _normalize_distance_groups(distance_template_selection)

    all_pairs, all_values = [], []
    for group in groups:
        dist = group.get("distance")
        if np.ndim(dist) >= 2:
            atoms = as_list(group["atoms"])
            idxs = [_resolve_single_atom_index(atom_array, s) for s in atoms]
            mat = np.asarray(dist, dtype=float)
            n = len(idxs)
            if mat.shape != (n, n):
                raise ValueError(
                    f"distance matrix shape {mat.shape} != ({n}, {n}) for group atoms {atoms}."
                )
            off_diag = ~np.eye(n, dtype=bool)
            a, b = np.where(
                off_diag
            )  # both (a,b) and (b,a), matching compute_distances
            pairs = np.stack([np.asarray(idxs)[a], np.asarray(idxs)[b]], axis=1)
            values = mat[a, b]
        else:
            mask = apply_atom_selection_mask(atom_array, as_list(group["atoms"]))
            pairs, values = compute_distances(
                atom_array, mask
            )  # within-group, from coords
            if dist is not None:
                values = np.full(len(pairs), float(dist), dtype=float)
        all_pairs.append(pairs)
        all_values.append(values)
    pairs = np.concatenate(all_pairs, axis=0)
    values = np.concatenate(all_values, axis=0)

    C_DIS.set_annotation(atom_array, pairs=pairs, values=values)
    # Drop NaN distances (unresolved coords), mirroring distance_condition.py.
    annot = C_DIS.annotation(atom_array)
    non_nan = ~np.isnan(annot.values)
    C_DIS.set_annotation(
        atom_array, pairs=annot.pairs[non_nan], values=annot.values[non_nan]
    )
    return atom_array


def atomize_selection_on_atom_array(
    atom_array: AtomArray, selection: list[str] | str | None
) -> AtomArray:
    """
    Atomize whole residues defined under ``distance_template_selection`` by setting the ``atomize`` flag.
    """
    if not selection:
        return atom_array
    import biotite.structure as struc

    sel = apply_atom_selection_mask(atom_array, as_list(selection))
    atomize = (
        atom_array.get_annotation("atomize").astype(bool).copy()
        if "atomize" in atom_array.get_annotation_categories()
        else np.zeros(len(atom_array), dtype=bool)
    )
    res_bounds = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for s, e in zip(res_bounds[:-1], res_bounds[1:]):
        if sel[s:e].any():
            atomize[s:e] = True
    atom_array.set_annotation("atomize", atomize)
    return atom_array


def apply_design_conditions(
    atom_array: AtomArray,
    *,
    hotspot_selection: list[str] | str | None = None,
    interface_type: dict[str, str] | str | None = None,
    rasa_selection: dict[str, str] | list[str] | str | None = None,
    hbond_selection: dict[str, str] | list[str] | str | None = None,
    distance_template_selection: dict | list | str | None = None,
) -> AtomArray:
    """Apply all design-task conditions as AtomArray annotations (each a no-op if None).

    Order: hotspot -> interface_type (uses the hotspot mask) -> rasa -> hbond -> distance.
    Return an AtomArrayPlus， if with atom_distance condition.
    """
    hotspot_mask = apply_hotspot_selection(atom_array, hotspot_selection)
    apply_interface_type(atom_array, interface_type, hotspot_mask)
    apply_rasa_selection(atom_array, rasa_selection)
    apply_hbond_selection(atom_array, hbond_selection)
    atom_array = apply_distance_template_selection(
        atom_array, distance_template_selection
    )
    return atom_array


def cyclize_atom_array(atom_array: AtomArray, cyclic_chains: list[str]) -> AtomArray:
    """Cyclize the atom array by positioining the termini properly if not already done.

    Behavior:
    - Positions the last carbon atom in the chain to be 1.3 Angstroms away from the first nitrogen atom if they are not already close.
    - Adds a bond between the termini for proper cif output.

    Args:
        atom_array: AtomArray to cyclize.
        cyclic_chains: List of chain IDs to cyclize.

    Returns:
        The same AtomArray with the specified chains cyclized.
    """
    for chain in cyclic_chains:
        # Find the first nitrogen atom in the chain
        nitrogen_mask = (atom_array.chain_id == chain) & (atom_array.atom_name == "N")
        nitrogen_mask_indices = np.where(nitrogen_mask)[0]
        first_nitrogen_index = nitrogen_mask_indices[0]
        nitrogen_coord = atom_array.coord[first_nitrogen_index]

        # move the last carbon atom in the chain to be 1.3 Angstroms away from the nitrogen
        carbon_mask = (atom_array.chain_id == chain) & (atom_array.atom_name == "C")
        carbon_mask_indices = np.where(carbon_mask)[0]
        last_carbon_index = carbon_mask_indices[-1]
        # check if the last carbon is already close to the nitrogen
        termini_distance = np.linalg.norm(
            atom_array.coord[last_carbon_index] - nitrogen_coord
        )
        if not (termini_distance < 1.5 and termini_distance > 0.5):
            atom_array.coord[last_carbon_index] = nitrogen_coord + np.array(
                [1.3, 0.0, 0.0]
            )

        # add a bond between the nitrogen and carbon so output cif has a connection
        atom_array.bonds.add_bond(first_nitrogen_index, last_carbon_index)
        atom_array.bonds.add_bond(last_carbon_index, first_nitrogen_index)

    return atom_array


class InferenceInputDataset(Dataset):
    """
    Dataset for inference inputs. Also has a length key telling you the number of tokens in each example for LoadBalancedDistributedSampler.

    To calculate the length of each example, we need to add the token_id annotation to the atom_array. If it doesn't exist yet, we add it,
    calculate the length, and then remove it since the downstream pipeline may not be expecting it. That means the num_tokens key may not ultimately
    be the same as what's actually used in the model, but this is a close enough approximation for load balancing.

    Args:
      inference_inputs: List of InferenceInput objects to wrap in a Dataset.
    """

    def __init__(self, inference_inputs: list[InferenceInput]):
        self.inference_inputs = inference_inputs
        self.key_to_balance = "num_tokens_approximate"

        # LoadBalancedDistributedSampler checks in dataset.data[key_to_balance] to determine balancing.
        # That means we need to make a dataframe in self.data that has a column with the key_to_balance.
        atom_array_token_lens = []
        for inf_input in self.inference_inputs:
            if "token_id" not in inf_input.atom_array.get_annotation_categories():
                inf_input.atom_array = add_global_token_id_annotation(
                    inf_input.atom_array
                )
                num_tokens = len(np.unique(inf_input.atom_array.token_id))

                # remove the token_id annotation since the pipeline may not be expecting it
                inf_input.atom_array.del_annotation("token_id")
            else:
                num_tokens = len(np.unique(inf_input.atom_array.token_id))
            atom_array_token_lens.append(num_tokens)
        self.data = pd.DataFrame({self.key_to_balance: atom_array_token_lens})

    def __len__(self):
        return len(self.inference_inputs)

    def __getitem__(self, idx):
        return self.inference_inputs[idx]
