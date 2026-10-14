"""Entrypoint for parsing and preparing atomic-level structures with AtomWorks.

We provide three public functions to cover the main use cases:

* :py:func:`parse` — load a file and return a full result dictionary
  (``chain_info``, ``assemblies``, ``metadata``, etc.).
* :py:func:`parse_atom_array` — process an existing AtomArray and return a
  result dictionary matching :py:func:`parse` output.
* :py:func:`prepare_atom_array` — process an existing AtomArray with AtomWorks' common
  annotations and return the processed atoms directly, without the surrounding metadata.

To control the options for parsing and preparing, either:
(a) Use the ``config`` argument to pass a configuration object (e.g. :py:class:`~atomworks.io.config.ParseConfig`) or preset name (e.g. ``"rcsb"``).
(b) (Legacy) Pass bare keyword arguments to :py:func:`parse` (e.g. ``add_missing_atoms=True``)
"""

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.common import string_to_md5_hash
from atomworks.io._loaders import load_cif, load_pdb
from atomworks.io._pipeline import _assemble_parse_result, _maybe_promote_to_plus, _prepare_atom_array_or_stack
from atomworks.io.config import ParseConfig, PrepareConfig, _resolve_config, get_config
from atomworks.io.utils.assembly import build_assemblies_from_asym_unit
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack
from atomworks.io.utils.catcif import resolve_catcif_source
from atomworks.io.utils.ccd import build_ccd_entries_from_cif_block, custom_ccd_residues, snapshot_custom_ccd_registry
from atomworks.io.utils.extra_fields import ExtraFieldsType
from atomworks.io.utils.io_utils import (
    apply_sharding_pattern,
    build_sharding_pattern,
    infer_pdb_file_type,
)
from atomworks.io.utils.standard_annotations.serialization import _deserialize_standard_annotations

logger = logging.getLogger("atomworks.io")

__all__ = ["ParseConfig", "get_config", "parse", "parse_atom_array", "prepare_atom_array"]

STANDARD_PARSER_ARGS = get_config("rcsb").to_dict()
STANDARD_PARSER_ARGS["model"] = None
"""Common parser arguments for many biomolecular use cases (deprecated, use ``get_config("rcsb")``). Will be removed in a future version."""

_CACHE_SHARDING_DEPTH = 2
_CACHE_SHARDING_CHARS_PER_DIR = 2


def _parse_args_to_hash(parse_arguments: dict[str, Any], truncate: int = 8) -> str:
    """Convert a dictionary of parsing arguments to a hash string for caching purposes"""
    args_string = ",".join(str(parse_arguments[k]) for k in sorted(parse_arguments.keys()))
    return string_to_md5_hash(args_string, truncate=truncate)


def _build_cache_file_path(cache_dir: Path, filename: os.PathLike, config: ParseConfig) -> Path:
    """Build a cache file path based: (a) the cache directory, (b) a hash of the parsing arguments, (c) the input filename, and (d) the assembly info."""
    structure_id = Path(filename).stem
    # Hash the full source path so distinct files always map to distinct cache entries.
    path_hash = string_to_md5_hash(str(filename), truncate=8)
    min_length = _CACHE_SHARDING_DEPTH * _CACHE_SHARDING_CHARS_PER_DIR
    structure_id_padded = structure_id.ljust(min_length, "_")

    sharding_pattern = build_sharding_pattern(depth=_CACHE_SHARDING_DEPTH, chars_per_dir=_CACHE_SHARDING_CHARS_PER_DIR)
    sharded_path = apply_sharding_pattern(structure_id_padded, sharding_pattern)

    # Remove variable arguments to avoid exploding hashes
    config_args = config.to_dict()
    build_assembly = config_args.pop("build_assembly")
    assembly_info = ",".join(build_assembly) if isinstance(build_assembly, list | tuple) else build_assembly
    altloc_seed = config_args.pop("altloc_seed")
    args_hash = _parse_args_to_hash(config_args)

    return (
        cache_dir
        / args_hash
        / sharded_path
        / f"{structure_id}_{path_hash}_assembly_{assembly_info}_altloc_{altloc_seed}.pkl.gz"
    )


def _attach_live_ccd_registry(results: list[dict[str, Any]]) -> None:
    """Snapshot the currently-scoped custom CCD registry and attach it to each result's ``asym_unit``."""
    ccd_registry = snapshot_custom_ccd_registry()

    if not ccd_registry:
        # No custom CCD entries currently registered
        return

    # Attach to all AtomArrayPlus-type objects
    for r in results:
        if isinstance(r["asym_unit"], AtomArrayPlus | AtomArrayPlusStack):
            r["asym_unit"]._custom_ccd_registry = ccd_registry
        for assembly in r["assemblies"].values():
            if isinstance(assembly, AtomArrayPlus | AtomArrayPlusStack):
                assembly._custom_ccd_registry = ccd_registry


def parse(
    source: os.PathLike | io.StringIO | io.BytesIO | None = None,
    *,
    config: str | ParseConfig | None = None,
    filename: os.PathLike | io.StringIO | io.BytesIO | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse structural files into an AtomArrayStack with standardized annotations and metadata.

    Processing behaviour is controlled by :py:class:`~atomworks.io.config.ParseConfig`.
    Legacy bare keyword arguments (e.g. ``add_missing_atoms=True``) are still accepted
    but will emit a :py:class:`DeprecationWarning`; prefer passing a ``config`` object.

    Args:
        source: Path or buffer to the structure file. May be any format of
            atomic-level structure (e.g. .cif, .bcif, .cif.gz, .pdb),
            although .cif files are strongly recommended.
        config: Processing configuration. Pass a preset name (``"default"``,
            ``"rcsb"``, ``"lightweight"``, ``"minimal"``), a
            :py:class:`~atomworks.io.config.ParseConfig` instance, or ``None``
            for defaults.
        filename: Deprecated alias for ``source``. Cannot be used together with ``source``.

    Returns:
        dict: A dictionary containing the following keys:
            chain_info
                A dictionary mapping chain ID to sequence, type (as an IntEnum), RCSB entity,
                EC number, and other information.
            ligand_info
                A dictionary containing ligand of interest information.
            asym_unit
                An AtomArrayStack instance representing the asymmetric unit.
            assemblies
                A dictionary mapping assembly IDs to AtomArrayStack instances.
            metadata
                A dictionary containing metadata about the structure
                (e.g., resolution, deposition date, etc.).
            extra_info
                A dictionary with information for cross-compatibility and caching.
                Should typically not be used directly.
    """
    config, kwargs = _resolve_config(config, kwargs, cls=ParseConfig)

    if filename is not None:
        if source is not None:
            raise TypeError("Cannot pass both 'source' and 'filename'")
        warnings.warn(
            "The 'filename' parameter is deprecated; use 'source' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        source = filename
    if source is None:
        raise TypeError("parse() requires a source (file path or buffer)")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    # Resolve catcif paths before any file-type inference or caching logic
    source, catcif_file_type = resolve_catcif_source(source)

    # Dispatch: file input
    file_type = config.file_type or catcif_file_type or infer_pdb_file_type(source)
    is_buffer = isinstance(source, io.StringIO | io.BytesIO)

    build_assembly = config.build_assembly
    extra_fields = config.extra_fields

    # +------ Cached route: loading ------+
    cache_file_path = None
    if config.cache_dir and not is_buffer:
        # Build a (order-invariant) hash of the parsing arguments
        cache_dir = Path(config.cache_dir)
        cache_file_path = _build_cache_file_path(cache_dir, source, config)

        # Load from cache if available and allowed by config
        if config.load_from_cache and cache_file_path.exists():
            try:
                result = pd.read_pickle(cache_file_path)
                extra_info = result["extra_info"]
                if "assembly_gen_category" in extra_info:
                    result["assemblies"] = build_assemblies_from_asym_unit(
                        assembly_gen_category=extra_info["assembly_gen_category"],
                        struct_oper_category=extra_info["struct_oper_category"],
                        asym_unit_atom_array_stack=result["asym_unit"],
                        build_assembly=build_assembly,
                        fix_symmetry_centers=config.fix_ligands_at_symmetry_centers,
                    )
                else:
                    # Fallback for cached files without assembly generation info
                    result["assemblies"] = {"1": result["asym_unit"]}

                # Early return
                return result
            except Exception as e:
                raise RuntimeError(f"Error loading from cache: {e}, tried path: {cache_file_path}") from e

    # +------ Uncached route: loading and processing ------+
    cif_block = None
    if file_type == "pdb":
        if config.altloc != "first":
            raise ValueError(
                f"altloc='{config.altloc}' is not supported for PDB files. "
                "PDB parsing always uses altloc='first'. Use CIF format for altloc selection."
            )
        atoms, metadata = load_pdb(source, model=config.model)
        extra_fields = None
        if build_assembly not in ("all", None):
            logger.warning(
                "PDB files always build all assemblies; ignoring build_assembly=%r",
                build_assembly,
            )
        build_assembly = "all"
    elif file_type in ("cif", "bcif"):
        atoms, _cif_file, cif_block, metadata, model_ids = load_cif(
            source,
            model=config.model,
            extra_fields=extra_fields,
            load_standard_annotations=config.load_standard_annotations,
            altloc=config.altloc,
            altloc_seed=config.altloc_seed,
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")

    # Build CCD entries from the CIF's chem_comp* categories and scope them
    # to this parse call so all downstream lookups (types, bonds, templates)
    # go through the registry.
    cif_ccd_entries = build_ccd_entries_from_cif_block(cif_block) if cif_block is not None else {}
    ctx = custom_ccd_residues(cif_ccd_entries) if cif_ccd_entries else contextlib.nullcontext()

    with ctx:
        if isinstance(atoms, list):
            results = []
            for arr, mn in zip(atoms, model_ids, strict=False):
                arr, chain_info = _prepare_atom_array_or_stack(
                    arr,
                    cif_block=cif_block,
                    config=config,
                    extra_fields=extra_fields,
                )
                if config.load_standard_annotations:
                    arr = _deserialize_standard_annotations(arr, cif_block, model_num=mn)
                results.append(
                    _assemble_parse_result(
                        atoms=arr,
                        chain_info=chain_info,
                        cif_block=cif_block,
                        config=config,
                        build_assembly=build_assembly,
                        metadata=metadata,
                        keep_cif_block=config.keep_cif_block,
                    )
                )
            _attach_live_ccd_registry(results)
            return results

        atoms, chain_info = _prepare_atom_array_or_stack(
            atoms,
            cif_block=cif_block,
            config=config,
            extra_fields=extra_fields,
        )

        if config.load_standard_annotations:
            atoms = _deserialize_standard_annotations(atoms, cif_block)

        result = _assemble_parse_result(
            atoms=atoms,
            chain_info=chain_info,
            cif_block=cif_block,
            config=config,
            build_assembly=build_assembly,
            metadata=metadata,
            keep_cif_block=config.keep_cif_block,
        )
        _attach_live_ccd_registry([result])

    # +------ Caching the result for the future ------+
    if not is_buffer and config.save_to_cache and cache_file_path and not cache_file_path.exists():
        try:
            from atomworks import __version__
        except ImportError:
            __version__ = "unknown"
        result.setdefault("metadata", {}).update(
            {"parse_arguments": config.to_dict(), "atomworks.version": __version__}
        )
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(PermissionError):
            cache_file_path.parent.chmod(0o777)  # avoid permissions issues with multiple users accessing the same cache

        pd.to_pickle({k: v for k, v in result.items() if k != "assemblies"}, cache_file_path)

    return result


def prepare_atom_array(
    source: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    *,
    config: str | PrepareConfig | None = None,
    cif_block: Any | None = None,
    extra_fields: ExtraFieldsType = None,
) -> AtomArray | AtomArrayStack:
    """Perform standard AtomWorks preparation of an AtomArray, returning the processed atoms directly.

    Runs the same core processing as :py:func:`parse` (standardize, add missing
    atoms, infer bonds, annotate) but returns the processed atoms instead of a
    full result dictionary. Single-model results are squeezed to
    :py:class:`~biotite.structure.AtomArray` for conciseness.

    For example, useful to add entity/molecule annotations to an AtomArray that already has a complete set of atoms and bonds.

    Args:
        source: The structure to process.
        config: Preset name (``"default"``, ``"rcsb"``, ``"lightweight"``),
            :py:class:`~atomworks.io.config.PrepareConfig`, or ``None`` for defaults.
        cif_block: Optional CIF block for richer processing (struct_conn bonds,
            entity categories, custom bonds).
        extra_fields: Extra CIF fields to preserve through processing.
    """
    config, _ = _resolve_config(config, {})
    atoms, _ = _prepare_atom_array_or_stack(
        source,
        config=config,
        cif_block=cif_block,
        extra_fields=extra_fields,
    )

    if config.return_atom_array_plus:
        atoms = _maybe_promote_to_plus(atoms)

    if isinstance(atoms, AtomArrayStack) and atoms.stack_depth() == 1:
        return atoms[0]

    return atoms


def parse_atom_array(
    source: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    *,
    config: str | PrepareConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Mimic of :py:func:`parse` that operates on an AtomArray or AtomArrayStack instead of a file input.

    Returns identical result dict to :py:func:`parse`, but takes an AtomArray or AtomArrayStack as input instead of a file path or buffer.

    Args:
        source: The AtomArray or AtomArrayStack to process.
        config: Preset name (``"default"``, ``"rcsb"``, ``"lightweight"``),
            :py:class:`~atomworks.io.config.PrepareConfig`, or ``None`` for defaults.
    """
    config, kwargs = _resolve_config(config, kwargs)

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    atoms, chain_info = _prepare_atom_array_or_stack(
        source,
        cif_block=None,
        config=config,
    )

    return _assemble_parse_result(
        atoms=atoms,
        chain_info=chain_info,
        cif_block=None,
        config=config,
        build_assembly="all",
        metadata={},
    )
