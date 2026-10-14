"""CIF-based dataset loaders."""

import functools
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from atomworks.io.config import ParseConfig
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.utils.io_utils import infer_pdb_file_type

from .base import _construct_metadata_hierarchy, _construct_structure_path


def _resolve_parse_config(
    parser_args: ParseConfig | dict | None,
    assembly_id: str,
    source: Path | io.BytesIO | io.StringIO,
    file_type: str | None = None,
    altloc_seed: int | None = None,
) -> ParseConfig:
    """Resolve ``parser_args`` to a :py:class:`ParseConfig`, stamping ``build_assembly`` from ``assembly_id`` for CIF/bCIF inputs.

    Accepts a ``ParseConfig`` (preferred) or a legacy dict of kwargs (merged
    with :py:data:`STANDARD_PARSER_ARGS` via :py:meth:`ParseConfig.from_dict`).

    When ``altloc_seed`` is provided, altloc selection is switched to
    ``"random_clash_aware"`` with the given seed for deterministic sampling.
    """
    if not isinstance(parser_args, ParseConfig):
        # Legacy dict support: merge with STANDARD_PARSER_ARGS and convert to ParseConfig
        parser_args = ParseConfig.from_dict({**STANDARD_PARSER_ARGS, **(parser_args or {})})

    if file_type is not None:
        # Overwrite file_type in parser_args if explicitly provided (e.g. for CIF bytes loader)
        parser_args = parser_args.replace(file_type=file_type)

    if (parser_args.file_type or infer_pdb_file_type(source)) in ("cif", "bcif"):
        parser_args = parser_args.replace(build_assembly=(assembly_id,))

        if altloc_seed is not None:
            # For CIF inputs, if altloc_seed is provided, switch to random_clash_aware altloc selection with the given seed
            parser_args = parser_args.replace(altloc="random_clash_aware", altloc_seed=altloc_seed)

    return parser_args


def _base_loader_function(
    row: pd.Series,
    example_id_colname: str,
    path_colname: str,
    assembly_id_colname: str | None,
    altloc_seed_colname: str | None,
    attrs: dict,
    base_path: str,
    extension: str,
    sharding_pattern: str | None,
    parser_args: ParseConfig | dict | None,
) -> dict[str, Any]:
    """Base loader function (picklable when used with functools.partial)."""
    # Prepare loader-specific attributes
    loader_attrs = attrs.copy()
    if base_path and "base_path" not in loader_attrs:
        loader_attrs["base_path"] = base_path
    if extension and "extension" not in loader_attrs:
        loader_attrs["extension"] = extension

    extra_info = _construct_metadata_hierarchy(row, loader_attrs)

    assembly_id = row[assembly_id_colname] if assembly_id_colname is not None and assembly_id_colname in row else "1"
    altloc_seed = row[altloc_seed_colname] if altloc_seed_colname is not None else None

    path = _construct_structure_path(
        row[path_colname], extra_info.get("base_path"), extra_info.get("extension"), sharding_pattern
    )
    result_dict = parse(path, config=_resolve_parse_config(parser_args, assembly_id, path, altloc_seed=altloc_seed))

    # Remove used columns from extra_info
    exclude_cols = (
        [example_id_colname, path_colname]
        + ([assembly_id_colname] if assembly_id_colname else [])
        + ([altloc_seed_colname] if altloc_seed_colname else [])
        + ["base_path", "extension"]
    )
    extra_info = {k: v for k, v in extra_info.items() if k not in exclude_cols}

    return {
        "example_id": row[example_id_colname],
        "path": path,
        "assembly_id": assembly_id,
        "altloc_seed": altloc_seed,
        "extra_info": extra_info,
        "atom_array": result_dict["assemblies"][assembly_id][0],
        "atom_array_stack": result_dict["assemblies"][assembly_id],
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }


def create_base_loader(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    attrs: dict | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable base loader for AtomWorks datasets.

    Args:
        example_id_colname: Name of column containing unique example identifiers
        path_colname: Name of column containing paths to structure files
        assembly_id_colname: Optional column name containing assembly IDs.
            If None, assembly_id defaults to "1" for all examples.
        attrs: Additional attributes to merge with highest precedence into the metadata hierarchy
            (and ultimately included in the output dictionary's "extra_info" key).
        base_path: Base path to prepend to file paths if not included in path column
        extension: File extension to add/replace if not included in path column
        sharding_pattern: Pattern for how files are organized in subdirectories, if not specified in the path
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)
        parser_args: Parser arguments — a :py:class:`ParseConfig` (preferred) or a
            legacy dict of kwargs merged with :py:data:`STANDARD_PARSER_ARGS`.

    Returns:
        A picklable loader function (via functools.partial) for multiprocessing.
    """
    return functools.partial(
        _base_loader_function,
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs or {},
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )


def _loader_with_query_pn_units_function(
    row: pd.Series,
    base_loader: Callable,
    pn_unit_iid_colnames: list[str],
) -> dict[str, Any]:
    """Loader with query pn_units (picklable when used with functools.partial)."""
    result = base_loader(row)
    result["extra_info"] = {k: v for k, v in result["extra_info"].items() if k not in pn_unit_iid_colnames}
    query_pn_unit_iids = [row[colname] for colname in pn_unit_iid_colnames]
    result["query_pn_unit_iids"] = query_pn_unit_iids
    return result


def create_loader_with_query_pn_units(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    pn_unit_iid_colnames: str | list[str] | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable loader for pipelines with query pn_units (chains).

    For instance, in the interfaces dataset, each sampled row contains two pn_unit instance IDs
    that should be included in the cropped structure.

    Examples:
        Interfaces dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"], assembly_id_colname="assembly_id"
            ... )

        Chains dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames="pn_unit_iid", base_path="/data/structures", extension=".cif.gz"
            ... )
    """
    # Normalize pn_unit_iid_colnames to list format
    if isinstance(pn_unit_iid_colnames, str):
        pn_unit_iid_colnames = [pn_unit_iid_colnames]
    pn_unit_iid_colnames = pn_unit_iid_colnames or []

    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    return functools.partial(
        _loader_with_query_pn_units_function,
        base_loader=base_loader,
        pn_unit_iid_colnames=pn_unit_iid_colnames,
    )


def _loader_with_interfaces_and_pn_units_to_score_function(
    row: pd.Series,
    base_loader: Callable,
    interfaces_to_score_colname: str | None,
    pn_units_to_score_colname: str | None,
) -> dict[str, Any]:
    """Loader with scoring info (picklable when used with functools.partial)."""
    result = base_loader(row)
    exclude_cols = [interfaces_to_score_colname, pn_units_to_score_colname]
    result["extra_info"] = {k: v for k, v in result["extra_info"].items() if k not in exclude_cols}

    interfaces_to_score = row[interfaces_to_score_colname] if interfaces_to_score_colname is not None else None
    pn_units_to_score = row[pn_units_to_score_colname] if pn_units_to_score_colname is not None else None

    result.update(
        {
            "interfaces_to_score": interfaces_to_score,
            "pn_units_to_score": pn_units_to_score,
        }
    )
    return result


def create_loader_with_interfaces_and_pn_units_to_score(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    interfaces_to_score_colname: str | None = "interfaces_to_score",
    pn_units_to_score_colname: str | None = "pn_units_to_score",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable loader for validation datasets with scoring information.

    Example:
        >>> loader = create_loader_with_interfaces_and_pn_units_to_score(
        ...     interfaces_to_score_colname="interfaces_to_score", pn_units_to_score_colname="pn_units_to_score"
        ... )
    """
    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    return functools.partial(
        _loader_with_interfaces_and_pn_units_to_score_function,
        base_loader=base_loader,
        interfaces_to_score_colname=interfaces_to_score_colname,
        pn_units_to_score_colname=pn_units_to_score_colname,
    )


def _cif_bytes_loader_function(
    raw_data: tuple,
    parser_args: ParseConfig | dict | None,
    assembly_id_colname: str | None,
    altloc_seed_colname: str | None = None,
) -> dict[str, Any]:
    """Loader for CIF bytes (picklable when used with functools.partial)."""
    cif_bytes, global_idx, metadata_row = raw_data

    # Extract assembly_id from metadata row when available
    assembly_id = "1"
    if metadata_row is not None and assembly_id_colname is not None and assembly_id_colname in metadata_row.index:
        assembly_id = str(metadata_row[assembly_id_colname])

    altloc_seed = None
    if metadata_row is not None and altloc_seed_colname is not None and altloc_seed_colname in metadata_row.index:
        altloc_seed = int(metadata_row[altloc_seed_colname])

    source = io.StringIO(cif_bytes.decode("utf-8"))
    result = parse(
        source,
        config=_resolve_parse_config(parser_args, assembly_id, source, file_type="cif", altloc_seed=altloc_seed),
    )
    # Build extra_info from metadata row
    extra_info: dict[str, Any] = {}
    if metadata_row is not None:
        extra_info = _construct_metadata_hierarchy(metadata_row, {})
        # Remove columns already represented in the output
        exclude_cols = {"example_id"}
        if assembly_id_colname:
            exclude_cols.add(assembly_id_colname)
        extra_info = {k: v for k, v in extra_info.items() if k not in exclude_cols}

    return {
        "assembly_id": assembly_id,
        "extra_info": extra_info,
        "atom_array": result["assemblies"][assembly_id][0],
        "atom_array_stack": result["assemblies"][assembly_id],
        "chain_info": result["chain_info"],
        "ligand_info": result["ligand_info"],
        "metadata": result["metadata"],
    }


def create_cif_bytes_loader(
    parser_args: ParseConfig | dict | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
) -> Callable:
    """Factory for loading CIF bytes (e.g. from LMDB) via :py:func:`~atomworks.io.parser.parse`.

    Args:
        parser_args: Parser arguments — a :py:class:`ParseConfig` (preferred) or a
            legacy dict of kwargs merged with :py:data:`STANDARD_PARSER_ARGS`.
        assembly_id_colname: Column name in the metadata row containing the assembly ID.
            If ``None``, assembly ID defaults to ``"1"``.
    """
    return functools.partial(
        _cif_bytes_loader_function,
        parser_args=parser_args,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
    )
