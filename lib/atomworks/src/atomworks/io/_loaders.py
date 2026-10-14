"""Internal loading helpers for the parse pipeline."""

import io
import logging
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.file import InvalidFileError
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from atomworks.io.transforms.categories import get_metadata_from_category
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.extra_fields import ExtraFieldsType, merge_extra_fields, normalize_extra_fields
from atomworks.io.utils.io_utils import get_structure, read_any
from atomworks.io.utils.standard_annotations.serialization import (
    STANDARD_ANNOTATIONS,
    _detect_standard_annotation_categories,
)

logger = logging.getLogger(__name__)


def load_cif(
    filename: str | Path | io.StringIO | io.BytesIO,
    *,
    model: int | None,
    extra_fields: ExtraFieldsType | None,
    load_standard_annotations: bool,
    altloc: str = "first",
    altloc_seed: int | None = None,
) -> tuple[AtomArrayStack | list[AtomArray], pdbx.CIFFile | pdbx.BinaryCIFFile, pdbx.CIFBlock, dict, list[int] | None]:
    """Load a CIF file and return ``(atoms, cif_file, cif_block, metadata)``."""
    # ... read the CIF file (bytes or path)
    cif_file = read_any(filename)
    cif_block = cif_file.block

    # +------------ Metadata ------------+

    # We require an "example_id" field in metadata; assign fallback if not present
    if isinstance(filename, io.StringIO | io.BytesIO):
        fallback_filename = next(iter(cif_file.keys()))
    else:
        fallback_filename = Path(filename).stem

    # ... initialize metadata dictionary from CIF categories (CIF-specific)
    metadata = get_metadata_from_category(cif_block, fallback_id=fallback_filename)

    # +------------ Annotations (CIF-specific) ------------+

    # NOTE: transformation_id is auto-injected by get_structure() for CIF files
    common_extra_fields = merge_extra_fields(
        ["label_entity_id", "atom_id", "b_factor", "occupancy", "charge", "auth_seq_id", "nhyd", "label_alt_id"],
        extra_fields,
    )

    # Sanity Check: extra_fields must not overlap with SA names saved as CIF extra-categories.
    # Scalar 1-body SAs live in atom_site and can coexist with user-provided extra_fields without
    # ambiguity. Nonscalar 1-body and 2-body SAs are saved as separate CIF categories, so
    # specifying their names in extra_fields is an error.
    if load_standard_annotations and extra_fields:
        sa_categories = _detect_standard_annotation_categories(cif_block)
        user_extra_fields = normalize_extra_fields(extra_fields)
        for field_name in user_extra_fields:
            if field_name in sa_categories:
                annotation_cls, _ = sa_categories[field_name]
                raise ValueError(
                    f"Field '{field_name}' conflicts with StandardAnnotation '{annotation_cls.name}'. "
                    "When load_standard_annotations=True, StandardAnnotation fields must not appear "
                    "in extra_fields. Set load_standard_annotations=False to load them manually via extra_fields."
                )

    # When loading StandardAnnotations, extend common_extra_fields with all registered 1-body SA names
    # so they are read from atom_site. Fields absent from the file are silently skipped.
    if load_standard_annotations:
        common_extra_fields = merge_extra_fields(STANDARD_ANNOTATIONS.get_field_names(n_body=1), common_extra_fields)

    # +------------ Structure ------------+

    try:
        asym_unit_stack = get_structure(
            cif_file, extra_fields=common_extra_fields, model=model, altloc=altloc, altloc_seed=altloc_seed
        )
    except InvalidFileError:
        if model is not None:
            logger.info("Invalid file error encountered; loading with only one model")
            asym_unit_stack = get_structure(cif_file, extra_fields=common_extra_fields, model=1)
            return asym_unit_stack, cif_file, cif_block, metadata, None
        # Variable-length multi-model: annotations differ between models, so biotite
        # cannot stack them. Load each model individually by 1-based position.
        atom_site = cif_block.get("atom_site")
        if atom_site is not None and "pdbx_PDB_model_num" in atom_site:
            model_ids_col = atom_site["pdbx_PDB_model_num"].as_array(int)
            unique_ids = list(np.unique(model_ids_col).tolist())
            arrays = [
                get_structure(cif_file, extra_fields=common_extra_fields, model=i)
                for i in range(1, len(unique_ids) + 1)
            ]
            return arrays, cif_file, cif_block, metadata, unique_ids
        logger.info("Invalid file error encountered; loading with only one model")
        asym_unit_stack = get_structure(
            cif_file, extra_fields=common_extra_fields, model=1, altloc=altloc, altloc_seed=altloc_seed
        )

    return asym_unit_stack, cif_file, cif_block, metadata, None


def load_pdb(
    filename: str | Path | io.StringIO | io.BytesIO,
    *,
    model: int | None,
) -> tuple[AtomArrayStack, dict]:
    """Load a PDB file and return ``(atoms, metadata)``.

    We require that a single chain contains either polymer or non-polymer residues,
    but not both. If the PDB file contains a chain with both, the non-polymer residues
    will be separated onto a new chain ID.

    Since PDB files don't have a standard way to represent bond types, we assign all
    bonds from PDB LINK records as SINGLE.
    """
    # ... read the PDB file (bytes or path)
    pdb_file = read_any(filename)

    # +------------ Metadata ------------+

    if isinstance(filename, io.StringIO | io.BytesIO):
        fallback_id = "unknown"
    else:
        fallback_id = Path(filename).stem.lower()
    metadata = {"id": fallback_id}

    # +------------ Structure ------------+

    # NOTE: PDB file support is more limited than CIF since PDB format doesn't support multiple models, extra fields, alternative locations, etc. etc.
    atom_array_stack = pdb_file.get_structure(
        model=model,
        altloc="first",
        extra_fields=["b_factor", "occupancy", "charge", "atom_id"],
        include_bonds=True,
    )

    # +------------ Normalizing PDB files ------------+

    # Normalize bond types from PDB LINK records: biotite assigns BondType.ANY (0) to LINK bonds
    # since PDB format doesn't specify bond order. Upgrade to SINGLE for correct formal charge computation.
    if atom_array_stack.bonds is not None:
        bonds_arr = atom_array_stack.bonds.as_array()
        any_mask = bonds_arr[:, 2] == struc.BondType.ANY
        if any_mask.any():
            bonds_arr[any_mask, 2] = struc.BondType.SINGLE
            atom_array_stack.bonds = struc.BondList(atom_array_stack.array_length(), bonds_arr)

    # ... if we have polymer and non-polymers on the same chain (as given by the HETATM field), we need to separate them for processing
    hetero_atom_mask = atom_array_stack.get_annotation("hetero")
    if np.any(hetero_atom_mask):
        original_chain_ids = np.unique(atom_array_stack.chain_id)
        chain_id_generator = create_chain_id_generator(unavailable_chain_ids=original_chain_ids)
        for chain_id in original_chain_ids:
            chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
            if np.any(chain_hetero_annotations) and np.any(~chain_hetero_annotations):
                hetero_chain_id = next(chain_id_generator)
                logger.warning(
                    f"Chain {chain_id} contains both polymer and non-polymer residues; "
                    f"separating them for processing, naming the non-polymer residues as {hetero_chain_id}."
                )
                atom_array_stack.chain_id[(atom_array_stack.chain_id == chain_id) & hetero_atom_mask] = hetero_chain_id
            updated_chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
            assert np.all(updated_chain_hetero_annotations) or np.all(~updated_chain_hetero_annotations)

    return atom_array_stack, metadata
