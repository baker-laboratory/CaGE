"""Test fixtures and utilities for RF3 tests."""

from dataclasses import dataclass, fields
from functools import cached_property, partial
from typing import Any, Dict, get_type_hints

import hydra
import pytest
import torch
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from hydra import compose, initialize
from hydra.utils import instantiate
from rf3.data.collate import collate_batch
from rf3.utils.inference import InferenceInput

from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Configure pytest for RF3 tests."""
    configure_pytest(config, __file__)


class ListDataset(torch.utils.data.Dataset):
    """A simple Dataset wrapping a list of items."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@pytest.fixture()
def two_training_examples(experiment):
    """Load two real examples using inference pipeline.

    Returns:
        List of two dictionaries containing model-ready inputs with all required features.
    """
    # Use two different structures from test data
    cif_1 = TEST_DATA_DIR / "5vht_from_file.cif"
    cif_2 = TEST_DATA_DIR / "8vkf_from_file.cif"

    with (
        initialize(config_path="../configs", version_base="1.3"),
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Load training config with quick-rf3 experiment
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
        )

        # Get the validation dataset config which has the transform pipeline
        # Use the first validation dataset
        first_val_dataset_key, first_val_dataset = next(iter(cfg.datasets.val.items()))

        # Instantiate the transform pipeline
        pipeline = instantiate(first_val_dataset.dataset.transform)

        # Create InferenceInput objects from CIF files
        input_1_spec = InferenceInput.from_cif_path(cif_1, example_id="example_1")
        input_2_spec = InferenceInput.from_cif_path(cif_2, example_id="example_2")

        # Run through transform pipeline to get model-ready inputs
        pipeline_output_1 = pipeline(input_1_spec.to_pipeline_input())
        pipeline_output_2 = pipeline(input_2_spec.to_pipeline_input())

    return [pipeline_output_1, pipeline_output_2]


def collate_fn_identity(batch):
    """Identity collate function - returns batch as-is."""
    return batch


def amplify_weights(module):
    """
    use random numbers instead of default initialization to amplify weights for testing purposes.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    for param in module.parameters():
        if param.requires_grad:
            if len(param.shape) >= 2:  # Weight matrices
                # Replace with large random values
                param.data = torch.randn_like(param) * 0.1
            elif len(param.shape) == 1:  # Biases
                param.data = torch.randn_like(param)


def paired_data_loaders_with_different_padding(
    two_training_examples, experiment, batch_size=2
):
    """Create two data loaders with different padding lengths for batching tests.

    Args:
        two_training_examples: Fixture providing two training examples.
        experiment: Experiment name for config loading.
        batch_size: Batch size for data loaders.
    Returns:
        Tuple of two DataLoader objects with different padding lengths.
    """
    dataset = ListDataset(two_training_examples)
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        # Load training config with quick-rf3 experiment
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
        )

    from rf3.data.schema.rf3 import (
        RF3Confidence,
        RF3DataSample,
        RF3DiffusionFeatures,
        RF3Features,
        RF3GroundTruth,
        RF3SymmetryResolution,
    )

    @dataclass
    class RF3FeaturesWithModifiedPadding(RF3Features):
        @cached_property
        def get_default_padding_value_overrides(self) -> Dict[str, Any]:
            padded_fields = {}
            hints = get_type_hints(self.__class__, include_extras=True)
            for field in fields(self):
                field_type = hints.get(field.name)
                if field_type is not None:
                    origin = getattr(field_type, "__origin__", None)
                    if origin is not None and origin.__name__ == "Float":
                        padded_fields[field.name] = (
                            1.1  # Use a distinct padding value for residue_index
                        )
                    elif origin is not None and origin.__name__ == "Int":
                        padded_fields[
                            field.name
                        ] = -5  # Use a distinct padding value for Int tensors
            return padded_fields

    @dataclass
    class RF3DataSampleWithModifiedPadding(RF3DataSample):
        features: RF3FeaturesWithModifiedPadding

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "RF3DataSample":
            data["diffusion_features"] = {
                "t": data["t"],
                "noise": data["noise"],
                "coord_atom_lvl_to_be_noised": data["coord_atom_lvl_to_be_noised"],
            }
            data["symmetry_resolution"].update(
                {
                    "automorphisms": data["automorphisms"],
                }
            )
            return cls(
                example_id=data["example_id"],
                features=RF3FeaturesWithModifiedPadding.from_dict(data["feats"]),
                ground_truth=RF3GroundTruth.from_dict(data["ground_truth"]),
                confidence=RF3Confidence.from_dict(data["confidence_feats"])
                if "confidence_feats" in data
                else None,
                diffusion_features=RF3DiffusionFeatures.from_dict(
                    data["diffusion_features"]
                ),
                symmetry_resolution=RF3SymmetryResolution.from_dict(
                    data["symmetry_resolution"]
                ),
                atom_array=data["atom_array"],
            )

    dataset = ListDataset(two_training_examples)
    items = [
        dataset[0],
        dataset[1],
    ]

    collate_batch_modified_padding = partial(
        collate_batch,
        schema=RF3DataSampleWithModifiedPadding,
    )
    collate_batch_default_padding = partial(
        collate_batch,
        schema=hydra.utils.get_class(cfg.datasets.input_feature_schema),
    )

    default_data = collate_batch_default_padding(items)
    modified_padding_data = collate_batch_modified_padding(items)
    return default_data, modified_padding_data


def paired_data_loaders_with_and_without_padding(
    two_training_examples, experiment, batch_size=2
):
    """Create two data loaders with different padding lengths for batching tests.

    Args:
        two_training_examples: Fixture providing two training examples.
        experiment: Experiment name for config loading.
        batch_size: Batch size for data loaders.
    Returns:
        Tuple of two DataLoader objects with different padding lengths.
    """
    dataset = ListDataset(two_training_examples)
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        # Load training config with quick-rf3 experiment
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_batch,
            schema=hydra.utils.get_class(cfg.datasets.input_feature_schema),
        ),
        shuffle=False,
    )

    dataloader_no_padding = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn_identity,
        shuffle=False,
    )
    return dataloader, dataloader_no_padding


def find_device():
    """Utility to find a GPU device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
