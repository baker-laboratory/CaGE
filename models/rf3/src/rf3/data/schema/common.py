from dataclasses import dataclass, fields, is_dataclass
from abc import ABC, abstractmethod
from torch.nn import functional as F
from typing import Any, Dict, get_type_hints
from functools import cached_property
import torch
import logging

logger = logging.getLogger(__name__)


def extract_dimension_sizes(obj: "Features") -> Dict[str, int]:
    """Auto-extract dimension sizes from jaxtyped fields.

    Returns:
        Dict like {'I': 128, 'C_s': 384, 'C_z': 128}
    """
    hints = get_type_hints(obj.__class__, include_extras=True)
    dim_sizes = {}

    for field in fields(obj):
        if field.name in ("is_padded", "is_expanded"):
            continue

        field_type = hints.get(field.name)
        if field_type is None or not hasattr(field_type, "dim_str"):
            logger.debug(f"Skipping field {field.name} as it lacks jaxtyping info.")
            continue

        # Extract shape string: "I C_s" -> ['I', 'C_s']
        shape_str = field_type.dim_str
        symbolic_dims = shape_str.split()

        # Get actual tensor
        tensor = getattr(obj, field.name)
        if not isinstance(tensor, torch.Tensor):
            continue

        # Map symbolic dims to actual sizes
        for i, dim_name in enumerate(symbolic_dims):
            if i >= len(tensor.shape):
                break

            actual_size = tensor.shape[i]

            if dim_name.isdecimal():
                assert actual_size == int(dim_name), (
                    f"Field {field.name} has fixed size dimension {dim_name} "
                    f"but actual size is {actual_size}"
                )
                continue  # Skip fixed-size dimensions

            # Validate consistency (all 'I' dims should have same size)
            if dim_name in dim_sizes:
                if dim_sizes[dim_name] != actual_size:
                    raise ValueError(
                        f"Dimension {dim_name} has inconsistent sizes: "
                        f"{dim_sizes[dim_name]} vs {actual_size}"
                        f" in field {field.name}"
                    )
            else:
                dim_sizes[dim_name] = actual_size

    return dim_sizes


def pad_dimension(
    tensor: torch.Tensor, dim: int, max_length: int, value: float = 0.0
) -> torch.Tensor:
    """Pad a specific dimension of a tensor to a maximum length.

    Args:
        tensor: The input tensor to be padded.
        dim: The dimension index to pad.
        max_length: The target length for the specified dimension.
        value: Fill value for padding. Defaults to 0.0.

    Returns:
        The padded tensor.
    """
    current_length = tensor.size(dim)

    # No padding needed
    if current_length >= max_length:
        return tensor

    padding_amount = max_length - current_length

    # F.pad expects padding in reverse dimension order:
    # (left_last, right_last, left_second_to_last, right_second_to_last, ...)
    # Build padding list: [0, 0] for all dims except the target
    padding = []

    for i in range(len(tensor.shape) - 1, -1, -1):
        if i == dim:
            # Pad this dimension on the right side
            padding.extend([0, padding_amount])
        else:
            # Don't pad this dimension
            padding.extend([0, 0])

    return F.pad(tensor, pad=padding, mode="constant", value=value)


def batch_dataclass_recursive(instances: list) -> Any:
    """
    Recursively batch a list of instances (could be dataclasses, tensors, primitives, etc.)
    """
    if not instances:
        raise ValueError("Cannot batch empty list")

    first = instances[0]

    # Base case 1: Tensor
    if isinstance(first, torch.Tensor):
        return torch.stack([x for x in instances], dim=0)

    # Base case 2: Primitives (int, float, str, bool, None)
    elif isinstance(first, (int, float, str, bool, type(None))):
        return instances  # Keep as list or convert to tensor if all same type

    # Base case 3: Lists/tuples
    elif isinstance(first, (list, tuple)):
        # Option A: Keep as nested list
        # Option B: Try to stack if all elements are tensors
        return instances

    # Recursive case: Dataclass
    elif is_dataclass(first):
        batched_fields = {}

        # Iterate through each field in the dataclass
        for field in fields(first):
            # Collect this field's values from all instances
            field_values = [getattr(instance, field.name) for instance in instances]

            # Recursively batch this field
            batched_fields[field.name] = batch_dataclass_recursive(field_values)

        # Construct new instance of the same dataclass type with batched fields
        return type(first)(**batched_fields)

    else:
        # Unknown type - keep as list
        return instances


def get_default_value(field, hints, padding_overrides):
    """
    Intended behavior: based on the type of the field, return a default padding value.
    For example, for Bool[torch.Tensor, ...], return False (0.0), for Int[torch.Tensor, ...], return 0,
    for Float[torch.Tensor, ...], return 0.0, etc.
    Currently, always returns 0.0 as a placeholder.
    """
    if field.name in padding_overrides:
        return padding_overrides[field.name]

    field_type = hints.get(field.name)
    if field_type is not None:
        origin = getattr(field_type, "__origin__", None)
        if origin is not None:
            if origin.__name__ == "Bool":
                return 0.0  # False
            elif origin.__name__ == "Int":
                return -1  # int is usually index tensor, pad with -1
            elif origin.__name__ == "Float":
                return 0.0
    return 0.0


@dataclass
class Features(ABC):
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Features":
        return cls(**data)

    @cached_property
    def get_default_padding_value_overrides(self) -> Dict[str, Any]:
        """
        Allows the user to specify custom default padding values for specific fields.
        Returns a dictionary mapping field names to their desired default padding values.
        """
        return {}

    def pad_to_max_lengths(
        self, max_lengths: Dict[str, int], exclude=None
    ) -> "Features":
        # use the jaxtyping annotations to determine which dimensions to pad
        # for each field, check if its a tensor, get its jaxtyping shape, and pad accordingly
        padded_fields = {}
        hints = get_type_hints(self.__class__, include_extras=True)
        for field in fields(self):
            field_value = getattr(self, field.name)
            field_type = hints.get(field.name)
            if field_type is None:
                padded_fields[field.name] = field_value
                continue

            if not isinstance(field_value, torch.Tensor):
                padded_fields[field.name] = field_value
                continue

            if field.name == exclude or (exclude is not None and field.name in exclude):
                padded_fields[field.name] = field_value
                continue

            shape_str = field_type.dim_str
            symbolic_dims = shape_str.split()

            padded_tensor = field_value
            for i, dim_name in enumerate(symbolic_dims):
                if dim_name in max_lengths:
                    target_length = max_lengths[dim_name]
                    padded_tensor = pad_dimension(
                        padded_tensor,
                        dim=i,
                        max_length=target_length,
                        value=get_default_value(
                            field, hints, self.get_default_padding_value_overrides
                        ),
                    )

            padded_fields[field.name] = padded_tensor
        return self.__class__(**padded_fields)


@dataclass
class DataSample(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSample":
        pass  # Placeholder for from_dict method

    @abstractmethod
    def pad_to_max_lengths(self, max_lengths: Dict[str, int]) -> "DataSample":
        pass  # Placeholder for padding method

    @cached_property
    def max_lengths(self) -> Dict[str, int]:
        """
        Get maximum lengths for each dimension in this data sample.
        """
        max_lengths = {}
        for field in fields(self):
            if isinstance(getattr(self, field.name), Features):
                feature_obj = getattr(self, field.name)
                feature_max_lengths = extract_dimension_sizes(feature_obj)
                for dim_name, size in feature_max_lengths.items():
                    if dim_name in max_lengths:
                        max_lengths[dim_name] = max(max_lengths[dim_name], size)
                    else:
                        max_lengths[dim_name] = size
        return max_lengths


@dataclass
class BatchedDataSample(ABC):
    """
    Three steps:
    1. From a list of DataSample instances, extract max lengths for each dimension across the batch.
    2. Pad each DataSample to these max lengths and record the mask of which tokens are padding.
    3. Stack the padded DataSamples into a single BatchedDataSample instance.
    """

    data_samples: list[DataSample]

    def __init__(self, data_samples: list[DataSample], default_padding_dimensions=None):
        super().__init__()
        self.data_samples = data_samples
        assert set(type(s) for s in data_samples) == {
            type(data_samples[0])
        }, "All padded samples must be of the same type"
        
        #  Optional: specify default padding dimensions for torch.compile 
        self.default_padding_dimensions = default_padding_dimensions or {}

    def get_max_lengths(self) -> Dict[str, int]:
        """
        Get maximum lengths for each dimension across the batch.
        First, retrieve the max lengths from each data sample,
        then compute the overall maximum for each dimension.
        If default_padding_dimensions are provided, use them as the target sizes.
        """
        max_lengths = {}
        for data_sample in self.data_samples:
            sample_max_lengths = data_sample.max_lengths
            for dim_name, size in sample_max_lengths.items():
                if dim_name in max_lengths:
                    max_lengths[dim_name] = max(max_lengths[dim_name], size)
                else:
                    max_lengths[dim_name] = size

        # Apply fixed padding dimensions (override with fixed sizes)
        for dim_name, fixed_size in self.default_padding_dimensions.items():
            if dim_name in max_lengths and max_lengths[dim_name] > fixed_size:
                raise ValueError(
                    f"Sample dimension {dim_name}={max_lengths[dim_name]} "
                    f"exceeds fixed size {fixed_size}"
                )
            max_lengths[dim_name] = fixed_size

        return max_lengths

    def get_padding_masks(self, max_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Generate padding masks for each dimension based on max lengths.
        Returns a dictionary mapping dimension names to boolean masks.
        """
        padding_masks = {}
        for dim_name, max_size in max_lengths.items():
            masks = []
            for data_sample in self.data_samples:
                sample_max_lengths = data_sample.max_lengths
                current_size = sample_max_lengths.get(dim_name, 0)
                mask = torch.zeros(max_size, dtype=torch.bool)
                if current_size < max_size:
                    mask[current_size:] = True  # Mark padding positions as True
                masks.append(mask)
            padding_masks[dim_name] = torch.stack(masks)
        return padding_masks

    def pad_and_stack(self) -> "DataSample":
        """
        Take a list of DataSample instances and batch them into a single DataSample instance,
        padding dimensions as necessary to ensure uniform sizes across the batch.
        """
        max_lengths = self.get_max_lengths()
        padding_masks = self.get_padding_masks(max_lengths)
        padded_samples = [
            sample.pad_to_max_lengths(max_lengths)
            for sample in getattr(self, "data_samples")
        ]

        # take all fields with same name and stack them
        stacked_fields = batch_dataclass_recursive(padded_samples)
        return PaddedBatchedDataSample(
            data_samples=stacked_fields, padding_masks=padding_masks
        )


@dataclass
class PaddedBatchedDataSample(BatchedDataSample):
    padding_masks: Dict[str, torch.Tensor]
