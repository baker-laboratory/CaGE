from typing import Dict, Optional
from rf3.data.schema.common import BatchedDataSample


def collate_batch(batch, schema, fixed_dimensions: Optional[Dict[str, int]] = None):
    """Collate a batch of data samples into a single batch with padding.

    Args:
        batch: A list of data samples (dictionaries).
        schema: The schema class to convert dicts to DataSample instances.
        fixed_dimensions: Optional dict of dimension names to fixed sizes for torch.compile.

    Returns:
        PaddedBatchedDataSample with padding masks.
    """
    samples = [schema.from_dict(item) for item in batch]
    batched_samples = BatchedDataSample(
        samples, default_padding_dimensions=fixed_dimensions
    )
    batched_sample = batched_samples.pad_and_stack()
    return batched_sample
