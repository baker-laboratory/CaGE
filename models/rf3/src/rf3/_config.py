"""Locate the Hydra configs without importing model or inference dependencies."""

from pathlib import Path


def get_rf3_config_dir() -> str:
    """Return the absolute config directory for wheel or editable installations."""
    package_dir = Path(__file__).resolve().parent
    candidates = (package_dir / "configs", package_dir.parents[1] / "configs")
    for candidate in candidates:
        if (candidate / "inference.yaml").is_file():
            return str(candidate)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"RF3 inference.yaml not found; searched: {searched}")
