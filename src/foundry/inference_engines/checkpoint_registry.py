"""Management of checkpoints"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import dotenv

DEFAULT_CHECKPOINT_DIR = Path.home() / ".foundry" / "checkpoints"


def _normalize_paths(paths: Iterable[Path]) -> list[Path]:
    """Return absolute, deduplicated paths in order."""
    seen = set()
    normalized: List[Path] = []
    for path in paths:
        resolved = path.expanduser().absolute()
        if resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)
    return normalized


def get_default_checkpoint_dirs() -> list[Path]:
    """Return checkpoint search paths.

    Always starts with the default ~/.foundry/checkpoints directory and then
    appends any additional directories from the colon-separated
    FOUNDRY_CHECKPOINT_DIRS environment variable.
    """
    env_dirs = os.environ.get("FOUNDRY_CHECKPOINT_DIRS", "")

    # For backward compatibility, also check FOUNDRY_CHECKPOINTS_DIR
    if not env_dirs:
        env_dirs = os.environ.get("FOUNDRY_CHECKPOINTS_DIR", "")

    extra_dirs: list[Path] = []
    if env_dirs:
        extra_dirs = [Path(p.strip()) for p in env_dirs.split(":") if p.strip()]
    return _normalize_paths([*extra_dirs, DEFAULT_CHECKPOINT_DIR])


def get_default_checkpoint_dir() -> Path:
    """Backward-compatible helper returning the primary checkpoint directory."""
    return get_default_checkpoint_dirs()[0]


def append_checkpoint_to_env(checkpoint_dirs: list[Path]) -> bool:
    dotenv_path = dotenv.find_dotenv()
    if dotenv_path:
        checkpoint_dirs = _normalize_paths(checkpoint_dirs)
        dotenv.set_key(
            dotenv_path=dotenv_path,
            key_to_set="FOUNDRY_CHECKPOINT_DIRS",
            value_to_set=":".join(str(path) for path in checkpoint_dirs),
            export=False,
        )
        return True
    else:
        return False


@dataclass
class RegisteredCheckpoint:
    url: str
    filename: str
    description: str
    sha256: str | None = None  # Optional: add checksum for verification
    doi: str | None = None  # Zenodo DOI of the release record

    def get_default_path(self):
        checkpoint_dirs = get_default_checkpoint_dirs()
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / self.filename
            if candidate.exists():
                return candidate
        return checkpoint_dirs[0] / self.filename


REGISTERED_CHECKPOINTS = {
    "run14": RegisteredCheckpoint(
        url="https://zenodo.org/records/<RECORD_ID>/files/run14.ckpt?download=1",
        filename="run14.ckpt",
        description=(
            "CaGE design-condition weights, structure only (402-d atom features "
            "including the hotspot / interface-type / h-bond / RASA conditions and the "
            "2-body distance processor). No confidence head: `confidence_streams` and "
            "`ranking_stream` fall back to the main stream."
        ),
        sha256="8aa680d8ebdb553f3d996ad4dd10292a371e0ef1c5b3c0a4acb760d6a9a267cf",
        doi="10.5281/zenodo.<RECORD_ID>",
    ),
    "run14_confhead": RegisteredCheckpoint(
        url="https://zenodo.org/records/<RECORD_ID>/files/run14_confhead.ckpt?download=1",
        filename="run14_confhead.ckpt",
        description=(
            "CaGE design-condition weights plus the AF3-style confidence head, for "
            "per-stream confidence and stream ranking. The default `ckpt_path` of the "
            "`rf3` inference engine, and a superset of `run14`."
        ),
        sha256="3c02431345f65580fbc60e10e5d4dae2c8d63988cd605e33db6392c9f8b88772",
        doi="10.5281/zenodo.<RECORD_ID>",
    ),
}

