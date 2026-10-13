"""Centralized paths for code, data, pretrained assets and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get(name, str(default))))).resolve()


DATA_ROOT = env_path(
    "MASKWAM_DATA_ROOT",
    PROJECT_ROOT / "realdata",
)
PRETRAINED_ROOT = env_path("MASKWAM_PRETRAINED_ROOT", PROJECT_ROOT / "pretrained_models")
WAN_ROOT = env_path(
    "MASKWAM_WAN_PATH",
    PRETRAINED_ROOT / "Wan2.2-TI2V-5B",
)
CHECKPOINT_ROOT = env_path(
    "MASKWAM_CHECKPOINT_ROOT",
    PROJECT_ROOT / "ckpt",
)
LIBERO_ROOT = env_path("LIBERO_ROOT", PROJECT_ROOT.parent / "LIBERO")


def require_path(path: Path | str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {resolved}. "
            "Set the correct path with a command-line argument or MASKWAM_* environment variable."
        )
    return resolved
