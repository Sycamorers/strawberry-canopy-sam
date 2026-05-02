"""Shared project paths.

The original scripts were developed from one local checkout. Keeping these
paths in one place makes the copied repository portable without changing the
benchmark logic.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("CANOPY_SAM_MODEL_DIR", PROJECT_ROOT / "models_wts"))
MOBILE_SAM_V2_ROOT = PROJECT_ROOT / "SAMs" / "MobileSAM" / "MobileSAMv2"
EFFICIENT_SAM_ROOT = PROJECT_ROOT / "SAMs" / "EfficientSAM"


def weight_path(filename: str) -> Path:
    """Return a model-weight path from the configured model directory."""
    return MODEL_DIR / filename


def require_file(path: Path, label: str) -> str:
    """Return a string path, raising a clear error when a local asset is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}. Place the file there or set CANOPY_SAM_MODEL_DIR."
        )
    return str(path)
