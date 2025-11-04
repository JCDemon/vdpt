from __future__ import annotations
from pathlib import Path

ARTIFACTS = Path("artifacts")
BUNDLED_IMAGES = ARTIFACTS / "bundled_images"

ALLOWED_IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".ppm"}


def list_bundled_images() -> list[str]:
    """Return file names (not full paths) under artifacts/bundled_images with allowed suffixes."""
    BUNDLED_IMAGES.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(BUNDLED_IMAGES.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_IMG_SUFFIXES:
            items.append(p.name)
    return items
