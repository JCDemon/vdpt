"""Test configuration for the VDPT project."""

import csv
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SAMPLE_IMAGE = ASSETS_DIR / "sample.jpg"
_SAMPLE_CSV = ASSETS_DIR / "sample.csv"


def pytest_configure(config):  # noqa: D401  - pytest hook to ensure fixtures exist
    """Create deterministic preview fixtures if they are missing."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_sample_image()
    _ensure_sample_csv()


def _ensure_sample_image() -> None:
    if _SAMPLE_IMAGE.exists():
        return
    image = Image.new("RGB", (512, 512), color=(32, 96, 160))
    image.save(_SAMPLE_IMAGE, format="JPEG", quality=95)


def _ensure_sample_csv() -> None:
    if _SAMPLE_CSV.exists():
        return
    with _SAMPLE_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b"])
        writer.writerows([[1, 0.5], [2, 1.5], [3, 2.5], [4, 3.5]])
