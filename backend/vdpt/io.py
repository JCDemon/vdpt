"""I/O helpers for VDPT data pipelines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def read_csv(path: PathLike) -> pd.DataFrame:
    """Read a CSV file into a dataframe with helpful errors."""

    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc
    except Exception as exc:  # pragma: no cover - pandas raises descriptive errors
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {exc}") from exc


def write_csv(df: pd.DataFrame, path: PathLike) -> None:
    """Persist a dataframe to disk as CSV, creating parent directories."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def sha256_bytes(blob: bytes) -> str:
    """Return the SHA-256 digest of the provided bytes as hex."""

    digest = hashlib.sha256()
    digest.update(blob)
    return digest.hexdigest()
