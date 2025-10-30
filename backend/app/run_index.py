"""Helpers for indexing recorded runs and their artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .types import RunSummary


def list_runs(artifacts_dir: Path | None = None) -> List[RunSummary]:
    """Return summaries for runs discovered under ``artifacts/``.

    Parameters
    ----------
    artifacts_dir:
        Optional override to the artifacts directory. Defaults to ``artifacts`` at
        the repository root.
    """

    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"

    runs: List[RunSummary] = []
    for run_dir in _iter_run_dirs(artifacts_dir):
        metadata = _read_metadata(run_dir / "metadata.json")
        run_id = metadata.get("id") if metadata else run_dir.name
        runs.append(RunSummary.from_metadata(run_id, metadata or {}))
    return runs


def _iter_run_dirs(artifacts_dir: Path) -> Iterable[Path]:
    if not artifacts_dir.exists():
        return []
    run_dirs = sorted(
        (p for p in artifacts_dir.iterdir() if p.is_dir() and p.name.startswith("run-")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return run_dirs


def _read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return {}
