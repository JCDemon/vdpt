"""Simple in-memory provenance recorder."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

_frequency_counter: Counter[str] = Counter()
_recency_index: dict[str, int] = {}
_tick: int = 0


def bump_frequency(kinds: Iterable[str]) -> None:
    """Increase the frequency count for each operation kind."""

    _frequency_counter.update(kinds)


def bump_recency(kinds: Iterable[str]) -> None:
    """Track the order in which operation kinds were executed."""

    global _tick
    for kind in kinds:
        _tick += 1
        _recency_index[kind] = _tick


def snapshot() -> dict[str, dict[str, float]]:
    """Return normalized frequency and recency statistics."""

    if not _frequency_counter and not _recency_index:
        return {}

    frequency: dict[str, float] = {}
    if _frequency_counter:
        max_count = max(_frequency_counter.values()) or 1
        frequency = {kind: count / max_count for kind, count in _frequency_counter.items()}

    recency: dict[str, float] = {}
    if _recency_index:
        max_tick = max(_recency_index.values()) or 1
        recency = {kind: tick / max_tick for kind, tick in _recency_index.items()}

    return {"frequency": frequency, "recency": recency}


def reset() -> None:
    """Clear all recorded provenance data."""

    global _tick
    _frequency_counter.clear()
    _recency_index.clear()
    _tick = 0
