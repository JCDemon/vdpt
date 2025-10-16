"""Tests for provenance recording utilities."""

from backend.app.provenance.recorder import bump_frequency, bump_recency, reset, snapshot


def test_snapshot_empty() -> None:
    reset()
    assert snapshot() == {}


def test_snapshot_range() -> None:
    reset()
    bump_frequency(["a", "a", "b"])
    bump_recency(["a"])
    bump_recency(["b"])

    stats = snapshot()

    assert 0.0 <= stats["frequency"]["a"] <= 1.0
    assert 0.0 <= stats["recency"]["b"] <= 1.0
