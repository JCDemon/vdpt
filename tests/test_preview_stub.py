"""Tests for the preview stubs."""

from backend.app.preview.preview_engine import preview_operation


def test_preview_segment_keys() -> None:
    result = preview_operation({"kind": "segment", "params": {}})
    assert "affected_pixels" in result


def test_preview_filter_keys() -> None:
    result = preview_operation({"kind": "filter", "params": {}})
    assert "kept_rows" in result
