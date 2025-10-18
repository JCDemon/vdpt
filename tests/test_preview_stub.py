from backend.app.preview.preview_engine import preview_operation


def test_preview_segment_keys():
    out = preview_operation({"kind": "segment", "params": {}})
    assert "affected_pixels" in out and "overlay" in out


def test_preview_filter_keys():
    out = preview_operation({"kind": "filter", "params": {}})
    assert "kept_rows" in out and "removed_rows" in out
    assert "column_summary" in out
