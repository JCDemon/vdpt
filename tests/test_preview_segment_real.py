from backend.app.preview.preview_engine import preview_operation


def test_preview_segment_box_uses_mask_counts():
    result = preview_operation({"kind": "segment", "params": {"box": [10, 10, 50, 50]}})

    assert result["overlay"] == "mask@alpha0.5"
    assert result["affected_pixels"] > 0
