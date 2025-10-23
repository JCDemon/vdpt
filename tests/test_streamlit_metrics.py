from ui.streamlit_app import _extract_operation_metrics


def test_extract_operation_metrics_normalizes_prefixed_keys():
    metrics = {
        "classify": 0.75,
        "op:img_caption": 0.5,
        "op:img_resize": 0.25,
        "img_caption": 0.9,
        42: 1.0,
        "ignored": "not-a-number",
    }

    result = _extract_operation_metrics(metrics)

    assert result == {
        "classify": 0.75,
        "img_caption": 0.9,
        "img_resize": 0.25,
    }


def test_extract_operation_metrics_prefers_largest_value():
    metrics = {
        "op:img_caption": 0.8,
        "img_caption": 0.6,
    }

    result = _extract_operation_metrics(metrics)

    assert result == {"img_caption": 0.8}
