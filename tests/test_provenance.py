from backend.app.provenance import recorder


def test_snapshot_normalized_range():
    recorder.bump_frequency(["x", "x", "y"])
    recorder.bump_recency(["x"])
    recorder.bump_recency(["y"])
    s = recorder.snapshot()
    for v in s.get("frequency", {}).values():
        assert 0.0 <= v <= 1.0
    for v in s.get("recency", {}).values():
        assert 0.0 <= v <= 1.0
