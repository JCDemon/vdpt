from backend.app.main import health


def test_health_ok():
    assert health()["ok"] is True
