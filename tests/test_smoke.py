"""Basic smoke tests for the VDPT API."""

from backend import create_app
from fastapi.testclient import TestClient


def test_health_endpoint_reports_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
