"""Integration tests for the /preview API endpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.vdpt import providers


class _StubProvider:
    """Stub provider returning deterministic responses for tests."""

    def chat(self, prompt: str, *, max_tokens: int = 128, **_: object) -> str:
        return "这是一段测试摘要"

    def caption(self, image_path: str, **_: object) -> str:
        return f"{Path(image_path).name} 的说明"


@pytest.fixture(autouse=True)
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the global provider with the stub implementation."""

    monkeypatch.setattr(providers, "current", _StubProvider())


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create temporary artifacts directories and sample CSV file."""

    uploads_dir = Path("artifacts") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    sample_src = Path("data") / "sample_news.csv"
    sample_dest = uploads_dir / "sample_news.csv"
    shutil.copy(sample_src, sample_dest)

    images_dir = Path("artifacts") / "bundled_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    created_images: list[Path] = []
    ppm_placeholder = ("P3\n# placeholder\n1 1\n255\n 255 0 0\n").encode("utf-8")
    for name in ("forest.ppm", "ocean.ppm", "sunrise.ppm"):
        image_path = images_dir / name
        if not image_path.exists():
            image_path.write_bytes(ppm_placeholder)
            created_images.append(image_path)

    created_dirs: list[Path] = []

    from backend.app import main as main_module

    def _create_artifact_dir() -> Path:
        run_dir = tmp_path / f"run-{len(created_dirs)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(run_dir)
        return run_dir

    monkeypatch.setattr(main_module, "_create_artifact_dir", _create_artifact_dir)

    try:
        yield
    finally:
        if sample_dest.exists():
            sample_dest.unlink()
        for image_path in created_images:
            if image_path.exists():
                image_path.unlink()
        try:
            uploads_dir.rmdir()
        except OSError:
            pass
        for directory in created_dirs:
            shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def api_client() -> TestClient:
    """Instantiate the FastAPI TestClient for the VDPT app."""

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.mark.usefixtures("tmp_artifacts_dir")
def test_preview_text_adds_summary_column(api_client: TestClient) -> None:
    payload = {
        "dataset": {
            "type": "csv",
            "path": "artifacts/uploads/sample_news.csv",
        },
        "preview_sample_size": 2,
        "operations": [
            {
                "kind": "summarize",
                "field": "text",
                "instructions": "用一句中文总结要点",
                "max_tokens": 80,
            }
        ],
    }

    response = api_client.post("/preview", json=payload)

    assert response.status_code == 200
    data = response.json()

    new_columns = data["schema"]["new_columns"]
    assert any(
        column == "text_summary"
        or (isinstance(column, dict) and column.get("name") == "text_summary")
        for column in new_columns
    )

    first_record = data["records"][0]
    assert "text_summary" in first_record
    assert isinstance(first_record["text_summary"], str)
    assert first_record["text_summary"].strip()


@pytest.mark.usefixtures("tmp_artifacts_dir")
def test_preview_images_captions_non_empty(api_client: TestClient) -> None:
    payload = {
        "dataset": {
            "type": "images",
            "path": "artifacts/bundled_images",
            "paths": ["forest.ppm", "ocean.ppm", "sunrise.ppm"],
        },
        "preview_sample_size": 3,
        "operations": [
            {
                "kind": "img_caption",
                "instructions": "用一句中文描述图片内容",
                "max_tokens": 80,
            }
        ],
    }

    response = api_client.post("/preview", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "artifacts" in data
    artifacts = data["artifacts"]
    assert "captions" in artifacts

    captions_path = Path(artifacts["captions"])
    assert captions_path.exists()

    captions = json.loads(captions_path.read_text())
    assert isinstance(captions, list)
    assert captions
    assert all(isinstance(entry, str) and entry.strip() for entry in captions)
