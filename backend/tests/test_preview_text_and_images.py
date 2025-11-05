import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.preview.preview_engine import preview_dataset  # noqa: E402
from backend.vdpt import providers  # noqa: E402


def test_preview_dataset_text_and_images(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "summarize", lambda *_, **__: "stub summary")
    monkeypatch.setattr(
        providers,
        "img_caption",
        lambda image_path, **__: f"caption for {Path(image_path).name}",
    )

    csv_records = [{"text": "hello world"}]
    csv_ops = [{"kind": "summarize", "params": {"field": "text"}}]
    csv_preview = preview_dataset("csv", csv_records, csv_ops, artifacts_dir=tmp_path)

    assert "text_summary" in csv_preview["schema"]["new_columns"]
    assert csv_preview["records"][0]["text_summary"] == "stub summary"

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_names = []
    for name in ("one.jpg", "two.jpg"):
        path = image_dir / name
        path.write_text("fake image contents")
        image_names.append(name)

    image_records = [{"image_path": name} for name in image_names]
    image_ops = [{"kind": "img_caption", "params": {}}]
    image_preview = preview_dataset(
        "images",
        image_records,
        image_ops,
        artifacts_dir=tmp_path / "artifacts",
        dataset={"path": str(image_dir), "paths": image_names},
    )

    assert all(record["caption"].startswith("caption for ") for record in image_preview["records"])

    captions_path = Path(image_preview["artifacts"]["captions"])
    assert captions_path.exists()
    captions = json.loads(captions_path.read_text())
    assert captions == [f"caption for {name}" for name in image_names]

    metadata_path = Path(image_preview["artifacts"]["metadata"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["count"] == len(image_names)
    assert "generated_at" in metadata
