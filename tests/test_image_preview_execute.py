import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from PIL import Image

from backend.app.main import Plan, Operation, execute, preview
from backend.vdpt.providers import mock as mock_provider


def _create_test_images(session_dir: Path) -> list[Path]:
    session_dir.mkdir(parents=True, exist_ok=True)
    colors = [(255, 0, 0), (0, 0, 255)]
    paths: list[Path] = []

    for idx, color in enumerate(colors):
        image = Image.new("RGB", (16, 16), color=color)
        path = session_dir / f"sample_{idx}.png"
        image.save(path)
        paths.append(path)

    return paths


def _expected_caption(path: Path) -> str:
    return mock_provider.caption(str(path), prompt="Describe")


def test_preview_and_execute_images(tmp_path):
    session_id = f"test-session-{tmp_path.name}-{uuid4().hex}"
    uploads_dir = Path("artifacts") / "uploads" / session_id
    image_paths = _create_test_images(uploads_dir)

    plan = Plan(
        dataset=Plan.ImageDataset(
            type="images",
            session=session_id,
            paths=[path.name for path in image_paths],
            sample_size=2,
        ),
        ops=[
            Operation(
                kind="img_caption",
                params={"prompt": "Describe"},
            ),
            Operation(
                kind="img_resize",
                params={"width": 8, "height": 8},
            ),
        ],
    )

    preview_result = preview(plan)

    assert preview_result["preview_sample_size"] == 2
    assert len(preview_result["records"]) == 2

    preview_artifacts = preview_result.get("artifacts")
    assert preview_artifacts is not None

    preview_captions_path = Path(preview_artifacts["captions"])
    preview_metadata_path = Path(preview_artifacts["metadata"])

    assert preview_captions_path.exists()
    assert preview_metadata_path.exists()

    preview_captions = json.loads(preview_captions_path.read_text())
    expected_preview_captions = [
        {
            "file": Path(record["image_path"]).name,
            "caption": record["caption"],
        }
        for record in preview_result["records"]
        if "caption" in record
    ]
    assert preview_captions == expected_preview_captions

    preview_metadata = json.loads(preview_metadata_path.read_text())
    assert preview_metadata["files"] == [path.name for path in image_paths]
    assert preview_metadata["preview_sample_size"] == preview_result["preview_sample_size"]
    assert preview_metadata["record_count"] == len(preview_result["records"])
    assert preview_metadata["provider"] == "mock"
    assert preview_metadata["model"] == "mock-vision"
    assert preview_metadata["args"]["dataset"]["session"] == session_id
    assert preview_metadata["artifacts"]["captions"] == str(preview_captions_path)
    assert preview_metadata["input_dir"] == str(uploads_dir.resolve())
    assert isinstance(preview_metadata["time"], str)

    new_columns = {col["name"]: col for col in preview_result["schema"]["new_columns"]}
    assert "caption" in new_columns
    assert "resized_path" in new_columns

    for original_path, record in zip(image_paths, preview_result["records"]):
        assert record["image_path"].endswith(original_path.name)
        assert record["caption"] == _expected_caption(Path(record["image_path"]))
        resized_path = Path(record["resized_path"])
        assert resized_path.exists()

    execute_result = execute(plan)

    assert execute_result["ok"] is True
    artifacts = execute_result["artifacts"]

    output_path = Path(artifacts["output_csv"])
    metadata_path = Path(artifacts["metadata"])
    captions_path = Path(artifacts["captions"])
    preview_path = Path(artifacts["preview"])

    assert output_path.exists()
    assert metadata_path.exists()
    assert captions_path.exists()
    assert preview_path.exists()

    generated_paths = [Path(p) for p in artifacts.get("generated", [])]
    for path in generated_paths:
        assert path.exists()

    output_df = pd.read_csv(output_path)
    assert "caption" in output_df.columns
    assert "resized_path" in output_df.columns

    metadata = json.loads(metadata_path.read_text())
    captions_payload = json.loads(captions_path.read_text())

    assert metadata["plan"]["dataset"]["session"] == session_id
    assert metadata["artifacts"]["captions"] == str(captions_path)
    assert sorted(metadata["source_images"]) == sorted(str(p) for p in image_paths)
    assert sorted(metadata["artifacts"]["generated"]) == sorted(str(p) for p in generated_paths)
    assert metadata["files"] == [path.name for path in image_paths]
    assert metadata["provider"] == "mock"
    assert metadata["model"] == "mock-vision"
    assert metadata["args"]["dataset"]["session"] == session_id
    assert metadata["record_count"] == len(image_paths)
    assert metadata["preview_sample_size"] == preview_result["preview_sample_size"]
    assert metadata["input_dir"] == str(uploads_dir.resolve())
    assert isinstance(metadata["time"], str)

    expected_captions = [
        {"file": path.name, "caption": _expected_caption(path)} for path in image_paths
    ]
    assert captions_payload == expected_captions
