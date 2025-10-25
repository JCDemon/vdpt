import json
from pathlib import Path

import pandas as pd

from backend.app.main import Plan, Operation, execute, preview
from backend.vdpt.io import sha256_bytes


def _make_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "text": [
                "The sky is blue and clear today.",
                "Rainy weather makes the streets shine.",
                "Snowfall covers everything in white.",
            ]
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _build_plan(csv_path: Path) -> Plan:
    return Plan(
        dataset=Plan.Dataset(
            type="csv",
            path=str(csv_path),
            sample_size=2,
            random_sample=False,
        ),
        ops=[
            Operation(
                kind="summarize",
                params={"field": "text", "instructions": "Summarize in five words."},
            ),
            Operation(
                kind="classify",
                params={"field": "text", "labels": ["positive", "negative"]},
            ),
        ],
    )


def test_preview_adds_new_columns(tmp_path):
    csv_path = _make_csv(tmp_path)
    plan = _build_plan(csv_path)

    result = preview(plan)

    assert result["preview_sample_size"] == 2
    assert len(result["records"]) == 2
    for record in result["records"]:
        assert "text" in record
        assert "text_summary" in record
        assert "text_classification" in record

    new_columns = {col["name"]: col for col in result["schema"]["new_columns"]}
    assert "text_summary" in new_columns
    assert new_columns["text_summary"]["operation"] == "summarize"
    assert "text_classification" in new_columns
    assert new_columns["text_classification"]["operation"] == "classify"


def test_execute_writes_artifacts(tmp_path):
    csv_path = _make_csv(tmp_path)
    plan = _build_plan(csv_path)

    exec_result = execute(plan)

    assert exec_result["ok"] is True
    artifacts = exec_result["artifacts"]

    output_path = Path(artifacts["output_csv"])
    metadata_path = Path(artifacts["metadata"])
    preview_path = Path(artifacts["preview"])

    assert output_path.exists()
    assert metadata_path.exists()
    assert preview_path.exists()

    output_df = pd.read_csv(output_path)
    assert "text_summary" in output_df.columns
    assert "text_classification" in output_df.columns

    metadata = json.loads(metadata_path.read_text())
    assert metadata["plan"]["dataset"]["path"] == str(csv_path)
    assert metadata["dataset_hash"] == sha256_bytes(csv_path.read_bytes())
    assert metadata["preview_sample_size"] == 2

    preview_payload = json.loads(preview_path.read_text())
    assert len(preview_payload["records"]) == metadata["preview_sample_size"]
    assert preview_payload["schema"]["new_columns"]
