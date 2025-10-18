from pathlib import Path

import pandas as pd

from backend.app.preview.preview_engine import preview_operation


ASSETS_CSV = Path("tests/assets/sample.csv")


def test_preview_filter_counts_rows():
    total_rows = pd.read_csv(ASSETS_CSV).shape[0]

    result = preview_operation(
        {
            "kind": "filter",
            "params": {"csv_path": str(ASSETS_CSV), "where": "a >= 2"},
        }
    )

    assert result["kept_rows"] + result["removed_rows"] == total_rows
    assert result["kept_rows"] > 0
    assert result["column_summary"]["unchanged"] == ["a", "b"]
    assert result["column_summary"]["removed"] == []
    assert result["column_summary"]["added"] == []
