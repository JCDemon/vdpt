from __future__ import annotations
from pathlib import Path
import csv

ART_ROOT = Path("artifacts")
UPLOADS = ART_ROOT / "uploads"


def ensure_sample_assets() -> None:
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    csv_path = UPLOADS / "sample_news.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow([1, "英超球队在转会市场上的策略发生了显著变化。"])
            w.writerow([2, "研究人员提出了一种新的模型来总结新闻要点。"])


def load_sample_plan() -> dict:
    # Minimal plan used by the sidebar tree; extend later as needed.
    return {
        "id": "sample-plan",
        "name": "Sample Plan",
        "operations": [
            {
                "kind": "summarize",
                "field": "text",
                "instructions": "用一句中文总结要点",
                "max_tokens": 80,
            }
        ],
    }
