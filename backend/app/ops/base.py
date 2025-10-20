from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

class OperationHandler:
    kind: str  # e.g., "summarize", "img_resize"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Return dict of {new_column_name: value} for preview (no disk writes required)."""
        raise NotImplementedError

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        """Return dict of {new_column_name: value} for full execution (persist outputs under out_dir)."""
        raise NotImplementedError
