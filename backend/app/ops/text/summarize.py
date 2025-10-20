from __future__ import annotations

from typing import Any, Dict

from ..base import OperationHandler
from ..registry import register


class SummarizeHandler(OperationHandler):
    kind = "summarize"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        field = params.get("field", "text")
        instr = params.get("instructions", "")
        src = str(row.get(field, ""))
        out = f"[mock-summary:{instr}] " + (src[:60] + "..." if len(src) > 60 else src)
        return {f"{field}_summary": out}

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir):
        # same as preview for mock; no disk writes
        return self.preview(row, params)


register(SummarizeHandler())
