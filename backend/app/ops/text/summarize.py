from __future__ import annotations

from typing import Any, Dict

from backend.vdpt import providers
from ..base import OperationHandler
from ..registry import register


class SummarizeHandler(OperationHandler):
    kind = "summarize"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        field = params.get("field", "text")
        text = str(row.get(field, ""))
        summary = _summarize(text, params)
        return {f"{field}_summary": summary}

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir):
        field = params.get("field", "text")
        text = str(row.get(field, ""))
        summary = _summarize(text, params)
        return {f"{field}_summary": summary}


def _summarize(text: str, params: Dict[str, Any]) -> str:
    instructions = str(params.get("instructions") or "Summarize the following text.").strip()
    max_tokens = _coerce_int(params.get("max_tokens"), default=256, minimum=1)
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            float(temperature)
        except (TypeError, ValueError):
            raise ValueError("summarize temperature must be a number")
    response = providers.summarize(
        text,
        instructions=instructions,
        max_tokens=max_tokens,
    )
    if not isinstance(response, str):
        return ""
    return response.strip()


def _coerce_int(value: Any, *, default: int, minimum: int = 1) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("summarize max_tokens must be an integer") from exc
    if coerced < minimum:
        raise ValueError("summarize max_tokens must be >= 1")
    return coerced


register(SummarizeHandler())
