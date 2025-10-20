"""Utility helpers for previewing operations without heavy dependencies."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from ...vdpt.providers import TextLLMProvider

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..ops.registry import get_handler
# Ensure operation handlers register themselves on import.
from ..ops.text import summarize as _text_summarize  # noqa: F401


_ASSETS_DIR = Path(__file__).resolve().parents[2] / "tests" / "assets"
_DEFAULT_IMAGE = _ASSETS_DIR / "sample.jpg"
_DEFAULT_CSV = _ASSETS_DIR / "sample.csv"
_POINT_RADIUS = 8
_ALLOWED_QUERY_CHARS = re.compile(r"^[\w\s><=.!&|()'\"+-/*]+$")
_ALLOWED_KEYWORDS = {"and", "or", "not", "True", "False"}


def run_operation(
    row: Dict[str, Any],
    op_kind: str,
    op_params: Dict[str, Any],
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """Dispatch an operation via the registry to preview or execute it."""

    handler = get_handler(op_kind)
    if out_dir is None:
        return handler.preview(row, op_params)
    return handler.execute(row, op_params, out_dir)


def preview_operation(op: dict):
    """Return a lightweight preview for the provided operation description."""

    kind = op.get("kind")
    params = op.get("params") or {}

    if kind == "segment":
        return _preview_segment(params)
    if kind == "filter":
        return _preview_filter(params)
    return {"note": "stub"}


def _preview_segment(params: dict) -> dict:
    image_path = params.get("image_path")
    image = _load_image(Path(image_path) if image_path else _DEFAULT_IMAGE)
    width, height = image.size

    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    if "box" in params:
        _draw_box(draw, params["box"], width, height)
    elif "points" in params:
        _draw_points(draw, params["points"], width, height)

    mask_array = np.array(mask_image, dtype=np.uint8)
    affected_pixels = int(np.count_nonzero(mask_array))
    return {"overlay": "mask@alpha0.5", "affected_pixels": affected_pixels}


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except FileNotFoundError:
        # Fallback to a deterministic blank canvas.
        return Image.new("RGB", (512, 512), color="black")


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], width: int, height: int) -> None:
    if len(box) != 4:
        return
    if width <= 0 or height <= 0:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    x1, x2 = sorted((_clamp(x1, 0, max_x), _clamp(x2, 0, max_x)))
    y1, y2 = sorted((_clamp(y1, 0, max_y), _clamp(y2, 0, max_y)))
    if x1 == x2 or y1 == y2:
        return
    draw.rectangle([x1, y1, x2, y2], fill=255)


def _draw_points(draw: ImageDraw.ImageDraw, points: Iterable[Sequence[float]], width: int, height: int) -> None:
    for point in points or []:
        if len(point) != 2:
            continue
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        draw.ellipse(
            [x - _POINT_RADIUS, y - _POINT_RADIUS, x + _POINT_RADIUS, y + _POINT_RADIUS],
            fill=255,
        )


def _preview_filter(params: dict) -> dict:
    csv_path = params.get("csv_path")
    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    df = _load_csv(path)

    where = params.get("where")
    if where:
        _validate_where(where, df.columns)
        df_filtered = df.query(where, engine="python")
    else:
        df_filtered = df

    kept_rows = int(df_filtered.shape[0])
    removed_rows = int(df.shape[0] - df_filtered.shape[0])
    column_summary = _summarize_columns(df.columns, df_filtered.columns)
    return {
        "kept_rows": kept_rows,
        "removed_rows": removed_rows,
        "column_summary": column_summary,
    }


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _validate_where(where: str, columns: Iterable[str]) -> None:
    if "@" in where:
        raise ValueError("query expressions using '@' variables are not supported")
    if not _ALLOWED_QUERY_CHARS.match(where):
        raise ValueError("query contains unsupported characters")

    allowed_columns: List[str] = [str(col) for col in columns]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", where)
    for token in tokens:
        if token not in allowed_columns and token not in _ALLOWED_KEYWORDS:
            raise ValueError(f"Unsupported token in query: {token}")


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _summarize_columns(
    original: Iterable[str], filtered: Iterable[str]
) -> dict[str, list[str]]:
    """Return a summary of column changes between the original and filtered data."""

    original_set = {str(col) for col in original}
    filtered_set = {str(col) for col in filtered}

    unchanged = sorted(original_set & filtered_set)
    removed = sorted(original_set - filtered_set)
    added = sorted(filtered_set - original_set)

    return {"unchanged": unchanged, "removed": removed, "added": added}


def preview_summarize(
    text: str,
    params: dict,
    provider: TextLLMProvider,
) -> str:
    """Generate a lightweight summary for preview responses."""

    instructions = params.get("instructions") or "Summarize the following text."
    max_tokens = int(params.get("max_tokens", 128))
    prompt = (
        f"{instructions.strip()}\n\n"
        "Text:\n"
        f"{text.strip()}\n"
        "Summary:"
    )
    try:
        return provider.generate(prompt, max_tokens=max_tokens).strip()
    except Exception as exc:  # pragma: no cover - provider errors
        raise RuntimeError(f"Failed to summarize text: {exc}") from exc


def preview_classify(
    text: str,
    labels: Sequence[str],
    params: dict,
    provider: TextLLMProvider,
) -> str:
    """Classify text into one of the provided labels for preview output."""

    if not labels:
        raise ValueError("classify operation requires at least one label")
    label_list = ", ".join(labels)
    instructions = params.get("instructions") or "Classify the text into one label."
    max_tokens = int(params.get("max_tokens", 16))
    prompt = (
        f"{instructions.strip()}\n\n"
        f"Possible labels: {label_list}\n"
        f"Text: {text.strip()}\n"
        "Label:"
    )
    try:
        return provider.generate(prompt, max_tokens=max_tokens).strip()
    except Exception as exc:  # pragma: no cover - provider errors
        raise RuntimeError(f"Failed to classify text: {exc}") from exc
