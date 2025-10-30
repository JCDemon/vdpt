"""Utility helpers for previewing operations without heavy dependencies."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from backend.vdpt import providers

from ..ops.registry import get_handler
from ..types import LogEntry, ProvenanceEdge, ProvenanceNode

# Ensure operation handlers register themselves on import.
from ..ops import embed as _embed_ops  # noqa: F401
from ..ops import image_ops as _image_ops  # noqa: F401
from ..ops.cluster import hdbscan_cluster, umap_reduce
from ..ops.text import summarize as _text_summarize  # noqa: F401


logger = logging.getLogger(__name__)


class _ProvenanceRun:
    """Capture provenance details, logs, and per-record data for a run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self.generated_at = datetime.now(timezone.utc)
        self.records: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []

    def start_record(self, row_index: int, inputs: Dict[str, Any]) -> dict[str, Any]:
        record_entry: dict[str, Any] = {
            "row_index": row_index,
            "inputs": _safe_json_value(inputs),
            "parameters": [],
            "outputs": {},
            "provenance": {"nodes": [], "edges": []},
            "logs": [],
        }
        input_node = ProvenanceNode(
            id=f"row-{row_index}",
            type="Entity",
            label=f"Row {row_index} Input",
            metadata={"fields": sorted(inputs.keys())},
        )
        record_entry["_input_node_id"] = input_node.id
        record_entry["provenance"]["nodes"].append(input_node.model_dump(mode="json"))
        self.records.append(record_entry)
        return record_entry

    def record_operation(
        self,
        record_entry: dict[str, Any],
        op_kind: str,
        params: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        op_index = len(record_entry["parameters"])
        activity_id = f"op-{record_entry['row_index']}-{op_index}"
        activity_node = ProvenanceNode(
            id=activity_id,
            type="Activity",
            label=op_kind,
            metadata={"params": _safe_json_value(params)},
        )
        record_entry["parameters"].append({"kind": op_kind, "params": _safe_json_value(params)})
        record_entry["provenance"]["nodes"].append(activity_node.model_dump(mode="json"))

        input_node_id = record_entry.get("_input_node_id")
        if input_node_id:
            edge = ProvenanceEdge(
                source=input_node_id,
                target=activity_id,
                relation="used",
            )
            record_entry["provenance"]["edges"].append(edge.model_dump(mode="json"))

        for name, value in outputs.items():
            entity_id = f"out-{record_entry['row_index']}-{op_index}-{name}"
            entity_node = ProvenanceNode(
                id=entity_id,
                type="Entity",
                label=name,
                metadata={"value": _safe_json_value(value)},
            )
            record_entry["provenance"]["nodes"].append(entity_node.model_dump(mode="json"))
            edge = ProvenanceEdge(
                source=activity_id,
                target=entity_id,
                relation="wasGeneratedBy",
            )
            record_entry["provenance"]["edges"].append(edge.model_dump(mode="json"))

        self.log(
            "INFO",
            f"operation '{op_kind}' completed",
            record_entry=record_entry,
            context={"outputs": sorted(outputs.keys())},
        )

    def finish_record(self, record_entry: dict[str, Any], outputs: Dict[str, Any]) -> None:
        record_entry["outputs"] = _safe_json_value(outputs)
        record_entry.pop("_input_node_id", None)

    def log(
        self,
        level: str,
        message: str,
        *,
        record_entry: dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        payload_context = dict(context or {})
        if record_entry is not None:
            payload_context.setdefault("row_index", record_entry.get("row_index"))
        entry = LogEntry(
            ts=datetime.now(timezone.utc),
            level=level,
            message=message,
            context=_safe_json_value(payload_context),
        )
        serialized = entry.model_dump(mode="json")
        self.logs.append(serialized)
        if record_entry is not None:
            record_entry.setdefault("logs", []).append(serialized)

    def write(self) -> Path:
        payload = {
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "records": [
                {
                    "row_index": record["row_index"],
                    "inputs": record["inputs"],
                    "parameters": record["parameters"],
                    "outputs": record["outputs"],
                    "provenance": record["provenance"],
                    "logs": record.get("logs", []),
                }
                for record in self.records
            ],
            "logs": self.logs,
        }
        path = self.run_dir / "provenance.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return path


def _safe_json_value(value: Any) -> Any:
    """Best-effort conversion to JSON-friendly data."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def start_provenance_run(
    artifacts_dir: Path | str | None = None,
) -> tuple[Path, _ProvenanceRun]:
    """Create a run directory and provenance recorder."""

    run_dir = _ensure_artifact_dir(artifacts_dir)
    return run_dir, _ProvenanceRun(run_dir)


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


def preview_dataset(
    dataset_kind: str,
    records: Sequence[Dict[str, Any]],
    ops: Sequence[Dict[str, Any]],
    *,
    artifacts_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Generate previews for tabular or image datasets with aggregated artifacts."""

    run_dir, provenance = start_provenance_run(artifacts_dir)
    processed_records: List[Dict[str, Any]] = []
    schema_new_columns: List[str] = []
    captions: List[str] = []
    new_columns_seen: set[str] = set()
    base_columns = set(records[0].keys()) if records else set()

    embedding_store: dict[str, List[np.ndarray | None]] = {}
    umap_results: dict[str, List[Optional[np.ndarray]]] = {}
    record_entries: List[dict[str, Any]] = []
    artifacts: Dict[str, str] = {}

    op_kinds = [op.get("kind") for op in ops]
    logger.debug("preview_dataset operations=%s", op_kinds)

    for index, original in enumerate(records):
        record = dict(original)
        record_errors = _ensure_error_list(record)
        record["error"] = record_errors
        record_entry = provenance.start_record(index, dict(original))
        record_entries.append(record_entry)

        for op in ops:
            kind = op.get("kind")
            params = op.get("params") or {}

            if dataset_kind == "csv" and kind == "summarize":
                source_field = str(params.get("field") or "text")
                column_name = _resolve_summary_column(params)
                summary_value = ""
                source_value = original.get(source_field)

                if isinstance(source_value, str):
                    try:
                        op_result = run_operation(record, kind, params)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.exception("summarize preview failed for column '%s'", column_name)
                        message = f"summarize failed: {exc}"
                        record_errors.append(message)
                        provenance.log(
                            "ERROR",
                            f"summarize failed for column '{column_name}'",
                            record_entry=record_entry,
                            context={"error": str(exc)},
                        )
                    else:
                        if column_name in op_result:
                            summary_value = str(op_result[column_name]).strip()
                        elif op_result:
                            summary_value = str(next(iter(op_result.values()))).strip()
                        provenance.record_operation(record_entry, kind, params, op_result)

                record[column_name] = summary_value
                if column_name not in base_columns and column_name not in new_columns_seen:
                    schema_new_columns.append(column_name)
                    new_columns_seen.add(column_name)
            elif dataset_kind == "images" and kind == "img_caption":
                caption_value = ""
                try:
                    op_result = run_operation(record, kind, params)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "image caption preview failed for record '%s'", record.get("id")
                    )
                    message = f"img_caption failed: {exc}"
                    record_errors.append(message)
                    provenance.log(
                        "ERROR",
                        "img_caption failed",
                        record_entry=record_entry,
                        context={"error": str(exc)},
                    )
                else:
                    raw_caption = op_result.get("caption") if op_result else ""
                    if raw_caption is not None:
                        caption_value = str(raw_caption).strip()
                    if op_result:
                        provenance.record_operation(record_entry, kind, params, op_result)
                record["caption"] = caption_value
                if caption_value:
                    captions.append(caption_value)
                if "caption" not in base_columns and "caption" not in new_columns_seen:
                    schema_new_columns.append("caption")
                    new_columns_seen.add("caption")
            elif kind == "field":
                source_field = str(params.get("field") or params.get("name") or "")
                if not source_field:
                    message = "field operation requires 'field'"
                    record_errors.append(message)
                    provenance.log("ERROR", message, record_entry=record_entry)
                    continue
                output_field = str(params.get("output_field") or source_field)
                value = record.get(source_field)
                record[output_field] = value
                provenance.record_operation(record_entry, kind, params, {output_field: value})
                if output_field not in base_columns and output_field not in new_columns_seen:
                    schema_new_columns.append(output_field)
                    new_columns_seen.add(output_field)
            elif dataset_kind == "csv" and kind == "embed_text":
                field = str(params.get("field") or params.get("source") or "")
                output_field = str(params.get("output_field") or "embedding")
                embeddings_list = embedding_store.setdefault(output_field, [])
                if not field:
                    message = "embed_text requires 'field'"
                    record_errors.append(message)
                    provenance.log("ERROR", message, record_entry=record_entry)
                    record[output_field] = None
                    embeddings_list.append(None)
                    continue
                try:
                    op_result = run_operation(record, kind, params)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("embed_text failed for field '%s'", field)
                    message = f"embed_text failed: {exc}"
                    record_errors.append(message)
                    provenance.log(
                        "ERROR",
                        "embed_text failed",
                        record_entry=record_entry,
                        context={"error": str(exc), "field": field},
                    )
                    record[output_field] = None
                    embeddings_list.append(None)
                else:
                    vector_array = _vector_from_output(op_result.get(output_field))
                    if vector_array is None:
                        record[output_field] = None
                        embeddings_list.append(None)
                    else:
                        embeddings_list.append(vector_array)
                        record[output_field] = vector_array.astype(np.float32).tolist()
                    if op_result:
                        provenance.record_operation(record_entry, kind, params, op_result)
                if output_field not in base_columns and output_field not in new_columns_seen:
                    schema_new_columns.append(output_field)
                    new_columns_seen.add(output_field)
            elif dataset_kind == "images" and kind == "embed_image":
                field = str(params.get("field") or params.get("path") or "image_path")
                output_field = str(params.get("output_field") or "embedding")
                embeddings_list = embedding_store.setdefault(output_field, [])
                value = record.get(field)
                if not value:
                    message = "embed_image requires an image path"
                    record_errors.append(message)
                    provenance.log(
                        "ERROR",
                        message,
                        record_entry=record_entry,
                        context={"field": field},
                    )
                    record[output_field] = None
                    embeddings_list.append(None)
                    continue
                try:
                    op_result = run_operation(record, kind, params)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("embed_image failed for path '%s'", value)
                    message = f"embed_image failed: {exc}"
                    record_errors.append(message)
                    provenance.log(
                        "ERROR",
                        "embed_image failed",
                        record_entry=record_entry,
                        context={"error": str(exc), "field": field},
                    )
                    record[output_field] = None
                    embeddings_list.append(None)
                else:
                    vector_array = _vector_from_output(op_result.get(output_field))
                    if vector_array is None:
                        record[output_field] = None
                        embeddings_list.append(None)
                    else:
                        embeddings_list.append(vector_array)
                        record[output_field] = vector_array.astype(np.float32).tolist()
                    if op_result:
                        provenance.record_operation(record_entry, kind, params, op_result)
                if output_field not in base_columns and output_field not in new_columns_seen:
                    schema_new_columns.append(output_field)
                    new_columns_seen.add(output_field)

        if record_errors:
            for err in record_errors:
                provenance.log("WARNING", err, record_entry=record_entry)

        processed_records.append(record)

    for op in ops:
        kind = op.get("kind")
        params = op.get("params") or {}

        if kind == "umap":
            source_field = str(params.get("source") or params.get("field") or "embedding")
            output_field = str(params.get("output_field") or "umap")
            embeddings_list = embedding_store.get(source_field)
            coords_by_record: List[Optional[np.ndarray]] = [None] * len(processed_records)

            if embeddings_list:
                valid = [(idx, vec) for idx, vec in enumerate(embeddings_list) if vec is not None]
                if len(valid) >= 2:
                    matrix = np.vstack([vec for _, vec in valid])
                    coords_valid = umap_reduce(
                        matrix,
                        n_neighbors=_coerce_positive_int(
                            params.get("n_neighbors"), default=15, minimum=2
                        ),
                        min_dist=_coerce_float(params.get("min_dist"), default=0.1, minimum=0.0),
                        metric=str(params.get("metric") or "cosine"),
                    )
                    coords_valid = np.asarray(coords_valid, dtype=np.float32)
                    for (idx, _), coord in zip(valid, coords_valid):
                        coords_by_record[idx] = coord.astype(np.float32)
                elif len(valid) == 1:
                    coords_by_record[valid[0][0]] = np.array([0.0, 0.0], dtype=np.float32)
            else:
                message = f"umap requires embeddings for '{source_field}'"
                for record, record_entry in zip(processed_records, record_entries):
                    _append_error(record, message)
                    provenance.log(
                        "ERROR",
                        message,
                        record_entry=record_entry,
                        context={"source": source_field},
                    )
                    provenance.log("WARNING", message, record_entry=record_entry)
                continue

            umap_results[output_field] = coords_by_record
            for idx, coord in enumerate(coords_by_record):
                if coord is None:
                    processed_records[idx][output_field] = None
                else:
                    processed_records[idx][output_field] = [float(coord[0]), float(coord[1])]
            for idx, record_entry in enumerate(record_entries):
                provenance.record_operation(
                    record_entry,
                    kind,
                    params,
                    {output_field: processed_records[idx].get(output_field)},
                )
            if output_field not in base_columns and output_field not in new_columns_seen:
                schema_new_columns.append(output_field)
                new_columns_seen.add(output_field)
            if any(coord is not None for coord in coords_by_record):
                coords_array = _coords_array(coords_by_record)
                filename = run_dir / f"{_sanitize_name(output_field)}_umap.npy"
                np.save(filename, coords_array)
                artifacts[_artifact_key("umap", output_field)] = str(filename)

        elif kind == "hdbscan":
            source_field = str(params.get("source") or "umap")
            output_field = str(params.get("output_field") or "cluster")
            coords_by_record = umap_results.get(source_field)
            labels_by_record: List[int] = [-1] * len(processed_records)

            if coords_by_record:
                valid = [
                    (idx, coord) for idx, coord in enumerate(coords_by_record) if coord is not None
                ]
                if valid:
                    matrix = np.vstack([coord for _, coord in valid])
                    labels_valid = hdbscan_cluster(
                        matrix,
                        min_cluster_size=_coerce_positive_int(
                            params.get("min_cluster_size"), default=10, minimum=2
                        ),
                        min_samples=_coerce_optional_int(params.get("min_samples")),
                        metric=str(params.get("metric") or "euclidean"),
                    )
                    labels_valid = np.asarray(labels_valid, dtype=np.int32)
                    for (idx, _), label in zip(valid, labels_valid):
                        labels_by_record[idx] = int(label)
            else:
                message = f"hdbscan requires UMAP coordinates from '{source_field}'"
                for record, record_entry in zip(processed_records, record_entries):
                    _append_error(record, message)
                    provenance.log(
                        "ERROR",
                        message,
                        record_entry=record_entry,
                        context={"source": source_field},
                    )
                    provenance.log("WARNING", message, record_entry=record_entry)
                continue

            for idx, label in enumerate(labels_by_record):
                processed_records[idx][output_field] = int(label)
                provenance.record_operation(
                    record_entries[idx], kind, params, {output_field: int(label)}
                )
            if output_field not in base_columns and output_field not in new_columns_seen:
                schema_new_columns.append(output_field)
                new_columns_seen.add(output_field)
            labels_array = np.asarray(labels_by_record, dtype=np.int32)
            filename = run_dir / f"{_sanitize_name(output_field)}_hdbscan.npy"
            np.save(filename, labels_array)
            artifacts[_artifact_key("hdbscan", output_field)] = str(filename)

    embedding_arrays: Dict[str, np.ndarray] = {}
    for name, vectors in embedding_store.items():
        dim = next((vec.shape[-1] for vec in vectors if vec is not None), None)
        if dim is None:
            continue
        filler = np.full((dim,), np.nan, dtype=np.float32)
        stacked = np.vstack(
            [vec.astype(np.float32) if vec is not None else filler for vec in vectors]
        )
        embedding_arrays[_sanitize_name(name)] = stacked
    if embedding_arrays:
        embeddings_path = run_dir / "embeddings.npz"
        np.savez(embeddings_path, **embedding_arrays)
        artifacts["embeddings"] = str(embeddings_path)

    for record_entry, record in zip(record_entries, processed_records):
        provenance.finish_record(record_entry, record)

    if captions:
        captions_path = run_dir / "captions.json"
        captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2))

        metadata_payload = {
            "count": len(captions),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2))

        artifacts["captions"] = str(captions_path)
        artifacts["metadata"] = str(metadata_path)

    provenance_path = provenance.write()
    artifacts["provenance"] = str(provenance_path)

    return {
        "ok": True,
        "schema": {"new_columns": schema_new_columns},
        "records": processed_records,
        "artifacts": artifacts,
        "logs": provenance.logs,
    }


def _ensure_error_list(record: Dict[str, Any]) -> List[str]:
    existing = record.get("error")
    if isinstance(existing, list):
        return [str(err) for err in existing]
    if existing:
        return [str(existing)]
    return []


def _append_error(record: Dict[str, Any], message: str) -> List[str]:
    errors = record.get("error")
    if isinstance(errors, list):
        errors.append(message)
        return errors
    if errors:
        new_errors = [str(errors), message]
    else:
        new_errors = [message]
    record["error"] = new_errors
    return new_errors


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "result"


def _artifact_key(prefix: str, name: str) -> str:
    return prefix if name == prefix else f"{prefix}_{name}"


def _vector_from_output(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return None
    return array.reshape(-1).astype(np.float32)


def _coords_array(coords: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    result = np.full((len(coords), 2), np.nan, dtype=np.float32)
    for idx, coord in enumerate(coords):
        if coord is not None and coord.size >= 2:
            result[idx, :] = coord[:2]
    return result


def _coerce_positive_int(value: Any, *, default: int, minimum: int) -> int:
    if value is None:
        return max(default, minimum)
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return max(default, minimum)
    return max(coerced, minimum)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, *, default: float, minimum: float) -> float:
    if value is None:
        return max(default, minimum)
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return max(default, minimum)
    return max(coerced, minimum)


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


def _draw_points(
    draw: ImageDraw.ImageDraw, points: Iterable[Sequence[float]], width: int, height: int
) -> None:
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


def _summarize_columns(original: Iterable[str], filtered: Iterable[str]) -> dict[str, list[str]]:
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
) -> str:
    """Generate a lightweight summary for preview responses."""

    instructions = params.get("instructions") or "Summarize the following text."
    max_tokens = int(params.get("max_tokens", 128))
    prompt = f"{instructions.strip()}\n\n" "Text:\n" f"{text.strip()}\n" "Summary:"
    try:
        response = providers.current.chat(prompt, max_tokens=max_tokens)
    except Exception as exc:  # pragma: no cover - provider errors
        raise RuntimeError(f"Failed to summarize text: {exc}") from exc
    return response.strip() if isinstance(response, str) else ""


def preview_classify(
    text: str,
    labels: Sequence[str],
    params: dict,
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
        response = providers.current.chat(prompt, max_tokens=max_tokens)
    except Exception as exc:  # pragma: no cover - provider errors
        raise RuntimeError(f"Failed to classify text: {exc}") from exc
    return response.strip() if isinstance(response, str) else ""


def _ensure_artifact_dir(artifacts_dir: Path | str | None) -> Path:
    if artifacts_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        artifacts_root = Path("artifacts")
        artifacts_root.mkdir(parents=True, exist_ok=True)
        target = artifacts_root / f"run-{timestamp}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = Path(artifacts_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_summary_column(params: Dict[str, Any]) -> str:
    for key in ("output_field", "output_column", "column_name", "column"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "text_summary"
