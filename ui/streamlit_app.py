"""Streamlit UI for interacting with the VDPT FastAPI backend."""

from __future__ import annotations

import base64
import binascii
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_LABEL_OPTIONS = ["positive", "negative", "neutral"]
TOP_K_PROVENANCE = 5
UPLOAD_DIR = Path("artifacts") / "uploads"
IMAGE_UPLOAD_SUBDIR = "images"

_REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_IMAGE_DIR = _REPO_ROOT / "artifacts" / "bundled_images"

_BUNDLED_IMAGE_PAYLOADS = {
    "sunrise.png": (
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABUElEQVR4nO3YsUsCUQDH8ffOgzAJEUKn"
        "AoeIlqBoCYqShsDIq6GmiHJJmiMyIpyci4Ki4f4AXWwNQRCRhnBpiVaHwAiTEKOU+gt83RG8n8Lvs97y"
        "+/J4d3Dyp5QS/cxAD/gvBqAxAI0BaAxAYwAaA9AYgMYANAagMQCNAWgMQDMVzyrPL0dX+e92x/QY9rE1E"
        "vRrm+Wc6gTi6Vs7aRUudhJrMweXd9o2uaI6gVq9+fnVFkLE5saDAZ+uSe6oAtJ7S/P7dnR2bGt5MjId1r"
        "bJFan+O13/aOWKT2eZ+/WFiVR8UdMoN7regdf3ZvmxGhjy7q5M5c+3r3MPOmc51zVASrl5mq3WGkKIt0Z"
        "rNNSLryChuAPD/sGbw9WNk6x3wPQYhp20dM5y7o870Pv6/kvMADQGoDEAjQFoDEBjABoD0BiAxgA0BqAx"
        "AI0BaAxAYwDaL9y0Qb5RRCsoAAAAAElFTkSuQmCC"
    ),
    "forest.png": (
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAuElEQVR4nO3TMQqDMABG4bbp7iR0cenq"
        "0tUb5AxOmXqGHqE36FrwEl4u0MFd0EIekfdNAZf/YXJO33Sq2YUe8C8DaAbQDKAZQDOAZgDNAJoBNANoB"
        "tAMoBlAu658m55Te2+Xc/fo+tgXmbTNWkAIIb5isSn7VH+Fqg9Yu0I55/k9L+chDc2tKTJpG98A7dAB42"
        "cstmO3Q/+BKhhAM4BmAM0AmgE0A2gG0AygGUAzgGYAzQCaAbTqA35QXA5MC4MMrAAAAABJRU5ErkJggg=="
    ),
    "ocean.png": (
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABR0lEQVR4nO3ZoU8CUQDH8Xd4XrhwQw1O"
        "CBQJyNk0iZE5qAaMaNH/gOE/IM2KTYsb2oVAwkrjgtsRIKCJqThxOxjYaM89dL7f2H6f9nbl993uXjk"
        "jf90RiyyEHvBXDEBjABoD0BiAxgA0BqAxAI0BaAxAYwAaA9AYgGb+8Mx/vH+q34TM5cl4lEgfb6YOtc1"
        "SJw3otRp+o3JQuLVsJxgO6pcn9sp6JJnSOU6F9BXyqle7uXPLdoQQlu3s5IreQ1njMFXSgPfn9mosOTu"
        "uxdy3nq9l0nyUP+LpVBjGfy75JWlAOBrvd73Zsd/1wtG4lknzkQa4mdNm5SL4+hBCBMNB8660nT3TOEy"
        "V9BaKuPufry+10tGSaU3Go0Q6v7G1p3OZIoN/aMAYgMYANAagMQCNAWgMQGMAGgPQGIDGADQGoDEAjQFo"
        "DED7BoxFQgUC1yeEAAAAAElFTkSuQmCC"
    ),
}

SAMPLE_CSV_CANDIDATES: List[Path] = [
    Path(__file__).resolve().with_name("sample.csv"),
    _REPO_ROOT / "data" / "sample_news.csv",
    Path(__file__).resolve().parents[1] / "tests" / "assets" / "sample.csv",
]
SAMPLE_PLAN_CANDIDATES: List[Path] = [
    Path(__file__).resolve().with_name("sample_plan.json"),
    _REPO_ROOT / "samples" / "plan_news_summarize.json",
    Path(__file__).resolve().parents[1] / "tests" / "assets" / "sample_plan.json",
]

IMAGE_SAMPLE_PLAN: Dict[str, Any] = {
    "dataset": {
        "type": "images",
        "path": str(BUNDLED_IMAGE_DIR.resolve()),
        "paths": sorted(_BUNDLED_IMAGE_PAYLOADS.keys()),
    },
    "ops": [
        {
            "kind": "img_caption",
            "params": {"instructions": "用一句中文描述图片内容", "max_tokens": 80},
        },
        {
            "kind": "img_resize",
            "params": {"width": 384, "height": 384, "keep_aspect": True},
        },
    ],
}


st.set_page_config(page_title="VDPT Preview & Execute", layout="wide")
st.title("VDPT Preview & Execution")


if "plan_ops" not in st.session_state:
    st.session_state.plan_ops = []  # type: ignore[attr-defined]
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 5
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = ""
if "preview_result" not in st.session_state:
    st.session_state.preview_result = None
if "execute_result" not in st.session_state:
    st.session_state.execute_result = None
if "dataset_kind" not in st.session_state:
    st.session_state.dataset_kind = "csv"
if "images_dir" not in st.session_state:
    st.session_state.images_dir = ""
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "use_bundled_images" not in st.session_state:
    st.session_state.use_bundled_images = False
if "previous_images_state" not in st.session_state:
    st.session_state.previous_images_state = None
if "prov_counts" not in st.session_state:
    st.session_state.prov_counts = {}  # type: ignore[attr-defined]
if "prov_history" not in st.session_state:
    st.session_state.prov_history = []  # type: ignore[attr-defined]
if "provenance_freq_items" not in st.session_state:
    st.session_state.provenance_freq_items = []  # type: ignore[attr-defined]
if "provenance_recency_items" not in st.session_state:
    st.session_state.provenance_recency_items = []  # type: ignore[attr-defined]


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for candidate in paths:
        if candidate and candidate.exists():
            return candidate
    return None


def _ensure_bundled_images_present() -> None:
    try:
        BUNDLED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.warning(f"Unable to create bundled image directory: {exc}")
        return

    for filename, encoded in _BUNDLED_IMAGE_PAYLOADS.items():
        target = BUNDLED_IMAGE_DIR / filename
        if target.exists():
            continue
        try:
            binary = base64.b64decode(encoded)
        except binascii.Error as exc:
            st.warning(f"Failed to decode bundled image {filename}: {exc}")
            continue
        try:
            target.write_bytes(binary)
        except OSError as exc:
            st.warning(f"Failed to write bundled image {filename}: {exc}")


def _load_sample_plan() -> Optional[Dict[str, Any]]:
    plan_path = _find_first_existing(SAMPLE_PLAN_CANDIDATES)
    if not plan_path:
        return None
    try:
        payload = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    ops: List[Dict[str, Any]] = []
    for op in payload.get("ops", []):
        kind = op.get("kind")
        params: Dict[str, Any] = {}
        params.update(op.get("params") or {})
        if kind == "summarize":
            params.update(op.get("summarize") or {})
            ops.append(
                {
                    "kind": "summarize",
                    "params": {
                        "field": params.get("field", ""),
                        "instructions": params.get("instructions", ""),
                        "max_tokens": int(params.get("max_tokens", 128) or 128),
                    },
                }
            )
        elif kind == "classify":
            params.update(op.get("classify") or {})
            labels = params.get("labels") or []
            if not isinstance(labels, list):
                labels = [str(labels)]
            ops.append(
                {
                    "kind": "classify",
                    "params": {
                        "field": params.get("field", ""),
                        "labels": [str(label) for label in labels],
                    },
                }
            )
    dataset = payload.get("dataset") or {}
    return {"ops": ops, "dataset": dataset}


def _list_bundled_images() -> List[Path]:
    _ensure_bundled_images_present()
    if not BUNDLED_IMAGE_DIR.exists():
        return []

    supported_suffixes = {".png", ".jpg", ".jpeg"}
    candidates = [
        path
        for path in BUNDLED_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    return sorted(candidates, key=lambda item: item.name.lower())


def _enable_bundled_images(*, remember_previous: bool = True) -> bool:
    bundled = _list_bundled_images()
    if not bundled:
        return False

    if remember_previous:
        st.session_state.previous_images_state = {
            "images_dir": st.session_state.get("images_dir", ""),
            "selected_images": list(st.session_state.get("selected_images", [])),
            "sample_size": int(st.session_state.get("sample_size", 1) or 1),
        }

    st.session_state.images_dir = str(BUNDLED_IMAGE_DIR.resolve())
    st.session_state.selected_images = [path.name for path in bundled]

    current_size = int(st.session_state.get("sample_size", 1) or 1)
    st.session_state.sample_size = min(max(1, current_size), len(bundled))
    st.session_state.use_bundled_images = True
    return True


def _disable_bundled_images() -> None:
    previous = st.session_state.get("previous_images_state")
    if isinstance(previous, dict):
        st.session_state.images_dir = previous.get("images_dir", "")
        st.session_state.selected_images = list(previous.get("selected_images", []))
        try:
            restored_size = int(previous.get("sample_size", st.session_state.sample_size))
            st.session_state.sample_size = max(1, restored_size)
        except (TypeError, ValueError):
            st.session_state.sample_size = max(1, st.session_state.sample_size)
    else:
        st.session_state.images_dir = ""
        st.session_state.selected_images = []
    st.session_state.use_bundled_images = False
    st.session_state.previous_images_state = None


def _persist_uploaded_file(upload) -> Optional[Path]:
    if upload is None:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / upload.name
    try:
        with target.open("wb") as fp:
            fp.write(upload.getbuffer())
    except OSError as exc:
        st.error(f"Failed to save uploaded file: {exc}")
        return None
    return target


def _ensure_images_dir() -> Path:
    images_dir_str = st.session_state.get("images_dir", "")
    if images_dir_str:
        images_dir = Path(images_dir_str)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        images_dir = UPLOAD_DIR / IMAGE_UPLOAD_SUBDIR / timestamp
        st.session_state.images_dir = str(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _persist_uploaded_images(uploads) -> List[Path]:
    saved: List[Path] = []
    if not uploads:
        return saved

    if st.session_state.get("use_bundled_images"):
        _disable_bundled_images()

    images_dir = _ensure_images_dir()

    for upload in uploads:
        if upload is None:
            continue
        filename = Path(upload.name).name
        target = images_dir / filename
        try:
            with target.open("wb") as fp:
                fp.write(upload.getbuffer())
        except OSError as exc:
            st.error(f"Failed to save uploaded image '{filename}': {exc}")
            continue
        saved.append(target)

    return saved


def _resolve_selected_image_paths() -> List[Path]:
    images_dir_str = st.session_state.get("images_dir", "")
    if not images_dir_str:
        return []

    images_dir = Path(images_dir_str)
    if not images_dir.exists():
        return []

    resolved: List[Path] = []
    remaining: List[str] = []
    for rel_path in st.session_state.selected_images:
        path = images_dir / rel_path
        if path.exists():
            resolved.append(path)
            remaining.append(rel_path)
    if remaining != st.session_state.selected_images:
        st.session_state.selected_images = remaining
    return resolved


def _format_size(num_bytes: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < step or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} B"


def _render_image_preview_table(records: List[Dict[str, Any]], ops: List[Dict[str, Any]]) -> None:
    if not records:
        st.info("No preview records to display.")
        return

    has_caption = any((op.get("kind") == "img_caption") for op in ops)
    if not has_caption:
        has_caption = any("caption" in record for record in records)
    has_resize = any((op.get("kind") == "img_resize") for op in ops)

    column_labels: List[str] = ["Thumb", "Filename"]
    column_weights: List[int] = [1, 3]
    if has_caption:
        column_labels.append("Caption")
        column_weights.append(4)
    if has_resize:
        column_labels.append("Resized path")
        column_weights.append(4)

    container = st.container()
    header_cols = container.columns(column_weights)
    for col, label in zip(header_cols, column_labels):
        col.markdown(f"**{label}**")

    for record in records:
        row_cols = container.columns(column_weights)
        image_value = record.get("image_path")
        image_path = Path(str(image_value)) if image_value else None
        if image_path and image_path.exists():
            row_cols[0].image(str(image_path), width=64)
        elif image_path:
            row_cols[0].markdown(f"`{image_path}`")
        else:
            row_cols[0].markdown("—")

        filename = image_path.name if image_path and image_path.name else str(image_path or "")
        row_cols[1].write(filename or "—")

        next_col = 2
        if has_caption:
            caption_value = record.get("caption")
            if caption_value:
                row_cols[next_col].write(caption_value)
            else:
                row_cols[next_col].markdown("—")
            next_col += 1
        if has_resize:
            resized_value = record.get("resized_path")
            if resized_value:
                row_cols[next_col].code(str(resized_value), language="plain")
            else:
                row_cols[next_col].markdown("—")


def _render_image_gallery(paths: List[Path], columns: int = 3) -> None:
    if not paths:
        return

    columns = max(1, min(columns, len(paths)))
    for start in range(0, len(paths), columns):
        row_paths = paths[start : start + columns]
        grid = st.columns(len(row_paths))
        for col, path in zip(grid, row_paths):
            with col:
                st.image(str(path), caption=path.name, use_container_width=True)


def sample_size_control(
    label: str,
    count: int,
    default: int = 5,
    *,
    allow_single_slider: bool = False,
) -> int:
    """Return a valid sample size. For count<=0, skip slider. Optionally allow single."""

    if count <= 0:
        st.caption(f"{label}: 0")
        return 0

    max_value = max(1, count)
    default = min(default, max_value)
    if count == 1 and not allow_single_slider:
        st.caption(f"{label}: 1")
        return 1

    return st.slider(label, min_value=1, max_value=max_value, value=default)


def drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with ``None`` values from ``values``."""

    return {key: val for key, val in values.items() if val is not None}


def build_dataset_payload_csv(csv_path: str) -> Dict[str, Any]:
    return {"type": "csv", "path": csv_path}


def build_dataset_payload_images(images_dir: Path, filenames: list[str]) -> Dict[str, Any]:
    try:
        resolved_dir = images_dir.resolve()
    except OSError:
        resolved_dir = images_dir if images_dir.is_absolute() else images_dir.absolute()

    return {
        "type": "images",
        "path": str(resolved_dir),
        "paths": [str(Path(name).name) for name in filenames],
    }


def _read_columns(path: str) -> List[str]:
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return []
    return [str(col) for col in df.columns]


def _default_params_for_kind(kind: str, columns: List[str]) -> Dict[str, Any]:
    if kind == "summarize":
        return {
            "field": columns[0] if columns else "",
            "instructions": "",
            "max_tokens": 128,
        }
    if kind == "classify":
        return {
            "field": columns[0] if columns else "",
            "labels": list(DEFAULT_LABEL_OPTIONS),
        }
    if kind == "img_caption":
        return {"instructions": "", "max_tokens": 80}
    if kind == "img_resize":
        return {"width": 512, "height": 512, "keep_aspect": True}
    return {}


def _ensure_operation_params(
    op: Dict[str, Any], kind: str, columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    params = op.get("params")
    if not isinstance(params, dict):
        params = {}

    legacy_keys = [
        "field",
        "instructions",
        "max_tokens",
        "labels",
        "prompt",
        "width",
        "height",
        "keep_aspect",
        "keep_ratio",
    ]
    for key in legacy_keys:
        if key in op and key not in params:
            params[key] = op[key]
        op.pop(key, None)

    if kind in {"summarize", "classify"}:
        params.setdefault("field", "")
        params["field"] = str(params.get("field", ""))
    if kind == "summarize":
        params["instructions"] = str(params.get("instructions", ""))
        try:
            params["max_tokens"] = int(params.get("max_tokens", 128) or 128)
        except (TypeError, ValueError):
            params["max_tokens"] = 128
    elif kind == "classify":
        labels = params.get("labels")
        if isinstance(labels, list):
            params["labels"] = [str(label) for label in labels]
        elif labels is None:
            params["labels"] = list(DEFAULT_LABEL_OPTIONS)
        else:
            params["labels"] = [str(labels)]
    elif kind == "img_caption":
        prompt_value = params.pop("prompt", None)
        if prompt_value is not None and not params.get("instructions"):
            params["instructions"] = str(prompt_value)
        params["instructions"] = str(params.get("instructions", ""))
        try:
            params["max_tokens"] = int(params.get("max_tokens", 80) or 80)
        except (TypeError, ValueError):
            params["max_tokens"] = 80
    elif kind == "img_resize":
        if "keep_ratio" in params and "keep_aspect" not in params:
            params["keep_aspect"] = bool(params.pop("keep_ratio"))
        params["keep_aspect"] = bool(params.get("keep_aspect", True))
        for key in ("width", "height"):
            try:
                value = int(params.get(key, 512) or 512)
            except (TypeError, ValueError):
                value = 512
            if value <= 0:
                value = 512
            params[key] = value

    if columns and kind in {"summarize", "classify"}:
        if params.get("field") not in columns and columns:
            params["field"] = columns[0]

    op["params"] = params
    return params


def _extract_run_directory(artifacts: Dict[str, Any]) -> Optional[Path]:
    if not isinstance(artifacts, dict):
        return None

    candidates: List[Path] = []

    def _collect_paths(value: Any) -> None:
        if isinstance(value, str) and value:
            try:
                candidates.append(Path(value))
            except (TypeError, ValueError):
                return
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _collect_paths(item)
        elif isinstance(value, dict):
            for item in value.values():
                _collect_paths(item)

    _collect_paths(artifacts)

    for path in candidates:
        for candidate in (path, *path.parents):
            if candidate == candidate.parent:
                continue
            name = candidate.name
            parent_name = candidate.parent.name if candidate.parent != candidate else ""
            if name.startswith("run-") and parent_name == "artifacts":
                return candidate
    for path in candidates:
        parent = path.parent
        if parent != path:
            return parent
    return None


# --- helpers: artifacts ---


def _read_bytes_safe(p: Path) -> bytes | None:
    try:
        return p.read_bytes()
    except Exception:
        return None


def render_artifact(label: str, rel_path: str, *, key_suffix: str | None = None):
    """Show an artifact row with a copyable path and a download button if readable."""
    import streamlit as st

    p = Path(rel_path)
    st.write(f"**{label}**")
    st.code(str(p), language="text")
    data = _read_bytes_safe(p)
    if data is not None:
        mime = "application/json" if p.suffix.lower() == ".json" else "text/plain"
        artifact_key = str(p).replace("/", "-")
        safe_label = label.replace(" ", "-")
        suffix_source = key_suffix or artifact_key
        suffix = suffix_source.replace("/", "-")
        key_value = f"dl-{suffix}-{safe_label}"
        st.download_button(
            label=f"Download {p.name}",
            data=data,
            file_name=p.name,
            mime=mime,
            use_container_width=True,
            key=key_value,
        )
    else:
        st.info("Artifact not readable from UI process; path is shown for reference.")


def _extract_artifact_run_identifier(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    for key in ("run_id", "runId", "runID", "time_id", "timeId", "timeID"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for value in artifacts.values():
            if isinstance(value, str) and value:
                parts = Path(value).parts
                for part in reversed(parts):
                    if part.startswith("run-"):
                        return part

    return None


def render_artifacts_section(resp_json: Any) -> None:
    arts = resp_json.get("artifacts") if isinstance(resp_json, dict) else None
    if not arts:
        return

    st.subheader("Artifacts")
    run_identifier = _extract_artifact_run_identifier(resp_json)
    for key in ("captions", "metadata", "output_csv", "preview"):
        if key in arts and arts[key]:
            render_artifact(key, arts[key], key_suffix=run_identifier)
    for key, val in arts.items():
        if key in ("captions", "metadata", "output_csv", "preview"):
            continue
        if isinstance(val, str):
            render_artifact(key, val, key_suffix=run_identifier)


def _prepare_plan_payload(
    ops: List[Dict[str, Any]],
    dataset: Optional[Dict[str, Any]],
    *,
    preview_sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    operations: List[Dict[str, Any]] = []
    for op in ops:
        kind = op.get("kind")
        params = op.get("params") if isinstance(op.get("params"), dict) else {}
        if not kind:
            continue

        if kind == "summarize":
            max_tokens_raw = params.get("max_tokens", 128)
            try:
                max_tokens_value = int(max_tokens_raw if max_tokens_raw is not None else 128)
            except (TypeError, ValueError):
                max_tokens_value = 128
            field_value = params.get("field")
            instructions_value = params.get("instructions")
            sanitized = drop_none(
                {
                    "kind": "summarize",
                    "field": "" if field_value is None else str(field_value),
                    "instructions": "" if instructions_value is None else str(instructions_value),
                    "max_tokens": max_tokens_value,
                }
            )
        elif kind == "classify":
            labels = params.get("labels")
            if isinstance(labels, list):
                sanitized_labels = [str(label) for label in labels]
            elif labels is None:
                sanitized_labels = list(DEFAULT_LABEL_OPTIONS)
            else:
                sanitized_labels = [str(labels)]
            field_value = params.get("field")
            sanitized = drop_none(
                {
                    "kind": "classify",
                    "field": "" if field_value is None else str(field_value),
                    "labels": sanitized_labels,
                }
            )
        elif kind == "img_caption":
            try:
                max_tokens = int(params.get("max_tokens", 80) or 80)
            except (TypeError, ValueError):
                max_tokens = 80
            instructions_value = params.get("instructions") or params.get("prompt") or ""
            sanitized = drop_none(
                {
                    "kind": "img_caption",
                    "instructions": str(instructions_value),
                    "max_tokens": max_tokens,
                }
            )
        elif kind == "img_resize":
            width_source = params.get("width", 512)
            height_source = params.get("height", width_source)
            try:
                width = int(width_source or 512)
            except (TypeError, ValueError):
                width = 512
            try:
                height = int(height_source or width)
            except (TypeError, ValueError):
                height = width
            keep_aspect = bool(params.get("keep_aspect", True))
            sanitized = drop_none(
                {
                    "kind": "img_resize",
                    "width": max(width, 1),
                    "height": max(height, 1),
                    "keep_aspect": keep_aspect,
                }
            )
        else:
            flattened: Dict[str, Any] = {"kind": str(kind)}
            if isinstance(params, dict):
                for key, value in params.items():
                    if value is None:
                        continue
                    flattened[key] = value
            sanitized = flattened

        operations.append(sanitized)

    payload: Dict[str, Any] = {"operations": operations}

    if dataset:
        dataset_copy: Dict[str, Any] = {}
        for key, value in dataset.items():
            if value is None:
                continue
            dataset_copy[key] = value
        dataset_type = dataset_copy.get("type") or dataset_copy.get("kind")
        if dataset_type is not None:
            dataset_copy["type"] = str(dataset_type)
        dataset_copy.pop("kind", None)
        path_value = dataset_copy.get("path")
        if path_value is not None:
            dataset_copy["path"] = str(path_value)
        if dataset_copy.get("type") == "images":
            paths_value = dataset_copy.get("paths")
            if isinstance(paths_value, list):
                dataset_copy["paths"] = [
                    Path(str(name)).name for name in paths_value if name is not None
                ]
        payload["dataset"] = dataset_copy

    if preview_sample_size is not None:
        try:
            sample_size = int(preview_sample_size)
        except (TypeError, ValueError):
            sample_size = None
        else:
            if sample_size >= 0:
                payload["preview_sample_size"] = sample_size

    return payload


def _post_json(url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None
    if response.status_code >= 400:
        try:
            detail = response.json()
        except json.JSONDecodeError:
            detail = response.text
        st.error(f"Error {response.status_code}: {detail}")
        return None
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON response: {exc}")
        return None


def _fetch_provenance(url: str) -> Dict[str, Dict[str, float]]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
    except requests.RequestException as exc:
        st.warning(f"Failed to load provenance: {exc}")
    except json.JSONDecodeError:
        st.warning("Provenance endpoint returned invalid JSON")
    return {}


def update_provenance_charts(counts: Dict[str, int]) -> None:
    filtered = [(str(kind), int(value)) for kind, value in counts.items() if int(value) > 0]
    filtered.sort(key=lambda item: item[1], reverse=True)
    st.session_state.provenance_freq_items = filtered[:TOP_K_PROVENANCE]

    history = st.session_state.get("prov_history") or []
    recency_scores: Dict[str, int] = {}
    if history:
        total = len(history)
        for offset, kind in enumerate(reversed(history)):
            score = total - offset
            existing = recency_scores.get(kind, 0)
            if score > existing:
                recency_scores[kind] = score

    recency_items = sorted(
        ((str(kind), int(score)) for kind, score in recency_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:TOP_K_PROVENANCE]
    st.session_state.provenance_recency_items = recency_items


st.sidebar.header("Configuration")
backend_url = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND_URL)

mode_labels = {"csv": "CSV", "images": "Images"}
mode_index = ["csv", "images"].index(st.session_state.dataset_kind)
selected_label = st.sidebar.radio(
    "Dataset type",
    options=[mode_labels[mode] for mode in ["csv", "images"]],
    index=mode_index,
)
selected_kind = "images" if selected_label == mode_labels["images"] else "csv"
if selected_kind != st.session_state.dataset_kind:
    st.session_state.dataset_kind = selected_kind
    st.session_state.plan_ops = []  # type: ignore[assignment]
    st.session_state.preview_result = None
    st.session_state.execute_result = None

dataset_kind = st.session_state.dataset_kind

current_dataset_path = ""
columns: List[str] = []
image_paths: List[Path] = []

if dataset_kind == "csv":
    sample_csv_path = _find_first_existing(SAMPLE_CSV_CANDIDATES)
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
    uploaded_path = _persist_uploaded_file(uploaded_file)

    use_sample = False
    if sample_csv_path and st.sidebar.checkbox(
        "Use bundled sample CSV", value=not bool(uploaded_path)
    ):
        use_sample = True
        st.sidebar.caption(f"Using sample at {sample_csv_path}")

    sample_plan_data = _load_sample_plan()
    if st.sidebar.button("Load sample plan", disabled=sample_plan_data is None):
        if sample_plan_data:
            st.session_state.plan_ops = sample_plan_data.get("ops", [])  # type: ignore[assignment]
            dataset = sample_plan_data.get("dataset") or {}
            sample_size = sample_plan_data.get("limit", dataset.get("sample_size"))
            if sample_size is not None:
                try:
                    st.session_state.sample_size = int(sample_size)
                except (TypeError, ValueError):
                    pass
            dataset_path = dataset.get("path")
            if dataset_path:
                st.session_state.dataset_path = str(dataset_path)
            st.sidebar.success("Sample plan loaded")
        else:
            st.sidebar.info("No sample plan found")

    if uploaded_path:
        current_dataset_path = str(uploaded_path)
    elif use_sample and sample_csv_path:
        current_dataset_path = str(sample_csv_path)
    elif st.session_state.dataset_path:
        current_dataset_path = st.session_state.dataset_path

    st.session_state.dataset_path = current_dataset_path
    if current_dataset_path:
        columns = _read_columns(current_dataset_path)
else:
    st.session_state.dataset_path = ""
    uploaded_images = st.sidebar.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    saved_images = _persist_uploaded_images(uploaded_images)
    if saved_images:
        images_dir = Path(st.session_state.images_dir)
        for path in saved_images:
            try:
                rel_path = path.relative_to(images_dir)
            except ValueError:
                rel_path = Path(path.name)
            rel_str = rel_path.as_posix()
            if rel_str not in st.session_state.selected_images:
                st.session_state.selected_images.append(rel_str)

    bundled_images = _list_bundled_images()
    bundled_available = bool(bundled_images)
    if st.session_state.use_bundled_images and not bundled_available:
        _disable_bundled_images()
        bundled_available = False

    checkbox_value = st.sidebar.checkbox(
        "Use bundled sample images",
        value=st.session_state.use_bundled_images if bundled_available else False,
        disabled=not bundled_available,
    )

    if checkbox_value != st.session_state.use_bundled_images:
        if checkbox_value:
            if _enable_bundled_images(remember_previous=not st.session_state.use_bundled_images):
                st.sidebar.success("Bundled sample images loaded.")
            else:
                st.sidebar.warning("Bundled sample images are unavailable.")
                st.session_state.use_bundled_images = False
        else:
            _disable_bundled_images()

    if bundled_available:
        st.sidebar.caption(f"{len(bundled_images)} bundled image(s) in {BUNDLED_IMAGE_DIR}")
    else:
        st.sidebar.caption(f"No bundled images found in {BUNDLED_IMAGE_DIR}")

    if st.sidebar.button("Load sample plan (images)", disabled=not bundled_available):
        activated = _enable_bundled_images(
            remember_previous=not st.session_state.use_bundled_images
        )
        if not activated:
            st.sidebar.warning("Bundled sample images are unavailable.")
        else:
            plan_copy = deepcopy(IMAGE_SAMPLE_PLAN)
            st.session_state.plan_ops = plan_copy.get("ops", [])  # type: ignore[assignment]
            st.session_state.images_dir = str(BUNDLED_IMAGE_DIR.resolve())
            st.session_state.selected_images = [path.name for path in bundled_images]
            st.session_state.sample_size = min(len(bundled_images), 3) or 1
            st.session_state.preview_result = None
            st.session_state.execute_result = None
            st.session_state.use_bundled_images = True
            st.sidebar.success("Sample image plan loaded.")

    image_paths = _resolve_selected_image_paths()
    images_dir_display = st.session_state.images_dir
    if images_dir_display:
        st.sidebar.caption(
            f"Images stored in {images_dir_display} ({len(image_paths)} file(s) saved)"
        )

    if image_paths:
        st.sidebar.markdown("#### Saved images")
        base_dir = Path(st.session_state.images_dir)
        for rel_path in list(st.session_state.selected_images):
            path = base_dir / rel_path
            if not path.exists():
                continue
            cols = st.sidebar.columns([4, 1])
            cols[0].write(f"{path.name} ({_format_size(path.stat().st_size)})")
            if cols[1].button("Remove", key=f"remove_{rel_path}"):
                st.session_state.selected_images = [
                    item for item in st.session_state.selected_images if item != rel_path
                ]
                st.rerun()
    else:
        st.sidebar.info("Upload PNG or JPG files to begin.")

dataset_payload: Optional[Dict[str, Any]] = None

main_col, provenance_col = st.columns([3, 1.2])

with main_col:
    st.subheader("Plan builder")

    if dataset_kind == "csv":
        if current_dataset_path:
            st.info(f"Dataset: {current_dataset_path}")
        else:
            st.warning("Upload a CSV file or enable the bundled sample to build a plan.")
    else:
        image_count = len(image_paths)
        if image_count:
            images_dir_display = st.session_state.images_dir
            location_note = f" from {images_dir_display}" if images_dir_display else ""
            st.info(f"{image_count} image(s) ready for processing{location_note}.")
            st.markdown("#### Uploaded images")
            _render_image_gallery(image_paths)
        else:
            st.warning("Upload one or more images to build a plan.")

    if dataset_kind == "csv":
        csv_count = 0
        if current_dataset_path:
            max_preview_rows = 50
            try:
                row_count = (
                    sum(
                        1
                        for _ in Path(current_dataset_path).open(
                            "r", encoding="utf-8", errors="ignore"
                        )
                    )
                    - 1
                )
            except Exception:
                row_count = 0
            if row_count > 0:
                csv_count = min(max_preview_rows, row_count)
        st.session_state.sample_size = sample_size_control("Preview sample size", csv_count)
    else:
        img_count = len(image_paths)
        default_value = min(5, max(1, img_count))
        st.session_state.sample_size = sample_size_control(
            "Preview sample size",
            img_count,
            default=default_value,
            allow_single_slider=True,
        )

    if dataset_kind == "csv" and current_dataset_path:
        dataset_payload = build_dataset_payload_csv(current_dataset_path)
    elif dataset_kind == "images" and image_paths:
        images_dir_str = st.session_state.images_dir
        images_dir_path = Path(images_dir_str) if images_dir_str else image_paths[0].parent
        dataset_payload = build_dataset_payload_images(
            images_dir_path.resolve(), list(st.session_state.selected_images)
        )

    st.markdown("### Operations")
    available_kinds = (
        ["summarize", "classify"] if dataset_kind == "csv" else ["img_caption", "img_resize"]
    )
    for idx, op_data in enumerate(st.session_state.plan_ops):
        op_kind = op_data.get("kind") or available_kinds[0]
        if op_kind not in available_kinds:
            op_kind = available_kinds[0]
            op_data["kind"] = op_kind
            op_data["params"] = _default_params_for_kind(
                op_kind, columns if dataset_kind == "csv" else []
            )
        expander_label = f"Operation {idx + 1}: {op_kind}"
        with st.expander(expander_label, expanded=True):
            kind_key = f"{dataset_kind}_op_kind_{idx}"
            selected_kind = st.selectbox(
                "Kind",
                options=available_kinds,
                index=available_kinds.index(op_kind),
                key=kind_key,
            )
            if selected_kind != op_kind:
                op_data["kind"] = selected_kind
                op_data["params"] = _default_params_for_kind(
                    selected_kind, columns if dataset_kind == "csv" else []
                )
            op_kind = op_data["kind"]
            params = _ensure_operation_params(
                op_data, op_kind, columns if dataset_kind == "csv" else None
            )

            if dataset_kind == "csv":
                field_key = f"{dataset_kind}_op_field_{idx}"
                if columns:
                    default_field = params.get("field")
                    if default_field not in columns:
                        default_field = columns[0]
                    params["field"] = st.selectbox(
                        "Field",
                        options=columns,
                        index=columns.index(default_field) if default_field in columns else 0,
                        key=field_key,
                    )
                else:
                    params["field"] = st.text_input(
                        "Field",
                        key=field_key,
                        value=params.get("field", ""),
                    )

                if op_kind == "summarize":
                    instructions_key = f"{dataset_kind}_instructions_{idx}"
                    params["instructions"] = st.text_area(
                        "Instructions",
                        key=instructions_key,
                        value=params.get("instructions", ""),
                        placeholder="Summarize the following text...",
                    )

                    max_tokens_key = f"{dataset_kind}_max_tokens_{idx}"
                    params["max_tokens"] = int(
                        st.number_input(
                            "Max tokens",
                            min_value=16,
                            max_value=1024,
                            value=int(params.get("max_tokens", 128) or 128),
                            step=16,
                            key=max_tokens_key,
                        )
                    )

                elif op_kind == "classify":
                    labels_key = f"{dataset_kind}_labels_{idx}"
                    label_options = sorted(
                        set(DEFAULT_LABEL_OPTIONS) | set(params.get("labels", []))
                    )
                    selected_labels = st.multiselect(
                        "Labels",
                        options=label_options,
                        default=params.get("labels", []),
                        key=labels_key,
                    )
                    params["labels"] = [str(label) for label in selected_labels]
            else:
                if op_kind == "img_caption":
                    instructions_key = f"img_caption_instructions_{idx}"
                    params["instructions"] = st.text_area(
                        "Instructions",
                        value=params.get("instructions", ""),
                        key=instructions_key,
                        placeholder="Describe the image...",
                    )
                    max_tokens_key = f"img_caption_max_tokens_{idx}"
                    params["max_tokens"] = int(
                        st.number_input(
                            "Max tokens",
                            min_value=1,
                            max_value=1024,
                            value=int(params.get("max_tokens", 80) or 80),
                            step=1,
                            key=max_tokens_key,
                        )
                    )
                elif op_kind == "img_resize":
                    width_key = f"img_width_{idx}"
                    height_key = f"img_height_{idx}"
                    keep_aspect_key = f"img_keep_aspect_{idx}"
                    width_value = int(
                        st.number_input(
                            "Width",
                            min_value=1,
                            max_value=8192,
                            value=int(params.get("width", 512) or 512),
                            step=1,
                            key=width_key,
                        )
                    )
                    params["width"] = width_value
                    keep_aspect_value = st.checkbox(
                        "Keep aspect ratio",
                        value=bool(params.get("keep_aspect", True)),
                        key=keep_aspect_key,
                    )
                    params["keep_aspect"] = bool(keep_aspect_value)
                    existing_height = params.get("height", width_value)
                    try:
                        existing_height_int = int(existing_height)
                    except (TypeError, ValueError):
                        existing_height_int = width_value
                    if existing_height_int <= 0:
                        existing_height_int = width_value
                    height_value = st.number_input(
                        "Height",
                        min_value=1,
                        max_value=8192,
                        value=existing_height_int,
                        step=1,
                        key=height_key,
                        disabled=keep_aspect_value,
                    )
                    if keep_aspect_value:
                        params["height"] = width_value
                    else:
                        params["height"] = int(height_value)

            remove_key = f"remove_{dataset_kind}_{idx}"
            if st.button("Remove", key=remove_key):
                st.session_state.plan_ops.pop(idx)
                st.rerun()

    add_col1, add_col2 = st.columns([1, 3])
    add_kind_key = f"add_kind_{dataset_kind}"
    with add_col2:
        add_kind = st.selectbox(
            "Operation",
            options=available_kinds,
            index=0,
            key=add_kind_key,
            label_visibility="collapsed",
        )
    if add_col1.button("Add operation"):
        new_params = _default_params_for_kind(add_kind, columns if dataset_kind == "csv" else [])
        st.session_state.plan_ops.append({"kind": add_kind, "params": new_params})
        st.rerun()

    if dataset_payload is not None:
        debug_plan_payload = _prepare_plan_payload(
            st.session_state.plan_ops,
            dataset_payload,
            preview_sample_size=st.session_state.sample_size,
        )
        with st.expander("Debug: request payload", expanded=False):
            st.json(debug_plan_payload)

    action_cols = st.columns(2)

    preview_disabled = False
    execute_disabled = False
    action_hint: Optional[str] = None

    if dataset_kind == "images" and not image_paths:
        preview_disabled = True
        execute_disabled = True
        action_hint = "Upload or select at least one image to enable preview and execution."

    preview_clicked = action_cols[0].button(
        "Preview", use_container_width=True, disabled=preview_disabled
    )
    execute_clicked = action_cols[1].button(
        "Execute", use_container_width=True, disabled=execute_disabled
    )

    if action_hint:
        st.caption(action_hint)

    preview_error = (
        "Select a dataset before previewing."
        if dataset_kind == "csv"
        else "Upload at least one image before previewing."
    )
    execute_error = (
        "Select a dataset before executing."
        if dataset_kind == "csv"
        else "Upload at least one image before executing."
    )

    if preview_clicked:
        if dataset_payload is None:
            st.error(preview_error)
        else:
            req_body = _prepare_plan_payload(
                st.session_state.plan_ops,
                dataset_payload,
                preview_sample_size=st.session_state.sample_size,
            )
            url = f"{backend_url.rstrip('/')}/preview"
            result = _post_json(url, req_body)
            if result is not None:
                st.session_state.preview_result = result
                st.session_state.execute_result = None
                ops = req_body.get("operations", [])
                if "prov_counts" not in st.session_state:
                    st.session_state.prov_counts = {}
                if "prov_history" not in st.session_state:
                    st.session_state.prov_history = []
                for op in ops:
                    kind = op.get("kind")
                    if kind:
                        st.session_state.prov_counts[kind] = (
                            st.session_state.prov_counts.get(kind, 0) + 1
                        )
                        st.session_state.prov_history.append(kind)
                update_provenance_charts(st.session_state.prov_counts)

    if execute_clicked:
        if dataset_payload is None:
            st.error(execute_error)
        else:
            req_body = _prepare_plan_payload(
                st.session_state.plan_ops,
                dataset_payload,
                preview_sample_size=st.session_state.sample_size,
            )
            url = f"{backend_url.rstrip('/')}/execute"
            result = _post_json(url, req_body)
            if result is not None:
                st.session_state.execute_result = result
                ops = req_body.get("operations", [])
                if "prov_counts" not in st.session_state:
                    st.session_state.prov_counts = {}
                if "prov_history" not in st.session_state:
                    st.session_state.prov_history = []
                for op in ops:
                    kind = op.get("kind")
                    if kind:
                        st.session_state.prov_counts[kind] = (
                            st.session_state.prov_counts.get(kind, 0) + 1
                        )
                        st.session_state.prov_history.append(kind)
                update_provenance_charts(st.session_state.prov_counts)

    if st.session_state.preview_result:
        st.markdown("### Preview output")
        preview = st.session_state.preview_result
        records = preview.get("records") or []
        if records:
            if dataset_kind == "images":
                _render_image_preview_table(records, st.session_state.plan_ops)
            else:
                st.dataframe(pd.DataFrame(records))
        else:
            st.write(preview)
        schema = preview.get("schema")
        if schema:
            st.json(schema)
        st.json(preview)
        preview_artifacts = preview.get("artifacts") if isinstance(preview, dict) else None
        if dataset_kind == "images":
            captions_path = None
            if isinstance(preview_artifacts, dict):
                captions_path = preview_artifacts.get("captions")
            if captions_path:
                render_artifacts_section(preview)
        else:
            render_artifacts_section(preview)

    if st.session_state.execute_result:
        st.markdown("### Execution results")
        result = st.session_state.execute_result
        artifacts = result.get("artifacts") or {}
        run_dir = _extract_run_directory(artifacts)

        if result.get("ok"):
            st.success("Execution finished")
            if run_dir:
                run_cols = st.columns([5, 1])
                run_dir_str = str(run_dir)
                run_cols[0].code(run_dir_str, language="bash")
                if run_cols[1].button("Copy path", key="copy_run_dir_button"):
                    st.session_state["copied_run_dir"] = run_dir_str
                    st.toast(f"Copied run directory path: {run_dir_str}")
                if run_dir.exists():
                    try:
                        run_cols[0].markdown(f"[Open folder]({run_dir.resolve().as_uri()})")
                    except ValueError:
                        pass
        elif run_dir:
            st.info(f"Artifacts saved under {run_dir}")

        st.json(result)
        render_artifacts_section(result)

with provenance_col:
    st.subheader("Provenance")
    freq_items = st.session_state.get("provenance_freq_items") or []
    recency_items = st.session_state.get("provenance_recency_items") or []
    if freq_items or recency_items:
        if freq_items:
            freq_df = pd.DataFrame(freq_items, columns=["operation", "count"]).set_index(
                "operation"
            )
            st.caption("Most frequent operations")
            st.bar_chart(freq_df)
        if recency_items:
            rec_df = pd.DataFrame(recency_items, columns=["operation", "recency_score"]).set_index(
                "operation"
            )
            st.caption("Most recent operations")
            st.bar_chart(rec_df)
    else:
        provenance_url = f"{backend_url.rstrip('/')}/provenance/snapshot"
        provenance_data = _fetch_provenance(provenance_url)
        if not provenance_data:
            st.info("No provenance data available yet.")
        else:
            frequency = provenance_data.get("frequency") or {}
            recency = provenance_data.get("recency") or {}
            if frequency:
                freq_items = sorted(frequency.items(), key=lambda item: item[1], reverse=True)[
                    :TOP_K_PROVENANCE
                ]
                freq_df = pd.DataFrame(
                    freq_items, columns=["operation", "normalized_frequency"]
                ).set_index("operation")
                st.caption("Most frequent operations")
                st.bar_chart(freq_df)
            if recency:
                rec_items = sorted(recency.items(), key=lambda item: item[1], reverse=True)[
                    :TOP_K_PROVENANCE
                ]
                rec_df = pd.DataFrame(rec_items, columns=["operation", "recency_score"]).set_index(
                    "operation"
                )
                st.caption("Most recent operations")
                st.bar_chart(rec_df)
