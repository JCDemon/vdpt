"""Streamlit UI for interacting with the VDPT FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_LABEL_OPTIONS = ["positive", "negative", "neutral"]
TOP_K_PROVENANCE = 5
UPLOAD_DIR = Path("artifacts") / "uploads"
IMAGE_UPLOAD_SUBDIR = "images"

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
if "dataset_mode" not in st.session_state:
    st.session_state.dataset_mode = "csv"
if "image_session_id" not in st.session_state:
    st.session_state.image_session_id = uuid4().hex
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for candidate in paths:
        if candidate and candidate.exists():
            return candidate
    return None


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
                    "field": params.get("field", ""),
                    "instructions": params.get("instructions", ""),
                    "max_tokens": int(params.get("max_tokens", 128) or 128),
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
                    "field": params.get("field", ""),
                    "labels": [str(label) for label in labels],
                }
            )
    dataset = payload.get("dataset") or {}
    return {"ops": ops, "dataset": dataset}


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


def _get_image_session_dir(session_id: str) -> Path:
    return UPLOAD_DIR / session_id / IMAGE_UPLOAD_SUBDIR


def _persist_uploaded_images(uploads) -> List[Path]:
    saved: List[Path] = []
    if not uploads:
        return saved

    session_id = st.session_state.image_session_id
    session_dir = _get_image_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    for upload in uploads:
        if upload is None:
            continue
        filename = Path(upload.name).name
        target = session_dir / filename
        try:
            with target.open("wb") as fp:
                fp.write(upload.getbuffer())
        except OSError as exc:
            st.error(f"Failed to save uploaded image '{filename}': {exc}")
            continue
        saved.append(target)

    return saved


def _list_image_paths(session_id: str) -> List[Path]:
    session_dir = _get_image_session_dir(session_id)
    if not session_dir.exists():
        return []
    return sorted(p for p in session_dir.iterdir() if p.is_file())


def _build_image_preview_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        image_path = Path(str(record.get("image_path", "")))
        filename = image_path.name if image_path.name else str(image_path)
        row: Dict[str, Any] = {"filename": filename}
        for key, value in record.items():
            if key == "image_path":
                continue
            row[key] = value
        rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["filename"])


def _render_image_gallery(paths: List[Path], columns: int = 3) -> None:
    if not paths:
        return

    columns = max(1, min(columns, len(paths)))
    for start in range(0, len(paths), columns):
        row_paths = paths[start : start + columns]
        grid = st.columns(len(row_paths))
        for col, path in zip(grid, row_paths):
            with col:
                st.image(str(path), caption=path.name, use_column_width=True)


def _read_columns(path: str) -> List[str]:
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return []
    return [str(col) for col in df.columns]


def _prepare_plan_payload(
    ops: List[Dict[str, Any]], dataset: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    plan_ops: List[Dict[str, Any]] = []
    for op in ops:
        kind = op.get("kind")
        if kind == "summarize":
            plan_ops.append(
                {
                    "kind": "summarize",
                    "params": {},
                    "summarize": {
                        "field": op.get("field"),
                        "instructions": op.get("instructions", ""),
                        "max_tokens": int(op.get("max_tokens") or 128),
                    },
                }
            )
        elif kind == "classify":
            plan_ops.append(
                {
                    "kind": "classify",
                    "params": {},
                    "classify": {
                        "field": op.get("field"),
                        "labels": [str(label) for label in op.get("labels", [])],
                    },
                }
            )
        elif kind == "img_caption":
            params: Dict[str, Any] = {}
            prompt = op.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                params["prompt"] = prompt
            plan_ops.append({"kind": "img_caption", "params": params})
        elif kind == "img_resize":
            params = {}
            try:
                params["width"] = int(op.get("width"))
            except (TypeError, ValueError):
                pass
            try:
                params["height"] = int(op.get("height"))
            except (TypeError, ValueError):
                pass
            if "keep_aspect" in op:
                params["keep_aspect"] = bool(op.get("keep_aspect"))
            plan_ops.append({"kind": "img_resize", "params": params})
    plan: Dict[str, Any] = {"ops": plan_ops}
    if dataset:
        plan["dataset"] = dataset
    return plan


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


st.sidebar.header("Configuration")
backend_url = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND_URL)

mode_labels = {"csv": "CSV", "images": "Images"}
mode_index = ["csv", "images"].index(st.session_state.dataset_mode)
selected_label = st.sidebar.radio(
    "Dataset type",
    options=[mode_labels[mode] for mode in ["csv", "images"]],
    index=mode_index,
)
selected_mode = "images" if selected_label == mode_labels["images"] else "csv"
if selected_mode != st.session_state.dataset_mode:
    st.session_state.dataset_mode = selected_mode
    st.session_state.plan_ops = []  # type: ignore[assignment]
    st.session_state.preview_result = None
    st.session_state.execute_result = None

dataset_mode = st.session_state.dataset_mode

current_dataset_path = ""
columns: List[str] = []
image_paths: List[Path] = []

if dataset_mode == "csv":
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
            sample_size = dataset.get("sample_size")
            if sample_size is not None:
                try:
                    st.session_state.sample_size = int(sample_size)
                except (TypeError, ValueError):
                    pass
            if dataset.get("path"):
                st.session_state.dataset_path = str(dataset["path"])
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
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    _persist_uploaded_images(uploaded_images)
    image_paths = _list_image_paths(st.session_state.image_session_id)
    st.session_state.image_paths = [str(path) for path in image_paths]
    st.sidebar.caption(
        f"Session: {st.session_state.image_session_id} — {len(image_paths)} image(s) uploaded"
    )

dataset_payload: Optional[Dict[str, Any]] = None

main_col, provenance_col = st.columns([3, 1.2])

with main_col:
    st.subheader("Plan builder")

    if dataset_mode == "csv":
        if current_dataset_path:
            st.info(f"Dataset: {current_dataset_path}")
        else:
            st.warning("Upload a CSV file or enable the bundled sample to build a plan.")
    else:
        session_id = st.session_state.image_session_id
        image_count = len(image_paths)
        if image_count:
            st.info(f"Session {session_id}: {image_count} image(s) ready for processing.")
            st.markdown("#### Uploaded images")
            _render_image_gallery(image_paths)
        else:
            st.warning("Upload one or more images to build a plan.")

    max_preview_rows = 50
    if dataset_mode == "csv" and current_dataset_path:
        try:
            row_count = (
                sum(
                    1
                    for _ in Path(current_dataset_path).open("r", encoding="utf-8", errors="ignore")
                )
                - 1
            )
            if row_count > 0:
                max_preview_rows = min(max_preview_rows, row_count)
        except Exception:
            pass
    elif dataset_mode == "images":
        max_preview_rows = len(image_paths) if image_paths else 1

    slider_max = max_preview_rows if max_preview_rows >= 1 else 1
    slider_value = min(st.session_state.sample_size, slider_max)
    st.session_state.sample_size = st.slider(
        "Preview sample size",
        min_value=1,
        max_value=slider_max,
        value=slider_value if slider_value >= 1 else 1,
        step=1,
    )

    if dataset_mode == "csv" and current_dataset_path:
        dataset_payload = {
            "type": "csv",
            "path": current_dataset_path,
            "sample_size": st.session_state.sample_size,
        }
    elif dataset_mode == "images" and image_paths:
        session_id = st.session_state.image_session_id
        base_dir = UPLOAD_DIR / session_id
        relative_paths: List[str] = []
        for path in image_paths:
            try:
                relative_paths.append(str(path.relative_to(base_dir)))
            except ValueError:
                relative_paths.append(str(path))
        dataset_payload = {
            "type": "images",
            "session": session_id,
            "paths": relative_paths,
            "sample_size": st.session_state.sample_size,
        }

    st.markdown("### Operations")
    available_kinds = (
        ["summarize", "classify"] if dataset_mode == "csv" else ["img_caption", "img_resize"]
    )
    for idx, op in enumerate(st.session_state.plan_ops):
        op_kind = op.get("kind") or available_kinds[0]
        if op_kind not in available_kinds:
            op_kind = available_kinds[0]
            st.session_state.plan_ops[idx]["kind"] = op_kind
        expander_label = f"Operation {idx + 1}: {op_kind}"
        with st.expander(expander_label, expanded=True):
            kind_key = f"{dataset_mode}_op_kind_{idx}"
            st.session_state.plan_ops[idx]["kind"] = st.selectbox(
                "Kind",
                options=available_kinds,
                index=available_kinds.index(op_kind),
                key=kind_key,
            )
            op_kind = st.session_state.plan_ops[idx]["kind"]

            if dataset_mode == "csv":
                field_key = f"{dataset_mode}_op_field_{idx}"
                if columns:
                    default_field = op.get("field") if op.get("field") in columns else columns[0]
                    st.session_state.plan_ops[idx]["field"] = st.selectbox(
                        "Field",
                        options=columns,
                        index=columns.index(default_field) if default_field in columns else 0,
                        key=field_key,
                    )
                else:
                    st.session_state.plan_ops[idx]["field"] = st.text_input(
                        "Field",
                        key=field_key,
                        value=op.get("field", ""),
                    )

                if op_kind == "summarize":
                    instructions_key = f"{dataset_mode}_instructions_{idx}"
                    st.session_state.plan_ops[idx]["instructions"] = st.text_area(
                        "Instructions",
                        key=instructions_key,
                        value=op.get("instructions", ""),
                        placeholder="Summarize the following text...",
                    )

                    max_tokens_key = f"{dataset_mode}_max_tokens_{idx}"
                    st.session_state.plan_ops[idx]["max_tokens"] = int(
                        st.number_input(
                            "Max tokens",
                            min_value=16,
                            max_value=1024,
                            value=int(op.get("max_tokens", 128) or 128),
                            step=16,
                            key=max_tokens_key,
                        )
                    )

                elif op_kind == "classify":
                    labels_key = f"{dataset_mode}_labels_{idx}"
                    label_options = sorted(set(DEFAULT_LABEL_OPTIONS) | set(op.get("labels", [])))
                    st.session_state.plan_ops[idx]["labels"] = st.multiselect(
                        "Labels",
                        options=label_options,
                        default=op.get("labels", []),
                        key=labels_key,
                    )
            else:
                if op_kind == "img_caption":
                    prompt_key = f"img_prompt_{idx}"
                    prompt = st.text_area(
                        "Prompt",
                        value=op.get("prompt", ""),
                        key=prompt_key,
                        placeholder="Describe the image...",
                    )
                    st.session_state.plan_ops[idx]["prompt"] = prompt
                elif op_kind == "img_resize":
                    width_key = f"img_width_{idx}"
                    height_key = f"img_height_{idx}"
                    keep_aspect_key = f"img_keep_aspect_{idx}"
                    width_value = st.number_input(
                        "Width",
                        min_value=1,
                        max_value=8192,
                        value=int(op.get("width", 512) or 512),
                        step=1,
                        key=width_key,
                    )
                    height_value = st.number_input(
                        "Height",
                        min_value=1,
                        max_value=8192,
                        value=int(op.get("height", 512) or 512),
                        step=1,
                        key=height_key,
                    )
                    keep_aspect_value = st.checkbox(
                        "Keep aspect ratio",
                        value=bool(op.get("keep_aspect", False)),
                        key=keep_aspect_key,
                    )
                    st.session_state.plan_ops[idx]["width"] = int(width_value)
                    st.session_state.plan_ops[idx]["height"] = int(height_value)
                    st.session_state.plan_ops[idx]["keep_aspect"] = bool(keep_aspect_value)

            remove_key = f"remove_{dataset_mode}_{idx}"
            if st.button("Remove", key=remove_key):
                st.session_state.plan_ops.pop(idx)
                st.experimental_rerun()

    add_col1, add_col2 = st.columns([1, 3])
    if add_col1.button("Add operation"):
        if dataset_mode == "csv":
            st.session_state.plan_ops.append(
                {
                    "kind": "summarize",
                    "field": columns[0] if columns else "",
                    "instructions": "",
                    "max_tokens": 128,
                    "labels": list(DEFAULT_LABEL_OPTIONS),
                }
            )
        else:
            st.session_state.plan_ops.append({"kind": "img_caption", "prompt": ""})
        st.experimental_rerun()

    action_cols = st.columns(2)
    preview_clicked = action_cols[0].button("Preview", use_container_width=True)
    execute_clicked = action_cols[1].button("Execute", use_container_width=True)

    plan_payload = _prepare_plan_payload(
        st.session_state.plan_ops,
        dataset_payload,
    )

    preview_error = (
        "Select a dataset before previewing."
        if dataset_mode == "csv"
        else "Upload at least one image before previewing."
    )
    execute_error = (
        "Select a dataset before executing."
        if dataset_mode == "csv"
        else "Upload at least one image before executing."
    )

    if preview_clicked:
        if dataset_payload is None:
            st.error(preview_error)
        else:
            url = f"{backend_url.rstrip('/')}/preview"
            result = _post_json(url, plan_payload)
            if result is not None:
                st.session_state.preview_result = result
                st.session_state.execute_result = None

    if execute_clicked:
        if dataset_payload is None:
            st.error(execute_error)
        else:
            url = f"{backend_url.rstrip('/')}/execute"
            result = _post_json(url, plan_payload)
            if result is not None:
                st.session_state.execute_result = result

    if st.session_state.preview_result:
        st.markdown("### Preview output")
        preview = st.session_state.preview_result
        records = preview.get("records") or []
        if records:
            if dataset_mode == "images":
                st.dataframe(_build_image_preview_table(records))
            else:
                st.dataframe(pd.DataFrame(records))
        else:
            st.write(preview)
        schema = preview.get("schema")
        if schema:
            st.json(schema)

    if st.session_state.execute_result:
        st.markdown("### Execution results")
        result = st.session_state.execute_result
        artifacts = result.get("artifacts") or {}
        if artifacts:
            st.markdown("#### Artifacts")
            for name, value in artifacts.items():
                if isinstance(value, list):
                    for item in value:
                        st.write(f"{name}: {item}")
                else:
                    st.write(f"{name}: {value}")
            output_csv = artifacts.get("output_csv")
            if output_csv:
                try:
                    csv_path = Path(output_csv)
                    if csv_path.exists():
                        st.download_button(
                            "Download output CSV",
                            data=csv_path.read_bytes(),
                            file_name=csv_path.name,
                            mime="text/csv",
                        )
                except OSError as exc:
                    st.warning(f"Unable to load CSV for download: {exc}")
        st.json(result)

with provenance_col:
    st.subheader("Provenance")
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
