"""Streamlit UI for interacting with the VDPT FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_LABEL_OPTIONS = ["positive", "negative", "neutral"]
TOP_K_PROVENANCE = 5
UPLOAD_DIR = Path("artifacts") / "uploads"

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


def _read_columns(path: str) -> List[str]:
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return []
    return [str(col) for col in df.columns]


def _prepare_plan_payload(
    ops: List[Dict[str, Any]], dataset_path: str, sample_size: int
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
    plan: Dict[str, Any] = {"ops": plan_ops}
    if dataset_path:
        plan["dataset"] = {
            "type": "csv",
            "path": dataset_path,
            "sample_size": sample_size,
        }
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

sample_csv_path = _find_first_existing(SAMPLE_CSV_CANDIDATES)
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
uploaded_path = _persist_uploaded_file(uploaded_file)

use_sample = False
if sample_csv_path and st.sidebar.checkbox("Use bundled sample CSV", value=not bool(uploaded_path)):
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


current_dataset_path = ""
if uploaded_path:
    current_dataset_path = str(uploaded_path)
elif use_sample and sample_csv_path:
    current_dataset_path = str(sample_csv_path)
elif st.session_state.dataset_path:
    current_dataset_path = st.session_state.dataset_path

st.session_state.dataset_path = current_dataset_path

columns: List[str] = []
if current_dataset_path:
    columns = _read_columns(current_dataset_path)

main_col, provenance_col = st.columns([3, 1.2])

with main_col:
    st.subheader("Plan builder")

    if current_dataset_path:
        st.info(f"Dataset: {current_dataset_path}")
    else:
        st.warning("Upload a CSV file or enable the bundled sample to build a plan.")

    max_preview_rows = 50
    if current_dataset_path:
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
    st.session_state.sample_size = st.slider(
        "Preview sample size",
        min_value=1,
        max_value=max_preview_rows if max_preview_rows >= 1 else 1,
        value=min(st.session_state.sample_size, max_preview_rows) if max_preview_rows >= 1 else 1,
        step=1,
    )

    st.markdown("### Operations")
    for idx, op in enumerate(st.session_state.plan_ops):
        expander_label = f"Operation {idx + 1}: {op.get('kind', 'summarize')}"
        with st.expander(expander_label, expanded=True):
            kind_key = f"op_kind_{idx}"
            st.session_state.plan_ops[idx]["kind"] = st.selectbox(
                "Kind",
                options=["summarize", "classify"],
                index=["summarize", "classify"].index(op.get("kind", "summarize")),
                key=kind_key,
            )
            op_kind = st.session_state.plan_ops[idx]["kind"]

            field_key = f"op_field_{idx}"
            if columns:
                default_field = op.get("field") if op.get("field") in columns else columns[0]
                st.session_state.plan_ops[idx]["field"] = st.selectbox(
                    "Field",
                    options=columns,
                    index=columns.index(default_field) if default_field in columns else 0,
                    key=field_key,
                )
            else:
                if field_key not in st.session_state:
                    st.session_state[field_key] = op.get("field", "")
                st.session_state.plan_ops[idx]["field"] = st.text_input("Field", key=field_key)

            if op_kind == "summarize":
                instructions_key = f"instructions_{idx}"
                if instructions_key not in st.session_state:
                    st.session_state[instructions_key] = op.get("instructions", "")
                st.session_state.plan_ops[idx]["instructions"] = st.text_area(
                    "Instructions",
                    key=instructions_key,
                    placeholder="Summarize the following text...",
                )

                max_tokens_key = f"max_tokens_{idx}"
                if max_tokens_key not in st.session_state:
                    st.session_state[max_tokens_key] = int(op.get("max_tokens", 128) or 128)
                st.session_state.plan_ops[idx]["max_tokens"] = st.number_input(
                    "Max tokens",
                    min_value=16,
                    max_value=1024,
                    value=int(st.session_state[max_tokens_key]),
                    step=16,
                    key=max_tokens_key,
                )

            elif op_kind == "classify":
                labels_key = f"labels_{idx}"
                label_options = sorted(set(DEFAULT_LABEL_OPTIONS) | set(op.get("labels", [])))
                st.session_state.plan_ops[idx]["labels"] = st.multiselect(
                    "Labels",
                    options=label_options,
                    default=op.get("labels", []),
                    key=labels_key,
                )

            remove_key = f"remove_{idx}"
            if st.button("Remove", key=remove_key):
                st.session_state.plan_ops.pop(idx)
                st.experimental_rerun()

    add_col1, add_col2 = st.columns([1, 3])
    if add_col1.button("Add operation"):
        st.session_state.plan_ops.append(
            {
                "kind": "summarize",
                "field": columns[0] if columns else "",
                "instructions": "",
                "max_tokens": 128,
                "labels": list(DEFAULT_LABEL_OPTIONS),
            }
        )
        st.experimental_rerun()

    action_cols = st.columns(2)
    preview_clicked = action_cols[0].button("Preview", use_container_width=True)
    execute_clicked = action_cols[1].button("Execute", use_container_width=True)

    plan_payload = _prepare_plan_payload(
        st.session_state.plan_ops,
        st.session_state.dataset_path,
        st.session_state.sample_size,
    )

    if preview_clicked:
        if not st.session_state.dataset_path:
            st.error("Select a dataset before previewing.")
        else:
            url = f"{backend_url.rstrip('/')}/preview"
            result = _post_json(url, plan_payload)
            if result is not None:
                st.session_state.preview_result = result
                st.session_state.execute_result = None

    if execute_clicked:
        if not st.session_state.dataset_path:
            st.error("Select a dataset before executing.")
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
            st.dataframe(pd.DataFrame(records))
        else:
            st.write(preview)
        schema = preview.get("schema")
        if schema:
            st.json(schema)

    if st.session_state.execute_result:
        st.markdown("### Execution results")
        result = st.session_state.execute_result
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
