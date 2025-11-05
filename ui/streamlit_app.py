"""Streamlit UI for interacting with the VDPT FastAPI backend."""

from __future__ import annotations

# ruff: noqa: E402

# --- import path guard (keep tiny) ---
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# (Dev tip) Optional: export PYTHONPATH=$(git rev-parse --show-toplevel)
# -------------------------------------

import io
import json
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import altair as alt
import math
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from ui.sample_data import load_sample_plan as _load_sample_plan, ensure_sample_assets
from ui.assets import list_bundled_images

_REPO_ROOT = Path(__file__).resolve().parents[1]

from ui.utils.keys import (
    _unique_widget_key as _base_widget_key,
    stable_key,
    stable_node_key,
    unique_key,
)
from backend.app.run_index import list_runs
from backend.app.types import RunSummary

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_LABEL_OPTIONS = ["positive", "negative", "neutral"]
TOP_K_PROVENANCE = 5
UPLOAD_DIR = Path("artifacts") / "uploads"
IMAGE_UPLOAD_SUBDIR = "images"

BUNDLED_IMAGE_DIR = _REPO_ROOT / "artifacts" / "bundled_images"

_BUNDLED_IMAGE_PAYLOADS = {
    "sunrise.ppm": (
        "P3\n"
        "# simple sunrise palette\n"
        "4 4\n"
        "255\n"
        " 255 69 0    255 99 71   255 140 0   255 215 0\n"
        " 255 165 0   255 140 0   255 215 0   173 216 230\n"
        " 238 130 238  216 191 216  173 216 230  135 206 235\n"
        " 70 130 180   65 105 225   25 25 112   0 0 128\n"
    ),
    "forest.ppm": (
        "P3\n"
        "# simple forest-like gradient\n"
        "4 4\n"
        "255\n"
        " 34 139 34   34 139 34   60 179 113  85 107 47\n"
        " 34 139 34   46 139 87   107 142 35  60 179 113\n"
        " 0 100 0     46 139 87   34 139 34   85 107 47\n"
        " 139 69 19   160 82 45   34 139 34   34 139 34\n"
    ),
    "ocean.ppm": (
        "P3\n"
        "# simple ocean palette\n"
        "4 4\n"
        "255\n"
        " 25 25 112  65 105 225  70 130 180  32 178 170\n"
        " 0 105 148  30 144 255  70 130 180  0 191 255\n"
        " 25 25 112  0 0 139      65 105 225  123 104 238\n"
        " 0 191 255  70 130 180  65 105 225  25 25 112\n"
    ),
}

SAMPLE_CSV_CANDIDATES: List[Path] = [
    Path(__file__).resolve().with_name("sample.csv"),
    _REPO_ROOT / "data" / "sample_news.csv",
    Path("artifacts") / "uploads" / "sample_news.csv",
    Path(__file__).resolve().parents[1] / "tests" / "assets" / "sample.csv",
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


_MASK_COLOR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (239, 83, 80),
    (30, 136, 229),
    (67, 160, 71),
    (255, 167, 38),
    (0, 172, 193),
    (142, 36, 170),
    (255, 112, 67),
    (38, 198, 218),
    (156, 204, 101),
    (171, 71, 188),
)
_MASK_NOISE_COLOR: Tuple[int, int, int] = (120, 120, 120)
_MASK_ALPHA: int = 120


st.set_page_config(page_title="VDPT Preview & Execute", layout="wide")
st.title("VDPT Preview & Execution")


_TREE_SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("datasets", "Datasets"),
    ("plans", "Plans"),
    ("runs", "Runs"),
    ("artifacts", "Artifacts"),
)


def _ensure_project_tree_state() -> None:
    if "project_tree" not in st.session_state:
        st.session_state.project_tree = {
            "expanded": {section: True for section, _ in _TREE_SECTIONS},
            "selected_node": None,
        }


_ensure_project_tree_state()


if "simple_mode" not in st.session_state:
    st.session_state.simple_mode = True
if "plan_ops" not in st.session_state:
    st.session_state.plan_ops = []  # type: ignore[attr-defined]
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 5
if "use_sample_csv" not in st.session_state:
    st.session_state.use_sample_csv = True
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
if "dataset_loader_state" not in st.session_state:
    st.session_state.dataset_loader_state = {"backend": None, "loaders": [], "error": None}
if "dataset_loader_id" not in st.session_state:
    st.session_state.dataset_loader_id = ""
if "dataset_loader_configs" not in st.session_state:
    st.session_state.dataset_loader_configs = {}
if "dataset_loader_limit" not in st.session_state:
    st.session_state.dataset_loader_limit = 12
if "dataset_loader_preview" not in st.session_state:
    st.session_state.dataset_loader_preview = None
if "dataset_loader_preview_error" not in st.session_state:
    st.session_state.dataset_loader_preview_error = None
if "prov_counts" not in st.session_state:
    st.session_state.prov_counts = {}  # type: ignore[attr-defined]
if "prov_history" not in st.session_state:
    st.session_state.prov_history = []  # type: ignore[attr-defined]
if "provenance_freq_items" not in st.session_state:
    st.session_state.provenance_freq_items = []  # type: ignore[attr-defined]
if "provenance_recency_items" not in st.session_state:
    st.session_state.provenance_recency_items = []  # type: ignore[attr-defined]
if "mask_analytics_result" not in st.session_state:
    st.session_state.mask_analytics_result = None
if "mask_prompt" not in st.session_state:
    st.session_state.mask_prompt = ""
if "mask_segmentation_mode" not in st.session_state:
    st.session_state.mask_segmentation_mode = "sam"
if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None


ensure_sample_assets()


def _render_project_tree_sidebar(
    container: "st.delta_generator.DeltaGenerator" | None = None,
) -> None:
    tree_state = st.session_state.project_tree
    section_nodes: Dict[str, List[Dict[str, Any]]] = {
        section_id: _gather_section_nodes(section_id) for section_id, _ in _TREE_SECTIONS
    }
    target = container or st.sidebar
    _handle_tree_keyboard_navigation(target, tree_state, section_nodes)
    for section_id, section_label in _TREE_SECTIONS:
        _render_tree_section(
            target, tree_state, section_nodes[section_id], section_id, section_label
        )


def _render_tree_section(
    container: "st.delta_generator.DeltaGenerator",
    tree_state: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    section_id: str,
    label: str,
) -> None:
    is_expanded: bool = tree_state["expanded"].get(section_id, True)
    toggle_key = stable_key("tree-section-toggle", section_id)
    toggle_label = f"{'▼' if is_expanded else '▶'} {label}"
    if container.button(toggle_label, key=toggle_key, use_container_width=True):
        tree_state["expanded"][section_id] = not is_expanded
        st.rerun()
        return

    if not is_expanded:
        return

    section_container = container.container()
    if not nodes:
        section_container.caption("No items available yet.")
        return

    for idx, node in enumerate(nodes):
        _render_tree_node(section_container, tree_state, section_id, node, idx)


def _render_tree_node(
    container: "st.delta_generator.DeltaGenerator",
    tree_state: Dict[str, Any],
    section_id: str,
    node: Dict[str, Any],
    idx: int,
) -> None:
    node_id = node["id"]
    full_node_id = f"{section_id}:{node_id}"
    selected_node = tree_state.get("selected_node")
    is_selected = selected_node == full_node_id
    label = node.get("label", node_id)
    badge = node.get("badge")
    subtitle = node.get("subtitle")
    description = node.get("description")
    button_label = f"{'●' if is_selected else '○'} {label}"
    if badge:
        button_label = f"{button_label} · {badge}"

    button_key = stable_node_key(section_id, node, idx)
    if container.button(button_label, key=button_key, use_container_width=True):
        _activate_tree_node(tree_state, section_id, node, trigger_rerun=True)
        return

    if subtitle:
        container.caption(subtitle)
    if description:
        container.write(description)


def _handle_tree_keyboard_navigation(
    container: "st.delta_generator.DeltaGenerator",
    tree_state: Dict[str, Any],
    section_nodes: Dict[str, List[Dict[str, Any]]],
) -> None:
    visible_nodes: List[Tuple[str, Dict[str, Any]]] = []
    for section_id, _ in _TREE_SECTIONS:
        if not tree_state["expanded"].get(section_id, True):
            continue
        for node in section_nodes.get(section_id, []):
            visible_nodes.append((section_id, node))

    if "tree_nav_value" not in st.session_state:
        st.session_state.tree_nav_value = ""

    def _on_nav_change() -> None:
        raw_value = st.session_state.tree_nav_value
        st.session_state.tree_nav_value = ""
        if not raw_value:
            return
        key = raw_value.strip().lower()
        if key not in {"j", "k", "enter"}:
            return
        _process_tree_nav_key(tree_state, visible_nodes, key)

    container.text_input(
        "Tree keyboard navigation",
        key="tree_nav_value",
        label_visibility="collapsed",
        placeholder="Focus here and press j / k / Enter",
        on_change=_on_nav_change,
    )


def _process_tree_nav_key(
    tree_state: Dict[str, Any],
    visible_nodes: List[Tuple[str, Dict[str, Any]]],
    key: str,
) -> None:
    if not visible_nodes:
        return

    selected = tree_state.get("selected_node")
    order = [f"{section}:{node['id']}" for section, node in visible_nodes]
    selected_index = order.index(selected) if selected in order else None

    if key == "j":
        next_index = 0 if selected_index is None else (selected_index + 1) % len(order)
        section_id, node = visible_nodes[next_index]
        _set_selected_tree_node(tree_state, section_id, node)
        st.rerun()
        return
    if key == "k":
        prev_index = len(order) - 1 if selected_index is None else (selected_index - 1) % len(order)
        section_id, node = visible_nodes[prev_index]
        _set_selected_tree_node(tree_state, section_id, node)
        st.rerun()
        return
    if key == "enter":
        target_index = 0 if selected_index is None else selected_index
        section_id, node = visible_nodes[target_index]
        _activate_tree_node(tree_state, section_id, node, trigger_rerun=True)


def _set_selected_tree_node(
    tree_state: Dict[str, Any], section_id: str, node: Dict[str, Any]
) -> None:
    full_node_id = f"{section_id}:{node['id']}"
    tree_state["selected_node"] = full_node_id


def _activate_tree_node(
    tree_state: Dict[str, Any],
    section_id: str,
    node: Dict[str, Any],
    *,
    trigger_rerun: bool = False,
) -> None:
    _set_selected_tree_node(tree_state, section_id, node)
    tree_state["active_section"] = section_id
    tree_state["active_payload"] = node.get("payload")
    if section_id == "runs":
        st.session_state.selected_run_id = node.get("payload", {}).get("run_id")
    else:
        st.session_state.selected_run_id = None
    if trigger_rerun:
        st.rerun()


def _format_badge(parts: Iterable[Optional[str]]) -> Optional[str]:
    values = [part for part in parts if part]
    return " · ".join(values) if values else None


def _gather_section_nodes(section_id: str) -> List[Dict[str, Any]]:
    if section_id == "datasets":
        return _dataset_tree_nodes()
    if section_id == "plans":
        return _plan_tree_nodes()
    if section_id == "runs":
        return _run_tree_nodes()
    if section_id == "artifacts":
        return _artifact_tree_nodes()
    return []


def _dataset_tree_nodes() -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    dataset_path = st.session_state.get("dataset_path")
    dataset_kind = st.session_state.get("dataset_kind", "csv")
    sample_size = st.session_state.get("sample_size")
    dataset_badge = None
    if dataset_path:
        dataset_badge = _format_badge(
            [
                dataset_kind.upper() if dataset_kind else None,
                f"sample {sample_size}" if sample_size else None,
            ]
        )
    if dataset_path:
        label = Path(dataset_path).name or dataset_path
        subtitle = "Current dataset"
        nodes.append(
            {
                "id": "current",
                "label": label,
                "badge": dataset_badge,
                "subtitle": subtitle,
                "payload": {"path": dataset_path},
            }
        )
    if st.session_state.get("use_bundled_images"):
        try:
            bundled_images = list_bundled_images()
        except Exception as e:  # pragma: no cover - defensive UI guard
            bundled_images = []
            st.sidebar.info(f"Sample images not found ({e}); you can upload your own.")
        image_badge = _format_badge(
            [
                "IMAGES",
                f"{len(bundled_images)} file(s)" if bundled_images else None,
            ]
        )
        nodes.append(
            {
                "id": "bundled-images",
                "label": "Bundled Images",
                "subtitle": "Sample image dataset",
                "badge": image_badge,
                "payload": {"path": str(BUNDLED_IMAGE_DIR)},
            }
        )
    return nodes


def _normalize_sample_plan(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not plan:
        return None

    normalized: Dict[str, Any] = {
        "id": plan.get("id", "sample-plan"),
        "name": plan.get("name", "Sample Plan"),
    }

    ops: List[Dict[str, Any]] = []
    if isinstance(plan.get("ops"), list):
        for op in plan.get("ops", []):  # type: ignore[arg-type]
            if not isinstance(op, dict):
                continue
            kind = op.get("kind")
            if not kind:
                continue
            params = dict(op.get("params") or {})
            ops.append({"kind": kind, "params": params})
    else:
        for op in plan.get("operations", []):  # type: ignore[arg-type]
            if not isinstance(op, dict):
                continue
            kind = op.get("kind")
            if not kind:
                continue
            params: Dict[str, Any] = {}
            if "field" in op:
                params["field"] = op.get("field")
            if "instructions" in op:
                params["instructions"] = op.get("instructions")
            if "max_tokens" in op:
                try:
                    params["max_tokens"] = int(op.get("max_tokens"))
                except (TypeError, ValueError):
                    pass
            for key, value in op.items():
                if key in {"kind", "field", "instructions", "max_tokens"}:
                    continue
                params[key] = value
            ops.append({"kind": kind, "params": params})

    normalized["ops"] = ops
    dataset = plan.get("dataset")
    if isinstance(dataset, dict):
        normalized["dataset"] = dataset
    else:
        normalized["dataset"] = {}
    if "limit" in plan:
        normalized["limit"] = plan.get("limit")

    return normalized


def _plan_tree_nodes() -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    plan_ops = st.session_state.get("plan_ops") or []
    if plan_ops:
        nodes.append(
            {
                "id": "active-plan",
                "label": "Active Plan",
                "badge": _format_badge([f"{len(plan_ops)} steps"]),
                "payload": {"ops": plan_ops},
            }
        )
    try:
        raw_sample_plan = _load_sample_plan()
    except Exception as e:
        st.warning(f"Failed to load sample plan: {e}")
        raw_sample_plan = {"id": "empty", "name": "Empty", "operations": []}

    sample_plan = _normalize_sample_plan(raw_sample_plan)
    if sample_plan:
        nodes.append(
            {
                "id": "sample-plan",
                "label": "Sample Plan",
                "badge": _format_badge([f"{len(sample_plan.get('ops', []))} steps"]),
                "payload": {"plan": sample_plan},
            }
        )
    return nodes


def _run_tree_nodes() -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for summary in list_runs():
        node = _run_summary_to_node(summary)
        run_id = summary.run_id
        node["id"] = run_id
        if not node.get("label"):
            human_label = (
                summary.extra.get("label")
                or summary.extra.get("name")
                or summary.extra.get("title")
                or run_id
            )
            node["label"] = human_label
        nodes.append(node)
    return nodes


def _run_summary_to_node(summary: RunSummary) -> Dict[str, Any]:
    label = (
        summary.extra.get("label")
        or summary.extra.get("name")
        or summary.extra.get("title")
        or summary.run_id
    )
    badge = _format_badge(
        [
            summary.status,
            f"{summary.num_artifacts} artifacts" if summary.num_artifacts is not None else None,
        ]
    )
    subtitle = None
    if summary.started_at:
        subtitle = summary.started_at.strftime("Started %Y-%m-%d %H:%M")
    return {
        "id": summary.run_id,
        "label": label,
        "badge": badge,
        "subtitle": subtitle,
        "payload": {"run_id": summary.run_id},
    }


def _artifact_tree_nodes() -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    artifact_dir = _REPO_ROOT / "artifacts"
    if not artifact_dir.exists():
        return nodes
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_dir():
            continue
        label = path.name
        nodes.append(
            {
                "id": label,
                "label": label,
                "subtitle": "directory",
                "payload": {"path": str(path)},
            }
        )
    return nodes
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

    for filename, payload in _BUNDLED_IMAGE_PAYLOADS.items():
        target = BUNDLED_IMAGE_DIR / filename
        if target.exists():
            continue
        try:
            if isinstance(payload, bytes):
                target.write_bytes(payload)
            else:
                target.write_text(str(payload))
        except OSError as exc:
            st.warning(f"Failed to write bundled image {filename}: {exc}")


def _fetch_dataset_loaders(backend_url: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    url = backend_url.rstrip("/") + "/datasets/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return [], str(exc)

    try:
        payload = response.json()
    except ValueError as exc:
        return [], f"Failed to decode loader list: {exc}"

    loaders = payload.get("loaders")
    if not isinstance(loaders, list):
        return [], "Backend did not return a loader list"

    normalized: List[Dict[str, Any]] = []
    for entry in loaders:
        if isinstance(entry, dict):
            normalized.append(entry)
    return normalized, None


def _request_dataset_preview(
    backend_url: str,
    loader_id: str,
    config: Dict[str, Any],
    limit: int,
    refresh: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = backend_url.rstrip("/") + "/datasets/preview"
    payload: Dict[str, Any] = {"loader": loader_id, "config": config, "limit": limit}
    if refresh:
        payload["refresh"] = True
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, str(exc)

    try:
        return response.json(), None
    except ValueError as exc:
        return None, f"Failed to decode preview response: {exc}"


def _render_dataset_loader_sidebar(
    backend_url: str, container: "st.delta_generator.DeltaGenerator" | None = None
) -> None:
    target = container or st.sidebar.container()
    target.markdown("### Datasets")

    loader_state = st.session_state.dataset_loader_state
    refresh_requested = target.button(
        "Refresh loaders",
        key="dataset_loader_refresh_button",
    )
    if refresh_requested:
        loader_state["backend"] = None

    if loader_state.get("backend") != backend_url:
        loaders, error = _fetch_dataset_loaders(backend_url)
        loader_state["backend"] = backend_url
        loader_state["loaders"] = loaders
        loader_state["error"] = error

    loaders = loader_state.get("loaders") or []
    error_message = loader_state.get("error")
    if error_message:
        target.warning(f"Unable to load dataset loaders: {error_message}")

    if not loaders:
        target.info("No dataset loaders available from the backend.")
        return

    loader_ids = [str(loader.get("id", "")) for loader in loaders]
    loader_labels = [
        f"{loader.get('name') or loader_id} ({loader_id})"
        for loader, loader_id in zip(loaders, loader_ids)
    ]

    selected_id = st.session_state.dataset_loader_id
    if selected_id not in loader_ids:
        selected_id = loader_ids[0]
    selected_index = loader_ids.index(selected_id)

    chosen_label = target.selectbox(
        "Loader",
        options=loader_labels,
        index=selected_index,
        key="dataset_loader_select",
    )
    selected_index = loader_labels.index(chosen_label)
    selected_id = loader_ids[selected_index]

    if selected_id != st.session_state.dataset_loader_id:
        st.session_state.dataset_loader_id = selected_id
        st.session_state.dataset_loader_preview = None
        st.session_state.dataset_loader_preview_error = None

    loader_definition = loaders[selected_index]
    description = loader_definition.get("description")
    if description:
        target.caption(description)

    config_store: Dict[str, Dict[str, Any]] = st.session_state.dataset_loader_configs
    config = dict(config_store.get(selected_id, {}))
    params = loader_definition.get("params") or []
    for param in params:
        name = str(param.get("name"))
        label = param.get("label") or name
        help_text = param.get("description")
        kind = param.get("kind", "string")
        default_value = param.get("default")
        choices = param.get("choices") or []
        widget_key = f"dataset_loader_param_{selected_id}_{name}"

        value = config.get(name, default_value)
        if kind == "select" and choices:
            str_choices = [str(choice) for choice in choices]
            if value not in str_choices:
                value = str_choices[0] if str_choices else ""
            selected_choice = target.selectbox(
                label,
                options=str_choices,
                index=str_choices.index(value) if value in str_choices else 0,
                key=widget_key,
                help=help_text,
            )
            config[name] = selected_choice
        else:
            text_value = target.text_input(
                label,
                value=str(value) if value is not None else "",
                key=widget_key,
                help=help_text,
            )
            config[name] = text_value.strip()

    config_store[selected_id] = config

    limit_value = int(st.session_state.get("dataset_loader_limit", 3))

    refresh_preview = target.checkbox(
        "Force refresh",
        value=False,
        key=f"dataset_loader_refresh_preview_{selected_id}",
        help="Bypass cached results on the backend.",
    )

    existing_preview = st.session_state.dataset_loader_preview
    if (
        existing_preview
        and isinstance(existing_preview, dict)
        and existing_preview.get("loader") == selected_id
        and existing_preview.get("summary")
    ):
        container.caption(f"Last preview: {existing_preview['summary']}")

    if st.session_state.dataset_loader_preview_error:
        target.error(st.session_state.dataset_loader_preview_error)

    if target.button(
        "Preview dataset",
        key=f"dataset_loader_preview_button_{selected_id}",
    ):
        sanitized_config = {key: value for key, value in config.items() if value not in ("", None)}
        preview, error = _request_dataset_preview(
            backend_url,
            selected_id,
            sanitized_config,
            int(limit_value),
            refresh_preview,
        )
        if error:
            st.session_state.dataset_loader_preview = None
            st.session_state.dataset_loader_preview_error = error
            target.error(f"Preview failed: {error}")
        else:
            st.session_state.dataset_loader_preview = preview
            st.session_state.dataset_loader_preview_error = None
            if preview and preview.get("summary"):
                target.success(preview["summary"])


def _preview_caption_from_record(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    file_name = record.get("file_name")
    if file_name:
        parts.append(str(file_name))
    image_id = record.get("image_id")
    if image_id and not file_name:
        parts.append(f"ID {image_id}")
    city = record.get("city")
    if city:
        parts.append(str(city))

    annotations = record.get("annotations")
    if isinstance(annotations, list):
        categories: List[str] = []
        for item in annotations:
            if not isinstance(item, dict):
                continue
            category = item.get("category") or item.get("label")
            if category and category not in categories:
                categories.append(str(category))
            if len(categories) >= 3:
                break
        if categories:
            parts.append(", ".join(categories))

    objects = record.get("objects")
    if isinstance(objects, list):
        labels: List[str] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label and label not in labels:
                labels.append(str(label))
            if len(labels) >= 3:
                break
        if labels:
            parts.append(", ".join(labels))

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        meta_items = []
        for key, value in list(metadata.items())[:3]:
            meta_items.append(f"{key}: {value}")
        if meta_items:
            parts.append("; ".join(meta_items))

    return " | ".join(parts)


def _resolve_preview_image_path(image_value: Any) -> Optional[Path]:
    if not image_value:
        return None

    raw_value = str(image_value)
    candidate = Path(raw_value)
    if candidate.exists():
        return candidate

    dataset_payload = st.session_state.get("last_dataset_payload")
    if isinstance(dataset_payload, dict):
        base_path = dataset_payload.get("path")
        if isinstance(base_path, str) and base_path:
            base_dir = Path(base_path)
            if not base_dir.is_absolute():
                base_dir = (Path.cwd() / base_dir).resolve()
            resolved = base_dir / raw_value
            if resolved.exists():
                return resolved

    return candidate


def _render_preview_thumbnail(record: Dict[str, Any]) -> None:
    image_value = record.get("image_path") or record.get("path")
    image_path = _resolve_preview_image_path(image_value)
    if image_path is None:
        st.write("No image path in record")
        return

    if not image_path.exists():
        st.warning(f"Missing image: {image_path}")
        return

    try:
        with Image.open(image_path) as preview_image:
            st.image(preview_image, use_column_width=True)
    except Exception as exc:
        st.error(f"Failed to open image: {exc}")
        return

    caption = _preview_caption_from_record(record)
    if caption:
        st.caption(caption)


def _render_dataset_preview(preview: Optional[Dict[str, Any]]) -> None:
    if not preview:
        return

    records = preview.get("records") or []
    summary = preview.get("summary")

    st.markdown("### Dataset preview")
    if summary:
        st.caption(summary)

    if not records:
        st.info("No preview records returned by the loader.")
        return

    max_items = min(len(records), 12)
    columns = st.columns(3)
    for idx in range(max_items):
        record = records[idx]
        column = columns[idx % 3]
        with column:
            if isinstance(record, dict):
                _render_preview_thumbnail(record)
            else:
                st.write(str(record))

    if len(records) > max_items:
        st.caption(f"Showing first {max_items} of {len(records)} record(s).")

    scan_results = preview.get("scan")
    if scan_results:
        with st.expander("Dataset scan details", expanded=False):
            st.json(scan_results)

    with st.expander("Preview records (JSON)", expanded=False):
        st.json(records[:max_items])


def _list_bundled_images() -> List[Path]:
    _ensure_bundled_images_present()
    try:
        filenames = list_bundled_images()
    except Exception:
        return []

    return [BUNDLED_IMAGE_DIR / name for name in filenames]


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
        image_path = _resolve_preview_image_path(image_value)
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
    if data:
        mime = "application/json" if p.suffix.lower() == ".json" else "text/plain"
        key_context = key_suffix or str(p)
        key_value = _unique_widget_key("artifact", label, key_context, p.name)
        st.download_button(
            label=f"Download {p.name}",
            data=data,
            file_name=p.name,
            mime=mime,
            use_container_width=True,
            key=key_value,
        )
    elif data == b"":
        st.info(f"Artifact {p.name} is empty.")
    else:
        st.info("Artifact not readable from UI process; path is shown for reference.")


# --- helpers: operation details ---


def _sanitize_streamlit_key(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in value)
    return sanitized.strip("-") or "key"


def _unique_widget_key(prefix: str, *parts: Any) -> str:
    sanitized_prefix = _sanitize_streamlit_key(str(prefix))
    sanitized_parts = [
        _sanitize_streamlit_key(str(part)) for part in parts if part not in (None, "")
    ]
    sanitized_parts = [part for part in sanitized_parts if part]
    if not sanitized_parts:
        return unique_key(sanitized_prefix)
    return _base_widget_key(sanitized_prefix, *sanitized_parts)


@st.cache_data(show_spinner=False)
def _load_provenance_payload(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    return payload, text


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _logs_dataframe(entries: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in entries or []:
        ts = entry.get("ts")
        level = entry.get("level")
        message = entry.get("message")
        context = entry.get("context")
        rows.append(
            {
                "ts": _stringify_value(ts),
                "level": level or "",
                "message": message or "",
                "context": _stringify_value(context),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ts", "level", "message", "context"])
    return pd.DataFrame(rows)


def _format_logs_for_download(entries: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in entries or []:
        ts = _stringify_value(entry.get("ts"))
        level = entry.get("level") or ""
        message = entry.get("message") or ""
        context_text = _stringify_value(entry.get("context"))
        parts = [part for part in (ts, level, message) if part]
        line = " ".join(parts)
        if context_text:
            line = f"{line} | {context_text}" if line else context_text
        lines.append(line)
    return "\n".join(lines)


def _key_with_suffix(base: str, suffix: str) -> str:
    return _unique_widget_key(base, suffix)


def render_operation_detail_drawers(section_label: str, payload: Dict[str, Any]) -> None:
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts, dict):
        return

    provenance_path = artifacts.get("provenance")
    if not provenance_path:
        st.info("No provenance artifact available for this run.")
        return

    provenance_data, provenance_text = _load_provenance_payload(str(provenance_path))
    if provenance_data is None:
        st.warning(f"Unable to load provenance details from {provenance_path}.")
        return

    st.subheader("Operation details")
    download_cols = st.columns(2)
    key_base = f"{section_label}-{provenance_path}"

    if provenance_text:
        download_cols[0].download_button(
            label="Download provenance.json",
            data=provenance_text,
            file_name="provenance.json",
            mime="application/json",
            use_container_width=True,
            key=_key_with_suffix(key_base, "prov"),
        )
    else:
        download_cols[0].caption("Provenance file not readable.")

    log_entries = payload.get("logs") or provenance_data.get("logs") or []
    logs_text = _format_logs_for_download(log_entries)
    if logs_text:
        download_cols[1].download_button(
            label="Download logs.txt",
            data=logs_text,
            file_name="logs.txt",
            mime="text/plain",
            use_container_width=True,
            key=_key_with_suffix(key_base, "logs"),
        )
    else:
        download_cols[1].caption("No logs available.")

    if log_entries:
        st.caption("Run logs")
        run_logs_df = _logs_dataframe(log_entries)
        if not run_logs_df.empty:
            st.dataframe(run_logs_df)

    records = provenance_data.get("records") or []
    if not records:
        st.info("No per-record provenance entries available.")
        return

    for record in records:
        row_index = record.get("row_index")
        label = f"Row {row_index}" if row_index is not None else "Record"
        with st.expander(f"Details · {label}"):
            inputs = record.get("inputs") or {}
            st.markdown("**Inputs**")
            if inputs:
                input_rows = [
                    {"field": str(key), "value": _stringify_value(value)}
                    for key, value in inputs.items()
                ]
                st.table(pd.DataFrame(input_rows))
            else:
                st.caption("No inputs recorded.")

            parameters = record.get("parameters") or []
            st.markdown("**Parameters**")
            if parameters:
                parameter_rows = [
                    {
                        "kind": param.get("kind"),
                        "params": _stringify_value(param.get("params")),
                    }
                    for param in parameters
                ]
                st.table(pd.DataFrame(parameter_rows))
            else:
                st.caption("No parameters recorded.")

            outputs = record.get("outputs") or {}
            st.markdown("**Outputs**")
            if outputs:
                output_rows = [
                    {"field": str(key), "value": _stringify_value(value)}
                    for key, value in outputs.items()
                ]
                st.table(pd.DataFrame(output_rows))
            else:
                st.caption("No outputs recorded.")

            provenance_graph = record.get("provenance") or {}
            graph_cols = st.columns(2)
            nodes = provenance_graph.get("nodes") or []
            graph_cols[0].markdown("**Nodes**")
            if nodes:
                graph_cols[0].dataframe(pd.DataFrame(nodes))
            else:
                graph_cols[0].caption("No nodes recorded.")
            edges = provenance_graph.get("edges") or []
            graph_cols[1].markdown("**Edges**")
            if edges:
                graph_cols[1].dataframe(pd.DataFrame(edges))
            else:
                graph_cols[1].caption("No edges recorded.")

            record_logs = record.get("logs") or []
            st.markdown("**Logs**")
            record_logs_df = _logs_dataframe(record_logs)
            if not record_logs_df.empty:
                st.table(record_logs_df)
            else:
                st.caption("No logs for this record.")


def render_cluster_view(
    records: List[Dict[str, Any]],
    *,
    key_prefix: str,
    dataset_kind: str,
    artifacts: Optional[Dict[str, Any]] = None,
) -> bool:
    if not records:
        return False

    umap_field = _detect_umap_field(records)
    cluster_field = _detect_cluster_field(records)
    if not umap_field or not cluster_field:
        return False

    label_field = _guess_label_field(records, dataset_kind)
    scatter_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        coords = record.get(umap_field)
        if not _valid_umap_point(coords):
            continue
        x_val = float(coords[0])
        y_val = float(coords[1])
        cluster_value = record.get(cluster_field)
        scatter_rows.append(
            {
                "row_index": idx,
                "umap_x": x_val,
                "umap_y": y_val,
                "cluster_label": _format_cluster_label(cluster_value),
                "label": _extract_label(record, label_field),
            }
        )
        detail_rows.extend(
            _detail_rows_for_record(
                idx,
                record,
                skip_keys={umap_field, cluster_field, "embedding"},
            )
        )

    if not scatter_rows:
        return False

    scatter_df = pd.DataFrame(scatter_rows)
    scatter_df["cluster_label"] = scatter_df["cluster_label"].astype(str)

    selection = alt.selection_point(
        fields=["row_index"],
        name=f"{key_prefix}_cluster_select",
    )
    tooltip_fields = ["row_index", "cluster_label"]
    if "label" in scatter_df.columns:
        tooltip_fields.append("label")

    points = (
        alt.Chart(scatter_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("umap_x:Q", title="UMAP 1"),
            y=alt.Y("umap_y:Q", title="UMAP 2"),
            color=alt.Color("cluster_label:N", title="Cluster"),
            tooltip=tooltip_fields,
            opacity=alt.condition(selection, alt.value(0.9), alt.value(0.3)),
        )
        .add_params(selection)
    )

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_chart = (
            alt.Chart(detail_df)
            .transform_filter(selection)
            .mark_text(align="left")
            .encode(
                y=alt.Y("order:O", axis=None),
                text="display:N",
            )
            .properties(height=max(140, int(detail_df["order"].max() + 1) * 18))
        )
    else:
        detail_chart = (
            alt.Chart(pd.DataFrame({"display": ["Select a point to view details."]}))
            .mark_text(align="left")
            .encode(text="display:N")
        )

    st.markdown("#### Cluster view")
    st.altair_chart(
        (points & detail_chart).resolve_scale(color="independent"), use_container_width=True
    )
    st.caption("Click a point to view the corresponding record.")

    if artifacts:
        cluster_artifacts = [
            f"{key}: {val}"
            for key, val in artifacts.items()
            if str(key).startswith(("umap", "hdbscan"))
        ]
        if cluster_artifacts:
            st.caption("Artifacts: " + ", ".join(cluster_artifacts))
    return True


def _detect_umap_field(records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    preferred = ("umap", "umap_coords", "coords")
    for field in preferred:
        if any(_valid_umap_point(record.get(field)) for record in records):
            return field
    for key in records[0].keys():
        if _valid_umap_point(records[0].get(key)):
            return key
    return None


def _detect_cluster_field(records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    preferred = ("cluster", "cluster_id", "cluster_label")
    for field in preferred:
        if any(field in record for record in records):
            if any(record.get(field) is not None for record in records):
                return field
    for key in records[0].keys():
        if "cluster" in key.lower():
            return key
    return None


def _valid_umap_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        x_val = float(value[0])
        y_val = float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(x_val) and math.isfinite(y_val)


def _guess_label_field(records: List[Dict[str, Any]], dataset_kind: str) -> Optional[str]:
    if not records:
        return None
    priority = ["text", "caption", "title", "headline"]
    if dataset_kind == "images":
        priority.insert(0, "image_path")
    for field in priority:
        if any(field in record and record[field] for record in records):
            return field
    for key, value in records[0].items():
        if isinstance(value, str) and value:
            return key
    return None


def _extract_label(record: Dict[str, Any], field: Optional[str]) -> str:
    if not field:
        return ""
    value = record.get(field)
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _detail_rows_for_record(
    row_index: int,
    record: Dict[str, Any],
    *,
    skip_keys: Optional[set[str]] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skip = set(skip_keys or set())
    skip.update({"error", "logs"})
    order = 0
    for key, value in record.items():
        if key in skip:
            continue
        display_value = _format_detail_value(value)
        rows.append({"row_index": row_index, "order": order, "display": f"{key}: {display_value}"})
        order += 1
        if order >= limit:
            break
    return rows


def _format_detail_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, str)):
        text = str(value)
        return text if len(text) <= 120 else text[:117] + "..."
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if len(value) > 12:
            return f"[{len(value)} items]"
        return "[" + ", ".join(_format_detail_value(item) for item in value) + "]"
    if isinstance(value, dict):
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except TypeError:
            serialized = str(value)
        return serialized if len(serialized) <= 120 else serialized[:117] + "..."
    return str(value)


def _format_cluster_label(value: Any) -> str:
    if isinstance(value, (int, float)):
        cluster_int = int(value)
        return "Noise (-1)" if cluster_int == -1 else f"Cluster {cluster_int}"
    if value is None:
        return "Unassigned"
    text = str(value)
    return text if text else "Unassigned"


def _cluster_color_rgb(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, (int, float)):
        cluster_int = int(value)
        if cluster_int < 0:
            return _MASK_NOISE_COLOR
        palette_index = cluster_int % len(_MASK_COLOR_PALETTE)
        return _MASK_COLOR_PALETTE[palette_index]
    return _MASK_NOISE_COLOR


def _build_mask_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        mask_entries = record.get("masks")
        if not isinstance(mask_entries, list):
            continue
        features = record.get("mask_embedding")
        coords_list = record.get("mask_umap") or record.get("umap")
        clusters_list = record.get("mask_cluster") or record.get("cluster")
        for mask_idx, mask_entry in enumerate(mask_entries):
            if not isinstance(mask_entry, dict):
                continue
            feature_entry = None
            if isinstance(features, list) and mask_idx < len(features):
                candidate = features[mask_idx]
                if isinstance(candidate, dict):
                    feature_entry = candidate
            coord_values = None
            if isinstance(coords_list, list) and mask_idx < len(coords_list):
                coord_candidate = coords_list[mask_idx]
                if isinstance(coord_candidate, (list, tuple)) and len(coord_candidate) >= 2:
                    try:
                        coord_values = (float(coord_candidate[0]), float(coord_candidate[1]))
                    except (TypeError, ValueError):
                        coord_values = None
            cluster_value = None
            if isinstance(clusters_list, list) and mask_idx < len(clusters_list):
                cluster_value = clusters_list[mask_idx]
            mask_id = mask_entry.get("id")
            if not mask_id and feature_entry and feature_entry.get("mask_id"):
                mask_id = feature_entry.get("mask_id")
            if not mask_id:
                mask_id = f"mask_{mask_idx}"
            mask_path = mask_entry.get("mask_path")
            if not mask_path and feature_entry:
                metadata = feature_entry.get("metadata")
                if isinstance(metadata, dict):
                    path_candidate = metadata.get("mask_path")
                    if path_candidate:
                        mask_path = str(path_candidate)
            row: Dict[str, Any] = {
                "record_index": record_index,
                "mask_index": mask_idx,
                "mask_id": str(mask_id),
                "image_path": str(record.get("image_path") or record.get("path") or ""),
                "mask_path": str(mask_path) if mask_path else None,
                "bbox": mask_entry.get("bbox"),
                "area": mask_entry.get("area"),
                "score": mask_entry.get("score"),
                "cluster": cluster_value,
                "umap": coord_values,
                "mask_metadata": (
                    mask_entry.get("metadata")
                    if isinstance(mask_entry.get("metadata"), dict)
                    else {}
                ),
                "feature": feature_entry or {},
            }
            if coord_values is not None:
                row["umap_x"], row["umap_y"] = coord_values
            else:
                row["umap_x"] = None
                row["umap_y"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def _mask_detail_rows(scatter_df: pd.DataFrame) -> List[Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    for _, row in scatter_df.iterrows():
        mask_key = row["mask_key"]
        label_items = [
            ("Mask", row.get("mask_id")),
            ("Image", Path(str(row.get("image_path") or "")).name),
            ("Cluster", row.get("cluster_label")),
        ]
        area_value = row.get("area")
        if pd.notna(area_value):
            try:
                label_items.append(("Area", f"{int(float(area_value))}"))
            except (TypeError, ValueError):
                pass
        score_value = row.get("score")
        if pd.notna(score_value):
            try:
                label_items.append(("Score", f"{float(score_value):.3f}"))
            except (TypeError, ValueError):
                pass
        for order, (label, value) in enumerate(label_items):
            if value in (None, ""):
                continue
            detail_rows.append(
                {"mask_key": mask_key, "order": order, "display": f"{label}: {value}"}
            )
    if not detail_rows:
        detail_rows.append(
            {"mask_key": "", "order": 0, "display": "Select a mask to view details."}
        )
    return detail_rows


def _generate_mask_overlay(image_path: str, mask_rows: pd.DataFrame) -> Optional[Image.Image]:
    if not image_path:
        return None
    image_file = Path(image_path)
    if not image_file.exists():
        return None
    try:
        base = Image.open(image_file).convert("RGBA")
    except Exception:
        return None
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for _, row in mask_rows.iterrows():
        mask_path = row.get("mask_path")
        if not mask_path:
            continue
        mask_file = Path(mask_path)
        if not mask_file.exists():
            continue
        try:
            mask_img = Image.open(mask_file).convert("L")
        except Exception:
            continue
        if mask_img.size != base.size:
            mask_img = mask_img.resize(base.size, resample=Image.NEAREST)
        binary_mask = mask_img.point(lambda p: 255 if p > 0 else 0)
        color_rgb = _cluster_color_rgb(row.get("cluster"))
        color_image = Image.new("RGBA", base.size, color_rgb + (_MASK_ALPHA,))
        overlay.paste(color_image, mask=binary_mask)
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def _render_mask_table(mask_rows: pd.DataFrame) -> None:
    if mask_rows.empty:
        st.caption("No masks available.")
        return
    table_df = mask_rows.copy()
    table_df["cluster_label"] = table_df["cluster"].apply(_format_cluster_label)
    display_columns = ["mask_id", "cluster_label", "area", "score", "mask_path"]
    for column in ["area", "score"]:
        if column in table_df.columns:
            table_df[column] = table_df[column].apply(
                lambda value: (
                    ""
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else value
                )
            )
    st.dataframe(table_df[display_columns])


def _render_mask_downloads(mask_rows: pd.DataFrame, record_label: str, record_index: int) -> None:
    mask_paths: List[Path] = []
    for path_str in mask_rows.get("mask_path", []):
        if not path_str:
            continue
        candidate = Path(str(path_str))
        if candidate.exists():
            mask_paths.append(candidate)
    if not mask_paths:
        st.caption("Mask files unavailable for download.")
        return
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in mask_paths:
            archive.write(path, arcname=path.name)
    buffer.seek(0)
    download_key = _unique_widget_key("mask-download", record_index, record_label)
    st.download_button(
        label=f"Download masks ({record_label})",
        data=buffer.getvalue(),
        file_name=f"{record_label}_masks.zip",
        mime="application/zip",
        key=download_key,
    )


def _render_mask_legend(entries: List[Tuple[str, Tuple[int, int, int]]]) -> None:
    if not entries:
        return
    legend_cols = st.columns(len(entries))
    for (label, color_rgb), col in zip(entries, legend_cols):
        hex_color = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        col.markdown(
            f"<div style='background-color:{hex_color}; width:100%; height:16px; border-radius:4px'></div>",
            unsafe_allow_html=True,
        )
        col.caption(label)


def _format_image_caption(record: Dict[str, Any], default_label: str) -> str:
    image_path = record.get("image_path") or record.get("path")
    if not image_path:
        return default_label
    return Path(str(image_path)).name or default_label


def _render_mask_scatter(mask_df: pd.DataFrame, key_prefix: str) -> None:
    scatter_df = mask_df.dropna(subset=["umap_x", "umap_y"])
    if scatter_df.empty:
        st.caption("Mask embeddings do not include UMAP coordinates.")
        return
    scatter_df = scatter_df.copy()
    scatter_df["cluster_label"] = scatter_df["cluster"].apply(_format_cluster_label)
    scatter_df["mask_key"] = scatter_df.apply(
        lambda row: f"{int(row['record_index'])}:{int(row['mask_index'])}", axis=1
    )
    selection = alt.selection_point(fields=["mask_key"], name=f"{key_prefix}_mask_select")
    tooltip_fields = ["mask_key", "mask_id", "cluster_label", "image_path"]
    chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("umap_x:Q", title="UMAP 1"),
            y=alt.Y("umap_y:Q", title="UMAP 2"),
            color=alt.Color("cluster_label:N", title="Cluster"),
            tooltip=tooltip_fields,
            opacity=alt.condition(selection, alt.value(0.9), alt.value(0.3)),
        )
        .add_params(selection)
    )
    detail_rows = _mask_detail_rows(scatter_df)
    detail_df = pd.DataFrame(detail_rows)
    detail_chart = (
        alt.Chart(detail_df)
        .transform_filter(selection)
        .mark_text(align="left")
        .encode(y=alt.Y("order:O", axis=None), text="display:N")
        .properties(height=max(140, int(detail_df["order"].max() + 1) * 18))
    )
    st.altair_chart(
        (chart & detail_chart).resolve_scale(color="independent"), use_container_width=True
    )
    st.caption("Select a mask to view details.")


def render_mask_analytics_result(result: Dict[str, Any]) -> None:
    st.session_state.mask_download_counter = 0
    records = result.get("records") or []
    if not records:
        st.info("No records returned by mask analytics.")
        return
    mask_df = _build_mask_dataframe(records)
    if mask_df.empty:
        st.info("No masks detected in the current selection.")
        return

    st.markdown("#### Mask embedding clusters")
    _render_mask_scatter(mask_df, key_prefix="mask")

    artifacts = result.get("artifacts") or {}
    mask_artifacts = [
        f"{key}: {val}" for key, val in artifacts.items() if "mask" in str(key).lower()
    ]
    if mask_artifacts:
        st.caption("Artifacts: " + ", ".join(mask_artifacts))

    for record_index, record in enumerate(records):
        record_mask_df = mask_df[mask_df["record_index"] == record_index]
        if record_mask_df.empty:
            continue
        image_label = _format_image_caption(record, f"Image {record_index}")
        st.markdown(f"#### {image_label}")
        overlay_image = _generate_mask_overlay(
            record.get("image_path") or record.get("path"), record_mask_df
        )
        if overlay_image:
            st.image(overlay_image, caption="Mask overlay", use_column_width=True)
        cluster_entries = sorted(
            {
                _format_cluster_label(value): _cluster_color_rgb(value)
                for value in record_mask_df["cluster"].unique()
            }.items(),
            key=lambda item: item[0],
        )
        _render_mask_legend(cluster_entries)
        _render_mask_table(record_mask_df)
        record_stem = Path(record_mask_df["image_path"].iloc[0]).stem or f"image_{record_index}"
        _render_mask_downloads(record_mask_df, record_stem, record_index)
        st.markdown("---")


def render_mask_analytics_panel(
    dataset_payload: Optional[Dict[str, Any]],
    backend_url: str,
    sample_size: int,
    run_requested: bool = False,
) -> None:
    st.markdown("### Mask analytics")
    if not dataset_payload:
        st.session_state.mask_analytics_result = None
        st.info("Upload images to enable mask analytics.")
        return

    segmentation_mode = st.session_state.get("mask_segmentation_mode", "sam")
    prompt_value = st.session_state.get("mask_prompt", "")

    mode_label = "Segment Anything (SAM)" if segmentation_mode == "sam" else "CLIPSeg (prompt)"
    prompt_note = ""
    if segmentation_mode == "clipseg" and prompt_value.strip():
        prompt_note = f" · prompt: {prompt_value.strip()}"
    st.caption(f"Segmentation: {mode_label}{prompt_note}")

    if run_requested:
        if segmentation_mode == "clipseg" and not prompt_value.strip():
            st.warning("Enter a prompt before running CLIPSeg segmentation.")
        else:
            operations: List[Dict[str, Any]] = []
            if segmentation_mode == "clipseg":
                operations.append(
                    {
                        "kind": "clipseg_segment",
                        "params": {"prompt": prompt_value.strip(), "output_field": "masks"},
                    }
                )
            else:
                operations.append({"kind": "sam_segment", "params": {"output_field": "masks"}})
            operations.extend(
                [
                    {
                        "kind": "embed_masks",
                        "params": {"source": "masks", "output_field": "mask_embedding"},
                    },
                    {
                        "kind": "umap",
                        "params": {
                            "source": "mask_embedding",
                            "output_field": "mask_umap",
                            "n_neighbors": 15,
                        },
                    },
                    {
                        "kind": "hdbscan",
                        "params": {"source": "mask_umap", "output_field": "mask_cluster"},
                    },
                ]
            )
            request_body = {
                "dataset": dataset_payload,
                "operations": operations,
                "preview_sample_size": sample_size,
            }
            with st.spinner("Running mask analytics..."):
                result = _post_json(f"{backend_url.rstrip('/')}/preview", request_body)
            if result is not None:
                st.session_state.mask_analytics_result = {
                    "result": result,
                    "params": {
                        "segmentation": segmentation_mode,
                        "prompt": prompt_value.strip() if segmentation_mode == "clipseg" else "",
                    },
                }

    stored_result = st.session_state.get("mask_analytics_result")
    if stored_result and stored_result.get("result"):
        render_mask_analytics_result(stored_result["result"])


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


def render_artifacts_section(resp_json: Any) -> bool:
    arts = resp_json.get("artifacts") if isinstance(resp_json, dict) else None
    if not arts:
        return False

    selected_run = st.session_state.get("selected_run_id")
    run_identifier = _extract_artifact_run_identifier(resp_json)
    if selected_run and run_identifier and selected_run != run_identifier:
        st.caption(f"Run filter active: {selected_run}. Artifacts for {run_identifier} hidden.")
        return False
    if selected_run and not run_identifier:
        st.caption(f"Run filter active: {selected_run}. Artifacts with no run metadata hidden.")
        return False

    st.subheader("Artifacts")
    if run_identifier:
        st.caption(f"Run: {run_identifier}")

    rendered_any = False
    for key in ("captions", "metadata", "output_csv", "preview"):
        if key in arts and arts[key]:
            render_artifact(key, arts[key], key_suffix=run_identifier)
            rendered_any = True
    for key, val in arts.items():
        if key in ("captions", "metadata", "output_csv", "preview"):
            continue
        if isinstance(val, str):
            render_artifact(key, val, key_suffix=run_identifier)
            rendered_any = True
    return rendered_any


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


backend_url = DEFAULT_BACKEND_URL
run_masks_clicked = False
clear_masks_clicked = False

with st.sidebar:
    st.header("Project & Tasks")
    with st.expander("Project tree (beta)", expanded=False):
        tree_container = st.container()
        _render_project_tree_sidebar(tree_container)

    st.divider()
    st.subheader("Dataset")

    simple_mode = st.toggle(
        "Simple mode",
        value=st.session_state.simple_mode,
        help="Hide advanced controls; great for first-time users.",
    )
    st.session_state.simple_mode = simple_mode

    dtype_options = ["CSV", "Images"]
    dtype_default_index = 1 if st.session_state.dataset_kind == "images" else 0
    dtype = st.radio(
        "Dataset type",
        dtype_options,
        horizontal=False,
        index=dtype_default_index,
        key="dtype",
        help="Choose CSV for text ops, Images for vision ops.",
    )

    selected_kind = "images" if dtype == "Images" else "csv"
    if selected_kind != st.session_state.dataset_kind:
        st.session_state.dataset_kind = selected_kind
        st.session_state.plan_ops = []  # type: ignore[assignment]
        st.session_state.preview_result = None
        st.session_state.execute_result = None
        st.session_state.mask_analytics_result = None

    dataset_kind = st.session_state.dataset_kind

    current_dataset_path = ""
    columns: List[str] = []
    image_paths: List[Path] = []

    if dataset_kind == "csv":
        sample_csv_path = _find_first_existing(SAMPLE_CSV_CANDIDATES)

        uploaded_file = None
        if simple_mode:
            with st.expander("Upload custom CSV", expanded=False):
                uploaded_file = st.file_uploader(
                    "Upload CSV dataset",
                    type=["csv"],
                    key="csv_uploader",
                )
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV dataset",
                type=["csv"],
                key="csv_uploader",
            )

        uploaded_path = _persist_uploaded_file(uploaded_file)

        default_use_sample = st.session_state.get("use_sample_csv", True)
        sample_checkbox_disabled = sample_csv_path is None
        if sample_checkbox_disabled:
            default_use_sample = False
        if uploaded_path and default_use_sample:
            default_use_sample = False
            st.session_state.use_sample_csv = False
        use_sample_csv = st.checkbox(
            "Use bundled sample CSV",
            value=default_use_sample,
            help="Loads data/sample_news.csv for a quick try.",
            disabled=sample_checkbox_disabled,
        )
        st.session_state.use_sample_csv = use_sample_csv

        if sample_csv_path and use_sample_csv:
            st.caption(f"Using sample at {sample_csv_path}")
        elif sample_checkbox_disabled:
            st.caption("No bundled sample CSV found.")

        try:
            raw_sample_plan = _load_sample_plan()
        except Exception as exc:
            st.warning(f"Failed to load sample plan: {exc}")
            raw_sample_plan = None

        sample_plan_data = _normalize_sample_plan(raw_sample_plan)
        if st.button("Load sample plan", disabled=sample_plan_data is None):
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
                if not dataset_path:
                    candidate_path = Path("artifacts") / "uploads" / "sample_news.csv"
                    if candidate_path.exists():
                        dataset_path = candidate_path
                if dataset_path:
                    st.session_state.dataset_path = str(dataset_path)
                st.success("Sample plan loaded")
            else:
                st.info("No sample plan found")

        if uploaded_path:
            current_dataset_path = str(uploaded_path)
        elif use_sample_csv and sample_csv_path:
            current_dataset_path = str(sample_csv_path)
        elif st.session_state.dataset_path:
            current_dataset_path = st.session_state.dataset_path

        st.session_state.dataset_path = current_dataset_path
        if current_dataset_path:
            columns = _read_columns(current_dataset_path)
    else:
        st.session_state.dataset_path = ""

        uploaded_images = None
        if simple_mode:
            with st.expander("Upload images", expanded=False):
                uploaded_images = st.file_uploader(
                    "Upload images",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key="image_uploader",
                )
        else:
            uploaded_images = st.file_uploader(
                "Upload images",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="image_uploader",
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

        use_sample_imgs = st.checkbox(
            "Use bundled sample images",
            value=st.session_state.use_bundled_images if bundled_available else False,
            help="Loads artifacts/bundled_images.",
            disabled=not bundled_available,
        )

        if use_sample_imgs != st.session_state.use_bundled_images:
            if use_sample_imgs:
                if _enable_bundled_images(remember_previous=not st.session_state.use_bundled_images):
                    st.success("Bundled sample images loaded.")
                else:
                    st.warning("Bundled sample images are unavailable.")
                    st.session_state.use_bundled_images = False
            else:
                _disable_bundled_images()

        if bundled_available and use_sample_imgs:
            st.caption(f"{len(bundled_images)} bundled image(s) found.")
        elif bundled_available:
            st.caption(f"{len(bundled_images)} bundled image(s) in {BUNDLED_IMAGE_DIR}")
        else:
            st.caption(f"No bundled images found in {BUNDLED_IMAGE_DIR}")

        if st.button("Load sample plan (images)", disabled=not bundled_available):
            activated = _enable_bundled_images(remember_previous=not st.session_state.use_bundled_images)
            if not activated:
                st.warning("Bundled sample images are unavailable.")
            else:
                plan_copy = deepcopy(IMAGE_SAMPLE_PLAN)
                st.session_state.plan_ops = plan_copy.get("ops", [])  # type: ignore[assignment]
                st.session_state.images_dir = str(BUNDLED_IMAGE_DIR.resolve())
                st.session_state.selected_images = [path.name for path in bundled_images]
                st.session_state.sample_size = min(len(bundled_images), 3) or 1
                st.session_state.preview_result = None
                st.session_state.execute_result = None
                st.session_state.use_bundled_images = True
                st.success("Sample image plan loaded.")

        image_paths = _resolve_selected_image_paths()
        images_dir_display = st.session_state.images_dir
        if images_dir_display:
            st.caption(
                f"Images stored in {images_dir_display} ({len(image_paths)} file(s) saved)"
            )

        if image_paths:
            st.markdown("#### Saved images")
            base_dir = Path(st.session_state.images_dir)
            for rel_path in list(st.session_state.selected_images):
                path = base_dir / rel_path
                if not path.exists():
                    continue
                cols = st.columns([4, 1])
                cols[0].write(f"{path.name} ({_format_size(path.stat().st_size)})")
                remove_key = _unique_widget_key("remove-image", rel_path)
                if cols[1].button("Remove", key=remove_key):
                    st.session_state.selected_images = [
                        item for item in st.session_state.selected_images if item != rel_path
                    ]
                    st.rerun()  # refresh UI with stable API
        else:
            st.info("Upload PNG, JPG, or PPM files to begin.")

    samples = st.slider(
        "Samples to preview",
        1,
        12,
        int(st.session_state.get("dataset_loader_limit", 3)),
        help="How many items to preview.",
    )
    st.session_state.dataset_loader_limit = int(samples)

    with st.expander("Advanced configuration", expanded=not simple_mode):
        backend_url = st.text_input(
            "Backend URL",
            value=backend_url,
            help="FastAPI server address.",
        )
        st.subheader("Datasets")
        loader_container = st.container()
        _render_dataset_loader_sidebar(backend_url, container=loader_container)

    if st.button("Preview dataset", key=stable_key("preview-dataset", dtype)):
        st.session_state.sidebar_preview_trigger = True

    st.divider()
    st.subheader("Mask analytics")

    seg_choice = st.radio(
        "Segmentation method",
        ["Segment Anything (SAM)", "CLIPSeg (prompt)"],
        key="seg_method",
        help="Choose SAM for general masks; CLIPSeg for text-prompt masks.",
    )
    seg_mode = "clipseg" if seg_choice == "CLIPSeg (prompt)" else "sam"
    st.session_state.mask_segmentation_mode = seg_mode

    prompt_disabled = seg_mode != "clipseg"
    prompt_value = st.text_input(
        "Text prompt",
        st.session_state.get("mask_prompt", ""),
        help="Used only for CLIPSeg.",
        disabled=prompt_disabled,
    )
    if not prompt_disabled:
        st.session_state.mask_prompt = prompt_value

    run_masks_clicked = st.button(
        "Run mask analytics",
        key=stable_key("run-masks", dtype),
    )
    clear_masks_clicked = st.button(
        "Clear results",
        key=stable_key("clear-masks", dtype),
    )

    if clear_masks_clicked:
        st.session_state.mask_analytics_result = None

dataset_payload: Optional[Dict[str, Any]] = None

main_col, provenance_col = st.columns([3, 1.2])

with main_col:
    active_run_filter = st.session_state.get("selected_run_id")
    if active_run_filter:
        info_cols = st.columns([4, 1])
        info_cols[0].caption(f"Run filter active: {active_run_filter}")
        if info_cols[1].button("Clear", key="clear_run_filter"):
            st.session_state.selected_run_id = None
            st.rerun()

    preview_payload = st.session_state.get("dataset_loader_preview")
    if isinstance(preview_payload, dict):
        _render_dataset_preview(preview_payload)

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
        if img_count <= 1:
            sample_size = 1
            st.caption("Preview sample size: 1 image")
        else:
            sample_size = st.slider(
                "Preview sample size",
                min_value=1,
                max_value=img_count,
                value=min(3, img_count),
            )
        st.session_state.sample_size = sample_size

    if dataset_kind == "csv" and current_dataset_path:
        dataset_payload = build_dataset_payload_csv(current_dataset_path)
    elif dataset_kind == "images" and image_paths:
        images_dir_str = st.session_state.images_dir
        images_dir_path = Path(images_dir_str) if images_dir_str else image_paths[0].parent
        dataset_payload = build_dataset_payload_images(
            images_dir_path.resolve(), list(st.session_state.selected_images)
        )

    st.session_state["last_dataset_payload"] = dataset_payload if dataset_payload else None

    if dataset_kind == "images":
        render_mask_analytics_panel(
            dataset_payload,
            backend_url,
            st.session_state.sample_size,
            run_requested=run_masks_clicked,
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

            remove_key = _unique_widget_key("remove-op", dataset_kind, idx)
            if st.button("Remove", key=remove_key):
                st.session_state.plan_ops.pop(idx)
                st.rerun()  # refresh UI with stable API

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
        st.rerun()  # refresh UI with stable API

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
    sidebar_preview_trigger = st.session_state.pop("sidebar_preview_trigger", False)
    if sidebar_preview_trigger:
        preview_clicked = True
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
        preview_artifacts = preview.get("artifacts") if isinstance(preview, dict) else None
        if records:
            render_cluster_view(
                records,
                key_prefix="preview",
                dataset_kind=dataset_kind,
                artifacts=preview_artifacts,
            )
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
        preview_displayed = False
        if dataset_kind == "images":
            captions_path = None
            if isinstance(preview_artifacts, dict):
                captions_path = preview_artifacts.get("captions")
            if captions_path:
                preview_displayed = render_artifacts_section(preview)
        else:
            preview_displayed = render_artifacts_section(preview)
        if st.session_state.get("selected_run_id") and preview_artifacts and not preview_displayed:
            st.caption("Preview artifacts hidden by active run filter.")

        render_operation_detail_drawers("preview", preview)

    if st.session_state.execute_result:
        st.markdown("### Execution results")
        result = st.session_state.execute_result
        artifacts = result.get("artifacts") or {}
        run_dir = _extract_run_directory(artifacts)
        records = result.get("records") or []
        if records:
            render_cluster_view(
                records,
                key_prefix="execute",
                dataset_kind=dataset_kind,
                artifacts=artifacts,
            )

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
        artifacts_shown = render_artifacts_section(result)
        if st.session_state.get("selected_run_id") and artifacts and not artifacts_shown:
            st.info(f"No execution artifacts matched run {st.session_state.selected_run_id}.")

        render_operation_detail_drawers("execute", result)

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
