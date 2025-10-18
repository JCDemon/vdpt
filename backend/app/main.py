import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ..schemas import TodoCreate, TodoRead
from ..vdpt.io import read_csv, sha256_bytes, write_csv
from ..vdpt.providers import MockProvider, QwenProvider, TextLLMProvider
from ..services import TodoService
from .preview.preview_engine import (
    preview_classify,
    preview_operation,
    preview_summarize,
)
from .provenance.recorder import bump_frequency, bump_recency, snapshot

__version__ = "0.1.0"

def get_text_provider() -> TextLLMProvider:
    """Return the configured text generation provider."""
    if os.getenv("DASHSCOPE_API_KEY"):
        return QwenProvider()
    return MockProvider()


todo_service = TodoService()


def get_service() -> TodoService:
    return todo_service


class Operation(BaseModel):
    kind: Literal["segment", "embed", "filter", "map", "summarize", "classify", "transform"]
    params: Dict[str, Any] = Field(default_factory=dict)
    summarize: Optional[Dict[str, Any]] = None
    classify: Optional[Dict[str, Any]] = None


class Plan(BaseModel):
    class Dataset(BaseModel):
        type: Literal["csv"]
        path: str
        sample_size: Optional[int] = Field(default=5, ge=1)
        random_sample: bool = False
        random_seed: Optional[int] = None

    ops: List[Operation] = Field(default_factory=list)
    dataset: Optional[Dataset] = None


def health() -> Dict[str, bool]:
    return {"ok": True}


def provenance_snapshot() -> Dict[str, Any]:
    return snapshot()


def preview(
    plan: Plan, provider: TextLLMProvider = Depends(get_text_provider)
) -> Dict[str, Any]:
    if plan.dataset and plan.dataset.type == "csv":
        return _preview_csv(plan, provider)

    details: Dict[str, Any] = {}
    for i, op in enumerate(plan.ops):
        details[str(i)] = preview_operation({"kind": op.kind, "params": op.params})
    return {
        "summary": f"dry-run ok; {len(plan.ops)} operation(s) would change data",
        "details": details,
    }


def execute(
    plan: Plan, provider: TextLLMProvider = Depends(get_text_provider)
) -> Dict[str, Any]:
    if plan.dataset and plan.dataset.type == "csv":
        return _execute_csv(plan, provider)

    keys = [op.kind for op in plan.ops]
    bump_frequency(keys)
    bump_recency(keys)
    return {"ok": True, "applied": len(plan.ops)}


def _preview_csv(plan: Plan, provider: TextLLMProvider) -> Dict[str, Any]:
    dataset = plan.dataset
    assert dataset is not None  # for type-checkers

    try:
        df = read_csv(dataset.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sample_df = _sample_dataframe(df, dataset)
    processed_df, new_columns = _apply_tabular_ops(sample_df, plan.ops, provider)

    records = processed_df.to_dict(orient="records")
    return {
        "records": records,
        "schema": {"new_columns": new_columns},
        "preview_sample_size": len(records),
        "summary": f"preview generated for {len(records)} row(s)",
    }


def _execute_csv(plan: Plan, provider: TextLLMProvider) -> Dict[str, Any]:
    dataset = plan.dataset
    assert dataset is not None

    csv_path = Path(dataset.path)
    try:
        df = read_csv(csv_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    processed_df, _ = _apply_tabular_ops(df.copy(), plan.ops, provider)

    artifacts_dir = _create_artifact_dir()
    output_path = artifacts_dir / "output.csv"
    write_csv(processed_df, output_path)

    preview_payload = _preview_csv(plan, provider)
    preview_path = artifacts_dir / "preview.json"
    preview_path.write_text(json.dumps(preview_payload, indent=2))

    dataset_hash = sha256_bytes(csv_path.read_bytes())
    metadata_path = artifacts_dir / "metadata.json"
    metadata = {
        "plan": plan.model_dump(),
        "dataset_hash": dataset_hash,
        "artifacts": {
            "output_csv": str(output_path),
            "preview_json": str(preview_path),
        },
        "preview_sample_size": preview_payload.get("preview_sample_size"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "ok": True,
        "applied": len(plan.ops),
        "artifacts": {
            "output_csv": str(output_path),
            "metadata": str(metadata_path),
            "preview": str(preview_path),
        },
    }


def _apply_tabular_ops(
    df: pd.DataFrame, ops: Sequence[Operation], provider: TextLLMProvider
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    working = df.copy()
    new_columns: List[Dict[str, Any]] = []
    provenance_keys: set[str] = set()

    for op in ops:
        params: Dict[str, Any] = dict(op.params)
        if op.summarize:
            params.update(op.summarize)
        if op.classify:
            params.update(op.classify)

        if op.kind == "summarize":
            field = params.get("field")
            if not field:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="summarize operation requires 'field'",
                )
            if field not in working.columns:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"summarize field '{field}' not found in dataset",
                )

            output_col = params.get("output_field") or f"{field}_summary"

            def summarize_value(value: Any) -> str:
                text = "" if pd.isna(value) else str(value)
                try:
                    return preview_summarize(text, params, provider)
                except RuntimeError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            working[output_col] = working[field].apply(summarize_value)
            new_columns.append({
                "name": output_col,
                "operation": "summarize",
                "source": field,
            })
            provenance_keys.update({f"op:{op.kind}", f"field:{field}"})
        elif op.kind == "classify":
            field = params.get("field")
            labels = params.get("labels")
            if not field:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="classify operation requires 'field'",
                )
            if field not in working.columns:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"classify field '{field}' not found in dataset",
                )
            if not labels:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="classify operation requires 'labels'",
                )

            output_col = params.get("output_field") or f"{field}_classification"

            def classify_value(value: Any) -> str:
                text = "" if pd.isna(value) else str(value)
                try:
                    return preview_classify(text, labels, params, provider)
                except RuntimeError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            working[output_col] = working[field].apply(classify_value)
            new_columns.append({
                "name": output_col,
                "operation": "classify",
                "source": field,
                "labels": list(labels),
            })
            provenance_keys.update({f"op:{op.kind}", f"field:{field}"})

    if provenance_keys:
        keys = sorted(provenance_keys)
        bump_frequency(keys)
        bump_recency(keys)

    return working, new_columns


def _sample_dataframe(df: pd.DataFrame, dataset: Plan.Dataset) -> pd.DataFrame:
    sample_size = dataset.sample_size or len(df)
    sample_size = min(sample_size, len(df)) if len(df) else 0
    if sample_size == 0:
        return df.head(0)
    if dataset.random_sample:
        return df.sample(n=sample_size, random_state=dataset.random_seed)
    return df.head(sample_size)


def _create_artifact_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    artifacts_root = Path("artifacts")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts_root / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_todos(service: TodoService = Depends(get_service)) -> List[TodoRead]:
    return [TodoRead(**todo) for todo in service.list_todos()]


def create_todo(
    payload: TodoCreate, service: TodoService = Depends(get_service)
) -> TodoRead:
    created = service.create(payload)
    return TodoRead(**created)


def complete_todo(
    todo_id: int, service: TodoService = Depends(get_service)
) -> TodoRead:
    try:
        completed = service.complete(todo_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo with id {todo_id} not found",
        ) from exc
    return TodoRead(**completed)


def create_app() -> FastAPI:
    app = FastAPI(title="VDPT API", version=__version__)

    app.get("/health", tags=["health"])(health)
    app.get("/provenance/snapshot", tags=["provenance"])(provenance_snapshot)
    app.post("/preview", tags=["preview"])(preview)
    app.post("/execute", tags=["execute"])(execute)

    app.get("/todos", response_model=List[TodoRead], tags=["todos"])(list_todos)
    app.post(
        "/todos",
        response_model=TodoRead,
        status_code=status.HTTP_201_CREATED,
        tags=["todos"],
    )(create_todo)
    app.post(
        "/todos/{todo_id}/complete",
        response_model=TodoRead,
        tags=["todos"],
    )(complete_todo)

    return app


app = create_app()
