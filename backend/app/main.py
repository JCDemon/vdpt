import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Union, cast

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ..schemas import TodoCreate, TodoRead
from backend.vdpt import providers
from ..vdpt.io import read_csv, sha256_bytes, write_csv
from ..services import TodoService
from .preview.preview_engine import (
    preview_classify,
    preview_operation,
    run_operation,
)
from .provenance.recorder import bump_frequency, bump_recency, snapshot

__version__ = "0.1.0"


todo_service = TodoService()


def get_service() -> TodoService:
    return todo_service


class Operation(BaseModel):
    kind: Literal[
        "segment",
        "embed",
        "filter",
        "map",
        "summarize",
        "classify",
        "transform",
        "img_resize",
        "img_caption",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    summarize: Optional[Dict[str, Any]] = None
    classify: Optional[Dict[str, Any]] = None


class CsvDataset(BaseModel):
    type: Literal["csv"]
    path: str
    sample_size: Optional[int] = Field(default=5, ge=1)
    random_sample: bool = False
    random_seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_kind(cls, values: Any) -> Any:
        if isinstance(values, dict):
            kind = values.pop("kind", None)
            if kind and "type" not in values:
                values["type"] = kind
        return values


class ImageDataset(BaseModel):
    type: Literal["images"]
    session: Optional[str] = None
    path: Optional[str] = None
    paths: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = Field(default=None, ge=1)
    random_sample: bool = False
    random_seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_kind(cls, values: Any) -> Any:
        if isinstance(values, dict):
            kind = values.pop("kind", None)
            if kind and "type" not in values:
                values["type"] = kind
        return values


DatasetType = Annotated[Union[CsvDataset, ImageDataset], Field(discriminator="type")]


class Plan(BaseModel):
    ops: List[Operation] = Field(default_factory=list)
    dataset: Optional[DatasetType] = None


# Backwards compatibility for existing callers instantiating Plan.Dataset
Plan.Dataset = CsvDataset  # type: ignore[attr-defined]
Plan.ImageDataset = ImageDataset  # type: ignore[attr-defined]


def health() -> Dict[str, bool]:
    return {"ok": True}


def provenance_snapshot() -> Dict[str, Any]:
    return snapshot()


def preview(plan: Plan) -> Dict[str, Any]:
    dataset = plan.dataset
    if isinstance(dataset, CsvDataset):
        return _preview_csv(plan)
    if isinstance(dataset, ImageDataset):
        return _preview_images(plan)

    details: Dict[str, Any] = {}
    for i, op in enumerate(plan.ops):
        details[str(i)] = preview_operation({"kind": op.kind, "params": op.params})
    return {
        "summary": f"dry-run ok; {len(plan.ops)} operation(s) would change data",
        "details": details,
    }


def execute(plan: Plan) -> Dict[str, Any]:
    dataset = plan.dataset
    if isinstance(dataset, CsvDataset):
        return _execute_csv(plan)
    if isinstance(dataset, ImageDataset):
        return _execute_images(plan)

    keys = [op.kind for op in plan.ops]
    bump_frequency(keys)
    bump_recency(keys)
    return {"ok": True, "applied": len(plan.ops)}


def _preview_csv(plan: Plan) -> Dict[str, Any]:
    dataset = cast(CsvDataset, plan.dataset)
    assert dataset is not None  # for type-checkers

    try:
        df = read_csv(dataset.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sample_df = _sample_dataframe(df, dataset)
    processed_df, new_columns = _apply_tabular_ops(sample_df, plan.ops)

    records = processed_df.to_dict(orient="records")
    return {
        "records": records,
        "schema": {"new_columns": new_columns},
        "preview_sample_size": len(records),
        "summary": f"preview generated for {len(records)} row(s)",
    }


def _execute_csv(plan: Plan) -> Dict[str, Any]:
    dataset = cast(CsvDataset, plan.dataset)
    assert dataset is not None

    csv_path = Path(dataset.path)
    try:
        df = read_csv(csv_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    artifacts_dir = _create_artifact_dir()
    processed_df, _ = _apply_tabular_ops(df.copy(), plan.ops, out_dir=artifacts_dir)
    output_path = artifacts_dir / "output.csv"
    write_csv(processed_df, output_path)

    preview_payload = _preview_csv(plan)
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


def _preview_images(plan: Plan) -> Dict[str, Any]:
    dataset = cast(ImageDataset, plan.dataset)
    assert dataset is not None

    image_paths = _resolve_image_paths(dataset)
    sample_paths = _sample_image_paths(image_paths, dataset)
    run_dir = _create_artifact_dir()
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    new_columns: Dict[str, Dict[str, Any]] = {}
    provenance_keys: set[str] = set()

    for index, path in enumerate(sample_paths):
        row_context: Dict[str, Any] = {
            "image_path": str(path),
            "__tmp_dir__": str(preview_dir),
            "__index__": index,
        }
        record: Dict[str, Any] = {"image_path": str(path)}

        for op in plan.ops:
            params = _operation_params(op)
            try:
                result = run_operation(row_context, op.kind, params)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            record.update(result)
            for column in result:
                new_columns.setdefault(column, {"name": column, "operation": op.kind})
            provenance_keys.add(f"op:{op.kind}")

        records.append(record)

    if provenance_keys:
        keys = sorted(provenance_keys)
        bump_frequency(keys)
        bump_recency(keys)

    new_column_list = [new_columns[name] for name in sorted(new_columns)]

    result = {
        "records": records,
        "schema": {"new_columns": new_column_list},
        "preview_sample_size": len(records),
        "summary": f"preview generated for {len(records)} image(s)",
    }

    captions_path = run_dir / "captions.json"
    captions_entries = _collect_image_captions(records)
    captions_path.write_text(json.dumps(captions_entries, ensure_ascii=False, indent=2))

    metadata_payload = _build_image_metadata(plan, dataset, sample_paths)
    metadata_payload["preview_sample_size"] = len(records)
    metadata_payload["record_count"] = len(records)
    metadata_payload["artifacts"] = {"captions": str(captions_path)}
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2))

    result["artifacts"] = {
        "captions": str(captions_path),
        "metadata": str(metadata_path),
    }

    return result


def _execute_images(plan: Plan) -> Dict[str, Any]:
    dataset = cast(ImageDataset, plan.dataset)
    assert dataset is not None

    image_paths = _resolve_image_paths(dataset)
    artifacts_dir = _create_artifact_dir()

    records: List[Dict[str, Any]] = []
    provenance_keys: set[str] = set()
    generated_files: set[str] = set()

    for index, path in enumerate(image_paths):
        row_context: Dict[str, Any] = {
            "image_path": str(path),
            "__index__": index,
        }
        record: Dict[str, Any] = {"image_path": str(path)}

        for op in plan.ops:
            params = _operation_params(op)
            try:
                result = run_operation(row_context, op.kind, params, out_dir=artifacts_dir)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            record.update(result)
            for value in result.values():
                if isinstance(value, str):
                    value_path = Path(value)
                    if value_path.exists() and str(value_path).startswith(str(artifacts_dir)):
                        generated_files.add(str(value_path))
            provenance_keys.add(f"op:{op.kind}")

        records.append(record)

    if provenance_keys:
        keys = sorted(provenance_keys)
        bump_frequency(keys)
        bump_recency(keys)

    if records:
        output_df = pd.DataFrame(records)
    else:
        output_df = pd.DataFrame(columns=["image_path"])

    output_path = artifacts_dir / "output.csv"
    output_df.to_csv(output_path, index=False)

    preview_payload = _preview_images(plan)
    preview_path = artifacts_dir / "preview.json"
    preview_path.write_text(json.dumps(preview_payload, indent=2))

    captions_path = artifacts_dir / "captions.json"
    captions_entries = _collect_image_captions(records)
    captions_path.write_text(json.dumps(captions_entries, ensure_ascii=False, indent=2))

    generated_list = sorted(generated_files)

    metadata_path = artifacts_dir / "metadata.json"
    metadata_artifacts: Dict[str, Any] = {
        "output_csv": str(output_path),
        "preview_json": str(preview_path),
        "captions": str(captions_path),
        "generated": generated_list,
    }
    metadata = {
        "plan": plan.model_dump(),
        "artifacts": metadata_artifacts,
        "preview_sample_size": preview_payload.get("preview_sample_size"),
        "source_images": [str(path) for path in image_paths],
    }
    metadata.update(_build_image_metadata(plan, dataset, image_paths))
    metadata["record_count"] = len(records)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    artifacts_payload: Dict[str, Any] = {
        "output_csv": str(output_path),
        "metadata": str(metadata_path),
        "preview": str(preview_path),
        "captions": str(captions_path),
    }
    if generated_files:
        artifacts_payload["generated"] = generated_list

    return {
        "ok": True,
        "applied": len(plan.ops),
        "artifacts": artifacts_payload,
    }


def _operation_params(op: Operation) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(op.params)
    if op.summarize:
        params.update(op.summarize)
    if op.classify:
        params.update(op.classify)
    return params


def _resolve_image_paths(dataset: ImageDataset) -> List[Path]:
    base_dir: Optional[Path] = None

    if dataset.path:
        base_dir = Path(dataset.path).expanduser()
        if not base_dir.is_absolute():
            base_dir = (Path.cwd() / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
        if not base_dir.exists():
            raise HTTPException(
                status_code=500,
                detail=f"images directory not found at {base_dir}",
            )
    elif dataset.session:
        base_dir = Path("artifacts") / "uploads" / dataset.session
        if not base_dir.exists() and not dataset.paths:
            raise HTTPException(
                status_code=500,
                detail=f"uploads session '{dataset.session}' not found",
            )

    candidates: List[Path] = []

    if dataset.paths:
        for raw in dataset.paths:
            candidate = Path(raw)
            if not candidate.is_absolute():
                if base_dir is None:
                    raise HTTPException(
                        status_code=500,
                        detail="relative image paths require a dataset 'path' or 'session'",
                    )
                candidate = base_dir / candidate
            candidates.append(candidate)
    else:
        if base_dir is None:
            raise HTTPException(
                status_code=500,
                detail="images dataset requires 'path' or 'session'",
            )
        if not base_dir.exists():
            raise HTTPException(
                status_code=500,
                detail=f"images directory not found at {base_dir}",
            )
        candidates.extend(sorted(p for p in base_dir.iterdir() if p.is_file()))

    resolved: List[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            raise HTTPException(status_code=500, detail=f"image not found at {candidate}")
        resolved.append(candidate)

    return resolved


def _sample_image_paths(paths: Sequence[Path], dataset: ImageDataset) -> List[Path]:
    all_paths = list(paths)
    if not all_paths:
        return []

    sample_size = dataset.sample_size or len(all_paths)
    sample_size = min(sample_size, len(all_paths))
    if sample_size <= 0:
        return []

    if dataset.random_sample:
        rng = random.Random(dataset.random_seed)
        return rng.sample(all_paths, sample_size)

    return all_paths[:sample_size]


def _collect_image_captions(records: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for record in records:
        image_path = record.get("image_path")
        caption = record.get("caption")
        if not image_path or caption is None:
            continue
        entries.append(
            {
                "file": Path(str(image_path)).name,
                "caption": str(caption),
            }
        )
    return entries


def _image_dataset_input_dir(dataset: ImageDataset) -> Optional[str]:
    if dataset.path:
        base_dir = Path(dataset.path).expanduser()
        if not base_dir.is_absolute():
            base_dir = (Path.cwd() / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
        return str(base_dir)
    if dataset.session:
        base_dir = Path("artifacts") / "uploads" / dataset.session
        return str(base_dir.resolve())
    return None


def _image_provider_details() -> tuple[str, str]:
    provider_name = (os.getenv("VDPT_PROVIDER") or "mock").strip().lower()
    provider_module = providers.current
    model_value = getattr(provider_module, "VISION_MODEL", None) or getattr(
        provider_module, "TEXT_MODEL", provider_name
    )
    return provider_name, str(model_value)


def _build_image_metadata(
    plan: Plan, dataset: ImageDataset, file_paths: Sequence[Path]
) -> Dict[str, Any]:
    provider_name, provider_model = _image_provider_details()
    files: List[str] = []
    seen: set[str] = set()
    for file_path in file_paths:
        name = Path(file_path).name
        if name in seen:
            continue
        seen.add(name)
        files.append(name)

    args_payload = {
        "dataset": dataset.model_dump(),
        "ops": [op.model_dump() for op in plan.ops],
    }

    return {
        "input_dir": _image_dataset_input_dir(dataset),
        "files": files,
        "provider": provider_name,
        "model": provider_model,
        "time": datetime.now(UTC).isoformat(),
        "args": args_payload,
    }


def _apply_tabular_ops(
    df: pd.DataFrame,
    ops: Sequence[Operation],
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    working = df.copy()
    new_columns: List[Dict[str, Any]] = []
    provenance_keys: set[str] = set()

    for op in ops:
        params = _operation_params(op)

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

            existing_columns = set(working.columns)
            results: List[Dict[str, Any]] = []
            indices: List[Any] = []
            new_column_names: set[str] = set()

            for idx, row in working.iterrows():
                row_dict = row.to_dict()
                try:
                    result = run_operation(row_dict, op.kind, params, out_dir=out_dir)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                results.append(result)
                indices.append(idx)
                new_column_names.update(result.keys())

            for column in sorted(new_column_names):
                if column not in working.columns:
                    working.loc[:, column] = pd.NA
                values = [result.get(column) for result in results]
                working.loc[indices, column] = values
                if column not in existing_columns:
                    new_columns.append(
                        {
                            "name": column,
                            "operation": "summarize",
                            "source": field,
                        }
                    )
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
                    return preview_classify(text, labels, params)
                except RuntimeError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            working[output_col] = working[field].apply(classify_value)
            new_columns.append(
                {
                    "name": output_col,
                    "operation": "classify",
                    "source": field,
                    "labels": list(labels),
                }
            )
            provenance_keys.update({f"op:{op.kind}", f"field:{field}"})

    if provenance_keys:
        keys = sorted(provenance_keys)
        bump_frequency(keys)
        bump_recency(keys)

    return working, new_columns


def _sample_dataframe(df: pd.DataFrame, dataset: CsvDataset) -> pd.DataFrame:
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


def create_todo(payload: TodoCreate, service: TodoService = Depends(get_service)) -> TodoRead:
    created = service.create(payload)
    return TodoRead(**created)


def complete_todo(todo_id: int, service: TodoService = Depends(get_service)) -> TodoRead:
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
