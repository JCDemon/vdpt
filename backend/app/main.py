import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ..schemas import TodoCreate, TodoRead
from backend.vdpt import providers
from ..vdpt.io import read_csv, sha256_bytes, write_csv
from ..services import TodoService
from .datasets import DatasetLoader, registry
from .preview.preview_engine import (
    preview_classify,
    preview_dataset,
    preview_operation,
    start_provenance_run,
    run_operation,
)
from .preview.schemas import Plan as PreviewRequestPlan
from .provenance.recorder import bump_frequency, bump_recency, snapshot
from .schemas import CsvDataset, Dataset, ImageDataset

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .preview.preview_engine import _ProvenanceRun

__version__ = "0.1.0"


todo_service = TodoService()

_dataset_scan_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[Dict[str, Any]]] = {}
_dataset_preview_cache: Dict[
    Tuple[str, Tuple[Tuple[str, Any], ...], int],
    List[Dict[str, Any]],
] = {}


def _normalize_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _normalize_cache_value(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_cache_value(item) for item in value)
    return value


def _make_cache_key(loader_id: str, config: Dict[str, Any], limit: Optional[int] = None):
    config_key = tuple(
        sorted((key, _normalize_cache_value(value)) for key, value in config.items())
    )
    if limit is None:
        return loader_id, config_key
    return loader_id, config_key, limit


def _instantiate_loader(loader_id: str, config: Dict[str, Any]) -> DatasetLoader:
    try:
        loader_cls = registry.get(loader_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        return loader_cls(**config)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


class DatasetPreviewRequest(BaseModel):
    loader: str
    config: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=12, ge=1, le=64)
    refresh: bool = False


def get_service() -> TodoService:
    return todo_service


def list_dataset_loaders() -> Dict[str, Any]:
    loaders = [loader_cls.metadata() for loader_cls in registry.all()]
    return {"loaders": loaders}


def preview_dataset_records(payload: DatasetPreviewRequest) -> Dict[str, Any]:
    loader = _instantiate_loader(payload.loader, payload.config)

    scan_key = _make_cache_key(payload.loader, payload.config)
    if not payload.refresh and scan_key in _dataset_scan_cache:
        scan_results = _dataset_scan_cache[scan_key]
        scan_cached = True
    else:
        try:
            scan_results = list(loader.scan())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        _dataset_scan_cache[scan_key] = scan_results
        scan_cached = False

    preview_key = _make_cache_key(payload.loader, payload.config, payload.limit)
    if not payload.refresh and preview_key in _dataset_preview_cache:
        records = _dataset_preview_cache[preview_key]
        preview_cached = True
    else:
        try:
            records = list(loader.iter_records(limit=payload.limit))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        _dataset_preview_cache[preview_key] = records
        preview_cached = False

    return {
        "loader": payload.loader,
        "config": payload.config,
        "records": records,
        "count": len(records),
        "scan": scan_results,
        "cached": {"scan": scan_cached, "preview": preview_cached},
        "summary": f"returned {len(records)} record(s)",
    }


class Operation(BaseModel):
    kind: Literal[
        "segment",
        "embed",
        "filter",
        "map",
        "summarize",
        "classify",
        "transform",
        "field",
        "embed_text",
        "embed_image",
        "umap",
        "hdbscan",
        "img_resize",
        "img_caption",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    summarize: Optional[Dict[str, Any]] = None
    classify: Optional[Dict[str, Any]] = None


class Plan(BaseModel):
    ops: List[Operation] = Field(default_factory=list)
    dataset: Optional[Dataset] = None


# Backwards compatibility for existing callers instantiating Plan.Dataset
Plan.Dataset = CsvDataset  # type: ignore[attr-defined]
Plan.ImageDataset = ImageDataset  # type: ignore[attr-defined]


def health() -> Dict[str, bool]:
    return {"ok": True}


def provenance_snapshot() -> Dict[str, Any]:
    return snapshot()


def config() -> Dict[str, Any]:
    provider_name = providers.current.__name__.split(".")[-1]
    return {"provider": provider_name, "mock": bool(os.getenv("VDPT_MOCK"))}


def preview_api(plan: PreviewRequestPlan) -> Dict[str, Any]:
    dataset = plan.dataset
    runtime_ops = plan.runtime_operations_for(dataset.type)

    if dataset.type == "csv":
        try:
            df = read_csv(dataset.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        sampling_dataset = dataset
        if plan.preview_sample_size is not None:
            sampling_dataset = dataset.model_copy(update={"sample_size": plan.preview_sample_size})

        sample_df = _sample_dataframe(df, sampling_dataset)
        records = sample_df.to_dict(orient="records")
        preview_result = preview_dataset("csv", records, runtime_ops)

        payload_records = preview_result.get("records", [])
        new_columns = preview_result.get("schema", {}).get("new_columns", [])
        response: Dict[str, Any] = {
            "records": payload_records,
            "schema": {"new_columns": new_columns},
            "preview_sample_size": len(payload_records),
            "summary": f"preview generated for {len(payload_records)} row(s)",
        }

        artifacts = preview_result.get("artifacts") or {}
        response["artifacts"] = artifacts
        response["logs"] = preview_result.get("logs", [])

        return response

    if dataset.type == "images":
        image_paths = _resolve_image_paths(dataset)

        sampling_dataset = dataset
        if plan.preview_sample_size is not None:
            sampling_dataset = dataset.model_copy(update={"sample_size": plan.preview_sample_size})

        sample_paths = _sample_image_paths(image_paths, sampling_dataset)
        records = [{"image_path": str(path)} for path in sample_paths]
        artifacts_dir = _create_artifact_dir()

        preview_result = preview_dataset(
            "images",
            records,
            runtime_ops,
            artifacts_dir=artifacts_dir,
        )

        payload_records = preview_result.get("records", [])
        preview_result["preview_sample_size"] = len(payload_records)
        preview_result["summary"] = (
            f"preview generated for {preview_result['preview_sample_size']} image(s)"
        )
        preview_result.setdefault("logs", [])
        return preview_result

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unsupported dataset type '{dataset.type}'",
    )


def preview(plan: Plan) -> Dict[str, Any]:
    dataset = plan.dataset
    if isinstance(dataset, CsvDataset):
        return _preview_csv(plan)
    if isinstance(dataset, ImageDataset):
        normalized_dataset = _normalize_image_dataset(dataset)
        normalized_plan = plan.model_copy(update={"dataset": normalized_dataset})
        return _preview_images(normalized_plan)

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
        normalized_dataset = _normalize_image_dataset(dataset)
        normalized_plan = plan.model_copy(update={"dataset": normalized_dataset})
        return _execute_images(normalized_plan)

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

    run_dir, provenance = start_provenance_run()
    processed_df, _ = _apply_tabular_ops(
        df.copy(),
        plan.ops,
        out_dir=run_dir,
        provenance=provenance,
    )
    output_path = run_dir / "output.csv"
    write_csv(processed_df, output_path)

    preview_payload = _preview_csv(plan)
    preview_path = run_dir / "preview.json"
    preview_path.write_text(json.dumps(preview_payload, indent=2))

    dataset_hash = sha256_bytes(csv_path.read_bytes())
    metadata_path = run_dir / "metadata.json"
    metadata = {
        "plan": plan.model_dump(),
        "dataset_hash": dataset_hash,
        "artifacts": {
            "output_csv": str(output_path),
            "preview_json": str(preview_path),
        },
        "preview_sample_size": preview_payload.get("preview_sample_size"),
    }

    provenance_path = provenance.write()
    metadata["artifacts"]["provenance"] = str(provenance_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "ok": True,
        "applied": len(plan.ops),
        "artifacts": {
            "output_csv": str(output_path),
            "metadata": str(metadata_path),
            "preview": str(preview_path),
            "provenance": str(provenance_path),
        },
        "logs": provenance.logs,
    }


def _preview_images(plan: Plan) -> Dict[str, Any]:
    dataset = cast(ImageDataset, plan.dataset)
    assert dataset is not None

    image_paths = _resolve_image_paths(dataset)
    sample_paths = _sample_image_paths(image_paths, dataset)
    run_dir, provenance = start_provenance_run()
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
        record_entry = provenance.start_record(index, {"image_path": str(path)})

        for op in plan.ops:
            params = _operation_params(op)
            try:
                result = run_operation(row_context, op.kind, params)
            except Exception as exc:
                provenance.log(
                    "ERROR",
                    f"{op.kind} failed",
                    record_entry=record_entry,
                    context={"error": str(exc)},
                )
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            record.update(result)
            if result:
                provenance.record_operation(record_entry, op.kind, params, result)
            for column in result:
                new_columns.setdefault(column, {"name": column, "operation": op.kind})
            provenance_keys.add(f"op:{op.kind}")

        provenance.finish_record(record_entry, record)
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

    provenance_path = provenance.write()
    result["artifacts"] = {
        "captions": str(captions_path),
        "metadata": str(metadata_path),
        "provenance": str(provenance_path),
    }
    result["logs"] = provenance.logs

    return result


def _execute_images(plan: Plan) -> Dict[str, Any]:
    dataset = cast(ImageDataset, plan.dataset)
    assert dataset is not None

    image_paths = _resolve_image_paths(dataset)
    run_dir, provenance = start_provenance_run()

    records: List[Dict[str, Any]] = []
    provenance_keys: set[str] = set()
    generated_files: set[str] = set()

    for index, path in enumerate(image_paths):
        row_context: Dict[str, Any] = {
            "image_path": str(path),
            "__index__": index,
        }
        record: Dict[str, Any] = {"image_path": str(path)}
        record_entry = provenance.start_record(index, {"image_path": str(path)})

        for op in plan.ops:
            params = _operation_params(op)
            try:
                result = run_operation(row_context, op.kind, params, out_dir=run_dir)
            except Exception as exc:
                provenance.log(
                    "ERROR",
                    f"{op.kind} failed",
                    record_entry=record_entry,
                    context={"error": str(exc)},
                )
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            record.update(result)
            if result:
                provenance.record_operation(record_entry, op.kind, params, result)
            for value in result.values():
                if isinstance(value, str):
                    value_path = Path(value)
                    if value_path.exists() and str(value_path).startswith(str(run_dir)):
                        generated_files.add(str(value_path))
            provenance_keys.add(f"op:{op.kind}")

        provenance.finish_record(record_entry, record)
        records.append(record)

    if provenance_keys:
        keys = sorted(provenance_keys)
        bump_frequency(keys)
        bump_recency(keys)

    if records:
        output_df = pd.DataFrame(records)
    else:
        output_df = pd.DataFrame(columns=["image_path"])

    output_path = run_dir / "output.csv"
    output_df.to_csv(output_path, index=False)

    preview_payload = _preview_images(plan)
    preview_path = run_dir / "preview.json"
    preview_path.write_text(json.dumps(preview_payload, indent=2))

    captions_path = run_dir / "captions.json"
    captions_entries = _collect_image_captions(records)
    captions_path.write_text(json.dumps(captions_entries, ensure_ascii=False, indent=2))

    generated_list = sorted(generated_files)

    metadata_path = run_dir / "metadata.json"
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

    provenance_path = provenance.write()

    artifacts_payload: Dict[str, Any] = {
        "output_csv": str(output_path),
        "metadata": str(metadata_path),
        "preview": str(preview_path),
        "captions": str(captions_path),
        "provenance": str(provenance_path),
    }
    if generated_files:
        artifacts_payload["generated"] = generated_list

    return {
        "ok": True,
        "applied": len(plan.ops),
        "artifacts": artifacts_payload,
        "logs": provenance.logs,
    }


def _operation_params(op: Operation) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(op.params)
    if op.summarize:
        params.update(op.summarize)
    if op.classify:
        params.update(op.classify)
    return params


def _normalize_image_dataset(dataset: ImageDataset) -> ImageDataset:
    updates: Dict[str, Any] = {}

    base_dir: Optional[Path] = None

    if dataset.path:
        base_dir = Path(dataset.path).expanduser()
        if not base_dir.is_absolute():
            base_dir = (Path.cwd() / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
        updates["path"] = str(base_dir)

    if dataset.paths:
        normalized_paths: List[str] = []
        for raw in dataset.paths:
            candidate = Path(raw)
            if candidate.is_absolute():
                normalized_paths.append(str(candidate.resolve()))
                continue

            if dataset.path and base_dir is not None:
                normalized_paths.append(str((base_dir / candidate).resolve()))
                continue

            if dataset.session:
                normalized_paths.append(str(candidate))
                continue

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="relative image paths require dataset.path",
            )

        if normalized_paths != dataset.paths:
            updates["paths"] = normalized_paths

    if updates:
        return dataset.model_copy(update=updates)
    return dataset


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
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="relative image paths require dataset.path",
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
    provenance: "_ProvenanceRun | None" = None,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    working = df.copy()
    new_columns: List[Dict[str, Any]] = []
    provenance_keys: set[str] = set()
    provenance_records: dict[Any, Any] = {}

    if provenance is not None:
        for row_index, (idx, row) in enumerate(working.iterrows()):
            provenance_records[idx] = provenance.start_record(row_index, row.to_dict())

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
                    record_entry = provenance_records.get(idx)
                    if provenance is not None and record_entry is not None:
                        provenance.log(
                            "ERROR",
                            f"summarize failed for row {idx}",
                            record_entry=record_entry,
                            context={"error": str(exc), "field": field},
                        )
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                results.append(result)
                indices.append(idx)
                new_column_names.update(result.keys())
                record_entry = provenance_records.get(idx)
                if provenance is not None and record_entry is not None and result:
                    provenance.record_operation(record_entry, op.kind, params, result)

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
            classification_results: List[tuple[Any, str]] = []

            for idx, row in working.iterrows():
                value = row[field]
                text = "" if pd.isna(value) else str(value)
                try:
                    label_value = preview_classify(text, labels, params)
                except RuntimeError as exc:
                    record_entry = provenance_records.get(idx)
                    if provenance is not None and record_entry is not None:
                        provenance.log(
                            "ERROR",
                            f"classify failed for row {idx}",
                            record_entry=record_entry,
                            context={"error": str(exc), "field": field},
                        )
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                classification_results.append((idx, label_value))
                record_entry = provenance_records.get(idx)
                if provenance is not None and record_entry is not None:
                    provenance.record_operation(
                        record_entry,
                        op.kind,
                        params,
                        {output_col: label_value},
                    )

            working.loc[:, output_col] = pd.NA
            for idx, label_value in classification_results:
                working.at[idx, output_col] = label_value
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

    if provenance is not None:
        for idx, record_entry in provenance_records.items():
            row_values = working.loc[idx].to_dict() if idx in working.index else {}
            provenance.finish_record(record_entry, row_values)

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
    app.get("/config", tags=["config"])(config)
    app.get("/provenance/snapshot", tags=["provenance"])(provenance_snapshot)
    app.get("/datasets/list", tags=["datasets"])(list_dataset_loaders)
    app.post("/datasets/preview", tags=["datasets"])(preview_dataset_records)
    app.post("/preview", tags=["preview"])(preview_api)
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
