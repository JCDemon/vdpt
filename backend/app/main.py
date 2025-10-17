"""Main FastAPI application for the VDPT backend."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..schemas import TodoCreate, TodoRead
from ..services import TodoService
from .preview.preview_engine import preview_operation
from .provenance import recorder


class Operation(BaseModel):
    """Representation of a single operation request."""

    kind: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class OperationsRequest(BaseModel):
    """Payload schema for preview/execute requests."""

    ops: list[Operation] = Field(default_factory=list)


def create_app(todo_service: TodoService | None = None) -> FastAPI:
    """Create a configured FastAPI application."""

    service = todo_service or TodoService()
    app = FastAPI(title="VDPT API", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> TodoService:
        return service

    @app.get("/health", tags=["health"])
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/preview", tags=["preview"])
    def preview(payload: OperationsRequest) -> dict[str, Any]:
        details = {str(idx): preview_operation(op.dict()) for idx, op in enumerate(payload.ops)}
        summary = f"dry-run ok; {len(payload.ops)} operation(s) would change data"
        return {"summary": summary, "details": details}

    @app.post("/execute", tags=["execute"])
    def execute(payload: OperationsRequest) -> dict[str, Any]:
        if payload.ops:
            kinds = [op.kind for op in payload.ops]
            recorder.bump_frequency(kinds)
            recorder.bump_recency(kinds)
        return {"ok": True, "applied": len(payload.ops)}

    @app.get("/provenance/snapshot", tags=["provenance"])
    def provenance_snapshot() -> dict[str, Any]:
        return recorder.snapshot()

    @app.get("/todos", response_model=list[TodoRead], tags=["todos"])
    def list_todos(todo_service: TodoService = Depends(get_service)) -> list[TodoRead]:
        return [TodoRead(**todo) for todo in todo_service.list_todos()]

    @app.post("/todos", response_model=TodoRead, status_code=status.HTTP_201_CREATED, tags=["todos"])
    def create_todo(payload: TodoCreate, todo_service: TodoService = Depends(get_service)) -> TodoRead:
        created = todo_service.create(payload)
        return TodoRead(**created)

    @app.post("/todos/{todo_id}/complete", response_model=TodoRead, tags=["todos"])
    def complete_todo(todo_id: int, todo_service: TodoService = Depends(get_service)) -> TodoRead:
        try:
            completed = todo_service.complete(todo_id)
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Todo with id {todo_id} not found",
            ) from exc
        return TodoRead(**completed)

    return app


app = create_app()
