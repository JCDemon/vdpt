from typing import Any, Dict, List, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ..schemas import TodoCreate, TodoRead
from ..services import TodoService
from .preview.preview_engine import preview_operation
from .provenance.recorder import bump_frequency, bump_recency, snapshot

__version__ = "0.1.0"

todo_service = TodoService()


def get_service() -> TodoService:
    return todo_service


class Operation(BaseModel):
    kind: Literal["segment", "embed", "filter", "map", "summarize", "classify", "transform"]
    params: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    ops: List[Operation] = Field(default_factory=list)


def health() -> Dict[str, bool]:
    return {"ok": True}


def provenance_snapshot() -> Dict[str, Any]:
    return snapshot()


def preview(plan: Plan) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for i, op in enumerate(plan.ops):
        details[str(i)] = preview_operation({"kind": op.kind, "params": op.params})
    return {
        "summary": f"dry-run ok; {len(plan.ops)} operation(s) would change data",
        "details": details,
    }


def execute(plan: Plan) -> Dict[str, Any]:
    keys = [op.kind for op in plan.ops]
    bump_frequency(keys)
    bump_recency(keys)
    return {"ok": True, "applied": len(plan.ops)}


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
