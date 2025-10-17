from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Literal

from ..schemas import TodoCreate, TodoRead
from ..services import TodoService
from .provenance.recorder import bump_frequency, bump_recency, snapshot
from .preview.preview_engine import preview_operation

app = FastAPI(title="VDPT API", version="0.1.0")

todo_service = TodoService()


def get_service() -> TodoService:
    return todo_service


class Operation(BaseModel):
    kind: Literal["segment", "embed", "filter", "map", "summarize", "classify", "transform"]
    params: Dict[str, Any]


class Plan(BaseModel):
    ops: List[Operation]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/provenance/snapshot")
def provenance_snapshot():
    return snapshot()


@app.post("/preview")
def preview(plan: Plan):
    details: Dict[str, Any] = {}
    for i, op in enumerate(plan.ops):
        details[str(i)] = preview_operation({"kind": op.kind, "params": op.params})
    return {
        "summary": f"dry-run ok; {len(plan.ops)} operation(s) would change data",
        "details": details,
    }


@app.post("/execute")
def execute(plan: Plan):
    keys = [op.kind for op in plan.ops]
    bump_frequency(keys)
    bump_recency(keys)
    return {"ok": True, "applied": len(plan.ops)}


@app.get("/todos", response_model=List[TodoRead])
def list_todos(service: TodoService = Depends(get_service)) -> List[TodoRead]:
    return [TodoRead(**todo) for todo in service.list_todos()]


@app.post("/todos", response_model=TodoRead, status_code=status.HTTP_201_CREATED)
def create_todo(payload: TodoCreate, service: TodoService = Depends(get_service)) -> TodoRead:
    created = service.create(payload)
    return TodoRead(**created)


@app.post("/todos/{todo_id}/complete", response_model=TodoRead)
def complete_todo(todo_id: int, service: TodoService = Depends(get_service)) -> TodoRead:
    try:
        completed = service.complete(todo_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo with id {todo_id} not found",
        ) from exc
    return TodoRead(**completed)
