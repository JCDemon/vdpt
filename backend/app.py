"""Application factory for the VDTP FastAPI app."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import HealthResponse, TodoCreate, TodoRead
from .services import TodoService


def create_app(todo_service: TodoService | None = None) -> FastAPI:
    """Create a configured FastAPI application."""

    service = todo_service or TodoService()
    app = FastAPI(title="VDTP MVP API", version="0.1.0", description="Simple todo API used for validation")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> TodoService:
        return service

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Todo with id {todo_id} not found") from exc
        return TodoRead(**completed)

    return app


app = create_app()
