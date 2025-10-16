"""Domain services for the VDTP API."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .schemas import TodoCreate


def _model_dump(model: TodoCreate) -> Dict[str, Any]:
    """Return a dictionary representation of a Pydantic model."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class TodoService:
    """Simple in-memory todo service used by the API."""

    def __init__(self, seed: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        self._todos: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        if seed:
            for item in seed:
                self._append_seed(item)

    def _append_seed(self, item: Dict[str, Any]) -> None:
        normalized = {
            "title": item.get("title", "Untitled"),
            "description": item.get("description"),
            "completed": bool(item.get("completed", False)),
        }
        normalized["id"] = self._next_id
        self._todos[self._next_id] = normalized
        self._next_id += 1

    def list_todos(self) -> List[Dict[str, Any]]:
        """Return all todos ordered by their identifier."""

        return [dict(todo) for _, todo in sorted(self._todos.items(), key=lambda entry: entry[0])]

    def create(self, payload: TodoCreate) -> Dict[str, Any]:
        """Create a new todo from the provided payload."""

        data = _model_dump(payload)
        todo = {
            "id": self._next_id,
            "title": data["title"],
            "description": data.get("description"),
            "completed": False,
        }
        self._todos[self._next_id] = todo
        self._next_id += 1
        return dict(todo)

    def complete(self, todo_id: int) -> Dict[str, Any]:
        """Mark a todo as completed."""

        if todo_id not in self._todos:
            raise KeyError(todo_id)
        todo = dict(self._todos[todo_id])
        todo["completed"] = True
        self._todos[todo_id] = todo
        return dict(todo)

    def reset(self) -> None:
        """Clear the service state."""

        self._todos.clear()
        self._next_id = 1


__all__ = ["TodoService"]
