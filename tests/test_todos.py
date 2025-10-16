"""Tests for the todo endpoints."""

from backend import create_app
from backend.services import TodoService
from fastapi.testclient import TestClient


def make_client(seed=None) -> tuple[TestClient, TodoService]:
    service = TodoService(seed=seed)
    app = create_app(todo_service=service)
    return TestClient(app), service


def test_list_todos_returns_seed_data() -> None:
    client, _ = make_client(
        seed=[{"title": "Bootstrap MVP", "description": "Create the first todo", "completed": True}]
    )

    response = client.get("/todos")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Bootstrap MVP"
    assert data[0]["completed"] is True


def test_create_todo_adds_new_record() -> None:
    client, _ = make_client()

    payload = {"title": "Write tests", "description": "Ensure API coverage"}
    response = client.post("/todos", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == payload["title"]
    assert body["completed"] is False

    list_response = client.get("/todos")
    assert any(todo["id"] == body["id"] for todo in list_response.json())


def test_complete_todo_marks_item_done() -> None:
    client, _ = make_client()

    created = client.post("/todos", json={"title": "Ship feature"}).json()

    complete_response = client.post(f"/todos/{created['id']}/complete")

    assert complete_response.status_code == 200
    assert complete_response.json()["completed"] is True


def test_completing_missing_todo_returns_not_found() -> None:
    client, _ = make_client()

    response = client.post("/todos/999/complete")

    assert response.status_code == 404
    assert response.json()["detail"] == "Todo with id 999 not found"
