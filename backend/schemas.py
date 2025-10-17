"""Pydantic schemas for the VDPT API."""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for the health check response."""

    status: str = Field(..., description="Status indicator for the API")


class TodoBase(BaseModel):
    """Common fields for todo payloads."""

    title: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=500)


class TodoCreate(TodoBase):
    """Schema for todo creation requests."""

    pass


class TodoRead(TodoBase):
    """Schema for todos returned to clients."""

    id: int = Field(..., ge=1)
    completed: bool = Field(False, description="Whether the todo is marked complete")


__all__ = [
    "HealthResponse",
    "TodoBase",
    "TodoCreate",
    "TodoRead",
]
