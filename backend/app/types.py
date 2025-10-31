"""Shared lightweight data structures for backend/app."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Sequence

from pydantic import BaseModel, Field


@dataclass
class RunSummary:
    """Summary metadata for a recorded run."""

    run_id: str
    started_at: Optional[datetime] = None
    status: Optional[str] = None
    num_artifacts: Optional[int] = None
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, run_id: str, metadata: Dict[str, Any]) -> "RunSummary":
        started_at = _parse_datetime(metadata.get("started_at")) if metadata else None
        num_artifacts = metadata.get("num_artifacts") if metadata else None
        status = metadata.get("status") if metadata else None
        description = metadata.get("description") if metadata else None
        extra = dict(metadata) if metadata else {}
        return cls(
            run_id=run_id,
            started_at=started_at,
            status=status,
            num_artifacts=num_artifacts,
            description=description,
            extra=extra,
        )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Support timestamps stored with a trailing ``Z``.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class ProvenanceNode(BaseModel):
    """Node within a provenance graph using a PROV-inspired vocabulary."""

    id: str
    type: Literal["Entity", "Activity", "Agent"]
    label: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceEdge(BaseModel):
    """Directed edge describing relationships between provenance nodes."""

    source: str
    target: str
    relation: Literal["wasGeneratedBy", "used"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """Structured log entry captured during a preview or execute run."""

    ts: datetime
    level: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class Mask:
    """Lightweight representation of a segmentation mask artifact."""

    id: str
    bbox: Sequence[float]
    mask_path: Optional[str] = None
    rle_path: Optional[str] = None
    area: Optional[int] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        bbox_list = [float(value) for value in self.bbox]
        payload: Dict[str, Any] = {"id": self.id, "bbox": bbox_list}
        if self.mask_path is not None:
            payload["mask_path"] = self.mask_path
        if self.rle_path is not None:
            payload["rle_path"] = self.rle_path
        if self.area is not None:
            payload["area"] = int(self.area)
        if self.score is not None:
            payload["score"] = float(self.score)
        if self.metadata:
            payload["metadata"] = _ensure_jsonable(self.metadata)
        return payload


@dataclass
class MaskFeature:
    """Embedding or analytic features associated with a mask."""

    mask_id: str
    embedding_path: Optional[str] = None
    vector: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"mask_id": self.mask_id}
        if self.embedding_path is not None:
            payload["embedding_path"] = self.embedding_path
        if self.vector is not None:
            payload["vector"] = [float(value) for value in self.vector]
        if self.metadata:
            payload["metadata"] = _ensure_jsonable(self.metadata)
        return payload


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _ensure_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
