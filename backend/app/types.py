"""Shared lightweight data structures for backend/app."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, List

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
    """Segmentation mask persisted as an artifact on disk."""

    id: str
    image_path: str
    mask_path: str
    bbox: Tuple[int, int, int, int]
    area: int
    score: Optional[float] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    rle_path: Optional[str] = None
    polygon_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        fallback_image_path: Optional[str] = None,
    ) -> "Mask":
        bbox = _coerce_bbox(payload.get("bbox"))
        area = _coerce_int(payload.get("area"))
        score = _coerce_float(payload.get("score"))
        mask_path = str(payload.get("mask_path") or payload.get("path") or "")
        image_path = str(payload.get("image_path") or fallback_image_path or "")
        return cls(
            id=str(payload.get("id") or payload.get("mask_id") or Path(mask_path).stem or "mask"),
            image_path=image_path,
            mask_path=mask_path,
            bbox=bbox,
            area=area,
            score=score,
            prompt=payload.get("prompt"),
            model=payload.get("model"),
            rle_path=payload.get("rle_path"),
            polygon_path=payload.get("polygon_path"),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass
class MaskFeature(Mask):
    """Mask enriched with embedding, projection, and clustering metadata."""

    embedding: List[float] = field(default_factory=list)
    umap: Optional[Tuple[float, float]] = None
    cluster: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["embedding"] = list(self.embedding)
        if self.umap is not None:
            payload["umap"] = [float(val) for val in self.umap]
        if self.cluster is not None:
            payload["cluster"] = int(self.cluster)
        return payload


def _coerce_bbox(value: Any) -> Tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        coords: List[int] = []
        for i in range(4):
            try:
                coords.append(int(float(value[i])))
            except (TypeError, ValueError):
                coords.append(0)
        x0, y0, x1, y1 = coords[:4]
        return (x0, y0, x1, y1)
    return (0, 0, 0, 0)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
