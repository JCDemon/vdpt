"""Shared lightweight data structures for backend/app."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


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
