"""Pydantic request models for the /preview endpoint."""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional, Set, Union
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from ..schemas import Dataset


class _OperationBase(BaseModel):
    """Common validation logic for preview operations."""

    kind: str

    @model_validator(mode="before")
    @classmethod
    def _ensure_kind(cls, values: object) -> object:
        if isinstance(values, dict) and "kind" in values:
            return values
        raise ValueError("operation requires 'kind' discriminator")

    def runtime_payload(self) -> Dict[str, object]:  # pragma: no cover - overridden
        return {"kind": self.kind, "params": {}}


class FieldOperation(_OperationBase):
    """Expose a dataset field in the preview output."""

    kind: Literal["field"] = "field"
    field: str
    output_field: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"field": self.field}
        if self.output_field:
            params["output_field"] = self.output_field
        return {"kind": self.kind, "params": params}


class SummarizeOperation(_OperationBase):
    """Summarize a text field."""

    kind: Literal["summarize"] = "summarize"
    field: str = "text"
    instructions: Optional[str] = None
    max_tokens: Optional[PositiveInt] = None
    temperature: Optional[float] = None
    output_field: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"field": self.field}
        if self.instructions is not None:
            params["instructions"] = self.instructions
        if self.max_tokens is not None:
            params["max_tokens"] = int(self.max_tokens)
        if self.temperature is not None:
            params["temperature"] = float(self.temperature)
        if self.output_field:
            params["output_field"] = self.output_field
        return {"kind": self.kind, "params": params}


class ImageCaptionOperation(_OperationBase):
    """Generate captions for image previews."""

    kind: Literal["img_caption"] = "img_caption"
    instructions: Optional[str] = None
    max_tokens: Optional[PositiveInt] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {}
        if self.instructions is not None:
            params["instructions"] = self.instructions
        if self.max_tokens is not None:
            params["max_tokens"] = int(self.max_tokens)
        return {"kind": self.kind, "params": params}


class ImageResizeOperation(_OperationBase):
    """Resize images for preview."""

    kind: Literal["img_resize"] = "img_resize"
    width: PositiveInt
    height: Optional[PositiveInt] = None
    keep_ratio: bool = False

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"width": int(self.width)}
        if self.height is not None:
            params["height"] = int(self.height)
        if self.keep_ratio:
            params["keep_ratio"] = self.keep_ratio
        return {"kind": self.kind, "params": params}


Operation = Annotated[
    Union[
        FieldOperation,
        SummarizeOperation,
        ImageCaptionOperation,
        ImageResizeOperation,
    ],
    Field(discriminator="kind"),
]


class Plan(BaseModel):
    """Preview request payload parsed by FastAPI."""

    dataset: Dataset
    operations: List[Operation] = Field(default_factory=list)
    preview_sample_size: Optional[PositiveInt] = Field(default=None)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_ops(cls, values: object) -> object:
        if isinstance(values, dict):
            if "operations" not in values and "ops" in values:
                values = dict(values)
                values["operations"] = values.pop("ops")
        return values

    def runtime_operations_for(self, dataset_kind: str) -> List[Dict[str, object]]:
        runtime: List[Dict[str, object]] = []
        allowed: Dict[str, Set[str]] = {
            "csv": {"field", "summarize"},
            "images": {"img_caption", "img_resize"},
        }
        permitted = allowed.get(dataset_kind, set())
        for operation in self.operations:
            payload = operation.runtime_payload()
            if payload["kind"] not in permitted:
                continue
            runtime.append(payload)
        return runtime


__all__ = [
    "FieldOperation",
    "ImageCaptionOperation",
    "ImageResizeOperation",
    "Operation",
    "Plan",
    "SummarizeOperation",
]
