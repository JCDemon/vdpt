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


class TextEmbedOperation(_OperationBase):
    """Generate embeddings for a text field."""

    kind: Literal["embed_text"] = "embed_text"
    field: str = "text"
    output_field: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"field": self.field}
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


class ImageEmbedOperation(_OperationBase):
    """Generate embeddings for an image field."""

    kind: Literal["embed_image"] = "embed_image"
    field: str = "image_path"
    output_field: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"field": self.field}
        if self.output_field:
            params["output_field"] = self.output_field
        return {"kind": self.kind, "params": params}


class MaskEmbedOperation(_OperationBase):
    """Generate embeddings for segmentation masks."""

    kind: Literal["embed_masks"] = "embed_masks"
    output_field: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {}
        if self.output_field:
            params["output_field"] = self.output_field
        return {"kind": self.kind, "params": params}


class SamSegmentOperation(_OperationBase):
    """Run SAM automatic segmentation for an image."""

    kind: Literal["sam_segment"] = "sam_segment"
    max_masks: Optional[PositiveInt] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {}
        if self.max_masks is not None:
            params["max_masks"] = int(self.max_masks)
        return {"kind": self.kind, "params": params}


class ClipSegOperation(_OperationBase):
    """Run CLIPSeg text-prompted segmentation for an image."""

    kind: Literal["clipseg_segment"] = "clipseg_segment"
    prompt: Optional[str] = None
    threshold: Optional[float] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {}
        if self.prompt is not None:
            params["prompt"] = self.prompt
        if self.threshold is not None:
            params["threshold"] = float(self.threshold)
        return {"kind": self.kind, "params": params}


class UmapOperation(_OperationBase):
    """Project embeddings into 2D using UMAP."""

    kind: Literal["umap"] = "umap"
    source: str = "embedding"
    output_field: Optional[str] = None
    n_neighbors: Optional[int] = None
    min_dist: Optional[float] = None
    metric: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"source": self.source}
        if self.output_field:
            params["output_field"] = self.output_field
        if self.n_neighbors is not None:
            params["n_neighbors"] = int(self.n_neighbors)
        if self.min_dist is not None:
            params["min_dist"] = float(self.min_dist)
        if self.metric:
            params["metric"] = self.metric
        return {"kind": self.kind, "params": params}


class HdbscanOperation(_OperationBase):
    """Cluster reduced coordinates using HDBSCAN."""

    kind: Literal["hdbscan"] = "hdbscan"
    source: str = "umap"
    output_field: Optional[str] = None
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    metric: Optional[str] = None

    def runtime_payload(self) -> Dict[str, object]:
        params: Dict[str, object] = {"source": self.source}
        if self.output_field:
            params["output_field"] = self.output_field
        if self.min_cluster_size is not None:
            params["min_cluster_size"] = int(self.min_cluster_size)
        if self.min_samples is not None:
            params["min_samples"] = int(self.min_samples)
        if self.metric:
            params["metric"] = self.metric
        return {"kind": self.kind, "params": params}


Operation = Annotated[
    Union[
        FieldOperation,
        SummarizeOperation,
        TextEmbedOperation,
        ImageCaptionOperation,
        ImageResizeOperation,
        ImageEmbedOperation,
        MaskEmbedOperation,
        SamSegmentOperation,
        ClipSegOperation,
        UmapOperation,
        HdbscanOperation,
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
            "csv": {"field", "summarize", "embed_text", "umap", "hdbscan"},
            "images": {
                "img_caption",
                "img_resize",
                "embed_image",
                "embed_masks",
                "sam_segment",
                "clipseg_segment",
                "umap",
                "hdbscan",
            },
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
    "ImageEmbedOperation",
    "MaskEmbedOperation",
    "SamSegmentOperation",
    "ClipSegOperation",
    "Operation",
    "Plan",
    "SummarizeOperation",
    "TextEmbedOperation",
    "UmapOperation",
    "HdbscanOperation",
]
