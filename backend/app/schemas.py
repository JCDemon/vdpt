"""Request schemas for the VDPT API."""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class CsvDataset(BaseModel):
    """CSV dataset request payload."""

    type: Literal["csv"]
    path: str
    sample_size: Optional[int] = Field(default=5, ge=1)
    random_sample: bool = False
    random_seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_kind(cls, values: object) -> object:
        if isinstance(values, dict):
            kind = values.pop("kind", None)
            if kind and "type" not in values:
                values["type"] = kind
        return values


class ImageDataset(BaseModel):
    """Image dataset request payload."""

    type: Literal["images"]
    session: Optional[str] = None
    path: Optional[str] = None
    paths: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = Field(default=None, ge=1)
    random_sample: bool = False
    random_seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_kind(cls, values: object) -> object:
        if isinstance(values, dict):
            kind = values.pop("kind", None)
            if kind and "type" not in values:
                values["type"] = kind
        return values


Dataset = Annotated[Union[CsvDataset, ImageDataset], Field(discriminator="type")]


__all__ = ["CsvDataset", "ImageDataset", "Dataset"]
