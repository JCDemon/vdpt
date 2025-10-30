from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from .base import OperationHandler
from .registry import register

_MODEL_NAME = "ViT-L-14"
_PRETRAINED = "openai"
_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER = None
_PREPROCESS = None
_DEVICE: Optional[torch.device] = None


def _load_model() -> tuple[torch.nn.Module, Any, Any, torch.device]:
    global _MODEL, _TOKENIZER, _PREPROCESS, _DEVICE
    if _MODEL is None or _TOKENIZER is None or _PREPROCESS is None or _DEVICE is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _MODEL = model
        _TOKENIZER = tokenizer
        _PREPROCESS = preprocess
        _DEVICE = device
    return _MODEL, _TOKENIZER, _PREPROCESS, _DEVICE  # type: ignore[return-value]


def _normalize_tensor(features: torch.Tensor) -> torch.Tensor:
    norm = features.norm(dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return features / norm


def embed_text(text: str) -> np.ndarray:
    model, tokenizer, _, device = _load_model()
    tokenized = tokenizer([text])
    tokenized = tokenized.to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = text_features.float()
        text_features = _normalize_tensor(text_features)
    vector = text_features[0].detach().cpu().numpy().astype(np.float32)
    return vector


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def embed_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found at {path}")
    model, _, preprocess, device = _load_model()
    image = _load_image(path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features.float()
        image_features = _normalize_tensor(image_features)
    vector = image_features[0].detach().cpu().numpy().astype(np.float32)
    return vector


def _coerce_field_name(params: Dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = params.get(key)
        if value:
            return str(value)
    return default


def _serialize_vector(vector: Sequence[float]) -> list[float]:
    return [float(value) for value in vector]


class TextEmbedHandler(OperationHandler):
    kind = "embed_text"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(row, params)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._run(row, params)

    def _run(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        field = _coerce_field_name(params, "field", "source", default="text")
        output_field = str(params.get("output_field") or "embedding")
        value = row.get(field)
        text = "" if value is None else str(value)
        vector = embed_text(text)
        return {output_field: _serialize_vector(vector)}


class ImageEmbedHandler(OperationHandler):
    kind = "embed_image"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(row, params)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._run(row, params)

    def _run(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        field = _coerce_field_name(params, "field", "path", default="image_path")
        output_field = str(params.get("output_field") or "embedding")
        value = row.get(field)
        if not value:
            raise ValueError("embed_image requires an image path")
        vector = embed_image(Path(value))
        return {output_field: _serialize_vector(vector)}


register(TextEmbedHandler())
register(ImageEmbedHandler())
