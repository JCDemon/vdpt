from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

from ..types import Mask, MaskFeature
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
    image = _load_image(path)
    return _encode_image(image)


def embed_mask(image_path: str | Path, mask_payload: Mask | Dict[str, Any]) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found at {path}")

    if isinstance(mask_payload, Mask):
        mask = mask_payload
    elif isinstance(mask_payload, dict):
        mask = Mask.from_payload(mask_payload, fallback_image_path=str(path))
    else:
        raise TypeError("mask_payload must be a Mask or dict")

    if not mask.mask_path:
        raise ValueError("embed_mask requires 'mask_path'")

    mask_path = Path(mask.mask_path)
    if not mask_path.is_absolute():
        mask_path = mask_path
    if not mask_path.exists():
        raise FileNotFoundError(f"mask image not found at {mask_path}")

    image = _load_image(path)
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.NEAREST)

    mask_array = np.array(mask_image)
    if not np.any(mask_array):
        return embed_image(path)

    bbox = _normalize_bbox(mask.bbox, image.size)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        ys, xs = np.where(mask_array > 0)
        if xs.size and ys.size:
            bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            )
        else:
            bbox = (0, 0, image.width, image.height)

    cropped_image = image.crop(bbox)
    cropped_mask = mask_image.crop(bbox)
    cropped_array = np.array(cropped_image)
    cropped_mask_array = np.array(cropped_mask)
    cropped_array[cropped_mask_array <= 0] = 0
    masked_image = Image.fromarray(cropped_array)
    return _encode_image(masked_image)


def _coerce_field_name(params: Dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = params.get(key)
        if value:
            return str(value)
    return default


def _serialize_vector(vector: Sequence[float]) -> list[float]:
    return [float(value) for value in vector]


def _encode_image(image: Image.Image) -> np.ndarray:
    model, _, preprocess, device = _load_model()
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features.float()
        image_features = _normalize_tensor(image_features)
    vector = image_features[0].detach().cpu().numpy().astype(np.float32)
    return vector


def _normalize_bbox(
    bbox: Tuple[int, int, int, int], size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width, int(x0)))
    y0 = max(0, min(height, int(y0)))
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    return (x0, y0, x1, y1)


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


class MaskEmbedHandler(OperationHandler):
    kind = "embed_masks"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(row, params)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._run(row, params)

    def _run(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = row.get("image_path")
        if not image_path:
            raise ValueError("embed_masks requires 'image_path' in the row")

        masks_payload = row.get("masks")
        if not isinstance(masks_payload, list) or not masks_payload:
            return {"mask_features": []}

        features: list[dict[str, Any]] = []
        for mask_payload in masks_payload:
            mask_obj = Mask.from_payload(mask_payload or {}, fallback_image_path=str(image_path))
            try:
                vector = embed_mask(image_path, mask_obj)
            except Exception as exc:  # pragma: no cover - defensive
                feature = mask_obj.to_dict()
                feature["embedding"] = []
                feature.setdefault("extra", {})["embedding_error"] = str(exc)
                features.append(feature)
                continue
            mask_feature = MaskFeature(
                id=mask_obj.id,
                image_path=mask_obj.image_path or str(image_path),
                mask_path=mask_obj.mask_path,
                bbox=mask_obj.bbox,
                area=mask_obj.area,
                score=mask_obj.score,
                prompt=mask_obj.prompt,
                model=mask_obj.model,
                rle_path=mask_obj.rle_path,
                polygon_path=mask_obj.polygon_path,
                extra=dict(mask_obj.extra),
                embedding=_serialize_vector(vector),
            )
            features.append(mask_feature.to_dict())

        return {"mask_features": features}


register(TextEmbedHandler())
register(ImageEmbedHandler())
register(MaskEmbedHandler())
