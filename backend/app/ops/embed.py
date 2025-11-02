from __future__ import annotations
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from .base import OperationHandler
from .registry import register
from ..types import Mask, MaskFeature


logger = logging.getLogger(__name__)

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


def _resolve_mask_input(
    mask: Mask | Mapping[str, Any],
) -> tuple[str, Optional[Sequence[float]], str]:
    if isinstance(mask, Mask):
        if not mask.mask_path:
            raise ValueError("Mask object is missing 'mask_path'")
        mask_id = mask.id or Path(mask.mask_path).stem
        return mask.mask_path, mask.bbox, mask_id
    if isinstance(mask, Mapping):
        mask_path = mask.get("mask_path") or mask.get("path")
        if not mask_path:
            raise ValueError("Mask mapping requires 'mask_path'")
        bbox = mask.get("bbox")
        mask_id_value = mask.get("id")
        mask_id = str(mask_id_value) if mask_id_value is not None else Path(str(mask_path)).stem
        return str(mask_path), bbox, mask_id
    raise TypeError("mask must be a Mask or mapping with mask metadata")


def _clamp_bbox(
    bbox: Optional[Sequence[float]], width: int, height: int
) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return 0, 0, width, height
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0, min(width, int(math.floor(x1))))
    y1 = max(0, min(height, int(math.floor(y1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y2))))
    return x1, y1, x2, y2


def _resolve_mask_path(mask_path: str, image_path: Path) -> Path:
    path = Path(mask_path)
    if path.is_absolute() and path.exists():
        return path
    candidates = [path, image_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path.resolve()


def _apply_mask(
    image: Image.Image, mask_path: Path, bbox: Optional[Sequence[float]]
) -> Image.Image:
    width, height = image.size
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=getattr(Image, "NEAREST", Image.NEAREST))
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)
    image_array = np.asarray(image, dtype=np.float32)
    masked = image_array * mask_array[..., None]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    masked = masked[y1:y2, x1:x2, :]
    if masked.size == 0:
        masked = np.zeros((1, 1, 3), dtype=np.float32)
    masked = np.clip(masked, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(masked, mode="RGB")


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


def embed_mask(image_path: str | Path, mask: Mask | Mapping[str, Any]) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found at {path}")
    mask_path_raw, bbox, _ = _resolve_mask_input(mask)
    mask_path = _resolve_mask_path(mask_path_raw, path)
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found at {mask_path}")

    image = _load_image(path)
    masked_image = _apply_mask(image, mask_path, bbox)
    model, _, preprocess, device = _load_model()
    image_tensor = preprocess(masked_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features.float()
        image_features = _normalize_tensor(image_features)
    vector = image_features[0].detach().cpu().numpy().astype(np.float32)
    return vector


def _sanitize_mask_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "mask"


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


class MaskEmbedHandler(OperationHandler):
    kind = "embed_masks"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("embed_masks preview requires execution mode with an output directory")

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        if out_dir is None:
            raise ValueError("embed_masks requires an output directory")

        image_path = row.get("image_path") or row.get("path")
        if not image_path:
            raise ValueError("embed_masks requires 'image_path' in the row")

        source_field = str(params.get("source") or params.get("field") or "masks")
        mask_entries = row.get(source_field)
        if mask_entries is None:
            raise ValueError(f"embed_masks requires '{source_field}' in the row")
        if not isinstance(mask_entries, list):
            raise ValueError(f"embed_masks expects '{source_field}' to be a list of masks")

        output_field = str(params.get("output_field") or "mask_embedding")
        embedding_dir = Path(out_dir) / "mask_embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)

        features: list[MaskFeature] = []
        used_names: set[str] = set()
        image_path_obj = Path(image_path)

        for index, mask_entry in enumerate(mask_entries):
            try:
                mask_path_raw, bbox, mask_id = _resolve_mask_input(mask_entry)
                resolved_mask_path = _resolve_mask_path(mask_path_raw, image_path_obj)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Skipping mask %s: %s", index, exc)
                continue

            try:
                vector = embed_mask(image_path_obj, mask_entry)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("embed_mask failed for '%s': %s", mask_id, exc)
                continue

            sanitized_base = _sanitize_mask_name(f"{image_path_obj.stem}_{mask_id}")
            filename = sanitized_base
            counter = 1
            while filename in used_names:
                filename = f"{sanitized_base}_{counter}"  # pragma: no cover - rare collision
                counter += 1
            used_names.add(filename)

            embedding_path = embedding_dir / f"{filename}.npy"
            np.save(embedding_path, vector.astype(np.float32))

            metadata = {
                "mask_path": str(resolved_mask_path),
                "source_field": source_field,
            }
            feature = MaskFeature(
                mask_id=str(mask_id),
                embedding_path=str(embedding_path),
                vector=_serialize_vector(vector),
                metadata=metadata,
            )
            features.append(feature)

        payload = {
            output_field: [feature.to_dict() for feature in features],
            "mask_embedding_dir": str(embedding_dir),
        }
        return payload


register(TextEmbedHandler())
register(ImageEmbedHandler())
register(MaskEmbedHandler())
