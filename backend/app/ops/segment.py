"""Segmentation operations using SAM and CLIPSeg backends."""

from __future__ import annotations

import logging
import os
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from ..types import Mask
from .base import OperationHandler
from .registry import register


logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    use_cuda = os.environ.get("SAM_USE_CUDA") in {"1", "true", "True"}
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _coerce_checkpoint(params: Mapping[str, Any]) -> Path:
    checkpoint = (
        params.get("checkpoint_path")
        or params.get("checkpoint")
        or os.environ.get("SAM_CHECKPOINT_PATH")
    )
    if not checkpoint:
        raise ValueError(
            "sam_segment requires 'checkpoint_path' parameter or SAM_CHECKPOINT_PATH environment variable"
        )
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
    return checkpoint_path


def _load_sam_model(checkpoint_path: Path, model_type: str) -> torch.nn.Module:
    key = (str(checkpoint_path), model_type)
    return _load_sam_model_cached(key, checkpoint_path, model_type)


@lru_cache(maxsize=2)
def _load_sam_model_cached(
    key: tuple[str, str], checkpoint_path: Path, model_type: str
) -> torch.nn.Module:
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    device = _select_device()
    sam.to(device)
    sam.eval()
    return sam


def _load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    with Image.open(path) as img:
        image = img.convert("RGB")
        return np.array(image)


_CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"
_CLIPSEG_MODEL: Optional[CLIPSegForImageSegmentation] = None
_CLIPSEG_PROCESSOR: Optional[CLIPSegProcessor] = None
_CLIPSEG_DEVICE: Optional[torch.device] = None


def _load_clipseg(
    model_name: str,
) -> tuple[CLIPSegForImageSegmentation, CLIPSegProcessor, torch.device]:
    global _CLIPSEG_MODEL, _CLIPSEG_PROCESSOR, _CLIPSEG_DEVICE
    if (
        _CLIPSEG_MODEL is None
        or _CLIPSEG_PROCESSOR is None
        or _CLIPSEG_DEVICE is None
        or getattr(_CLIPSEG_MODEL, "name_or_path", None) != model_name
    ):
        device = _select_device()
        processor = CLIPSegProcessor.from_pretrained(model_name)
        model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _CLIPSEG_MODEL = model
        _CLIPSEG_PROCESSOR = processor
        _CLIPSEG_DEVICE = device
    return _CLIPSEG_MODEL, _CLIPSEG_PROCESSOR, _CLIPSEG_DEVICE  # type: ignore[return-value]


def _sam_generator_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "points_per_side",
        "points_per_batch",
        "pred_iou_thresh",
        "stability_score_thresh",
        "crop_n_layers",
        "crop_n_points_downscale_factor",
        "min_mask_region_area",
    }
    provided = params.get("generator") or {}
    if not isinstance(provided, Mapping):
        return {}
    return {key: provided[key] for key in allowed_keys if key in provided}


def _bbox_from_sam(raw_bbox: Iterable[float]) -> tuple[float, float, float, float]:
    values = list(raw_bbox or [])
    if len(values) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    x, y, w, h = values
    return (float(x), float(y), float(x + w), float(y + h))


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    positions = np.argwhere(mask > 0)
    if positions.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)
    return (float(x_min), float(y_min), float(x_max + 1), float(y_max + 1))


def _save_mask(segmentation: np.ndarray, path: Path) -> None:
    binary = np.asarray(segmentation, dtype=np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    image.save(path)


def sam_segment(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path | None,
    output_dir: Optional[Path] = None,
    model_type: str = "vit_b",
    generator_params: Optional[Mapping[str, Any]] = None,
    mock: bool = False,
) -> list[Mask]:
    """Generate segmentation masks for an image using SAM."""

    if mock:
        image = _load_image(image_path)
        height, width = image.shape[:2]
        binary = np.ones((height, width), dtype=np.uint8)
        mask_dir: Optional[Path] = None
        if output_dir is not None:
            mask_dir = Path(output_dir)
            mask_dir.mkdir(parents=True, exist_ok=True)
        mask_id = f"{Path(image_path).stem}_mock"
        mask_path: Optional[str] = None
        if mask_dir is not None:
            mask_file = mask_dir / f"{mask_id}.png"
            _save_mask(binary, mask_file)
            mask_path = str(mask_file)
        logger.info("sam_segment mock mode enabled for '%s'", image_path)
        mask = Mask(
            id=mask_id,
            bbox=(0.0, 0.0, float(width), float(height)),
            mask_path=mask_path,
            area=int(width * height),
            score=1.0,
            metadata={"mock": True},
        )
        return [mask]

    if checkpoint_path is None:
        raise ValueError("sam_segment requires 'checkpoint_path' when mock is disabled")

    checkpoint = Path(checkpoint_path)
    sam = _load_sam_model(checkpoint, model_type)
    generator_kwargs = dict(generator_params or {})
    mask_generator = SamAutomaticMaskGenerator(sam, **generator_kwargs)
    image = _load_image(image_path)
    raw_masks = mask_generator.generate(image)
    masks: list[Mask] = []

    mask_dir: Optional[Path] = None
    if output_dir is not None:
        mask_dir = Path(output_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    for index, raw_mask in enumerate(raw_masks):
        mask_id = f"{stem}_mask_{index:03d}"
        mask_path: Optional[str] = None
        segmentation = raw_mask.get("segmentation")
        if mask_dir is not None and segmentation is not None:
            filename = f"{mask_id}.png"
            path = mask_dir / filename
            _save_mask(segmentation, path)
            mask_path = str(path)
        mask = Mask(
            id=mask_id,
            bbox=_bbox_from_sam(raw_mask.get("bbox", [])),
            mask_path=mask_path,
            area=int(raw_mask.get("area")) if raw_mask.get("area") is not None else None,
            score=(
                float(raw_mask.get("predicted_iou"))
                if raw_mask.get("predicted_iou") is not None
                else None
            ),
            metadata={
                "stability_score": raw_mask.get("stability_score"),
            },
        )
        masks.append(mask)
    if mask_dir is not None and not masks:
        logger.warning("sam_segment produced no masks for image '%s'", Path(image_path).name)
    return masks


def clipseg_segment(
    image_path: str | Path,
    prompt: str,
    *,
    output_dir: Optional[Path] = None,
    model_name: str = _CLIPSEG_MODEL_NAME,
    threshold: float = 0.5,
) -> list[Mask]:
    """Generate a segmentation mask conditioned on a text prompt using CLIPSeg."""

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        raise ValueError("clipseg_segment requires a non-empty prompt")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    model, processor, device = _load_clipseg(model_name)
    with Image.open(path) as pil_image:
        image = pil_image.convert("RGB")

    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    probs = torch.sigmoid(logits)
    mask_tensor = probs[0, 0]
    mask_np = mask_tensor.detach().cpu().numpy()
    binary = (mask_np >= float(threshold)).astype(np.uint8)

    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8]
    mask_id = f"{path.stem}_clipseg_{digest}"
    mask_path: Optional[str] = None
    mask_dir: Optional[Path] = None
    if output_dir is not None:
        mask_dir = Path(output_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

    if binary.max() == 0:
        if mask_dir is not None:
            logger.warning("clipseg_segment produced an empty mask for prompt '%s'", prompt_text)
        return []

    if mask_dir is not None:
        mask_file = mask_dir / f"{mask_id}.png"
        _save_mask(binary, mask_file)
        mask_path = str(mask_file)

    mask = Mask(
        id=mask_id,
        bbox=_bbox_from_mask(binary),
        mask_path=mask_path,
        area=int(binary.sum()),
        score=float(mask_tensor.mean().item()),
        metadata={
            "prompt": prompt_text,
            "threshold": float(threshold),
        },
    )
    return [mask]


class SamSegmentHandler(OperationHandler):
    kind = "sam_segment"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("sam_segment preview requires execution mode with an output directory")

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        if out_dir is None:
            raise ValueError("sam_segment requires an output directory")
        image_path = row.get("image_path") or row.get("path")
        if not image_path:
            raise ValueError("sam_segment requires 'image_path' in the row")

        model_type = str(params.get("model_type") or "vit_b")
        generator_kwargs = _sam_generator_kwargs(params)
        mock = bool(params.get("mock", False))
        checkpoint = None if mock else _coerce_checkpoint(params)

        mask_dir = Path(out_dir) / "masks"
        masks = sam_segment(
            image_path,
            checkpoint_path=checkpoint,
            output_dir=mask_dir,
            model_type=model_type,
            generator_params=generator_kwargs,
            mock=mock,
        )

        output_field = str(params.get("output_field") or "masks")
        payload = {
            output_field: [mask.to_dict() for mask in masks],
            "mask_artifact_dir": str(mask_dir),
        }
        return payload


class ClipSegSegmentHandler(OperationHandler):
    kind = "clipseg_segment"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(
            "clipseg_segment preview requires execution mode with an output directory"
        )

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        if out_dir is None:
            raise ValueError("clipseg_segment requires an output directory")

        image_path = row.get("image_path") or row.get("path")
        if not image_path:
            raise ValueError("clipseg_segment requires 'image_path' in the row")

        prompt_value = params.get("prompt") or params.get("text")
        if not prompt_value:
            raise ValueError("clipseg_segment requires a 'prompt' parameter")

        model_name = str(params.get("model_name") or _CLIPSEG_MODEL_NAME)
        threshold_raw = params.get("threshold", 0.5)
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("clipseg_segment threshold must be numeric") from exc
        threshold = float(np.clip(threshold, 0.0, 1.0))

        mask_dir = Path(out_dir) / "masks"
        masks = clipseg_segment(
            image_path,
            str(prompt_value),
            output_dir=mask_dir,
            model_name=model_name,
            threshold=threshold,
        )

        output_field = str(params.get("output_field") or "masks")
        payload = {
            output_field: [mask.to_dict() for mask in masks],
            "mask_artifact_dir": str(mask_dir),
        }
        return payload


register(SamSegmentHandler())
register(ClipSegSegmentHandler())
