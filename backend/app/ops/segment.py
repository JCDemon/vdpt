from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from ..types import Mask
from .base import OperationHandler
from .registry import register

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception:  # pragma: no cover - defensive import guard
    SamAutomaticMaskGenerator = None  # type: ignore[assignment]
    sam_model_registry = {}  # type: ignore[assignment]


try:  # pragma: no cover - optional dependency
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
except Exception:  # pragma: no cover - defensive import guard
    CLIPSegForImageSegmentation = None  # type: ignore[assignment]
    CLIPSegProcessor = None  # type: ignore[assignment]


_SAM_MODEL_KEY = "vit_b"
_SAM_GENERATOR: Optional[SamAutomaticMaskGenerator] = None  # type: ignore[name-defined]
_CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
_CLIPSEG_PROCESSOR: Optional[CLIPSegProcessor] = None  # type: ignore[name-defined]
_CLIPSEG_MODEL: Optional[CLIPSegForImageSegmentation] = None  # type: ignore[name-defined]
_CLIPSEG_DEVICE: Optional[torch.device] = None


def _resolve_sam_checkpoint() -> Optional[Path]:
    """Locate a SAM checkpoint on disk or via environment configuration."""

    candidates: List[Path] = []
    env_path = os.getenv("SAM_CHECKPOINT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "artifacts" / "models" / "sam_vit_b.pth")
    candidates.append(repo_root / "artifacts" / "models" / "sam_vit_b_01ec64.pth")

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _load_sam_generator() -> Optional[SamAutomaticMaskGenerator]:  # type: ignore[name-defined]
    """Create (and cache) a SAM automatic mask generator if possible."""

    global _SAM_GENERATOR
    if _SAM_GENERATOR is not None:
        return _SAM_GENERATOR
    if SamAutomaticMaskGenerator is None or not sam_model_registry:  # type: ignore[name-defined]
        logger.warning("segment-anything not installed; using fallback segmentation")
        return None

    checkpoint = _resolve_sam_checkpoint()
    if checkpoint is None:
        logger.warning("SAM checkpoint not found; using fallback segmentation")
        return None

    try:
        sam_builder = sam_model_registry[_SAM_MODEL_KEY]  # type: ignore[index]
    except KeyError:
        logger.error("SAM model '%s' unavailable; using fallback segmentation", _SAM_MODEL_KEY)
        return None

    try:
        sam = sam_builder(checkpoint=checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam.to(device)
        generator = SamAutomaticMaskGenerator(sam)  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to initialize SAM generator: %s", exc)
        return None

    _SAM_GENERATOR = generator
    logger.info("Loaded SAM checkpoint from %s", checkpoint)
    return _SAM_GENERATOR


def _load_clipseg() -> tuple[
    Optional[CLIPSegProcessor],
    Optional[CLIPSegForImageSegmentation],
    Optional[torch.device],
]:
    """Create (and cache) CLIPSeg processor/model for prompted segmentation."""

    global _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL, _CLIPSEG_DEVICE
    if (
        _CLIPSEG_PROCESSOR is not None
        and _CLIPSEG_MODEL is not None
        and _CLIPSEG_DEVICE is not None
    ):
        return _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL, _CLIPSEG_DEVICE

    if CLIPSegProcessor is None or CLIPSegForImageSegmentation is None:  # type: ignore[name-defined]
        logger.warning("CLIPSeg dependencies missing; using fallback segmentation")
        return None, None, None

    try:
        processor = CLIPSegProcessor.from_pretrained(_CLIPSEG_MODEL_ID)  # type: ignore[arg-type]
        model = CLIPSegForImageSegmentation.from_pretrained(_CLIPSEG_MODEL_ID)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to initialize CLIPSeg model '%s': %s", _CLIPSEG_MODEL_ID, exc)
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _CLIPSEG_PROCESSOR = processor
    _CLIPSEG_MODEL = model
    _CLIPSEG_DEVICE = device
    logger.info("Loaded CLIPSeg model %s on %s", _CLIPSEG_MODEL_ID, device)
    return processor, model, device


def _ensure_mask_dir(run_dir: Path, record_index: int) -> Path:
    mask_root = run_dir / "masks"
    record_dir = mask_root / f"record-{record_index:04d}"
    record_dir.mkdir(parents=True, exist_ok=True)
    return record_dir


def _save_mask_image(mask_dir: Path, base_name: str, segmentation: np.ndarray) -> Path:
    binary = np.asarray(segmentation).astype(np.uint8)
    mask_img = Image.fromarray(binary * 255)
    mask_path = mask_dir / f"{base_name}.png"
    mask_img.save(mask_path)
    return mask_path


def _fallback_masks(
    image: Image.Image,
    image_path: Path,
    mask_dir: Path,
    record_index: int,
    *,
    reason: str,
    model_name: str,
    prompt: Optional[str] = None,
) -> List[Mask]:
    mask_id = f"mask-{record_index:04d}-0"
    mask_path = mask_dir / f"{image_path.stem}_{mask_id}.png"
    Image.new("L", image.size, color=255).save(mask_path)
    bbox = (0, 0, image.width, image.height)
    area = image.width * image.height
    mask = Mask(
        id=mask_id,
        image_path=str(image_path),
        mask_path=str(mask_path),
        bbox=bbox,
        area=area,
        model=model_name,
        prompt=prompt,
        extra={"reason": reason},
    )
    logger.warning("%s fallback segmentation generated single mask for %s", model_name, image_path)
    return [mask]


def _coerce_masks(
    raw_masks: Iterable[Dict[str, Any]],
    image: Image.Image,
    image_path: Path,
    mask_dir: Path,
    record_index: int,
    max_masks: Optional[int] = None,
) -> List[Mask]:
    results: List[Mask] = []
    for idx, payload in enumerate(
        sorted(raw_masks, key=lambda item: item.get("area", 0), reverse=True)
    ):
        if max_masks is not None and idx >= max_masks:
            break
        segmentation = payload.get("segmentation")
        if segmentation is None:
            continue
        bbox_raw = payload.get("bbox") or [0, 0, image.width, image.height]
        try:
            x, y, w, h = [int(v) for v in bbox_raw]
        except Exception:
            x, y, w, h = 0, 0, image.width, image.height
        bbox = (max(0, x), max(0, y), min(image.width, x + w), min(image.height, y + h))
        base_name = f"{image_path.stem}_mask_{record_index:04d}_{idx:02d}"
        mask_path = _save_mask_image(mask_dir, base_name, np.asarray(segmentation, dtype=bool))
        area_val = int(payload.get("area") or np.count_nonzero(segmentation))
        score = payload.get("predicted_iou")
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        mask = Mask(
            id=f"mask-{record_index:04d}-{idx}",
            image_path=str(image_path),
            mask_path=str(mask_path),
            bbox=bbox,
            area=area_val,
            score=score_val,
            model=f"sam-{_SAM_MODEL_KEY}",
        )
        results.append(mask)
    return results


def sam_segment(
    image_path: str | Path,
    *,
    run_dir: Path,
    record_index: int,
    max_masks: Optional[int] = None,
) -> List[Mask]:
    """Segment an image using Segment Anything (SAM)."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found at {path}")

    mask_dir = _ensure_mask_dir(run_dir, record_index)
    with Image.open(path) as img:
        image = img.convert("RGB")

    generator = _load_sam_generator()
    if generator is None:
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="sam_unavailable",
            model_name="sam-fallback",
        )

    try:
        raw_masks = generator.generate(np.array(image))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("SAM segmentation failed for %s: %s", path, exc)
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="sam_error",
            model_name="sam-fallback",
        )

    masks = _coerce_masks(raw_masks, image, path, mask_dir, record_index, max_masks=max_masks)
    if not masks:
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="sam_empty",
            model_name="sam-fallback",
        )
    return masks


def clipseg_segment(
    image_path: str | Path,
    prompt: str,
    *,
    run_dir: Path,
    record_index: int,
    threshold: float = 0.5,
) -> List[Mask]:
    """Segment an image with a text prompt using CLIPSeg."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found at {path}")

    normalized_prompt = prompt.strip() or "object"
    mask_dir = _ensure_mask_dir(run_dir, record_index)
    with Image.open(path) as img:
        image = img.convert("RGB")

    processor, model, device = _load_clipseg()
    if processor is None or model is None or device is None:
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="clipseg_unavailable",
            model_name="clipseg-fallback",
            prompt=normalized_prompt,
        )

    try:
        inputs = processor(text=[normalized_prompt], images=[image], return_tensors="pt")  # type: ignore[call-arg]
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        scores = torch.sigmoid(logits)[0][0].cpu().numpy()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("CLIPSeg inference failed for %s: %s", path, exc)
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="clipseg_error",
            model_name="clipseg-fallback",
            prompt=normalized_prompt,
        )

    if scores.size == 0:
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="clipseg_empty",
            model_name="clipseg-fallback",
            prompt=normalized_prompt,
        )

    scaled = scores - scores.min()
    max_val = float(scaled.max())
    if max_val > 0:
        scaled = scaled / max_val

    mask_resized = Image.fromarray((scaled * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    binary = np.asarray(mask_resized) >= int(np.clip(threshold, 0.01, 0.99) * 255)

    if not np.any(binary):
        return _fallback_masks(
            image,
            path,
            mask_dir,
            record_index,
            reason="clipseg_threshold_empty",
            model_name="clipseg-fallback",
            prompt=normalized_prompt,
        )

    ys, xs = np.where(binary)
    x_min, x_max = int(xs.min()), int(xs.max() + 1)
    y_min, y_max = int(ys.min()), int(ys.max() + 1)
    bbox = (
        max(0, x_min),
        max(0, y_min),
        min(image.width, x_max),
        min(image.height, y_max),
    )
    base_name = f"{path.stem}_clipseg_{record_index:04d}"
    mask_path = _save_mask_image(mask_dir, base_name, binary)
    area = int(np.count_nonzero(binary))
    score = float(np.mean(scaled[ys, xs])) if area else None
    mask = Mask(
        id=f"mask-{record_index:04d}-clipseg-0",
        image_path=str(path),
        mask_path=str(mask_path),
        bbox=bbox,
        area=area,
        score=score,
        prompt=normalized_prompt,
        model="clipseg-rd64",
        extra={"threshold": float(np.clip(threshold, 0.01, 0.99))},
    )
    return [mask]


class SamSegmentHandler(OperationHandler):
    kind = "sam_segment"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._segment(row, params, out_dir=None)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._segment(row, params, out_dir=out_dir)

    def _segment(
        self,
        row: Dict[str, Any],
        params: Dict[str, Any],
        *,
        out_dir: Path | None,
    ) -> Dict[str, Any]:
        image_value = row.get("image_path")
        if not image_value:
            raise ValueError("sam_segment requires 'image_path' in the row")

        record_index = int(row.get("__record_index__", 0))
        run_dir = _resolve_run_dir(row, out_dir)
        max_masks = _parse_positive_int(params.get("max_masks"))

        masks = sam_segment(
            image_value, run_dir=run_dir, record_index=record_index, max_masks=max_masks
        )
        return {"masks": [mask.to_dict() for mask in masks]}


class ClipSegSegmentHandler(OperationHandler):
    kind = "clipseg_segment"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._segment(row, params, out_dir=None)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._segment(row, params, out_dir=out_dir)

    def _segment(
        self,
        row: Dict[str, Any],
        params: Dict[str, Any],
        *,
        out_dir: Path | None,
    ) -> Dict[str, Any]:
        image_value = row.get("image_path")
        if not image_value:
            raise ValueError("clipseg_segment requires 'image_path' in the row")

        record_index = int(row.get("__record_index__", 0))
        run_dir = _resolve_run_dir(row, out_dir)
        prompt_value = (
            params.get("prompt")
            or params.get("text")
            or params.get("instructions")
            or params.get("query")
        )
        prompt = str(prompt_value).strip() if prompt_value is not None else ""
        threshold = _parse_threshold(params.get("threshold"))

        masks = clipseg_segment(
            image_value,
            prompt,
            run_dir=run_dir,
            record_index=record_index,
            threshold=threshold,
        )
        return {"masks": [mask.to_dict() for mask in masks]}


def _resolve_run_dir(row: Dict[str, Any], out_dir: Path | None) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    hinted = row.get("__run_dir__")
    if hinted:
        hinted_path = Path(hinted)
        hinted_path.mkdir(parents=True, exist_ok=True)
        return hinted_path
    fallback = Path("artifacts") / "runs" / "preview"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _parse_positive_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("max_masks must be an integer") from exc
    if parsed <= 0:
        raise ValueError("max_masks must be positive")
    return parsed


def _parse_threshold(value: Any) -> float:
    if value in (None, "", False):
        return 0.5
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("clipseg threshold must be numeric") from exc
    return float(min(max(parsed, 0.0), 1.0))


register(SamSegmentHandler())
register(ClipSegSegmentHandler())


__all__ = ["sam_segment", "clipseg_segment"]
