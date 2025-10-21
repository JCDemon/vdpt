from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .base import OperationHandler
from .registry import register


_RESAMPLING = getattr(Image, "Resampling", Image)
_RESAMPLE_FILTER = getattr(_RESAMPLING, "LANCZOS", Image.LANCZOS)


class ImageResizeHandler(OperationHandler):
    kind = "img_resize"

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        src_path = self._get_source(row)
        width, height, keep_aspect = self._parse_params(params)

        tmp_dir = Path(row.get("__tmp_dir__", Path("artifacts") / "tmp"))
        dest_dir = tmp_dir / self.kind
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / self._output_name(src_path, width, height)
        self._resize_image(src_path, width, height, keep_aspect).save(dest_path)
        return {"resized_path": str(dest_path)}

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        if out_dir is None:
            raise ValueError("img_resize execute requires an output directory")

        src_path = self._get_source(row)
        width, height, keep_aspect = self._parse_params(params)

        dest_dir = out_dir / self.kind
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / self._output_name(src_path, width, height)
        self._resize_image(src_path, width, height, keep_aspect).save(dest_path)
        return {"resized_path": str(dest_path)}

    @staticmethod
    def _parse_params(params: Dict[str, Any]) -> tuple[int, int, bool]:
        try:
            width = int(params["width"])
            height = int(params["height"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("img_resize requires 'width' and 'height'") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("img_resize width/height must be integers") from exc

        if width <= 0 or height <= 0:
            raise ValueError("img_resize width/height must be positive")

        keep_aspect = bool(params.get("keep_aspect", False))
        return width, height, keep_aspect

    @staticmethod
    def _output_name(src_path: Path, width: int, height: int) -> str:
        suffix = src_path.suffix or ".png"
        return f"{src_path.stem}_{width}x{height}{suffix}"

    @staticmethod
    def _get_source(row: Dict[str, Any]) -> Path:
        value = row.get("image_path")
        if not value:
            raise ValueError("img_resize requires 'image_path' in the row")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"image not found at {path}")
        return path

    @staticmethod
    def _resize_image(src_path: Path, width: int, height: int, keep_aspect: bool) -> Image.Image:
        with Image.open(src_path) as img:
            img.load()
            if keep_aspect:
                working = img.copy()
                working.thumbnail((width, height), resample=_RESAMPLE_FILTER)
                return working
            return img.resize((width, height), resample=_RESAMPLE_FILTER)


class ImageCaptionHandler(OperationHandler):
    kind = "img_caption"

    _CAPTIONS = [
        "A vibrant abstract scene.",
        "A close-up portrait with warm light.",
        "A calm landscape with open skies.",
        "An energetic snapshot full of motion.",
    ]

    def preview(self, row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._caption(row)

    def execute(self, row: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        return self._caption(row)

    def _caption(self, row: Dict[str, Any]) -> Dict[str, Any]:
        value = row.get("image_path")
        if not value:
            raise ValueError("img_caption requires 'image_path' in the row")
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        index = int(digest, 16) % len(self._CAPTIONS)
        return {"caption": self._CAPTIONS[index]}


register(ImageResizeHandler())
register(ImageCaptionHandler())
