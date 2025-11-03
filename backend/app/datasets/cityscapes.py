from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .base import DatasetLoader, LoaderParam, register_loader


def _iter_city_files(root: Path, split: str) -> Iterator[tuple[Path, Path]]:
    image_root = root / "leftImg8bit" / split
    annotation_root = root / "gtFine" / split
    if not image_root.exists():
        raise FileNotFoundError(
            "Cityscapes leftImg8bit directory not found. Ensure the dataset is installed."
        )
    if not annotation_root.exists():
        raise FileNotFoundError("Cityscapes gtFine directory not found.")

    for city_dir in sorted(image_root.glob("*")):
        if not city_dir.is_dir():
            continue
        for image_file in sorted(city_dir.glob("*_leftImg8bit.png")):
            stem = image_file.name.replace("_leftImg8bit.png", "")
            annotation_file = annotation_root / city_dir.name / f"{stem}_gtFine_polygons.json"
            if annotation_file.exists():
                yield image_file, annotation_file


@register_loader
class CityscapesFineLoader(DatasetLoader):
    id = "cityscapes_fine"
    name = "Cityscapes (fine annotations)"
    description = (
        "Cityscapes fine annotations. Requires registration at https://www.cityscapes-dataset.com/."
    )
    params = (
        LoaderParam(
            name="root",
            label="Dataset root",
            kind="path",
            description="Root directory containing 'leftImg8bit' and 'gtFine'.",
        ),
        LoaderParam(
            name="split",
            label="Split",
            kind="select",
            choices=("train", "val"),
            default="val",
        ),
    )

    def scan(self) -> Iterable[Dict[str, Any]]:
        root = Path(self.config.get("root", "")).expanduser().resolve()
        split = self.config.get("split", "val")

        files = list(_iter_city_files(root, split))
        return [
            {
                "id": f"cityscapes:{split}",
                "split": split,
                "root": str(root),
                "num_images": len(files),
            }
        ]

    def iter_records(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        root = Path(self.config.get("root", "")).expanduser().resolve()
        split = self.config.get("split", "val")

        yielded = 0
        for image_path, annotation_path in _iter_city_files(root, split):
            with annotation_path.open("r", encoding="utf-8") as fh:
                annotation_payload = json.load(fh)

            objects: List[Dict[str, Any]] = []
            for obj in annotation_payload.get("objects", []):
                polygons = obj.get("polygon")
                if isinstance(polygons, list):
                    polygon_count = len(polygons)
                else:
                    polygon_count = 0
                objects.append(
                    {
                        "label": obj.get("label"),
                        "polygon_count": polygon_count,
                    }
                )

            yield {
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "city": image_path.parent.name,
                "objects": objects,
            }
            yielded += 1
            if limit is not None and yielded >= limit:
                break
