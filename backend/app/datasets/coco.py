from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .base import DatasetLoader, LoaderParam, register_loader


@register_loader
class Coco2017Loader(DatasetLoader):
    id = "coco2017"
    name = "COCO 2017"
    description = "Microsoft COCO 2017 images and instance annotations."
    params = (
        LoaderParam(
            name="root",
            label="Dataset root",
            kind="path",
            description=(
                "Directory containing 'images/<split>2017' and "
                "'annotations/instances_<split>2017.json'."
            ),
        ),
        LoaderParam(
            name="split",
            label="Split",
            kind="select",
            choices=("train", "val"),
            default="val",
            required=True,
        ),
    )

    def scan(self) -> Iterable[Dict[str, Any]]:
        split = self.config.get("split", "val")
        root = Path(self.config.get("root", "")).expanduser().resolve()
        image_dir = root / "images" / f"{split}2017"
        annotation_file = root / "annotations" / f"instances_{split}2017.json"

        if not image_dir.exists():
            raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
        if not annotation_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")

        with annotation_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        image_count = 0
        for image in data.get("images", []):
            file_name = image.get("file_name")
            if not file_name:
                continue
            if (image_dir / file_name).exists():
                image_count += 1
        return [
            {
                "id": f"coco2017:{split}",
                "split": split,
                "root": str(root),
                "image_dir": str(image_dir),
                "annotation_file": str(annotation_file),
                "num_images": image_count,
            }
        ]

    def iter_records(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        split = self.config.get("split", "val")
        root = Path(self.config.get("root", "")).expanduser().resolve()
        image_dir = root / "images" / f"{split}2017"
        annotation_file = root / "annotations" / f"instances_{split}2017.json"

        if not annotation_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")

        with annotation_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        category_lookup = {item["id"]: item["name"] for item in data.get("categories", [])}
        annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            annotations_by_image[int(ann["image_id"])].append(ann)

        yielded = 0
        for image in data.get("images", []):
            file_name = image.get("file_name")
            if not file_name:
                continue
            image_path = image_dir / file_name
            if not image_path.exists():
                continue

            annotations: List[dict] = []
            for ann in annotations_by_image.get(int(image["id"]), []):
                annotations.append(
                    {
                        "id": ann.get("id"),
                        "category": category_lookup.get(ann.get("category_id")),
                        "bbox": ann.get("bbox"),
                        "area": ann.get("area"),
                        "iscrowd": ann.get("iscrowd"),
                    }
                )

            record = {
                "image_path": str(image_path),
                "image_id": image.get("id"),
                "file_name": file_name,
                "width": image.get("width"),
                "height": image.get("height"),
                "annotations": annotations,
            }
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                break
