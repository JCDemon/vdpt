from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .base import DatasetLoader, LoaderParam, register_loader

try:  # pragma: no cover - optional dependency
    from datasets import (
        get_dataset_config_names,
        get_dataset_split_names,
        load_dataset,
    )
except Exception:  # pragma: no cover - handled at runtime
    get_dataset_config_names = None  # type: ignore[assignment]
    get_dataset_split_names = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - handled at runtime
    Image = None  # type: ignore[assignment]


_CACHE_DIR = Path("artifacts") / "datasets" / "hf_cache"


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return str(value)


def _prepare_image(example: Dict[str, Any], field: str) -> Optional[Image.Image]:
    if Image is None:
        return None
    image_value = example.get(field)
    if image_value is None:
        return None
    if isinstance(image_value, Image.Image):
        return image_value
    if hasattr(image_value, "to_pil"):
        return image_value.to_pil()
    try:
        return Image.fromarray(image_value)
    except Exception:
        return None


@register_loader
class HuggingFaceImageLoader(DatasetLoader):
    id = "hf_images"
    name = "HuggingFace datasets"
    description = "Image datasets available through the HuggingFace datasets hub."
    params = (
        LoaderParam(
            name="dataset_name",
            label="Dataset name",
            description="Name passed to datasets.load_dataset (e.g. 'beans').",
        ),
        LoaderParam(
            name="subset",
            label="Config/subset",
            description="Optional dataset configuration name.",
            required=False,
        ),
        LoaderParam(
            name="split",
            label="Split",
            kind="string",
            description="Split name to preview (e.g. 'train').",
            default="train",
            required=False,
        ),
        LoaderParam(
            name="image_field",
            label="Image field",
            kind="string",
            description="Column containing images.",
            default="image",
            required=False,
        ),
    )

    def _ensure_dependency(self) -> None:
        if load_dataset is None:
            raise RuntimeError(
                "The 'datasets' library is required for HuggingFace dataset loading. Install it "
                "with `pip install datasets`."
            )
        if Image is None:
            raise RuntimeError("Pillow is required for handling dataset images.")

    def _resolve_config(self, dataset_name: str, subset: Optional[str]) -> Optional[str]:
        if subset:
            return subset
        if get_dataset_config_names is None:
            return None
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception:
            return None
        if not configs:
            return None
        if len(configs) == 1:
            return configs[0]
        return None

    def scan(self) -> Iterable[Dict[str, Any]]:
        dataset_name = self.config.get("dataset_name")
        subset = self.config.get("subset")
        split = self.config.get("split", "train")
        if not dataset_name:
            raise ValueError("'dataset_name' must be provided for HuggingFace datasets.")

        self._ensure_dependency()
        resolved_subset = self._resolve_config(dataset_name, subset)

        summary: List[Dict[str, Any]] = []
        if resolved_subset is not None:
            if get_dataset_split_names is not None:
                try:
                    splits = get_dataset_split_names(dataset_name, resolved_subset)
                except Exception:
                    splits = [split]
            else:
                splits = [split]
            for split_name in splits:
                summary.append(
                    {
                        "id": f"{dataset_name}:{resolved_subset}:{split_name}",
                        "dataset": dataset_name,
                        "config": resolved_subset,
                        "split": split_name,
                    }
                )
        else:
            if get_dataset_config_names is not None:
                try:
                    configs = get_dataset_config_names(dataset_name)
                except Exception:
                    configs = []
            else:
                configs = []
            if configs:
                summary.extend(
                    {
                        "id": f"{dataset_name}:{config}",
                        "dataset": dataset_name,
                        "config": config,
                        "split": split,
                    }
                    for config in configs
                )
            else:
                summary.append(
                    {
                        "id": f"{dataset_name}:{split}",
                        "dataset": dataset_name,
                        "config": subset,
                        "split": split,
                    }
                )
        return summary

    def iter_records(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        dataset_name = self.config.get("dataset_name")
        subset = self.config.get("subset")
        split = self.config.get("split", "train")
        image_field = self.config.get("image_field", "image")
        if not dataset_name:
            raise ValueError("'dataset_name' must be provided for HuggingFace datasets.")

        self._ensure_dependency()
        resolved_subset = self._resolve_config(dataset_name, subset)
        cache_dir = _ensure_cache_dir()

        dataset = load_dataset(
            dataset_name,
            resolved_subset,
            split=split,
            streaming=True,
        )

        count = 0
        for idx, example in enumerate(dataset):
            image = _prepare_image(example, image_field)
            if image is None:
                continue
            identifier = sha1(f"{dataset_name}:{resolved_subset}:{split}:{idx}".encode()).hexdigest()
            image_path = cache_dir / f"{identifier}.png"
            try:
                image.save(image_path)
            except Exception:
                continue

            metadata = {key: _serialize(value) for key, value in example.items() if key != image_field}
            yield {
                "image_path": str(image_path),
                "dataset": dataset_name,
                "config": resolved_subset,
                "split": split,
                "index": idx,
                "metadata": metadata,
            }
            count += 1
            if limit is not None and count >= limit:
                break
