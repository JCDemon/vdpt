from .base import DatasetLoader, LoaderParam, register_loader, registry

__all__ = [
    "DatasetLoader",
    "LoaderParam",
    "register_loader",
    "registry",
]


# Register built-in loaders
from . import coco, cityscapes, hf  # noqa: F401
