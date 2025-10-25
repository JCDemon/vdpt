import importlib
import os

_PROVIDER_NAME = os.getenv("VDPT_PROVIDER", "mock")
current = importlib.import_module(f".{_PROVIDER_NAME}", __name__)

__all__ = ["current"]
