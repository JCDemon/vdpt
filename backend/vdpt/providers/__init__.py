import importlib
import os
from typing import Literal, cast


def detect_provider() -> Literal["qwen", "mock"]:
    env_provider = os.getenv("VDPT_PROVIDER")
    if env_provider:
        return cast(Literal["qwen", "mock"], env_provider)

    if os.getenv("DASHSCOPE_API_KEY"):
        return "qwen"

    return "mock"


current = importlib.import_module(f".{detect_provider()}", package=__name__)

__all__ = ["current", "detect_provider"]
