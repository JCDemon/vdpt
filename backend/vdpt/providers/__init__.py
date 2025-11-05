"""Provider selection and narrow API surface for VDPT."""

from __future__ import annotations

import importlib
import os
from typing import Final


_PROVIDER_MODULES: Final[dict[str, str]] = {
    "qwen": ".qwen",
    "openai": ".openai",
    "mock": ".mock",
    "dummy": ".dummy",
}


def _detect_provider() -> str:
    """Determine which provider module should be used."""

    env_value = os.getenv("VDPT_PROVIDER")
    if env_value:
        return env_value.lower()

    if os.getenv("DASHSCOPE_API_KEY"):
        return "qwen"

    return "openai"


_detected = _detect_provider()
module_name = _PROVIDER_MODULES.get(_detected)

if module_name is None:
    module_name = _PROVIDER_MODULES["dummy"]
    PROVIDER_NAME = "dummy"
else:
    PROVIDER_NAME = _detected

provider = importlib.import_module(module_name, package=__name__)

summarize = provider.summarize  # type: ignore[attr-defined]
img_caption = provider.img_caption  # type: ignore[attr-defined]

# Backwards compatibility for modules that still expect the provider module.
current = provider

__all__ = ["PROVIDER_NAME", "provider", "summarize", "img_caption", "current"]
