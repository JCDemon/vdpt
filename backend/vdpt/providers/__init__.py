"""Provider selection and narrow API surface for VDPT."""

from __future__ import annotations

import os

PROVIDER_NAME = os.getenv("VDPT_PROVIDER", "openai").lower()

if PROVIDER_NAME == "qwen":
    from . import qwen as provider  # noqa: F401
elif PROVIDER_NAME == "openai":
    from . import openai as provider  # noqa: F401
else:
    from . import dummy as provider  # noqa: F401

summarize = provider.summarize  # type: ignore[attr-defined]
img_caption = provider.img_caption  # type: ignore[attr-defined]

# Backwards compatibility for modules that still expect the provider module.
current = provider

__all__ = ["PROVIDER_NAME", "provider", "summarize", "img_caption", "current"]
