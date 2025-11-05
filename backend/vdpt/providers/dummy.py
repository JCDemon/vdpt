"""Deterministic fallback provider used when no real provider is available."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

TEXT_MODEL = "dummy-text"
VISION_MODEL = "dummy-vision"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def summarize(text: str, instructions: str = "", max_tokens: int = 128) -> str:
    base = text.strip()
    if instructions:
        base = f"{instructions.strip()}::{base}"
    base = f"{base}:{max_tokens}"
    return f"[dummy-summarize:{_digest(base)}]"


def chat(prompt: str, **kwargs: Any) -> str:
    system = kwargs.get("system")
    temperature = kwargs.get("temperature")
    digest_source = prompt
    if system:
        digest_source = f"{system}::{digest_source}"
    if temperature is not None:
        digest_source = f"{digest_source}:{temperature}"
    return f"[dummy-chat:{_digest(digest_source)}]"


def img_caption(image_path: str, instructions: str = "", max_tokens: int = 80) -> str:
    prompt = instructions or "用一句中文描述图片内容"
    digest_source = f"{Path(image_path).name}:{prompt}:{max_tokens}"
    return f"[dummy-img-caption:{_digest(digest_source)}]"


def caption(image_path: str, **kwargs: Any) -> str:
    prompt = kwargs.get("prompt") or kwargs.get("instructions") or ""
    max_tokens = int(kwargs.get("max_tokens", 80))
    return img_caption(image_path, instructions=str(prompt), max_tokens=max_tokens)


__all__ = [
    "TEXT_MODEL",
    "VISION_MODEL",
    "summarize",
    "img_caption",
    "chat",
    "caption",
]
