"""Lightweight OpenAI provider shim with deterministic fallback."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def summarize(text: str, instructions: str = "", max_tokens: int = 128) -> str:
    base = text.strip()
    if instructions:
        base = f"{instructions.strip()}::{base}"
    payload = f"{base}:{max_tokens}:{TEXT_MODEL}"
    return f"[openai-summarize:{_digest(payload)}]"


def chat(prompt: str, **kwargs: Any) -> str:
    system = kwargs.get("system")
    instructions = kwargs.get("instructions")
    temperature = kwargs.get("temperature")
    parts = [prompt]
    if system:
        parts.append(str(system))
    if instructions:
        parts.append(str(instructions))
    if temperature is not None:
        parts.append(str(temperature))
    payload = "::".join(parts)
    return f"[openai-chat:{_digest(payload)}]"


def img_caption(image_path: str, instructions: str = "", max_tokens: int = 80) -> str:
    prompt = instructions or "Describe the image in Chinese"
    payload = f"{Path(image_path).name}:{prompt}:{max_tokens}:{VISION_MODEL}"
    return f"[openai-img-caption:{_digest(payload)}]"


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
