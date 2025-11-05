"""Mock provider implementation for tests and offline usage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

TEXT_MODEL = "mock-text"
VISION_MODEL = "mock-vision"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    parts = ["[mock-chat]"]
    if system:
        parts.append(f"system={_digest(system)}")
    parts.append(_digest(prompt))
    return " ".join(parts)


def summarize(
    text: str,
    instructions: str = "",
    max_tokens: int = 128,
    *,
    model: Optional[str] = None,
) -> str:
    base = text.strip()
    if instructions:
        base = f"{instructions.strip()}::{base}"
    payload = f"{base}:{max_tokens}:{model or TEXT_MODEL}"
    return f"[mock-summarize] {_digest(payload)}"


def caption(
    image_path: str,
    *,
    prompt: str = "请用一句中文描述图片内容",
    max_tokens: int = 80,
    model: Optional[str] = None,
) -> str:
    name = Path(image_path).name
    prompt_digest = _digest(f"{prompt}:{model or VISION_MODEL}:{max_tokens}")
    return f"[mock-caption] {name}:{prompt_digest}"


def img_caption(image_path: str, instructions: str = "", max_tokens: int = 80) -> str:
    prompt = instructions or "请用一句中文描述图片内容"
    return caption(image_path, prompt=prompt, max_tokens=max_tokens)


__all__ = ["TEXT_MODEL", "VISION_MODEL", "chat", "caption", "summarize", "img_caption"]
