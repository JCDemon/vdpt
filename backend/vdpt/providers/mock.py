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


def caption(
    image_path: str,
    *,
    prompt: str = "请用一句中文描述图片内容",
    max_tokens: int = 80,
    model: Optional[str] = None,
) -> str:
    name = Path(image_path).name
    prompt_digest = _digest(prompt)
    return f"[mock-caption] {name}:{prompt_digest}"


__all__ = ["TEXT_MODEL", "VISION_MODEL", "chat", "caption"]
