from __future__ import annotations

import hashlib
import os
from typing import Any

from dashscope import Generation, MultiModalConversation

_QWEN_TEXT_MODEL = os.getenv("VDPT_QWEN_TEXT_MODEL") or os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
_QWEN_VL_MODEL = os.getenv("VDPT_QWEN_VISION_MODEL") or os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
_USE_MOCK = os.getenv("VDPT_MOCK", "0") == "1"


def _mock_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _call_generation(prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        return ""

    try:
        return response["output"]["text"].strip()
    except Exception:
        return ""


def summarize(text: str, instructions: str = "", max_tokens: int = 128) -> str:
    if _USE_MOCK:
        return f"[mock-qwen-summarize:{_mock_digest(text)}]"

    prompt = f"{instructions}\n\nTEXT:\n{text}" if instructions else text
    return _call_generation(
        prompt,
        model=_QWEN_TEXT_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
    )


def chat(prompt: str, **kwargs: Any) -> str:
    if _USE_MOCK:
        system = kwargs.get("system")
        digest_source = f"{system or ''}:{prompt}" if system else prompt
        return f"[mock-qwen-chat:{_mock_digest(digest_source)}]"

    system = kwargs.get("system")
    instructions = kwargs.get("instructions")
    max_tokens = int(kwargs.get("max_tokens", 256))
    temperature = float(kwargs.get("temperature", 0.7))
    model = str(kwargs.get("model") or _QWEN_TEXT_MODEL)

    prefix = str(instructions or "")
    if system:
        prefix = f"{system}\n\n{prefix}" if prefix else str(system)
    prompt_text = f"{prefix}\n\n{prompt}" if prefix else prompt

    return _call_generation(
        prompt_text,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def img_caption(image_path: str, instructions: str = "", max_tokens: int = 80) -> str:
    if _USE_MOCK:
        digest_source = (
            f"{os.path.basename(image_path)}:{instructions}"
            if instructions
            else os.path.basename(image_path)
        )
        return f"[mock-qwen-caption:{_mock_digest(digest_source)}]"

    prompt_text = instructions or "用一句中文描述图片内容"
    messages = [
        {"role": "system", "content": "You are an assistant that captions images in Chinese."},
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": prompt_text},
            ],
        },
    ]

    try:
        response = MultiModalConversation.call(
            model=_QWEN_VL_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
    except Exception:
        return ""

    try:
        return response["output"]["choices"][0]["message"]["content"][0]["text"].strip()
    except Exception:
        return ""


def caption(image_path: str, **kwargs: Any) -> str:
    prompt = kwargs.get("prompt") or kwargs.get("instructions") or ""
    max_tokens = int(kwargs.get("max_tokens", 80))
    return img_caption(
        image_path,
        instructions=str(prompt),
        max_tokens=max_tokens,
    )


TEXT_MODEL = _QWEN_TEXT_MODEL
VISION_MODEL = _QWEN_VL_MODEL

__all__ = [
    "TEXT_MODEL",
    "VISION_MODEL",
    "chat",
    "caption",
    "summarize",
    "img_caption",
]
