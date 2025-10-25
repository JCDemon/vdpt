from __future__ import annotations

import base64
import hashlib
import os
from typing import Any

import httpx

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

TEXT_MODEL = os.getenv("VDPT_QWEN_TEXT_MODEL", "qwen-turbo")
VISION_MODEL = os.getenv("VDPT_QWEN_VISION_MODEL", "qwen-vl-plus")
use_mock = os.getenv("VDPT_MOCK", "0") == "1"


def _require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DashScope provider requires the DASHSCOPE_API_KEY environment variable."
        )
    return api_key


def _post(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def _mock_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def chat(prompt: str, **kwargs: Any) -> str:
    if use_mock:
        system = kwargs.get("system")
        digest_source = f"{system or ''}:{prompt}" if system else prompt
        return f"[mock-qwen-chat:{_mock_digest(digest_source)}]"

    system = kwargs.get("system")
    max_tokens = int(kwargs.get("max_tokens", 256))
    temperature = float(kwargs.get("temperature", 0.7))
    model = kwargs.get("model") or TEXT_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": prompt})
    data = _post(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    return data["choices"][0]["message"]["content"]


def _image_to_data_url(path: str) -> str:
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def caption(image_path: str, **kwargs: Any) -> str:
    if use_mock:
        prompt = kwargs.get("prompt")
        digest_source = (
            f"{os.path.basename(image_path)}:{prompt or ''}"
            if prompt
            else os.path.basename(image_path)
        )
        return f"[mock-qwen-caption:{_mock_digest(digest_source)}]"

    prompt = str(kwargs.get("prompt") or "请用一句中文描述图片内容")
    max_tokens = int(kwargs.get("max_tokens", 80))
    model = kwargs.get("model") or VISION_MODEL
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
    ]
    data = _post(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
    )
    return data["choices"][0]["message"]["content"]


__all__ = ["TEXT_MODEL", "VISION_MODEL", "chat", "caption"]
