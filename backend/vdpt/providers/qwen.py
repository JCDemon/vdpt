from __future__ import annotations

import base64
import os
from typing import Optional

import httpx

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

TEXT_MODEL = os.getenv("VDPT_QWEN_TEXT_MODEL", "qwen-turbo")
VISION_MODEL = os.getenv("VDPT_QWEN_VISION_MODEL", "qwen-vl-plus")


def _post(payload: dict) -> dict:
    if not API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = _post(
        {
            "model": model or TEXT_MODEL,
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


def caption(
    image_path: str,
    *,
    prompt: str = "请用一句中文描述图片内容",
    max_tokens: int = 80,
    model: Optional[str] = None,
) -> str:
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
    ]
    data = _post(
        {
            "model": model or VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
    )
    return data["choices"][0]["message"]["content"]
