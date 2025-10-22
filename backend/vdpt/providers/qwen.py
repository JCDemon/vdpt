"""DashScope Tongyi Qianwen providers."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # pragma: no cover - optional dependency may be absent in CI
    import dashscope
    from dashscope import Generation, MultiModalConversation
    from dashscope.common.error import DashScopeException
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without dashscope
    dashscope = None
    Generation = None
    MultiModalConversation = None

    class DashScopeException(Exception):
        """Fallback exception when DashScope SDK is unavailable."""

        pass


from .base import TextLLMProvider
from .vision import VisionProvider, register_vision_provider


DEFAULT_QWEN_VL_MODEL = os.getenv("VDPT_QWEN_VL_MODEL", "qwen-vl-plus")


def _read_image_b64(path: Path) -> str:
    """Return the contents of *path* encoded as a base64 PNG data URI."""

    with path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def caption_image(
    image_path: str,
    instructions: str = "Describe this image in one concise Chinese sentence.",
    max_tokens: int = 80,
    model: Optional[str] = None,
) -> str:
    """Return an image caption using Qwen-VL via DashScope."""

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope is None or MultiModalConversation is None or not api_key:
        return f"[mock] {Path(image_path).name} 的简要描述"

    dashscope.api_key = api_key
    img_b64 = _read_image_b64(Path(image_path))
    payload_model = model or DEFAULT_QWEN_VL_MODEL
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{img_b64}"},
                {"text": instructions},
            ],
        }
    ]

    response = MultiModalConversation.call(
        model=payload_model,
        messages=messages,
        max_tokens=max_tokens,
    )
    try:
        return response["output"]["choices"][0]["message"]["content"][0]["text"].strip()
    except Exception:  # pragma: no cover - fall back to a readable string
        return str(response)


class QwenProvider(TextLLMProvider):
    """Text provider backed by Alibaba Cloud's Tongyi Qianwen."""

    def __init__(
        self,
        *,
        model: str = "qwen-plus",
        timeout: int = 15,
        max_retries: int = 3,
        retry_delay: float = 1.5,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._generation = Generation

    def generate(self, prompt: str, **kwargs: Any) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self._generation is None:
                    raise RuntimeError(
                        "dashscope package is required to use QwenProvider; install dashscope>=1.15.0"
                    )
                if not os.getenv("DASHSCOPE_API_KEY"):
                    raise RuntimeError(
                        "DASHSCOPE_API_KEY environment variable must be set for QwenProvider"
                    )

                response = self._generation.call(
                    model=self.model,
                    prompt=prompt,
                    timeout=self.timeout,
                    **kwargs,
                )
                text = self._extract_text(response)
                if text:
                    return text
                last_error = RuntimeError("DashScope response did not contain any text output")
            except DashScopeException as err:
                last_error = err
            except Exception as err:  # pragma: no cover - defensive guard
                last_error = err

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        message = "Failed to generate text with Qwen:"
        if last_error is not None:
            message = f"{message} {last_error}"
        raise RuntimeError(message)

    @staticmethod
    def _extract_text(response: Any) -> Optional[str]:
        if response is None:
            return None

        # DashScope wraps the payload in response.output
        output = getattr(response, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            if "text" in output and isinstance(output["text"], str):
                return output["text"]
            choices = output.get("choices")
            if isinstance(choices, Iterable):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, Iterable):
                            for part in content:
                                if isinstance(part, dict) and part.get("text"):
                                    return str(part["text"])
                    content = choice.get("content")
                    if isinstance(content, str):
                        return content
        return None


class QwenVisionProvider(VisionProvider):
    """Vision provider that uses Qwen-VL models via DashScope."""

    def __init__(
        self,
        *,
        default_instructions: str = "Describe this image in one concise Chinese sentence.",
        default_max_tokens: int = 80,
        model: Optional[str] = None,
    ) -> None:
        self._default_instructions = default_instructions
        self._default_max_tokens = default_max_tokens
        self._model = model or DEFAULT_QWEN_VL_MODEL

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        del seed  # unused
        instructions = prompt or self._default_instructions
        token_limit = max_tokens if max_tokens is not None else self._default_max_tokens
        return caption_image(
            str(image_path),
            instructions=instructions,
            max_tokens=token_limit,
            model=self._model,
        )


register_vision_provider("qwen", QwenVisionProvider)
register_vision_provider("qwen_vl", QwenVisionProvider)
