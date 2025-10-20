"""DashScope Tongyi Qianwen provider."""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Optional

try:  # pragma: no cover - optional dependency may be absent in CI
    from dashscope import Generation
    from dashscope.common.error import DashScopeException
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without dashscope
    Generation = None

    class DashScopeException(Exception):
        """Fallback exception when DashScope SDK is unavailable."""

        pass


from .base import TextLLMProvider


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
