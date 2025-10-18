"""Mock text generation provider for offline testing."""

from __future__ import annotations

import hashlib
from typing import Any

from .base import TextLLMProvider


class MockProvider(TextLLMProvider):
    """Deterministic mock provider useful for tests."""

    def __init__(self, template: str | None = None) -> None:
        self._template = template or "[MOCK]{digest}"

    def generate(self, prompt: str, **_: Any) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return self._template.format(digest=digest, prompt=prompt)
