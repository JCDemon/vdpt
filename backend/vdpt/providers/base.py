"""Abstract base classes for text LLM providers."""

from __future__ import annotations

import abc
from typing import Any


class TextLLMProvider(abc.ABC):
    """Interface for text generation providers."""

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text completion for the given prompt."""
        raise NotImplementedError
