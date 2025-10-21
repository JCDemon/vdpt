"""Vision provider interfaces and registry utilities."""

from __future__ import annotations

import abc
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable


class VisionProvider(abc.ABC):
    """Interface for computer vision model providers."""

    @abc.abstractmethod
    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate a caption for the given image."""
        raise NotImplementedError


class MockVisionProvider(VisionProvider):
    """Deterministic mock provider that mirrors the legacy behaviour."""

    _CAPTIONS: tuple[str, ...] = (
        "A vibrant abstract scene.",
        "A close-up portrait with warm light.",
        "A calm landscape with open skies.",
        "An energetic snapshot full of motion.",
    )

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Return a deterministic caption based on the image path."""

        del prompt, max_tokens  # Unused in the mock provider.
        key_material = str(Path(image_path))
        if seed is not None:
            key_material = f"{key_material}:{seed}"
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        index = int(digest, 16) % len(self._CAPTIONS)
        return self._CAPTIONS[index]


ProviderFactory = Callable[[], VisionProvider]


class _UnimplementedVisionProvider(VisionProvider):
    """Placeholder provider for backends that are not yet implemented."""

    def __init__(self, name: str) -> None:
        self._name = name

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:  # pragma: no cover - defensive guard
        del image_path, prompt, max_tokens, seed
        raise RuntimeError(f"Vision provider '{self._name}' is not implemented yet")


_PROVIDERS: Dict[str, ProviderFactory] = {}
_DEFAULT_PROVIDER = "mock"


def register_vision_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under the given name."""

    key = name.strip().lower()
    if not key:
        raise ValueError("vision provider name must be non-empty")
    _PROVIDERS[key] = factory


def available_vision_providers() -> Iterable[str]:
    """Return an iterable of registered provider names."""

    return sorted(_PROVIDERS)


def create_vision_provider(name: str | None = None) -> VisionProvider:
    """Instantiate a vision provider by name or from the environment."""

    provider_name = (name or os.getenv("VDPT_VISION_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()
    try:
        factory = _PROVIDERS[provider_name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown vision provider '{provider_name}'. Available providers: {available}"
        ) from exc
    provider = factory()
    if not isinstance(provider, VisionProvider):  # pragma: no cover - defensive guard
        raise TypeError("vision provider factory must return a VisionProvider instance")
    return provider


@lru_cache(maxsize=1)
def get_vision_provider() -> VisionProvider:
    """Return the configured vision provider, caching the instance."""

    return create_vision_provider()


def reset_vision_provider_cache() -> None:
    """Clear the cached provider instance (useful for tests)."""

    get_vision_provider.cache_clear()  # type: ignore[attr-defined]


# Register built-in providers.
register_vision_provider("mock", MockVisionProvider)
register_vision_provider("qwen_vl", lambda: _UnimplementedVisionProvider("qwen_vl"))
