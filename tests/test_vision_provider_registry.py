import pytest

from backend.vdpt.providers.vision import (
    MockVisionProvider,
    create_vision_provider,
    get_vision_provider,
    reset_vision_provider_cache,
)


def test_create_vision_provider_explicit_mock():
    provider = create_vision_provider("mock")
    assert isinstance(provider, MockVisionProvider)


def test_create_vision_provider_from_env(monkeypatch):
    monkeypatch.setenv("VDPT_VISION_PROVIDER", "mock")
    provider = create_vision_provider()
    assert isinstance(provider, MockVisionProvider)


def test_get_vision_provider_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("VDPT_VISION_PROVIDER", raising=False)
    reset_vision_provider_cache()
    provider = get_vision_provider()
    assert isinstance(provider, MockVisionProvider)
    reset_vision_provider_cache()


def test_get_vision_provider_is_cached(monkeypatch):
    monkeypatch.setenv("VDPT_VISION_PROVIDER", "mock")
    reset_vision_provider_cache()
    provider_one = get_vision_provider()
    provider_two = get_vision_provider()
    assert provider_one is provider_two
    reset_vision_provider_cache()


def test_reset_cache_creates_new_instance(monkeypatch):
    monkeypatch.setenv("VDPT_VISION_PROVIDER", "mock")
    reset_vision_provider_cache()
    provider_one = get_vision_provider()
    reset_vision_provider_cache()
    provider_two = get_vision_provider()
    assert provider_one is not provider_two
    reset_vision_provider_cache()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("VDPT_VISION_PROVIDER", "unknown")
    with pytest.raises(ValueError):
        create_vision_provider()
