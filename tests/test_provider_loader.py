import importlib

import pytest

from backend.vdpt import providers


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    monkeypatch.setenv("VDPT_PROVIDER", "dummy")
    importlib.reload(providers)
    yield
    monkeypatch.setenv("VDPT_PROVIDER", "dummy")
    importlib.reload(providers)


def test_loader_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("VDPT_PROVIDER", raising=False)
    importlib.reload(providers)
    assert providers.PROVIDER_NAME == "openai"
    assert providers.provider.__name__.endswith(".openai")


def test_loader_switches_to_named_provider(monkeypatch):
    monkeypatch.setenv("VDPT_PROVIDER", "qwen")
    importlib.reload(providers)
    assert providers.PROVIDER_NAME == "qwen"
    assert providers.provider.__name__.endswith(".qwen")


def test_loader_falls_back_to_dummy(monkeypatch):
    monkeypatch.setenv("VDPT_PROVIDER", "unknown")
    importlib.reload(providers)
    assert providers.provider.__name__.endswith(".dummy")
