import importlib

import pytest

from backend.vdpt import providers


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    monkeypatch.setenv("VDPT_PROVIDER", "mock")
    importlib.reload(providers)
    yield
    monkeypatch.setenv("VDPT_PROVIDER", "mock")
    importlib.reload(providers)


def test_loader_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("VDPT_PROVIDER", raising=False)
    importlib.reload(providers)
    assert providers.current.__name__.endswith(".mock")


def test_loader_switches_to_named_provider(monkeypatch):
    monkeypatch.setenv("VDPT_PROVIDER", "qwen")
    importlib.reload(providers)
    assert providers.current.__name__.endswith(".qwen")
