from backend.app.main import get_text_provider
from backend.vdpt.providers import MockProvider, QwenProvider


def test_factory_returns_mock_when_no_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    provider = get_text_provider()

    assert isinstance(provider, MockProvider)


def test_factory_returns_qwen_when_api_key_present(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    provider = get_text_provider()

    assert isinstance(provider, QwenProvider)
