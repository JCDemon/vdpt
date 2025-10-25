from backend.vdpt.providers import mock as mock_provider


def test_mock_provider_is_deterministic():
    prompt = "Tell me a story"

    first = mock_provider.chat(prompt)
    second = mock_provider.chat(prompt)

    assert first == second
    assert first.startswith("[mock-chat]")
