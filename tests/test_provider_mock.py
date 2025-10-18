from backend.vdpt.providers import MockProvider


def test_mock_provider_is_deterministic():
    provider = MockProvider()
    prompt = "Tell me a story"

    first = provider.generate(prompt)
    second = provider.generate(prompt)

    assert first == second
    assert first.startswith("[MOCK]")
