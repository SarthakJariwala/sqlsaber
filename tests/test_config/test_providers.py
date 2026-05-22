import pytest

from sqlsaber.config import providers


def test_all_keys_contains_expected_providers():
    keys = providers.all_keys()
    # Stable core set
    for k in [
        "anthropic",
        "openai",
        "google",
        "groq",
        "mistral",
        "cohere",
        "huggingface",
    ]:
        assert k in keys


def test_env_var_name_mapping():
    assert providers.env_var_name("openai") == "OPENAI_API_KEY"
    assert providers.env_var_name("anthropic") == "ANTHROPIC_API_KEY"
    assert providers.env_var_name("unknown") == "AI_API_KEY"


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-7", True),
        ("anthropic:claude-opus-4-7-20260501", True),
        ("claude-opus-4-6", False),
        ("anthropic:claude-sonnet-4-6", False),
        ("openai:gpt-5", False),
        ("", False),
    ],
)
def test_requires_anthropic_adaptive_thinking(model: str, expected: bool):
    assert providers.requires_anthropic_adaptive_thinking(model) is expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("anthropic:claude-3", "anthropic"),
        ("openai:gpt-4o", "openai"),
        ("google:gemini-1.5-pro", "google"),
        ("google-gla:gemini-1.5-pro", "google"),
        ("mistral:large", "mistral"),
        ("unknown:model", None),
        ("", None),
    ],
)
def test_provider_from_model(model: str, expected: str | None):
    assert providers.provider_from_model(model) == expected
