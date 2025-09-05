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
