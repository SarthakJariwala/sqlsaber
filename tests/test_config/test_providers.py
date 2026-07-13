import importlib

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
    ("module_name", "model_class"),
    [
        ("pydantic_ai.models.groq", "GroqModel"),
        ("pydantic_ai.models.mistral", "MistralModel"),
        ("pydantic_ai.models.cohere", "CohereModel"),
        ("pydantic_ai.models.huggingface", "HuggingFaceModel"),
    ],
)
def test_supported_provider_dependencies_are_installed(
    module_name: str, model_class: str
) -> None:
    module = importlib.import_module(module_name)
    assert getattr(module, model_class) is not None


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
