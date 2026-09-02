from __future__ import annotations

from types import MappingProxyType

import pytest

from sqlsaber.cli.models import ModelManager
from sqlsaber.config.settings import ModelConfigManager

CURRENT_QUALIFIED_RECS = {
    "google": "google:gemini-2.5-pro",
    "groq": "groq:llama-3-3-70b-versatile",
    "mistral": "mistral:mistral-large-latest",
    "cohere": "cohere:command-r-plus",
}


def test_openai_recommendation_is_product_default():
    recommended = ModelManager.recommended_model_id("openai")
    assert recommended == ModelConfigManager.DEFAULT_MODEL
    assert recommended == "openai:gpt-5.6-sol"


def test_anthropic_recommendation_is_opus_5():
    assert ModelManager.recommended_model_id("anthropic") == "anthropic:claude-opus-5"


def test_huggingface_has_no_recommendation():
    assert ModelManager.recommended_model_id("huggingface") is None


@pytest.mark.parametrize(
    ("provider", "expected"),
    list(CURRENT_QUALIFIED_RECS.items()),
)
def test_other_provider_recommendations_stay_qualified(provider: str, expected: str):
    assert ModelManager.recommended_model_id(provider) == expected


def test_recommendation_registry_is_immutable():
    registry = ModelManager._RECOMMENDED_MODEL_IDS
    assert isinstance(registry, MappingProxyType)
    with pytest.raises(TypeError):
        registry["openai"] = "openai:mutated"


def test_public_recommended_models_table_is_removed():
    assert not hasattr(ModelManager, "RECOMMENDED_MODELS")


def test_builder_rejects_non_openai_product_default():
    from sqlsaber.cli.models import (
        _RecommendationSource,
        _build_recommendation_registry,
    )

    with pytest.raises(ValueError, match="openai"):
        _build_recommendation_registry(
            {
                "openai": _RecommendationSource.PRODUCT_DEFAULT,
                "anthropic": "claude-opus-5",
            },
            product_default="anthropic:claude-opus-5",
        )


def test_builder_rejects_openai_string_literal():
    from sqlsaber.cli.models import _build_recommendation_registry

    with pytest.raises(ValueError, match="openai"):
        _build_recommendation_registry(
            {
                "openai": "gpt-5.6-sol",
                "anthropic": "claude-opus-5",
            },
            product_default=ModelConfigManager.DEFAULT_MODEL,
        )


def test_builder_qualifies_and_freezes_specs():
    from sqlsaber.cli.models import (
        _RecommendationSource,
        _build_recommendation_registry,
    )

    built = _build_recommendation_registry(
        {
            "anthropic": "claude-opus-5",
            "openai": _RecommendationSource.PRODUCT_DEFAULT,
        },
        product_default=ModelConfigManager.DEFAULT_MODEL,
    )
    assert built["openai"] == ModelConfigManager.DEFAULT_MODEL
    assert built["anthropic"] == "anthropic:claude-opus-5"
    assert isinstance(built, MappingProxyType)
    with pytest.raises(TypeError):
        built["anthropic"] = "anthropic:mutated"


@pytest.mark.asyncio
async def test_select_model_empty_fetch_uses_qualified_recommendation(monkeypatch):
    from sqlsaber.cli.onboarding import select_model_for_provider

    async def empty_fetch(self, providers=None):
        return []

    monkeypatch.setattr(ModelManager, "fetch_available_models", empty_fetch)

    assert await select_model_for_provider("openai") == ModelConfigManager.DEFAULT_MODEL
    assert await select_model_for_provider("anthropic") == "anthropic:claude-opus-5"
    assert await select_model_for_provider("huggingface") == ModelManager.DEFAULT_MODEL


@pytest.mark.asyncio
async def test_select_model_fetch_exception_uses_qualified_recommendation(
    monkeypatch,
):
    from sqlsaber.cli.onboarding import select_model_for_provider

    async def boom(self, providers=None):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(ModelManager, "fetch_available_models", boom)

    assert await select_model_for_provider("openai") == ModelConfigManager.DEFAULT_MODEL
    assert await select_model_for_provider("anthropic") == "anthropic:claude-opus-5"
    assert await select_model_for_provider("huggingface") == ModelManager.DEFAULT_MODEL
