from __future__ import annotations

from typing import Any

import pytest

from sqlsaber.cli.models import FetchedModel
from sqlsaber.cli.workflows.model_selection import choose_model
from sqlsaber.config.settings import ModelConfigManager


def _model(model_id: str, provider: str, name: str) -> FetchedModel:
    return FetchedModel(
        id=model_id,
        provider=provider,
        name=name,
        description="$1/$2 per 1M tokens",
        context_length=128000,
        knowledge="",
    )


class CancelPrompter:
    def __init__(self) -> None:
        self.choices: list[Any] = []

    async def select(
        self,
        message: str,
        choices: list[Any] | None = None,
        default: Any = None,
        use_search_filter: bool = False,
        use_jk_keys: bool = True,
    ) -> Any:
        self.choices = list(choices or [])
        return None


@pytest.mark.asyncio
async def test_choose_model_labels_qualified_recommendation_and_returns_it_on_cancel():
    prompter = CancelPrompter()
    models = [
        _model("openai:gpt-4.1", "openai", "GPT-4.1"),
        _model("openai:gpt-5", "openai", "GPT-5"),
        _model("openai:gpt-5.6-sol", "openai", "GPT-5.6 Sol"),
    ]

    selected = await choose_model(prompter, models, restrict_provider="openai")

    assert selected == ModelConfigManager.DEFAULT_MODEL
    assert selected == "openai:gpt-5.6-sol"
    assert prompter.choices[0].title == "GPT-5.6 Sol (Recommended)"
    assert prompter.choices[0].value == "openai:gpt-5.6-sol"
    assert all("GPT-5 (Recommended)" not in choice.title for choice in prompter.choices)
