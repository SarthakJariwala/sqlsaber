from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx
import pytest

from sqlsaber.cli import models as models_cli


class _CatalogClient:
    def __init__(self, catalog: dict[str, Any]) -> None:
        self._catalog = catalog

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    async def get(self, url: str) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(200, json=self._catalog, request=request)


@pytest.fixture
def openai_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = {
        "openai": {
            "models": {
                "gpt-catalog": {
                    "name": "GPT Catalog",
                    "knowledge": "2025-06",
                    "limit": {"context": 123_456},
                    "cost": {"input": 1.25, "output": 10},
                }
            }
        }
    }
    monkeypatch.setattr(
        models_cli.httpx,
        "AsyncClient",
        lambda **_: _CatalogClient(catalog),
    )


@pytest.mark.asyncio
async def test_openai_catalog_keeps_openai_qualified_ids(openai_catalog) -> None:
    models = await models_cli.ModelManager().fetch_available_models(
        providers=["openai"]
    )

    assert models == [
        {
            "id": "openai:gpt-catalog",
            "provider": "openai",
            "name": "GPT Catalog",
            "description": "$1.25/10 per 1M tokens",
            "context_length": 123_456,
            "knowledge": "2025-06",
        }
    ]


@pytest.mark.asyncio
async def test_codex_catalog_projects_openai_models_to_codex_ids(
    openai_catalog,
) -> None:
    models = await models_cli.ModelManager().fetch_available_models(
        providers=["openai-codex"]
    )

    assert models == [
        {
            "id": "openai-codex:gpt-catalog",
            "provider": "openai-codex",
            "name": "GPT Catalog",
            "description": "$1.25/10 per 1M tokens",
            "context_length": 123_456,
            "knowledge": "2025-06",
        }
    ]


@pytest.mark.asyncio
async def test_default_catalog_lists_openai_and_codex_ids(openai_catalog) -> None:
    models = await models_cli.ModelManager().fetch_available_models()

    assert models == [
        {
            "id": "openai:gpt-catalog",
            "provider": "openai",
            "name": "GPT Catalog",
            "description": "$1.25/10 per 1M tokens",
            "context_length": 123_456,
            "knowledge": "2025-06",
        },
        {
            "id": "openai-codex:gpt-catalog",
            "provider": "openai-codex",
            "name": "GPT Catalog",
            "description": "$1.25/10 per 1M tokens",
            "context_length": 123_456,
            "knowledge": "2025-06",
        },
    ]
