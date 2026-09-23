"""Tests for capability plugin discovery."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from sqlsaber.capabilities import plugins
from sqlsaber.capabilities.plugins import (
    PluginContext,
    _load_capability_factories,
    discover_capabilities,
    resolve_capability_specs,
)
from sqlsaber.config.settings import Config
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.query_results import InMemoryQueryResultStore


class FakeCodexCredentialStore:
    def preflight(self) -> None:
        pass

    async def load(self) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(
            access_token="fake-access",
            refresh_token="fake-refresh",
            account_id="fake-account",
        )

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        del credentials


def _context(config: Config | None = None) -> PluginContext:
    registry = DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name="test",
                connection=SQLiteConnection("sqlite:///:memory:"),
                description=None,
                excluded_schemas=[],
            )
        ]
    )
    return PluginContext(
        registry=registry,
        knowledge_manager=KnowledgeManager(),
        allow_dangerous=True,
        config=config
        or Config.in_memory(
            model_name="anthropic:claude-main",
            api_keys={"anthropic": "main-key", "openai": "openai-key"},
        ),
        main_model_name="anthropic:claude-main",
        main_api_key="main-key",
        query_result_store=InMemoryQueryResultStore(),
    )


def test_discover_capabilities_delivers_plugin_context(monkeypatch) -> None:
    received: list[PluginContext] = []

    def factory(context: PluginContext):
        received.append(context)
        return Capability(id="test-plugin", instructions="plugin instructions")

    entry_point = SimpleNamespace(name="test", load=lambda: factory)
    monkeypatch.setattr(
        plugins,
        "_select_entry_points",
        lambda group: [entry_point] if group == "sqlsaber.capabilities" else [],
    )
    context = _context()

    discovered = discover_capabilities(context)

    assert [capability.id for capability in discovered] == ["test-plugin"]
    assert received == [context]
    assert received[0].allow_dangerous is True


def test_discover_capabilities_sorts_entry_points_by_name(monkeypatch) -> None:
    def entry_point(name: str):
        return SimpleNamespace(
            name=name,
            load=lambda: (
                lambda context: Capability(
                    id=name, instructions=context.main_model_name
                )
            ),
        )

    monkeypatch.setattr(
        plugins,
        "_select_entry_points",
        lambda group: [entry_point("zeta"), entry_point("alpha")],
    )

    discovered = discover_capabilities(_context())

    assert [capability.id for capability in discovered] == ["alpha", "zeta"]


@pytest.mark.parametrize(
    (
        "explicit_model",
        "explicit_key",
        "expected_model",
        "expected_key",
    ),
    [
        (
            "openai:gpt-explicit",
            "explicit-key",
            "openai:gpt-explicit",
            "explicit-key",
        ),
        ("openai:gpt-explicit", None, "openai:gpt-explicit", None),
        (None, None, "anthropic:claude-main", "main-key"),
        (None, "explicit-key", "anthropic:claude-main", "explicit-key"),
    ],
)
def test_plugin_context_model_precedence(
    monkeypatch,
    explicit_model,
    explicit_key,
    expected_model,
    expected_key,
) -> None:
    context = _context()
    resolved_model = object()
    resolve = Mock(
        return_value=SimpleNamespace(model=resolved_model, provider="test-provider")
    )
    monkeypatch.setattr("sqlsaber.agents.model_factory.resolve_model", resolve)

    model_name, model, provider = context.resolve_subagent_model(
        model_name=explicit_model,
        api_key=explicit_key,
    )

    assert (model_name, model, provider) == (
        expected_model,
        resolved_model,
        "test-provider",
    )
    resolve.assert_called_once_with(
        context.config.auth,
        expected_model,
        api_key_override=expected_key,
    )


def test_plugin_context_ignores_persisted_legacy_subagent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    config = Config()
    legacy_payload = {
        "version": 2,
        "model": "anthropic:claude-main",
        "thinking": {"enabled": True, "level": "medium"},
        "subagents": {"notebook": "openai:gpt-legacy"},
    }
    legacy_text = json.dumps(legacy_payload, indent=2)
    config.model._manager.config_file.write_text(legacy_text)
    context = _context(config)
    resolve = Mock(
        return_value=SimpleNamespace(model="resolved-model", provider="anthropic")
    )
    monkeypatch.setattr("sqlsaber.agents.model_factory.resolve_model", resolve)

    model_name, _, _ = context.resolve_subagent_model()

    assert model_name == "anthropic:claude-main"
    resolve.assert_called_once_with(
        config.auth,
        "anthropic:claude-main",
        api_key_override="main-key",
    )
    assert config.model._manager.config_file.read_text() == legacy_text


def test_plugin_context_resolves_explicit_codex_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "sqlsaber.config.openai_codex.OpenAICodexCredentialStore",
        FakeCodexCredentialStore,
    )
    context = _context()

    model_name, model, provider = context.resolve_subagent_model(
        model_name="openai-codex:gpt-test"
    )

    assert model_name == "openai-codex:gpt-test"
    assert isinstance(model, OpenAICodexModel)
    assert provider == "openai-codex"


def test_discover_capabilities_isolates_broken_plugin(monkeypatch) -> None:
    broken = SimpleNamespace(
        name="broken", load=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(plugins, "_select_entry_points", lambda group: [broken])

    assert discover_capabilities(_context()) == []


def test_plugin_context_requires_query_result_store() -> None:
    registry = DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name="test",
                connection=SQLiteConnection("sqlite:///:memory:"),
                description=None,
                excluded_schemas=[],
            )
        ]
    )
    with pytest.raises(TypeError, match="query_result_store"):
        PluginContext(
            registry=registry,
            knowledge_manager=KnowledgeManager(),
            allow_dangerous=True,
            config=Config.in_memory(
                model_name="anthropic:claude-main",
                api_keys={"anthropic": "main-key"},
            ),
            main_model_name="anthropic:claude-main",
        )


def test_entry_point_loader_does_not_invoke(monkeypatch) -> None:
    called: list[PluginContext] = []

    def factory(context: PluginContext):
        called.append(context)
        return Capability(id="lazy", instructions="plugin instructions")

    entry_point = SimpleNamespace(name="lazy", load=lambda: factory)
    monkeypatch.setattr(
        plugins,
        "_select_entry_points",
        lambda group: [entry_point] if group == "sqlsaber.capabilities" else [],
    )

    factories = _load_capability_factories()
    assert called == []
    assert [item.name for item in factories] == ["lazy"]

    context = _context()
    discovered = resolve_capability_specs(factories, context)
    assert [capability.id for capability in discovered] == ["lazy"]
    assert called == [context]


def test_resolve_capability_specs_keeps_instances_and_invokes_factories() -> None:
    extra = Capability(id="custom", instructions="Custom capability")
    context = _context()

    def factory(plugin_context: PluginContext):
        assert plugin_context is context
        return Capability(id="from-factory", instructions="factory")

    resolved = resolve_capability_specs([extra, factory], context)
    assert [capability.id for capability in resolved] == ["custom", "from-factory"]
