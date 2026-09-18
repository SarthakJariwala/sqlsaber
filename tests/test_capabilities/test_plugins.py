"""Tests for capability plugin discovery."""

from types import SimpleNamespace

import pytest
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from sqlsaber.agents.model_factory import resolve_model
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
from sqlsaber.nested_model import INHERIT, pin
from sqlsaber.overrides import normalize_tool_overides
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


def _context() -> PluginContext:
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
    config = Config.in_memory(
        model_name="anthropic:claude-main",
        api_keys={"anthropic": "main-key", "openai": "openai-key"},
    )
    main = resolve_model(config.auth, "anthropic:claude-main")
    return PluginContext(
        registry=registry,
        knowledge_manager=KnowledgeManager(),
        allow_dangerous=True,
        tool_overrides=normalize_tool_overides({"viz": pin("openai:gpt-test")}),
        auth=config.auth,
        main=main,
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
    assert str(received[0].tool_overrides["viz"].id) == "openai:gpt-test"


def test_discover_capabilities_sorts_entry_points_by_name(monkeypatch) -> None:
    def entry_point(name: str):
        return SimpleNamespace(
            name=name,
            load=lambda: (
                lambda context: Capability(
                    id=name, instructions=context.main.model_name
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


def test_plugin_context_resolves_subagent_precedence(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = _context()

    inherited = context.resolve_subagent_model(INHERIT, tool="analyze_data")
    assert inherited is context.main
    assert inherited.model_name == "anthropic:claude-main"

    pinned = context.resolve_subagent_model(
        pin("openai:gpt-notebook"), tool="analyze_data"
    )
    assert pinned.model_name == "openai:gpt-notebook"
    assert pinned.provider == "openai"
    assert pinned.api_key == "openai-key"
    assert pinned.api_key != context.main.api_key


def test_plugin_context_tool_override_wins(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = _context()

    resolved = context.resolve_subagent_model(pin("openai:gpt-notebook"), tool="viz")

    assert resolved.model_name == "openai:gpt-test"
    assert resolved.provider == "openai"


def test_plugin_context_resolves_codex_subagent_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "sqlsaber.config.openai_codex.OpenAICodexCredentialStore",
        FakeCodexCredentialStore,
    )
    context = _context()

    resolved = context.resolve_subagent_model(
        pin("openai-codex:gpt-test"), tool="analyze_data"
    )

    assert resolved.model_name == "openai-codex:gpt-test"
    assert isinstance(resolved.model, OpenAICodexModel)
    assert resolved.provider == "openai-codex"


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
            tool_overrides={},
            auth=Config.in_memory(
                model_name="anthropic:claude-main",
                api_keys={"anthropic": "main-key"},
            ).auth,
            main=resolve_model(
                Config.in_memory(
                    model_name="anthropic:claude-main",
                    api_keys={"anthropic": "main-key"},
                ).auth,
                "anthropic:claude-main",
            ),
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
