"""Tests for capability plugin discovery."""

from types import SimpleNamespace

from pydantic_ai.capabilities import Capability

from sqlsaber.capabilities import plugins
from sqlsaber.capabilities.plugins import PluginContext, discover_capabilities
from sqlsaber.config.settings import Config
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.overrides import ModelOverides


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
    return PluginContext(
        registry=registry,
        knowledge_manager=KnowledgeManager(),
        allow_dangerous=True,
        tool_overrides={"viz": ModelOverides(model_name="openai:gpt-test")},
        config=Config.in_memory(
            model_name="anthropic:claude-main",
            api_keys={"anthropic": "main-key", "openai": "openai-key"},
        ),
        main_model_name="anthropic:claude-main",
        main_api_key="main-key",
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
    assert received[0].tool_overrides["viz"].model_name == "openai:gpt-test"


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


def test_plugin_context_resolves_subagent_precedence(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = _context()

    model_name, model, provider = context.resolve_subagent_model("notebook")
    assert model_name == "anthropic:claude-main"
    assert model.model_name == "claude-main"
    assert provider == "anthropic"

    context.config.model.set_subagent_model("notebook", "openai:gpt-notebook")
    model_name, model, provider = context.resolve_subagent_model("notebook")
    assert model_name == "openai:gpt-notebook"
    assert model.model_name == "gpt-notebook"
    assert provider == "openai"


def test_plugin_context_tool_override_wins(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = _context()

    model_name, model, provider = context.resolve_subagent_model(
        "notebook", tool_name="viz"
    )

    assert model_name == "openai:gpt-test"
    assert model.model_name == "gpt-test"
    assert provider == "openai"


def test_discover_capabilities_isolates_broken_plugin(monkeypatch) -> None:
    broken = SimpleNamespace(
        name="broken", load=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(plugins, "_select_entry_points", lambda group: [broken])

    assert discover_capabilities(_context()) == []
