"""Tests for capability plugin discovery."""

from types import SimpleNamespace

from pydantic_ai.capabilities import Capability

from sqlsaber.capabilities import plugins
from sqlsaber.capabilities.plugins import PluginContext, discover_capabilities
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


def test_discover_capabilities_isolates_broken_plugin(monkeypatch) -> None:
    broken = SimpleNamespace(
        name="broken", load=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(plugins, "_select_entry_points", lambda group: [broken])

    assert discover_capabilities(_context()) == []
