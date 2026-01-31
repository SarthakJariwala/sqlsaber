"""Tests for tool plugin discovery."""

import pytest

from sqlsaber.tools.base import Tool
from sqlsaber.tools.registry import ToolRegistry, discover_plugins


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy_tool"

    async def execute(self) -> str:
        return "{}"


class DummyToolAlt(Tool):
    @property
    def name(self) -> str:
        return "dummy_tool_alt"

    async def execute(self) -> str:
        return "{}"


def _fake_entry_points(group: str):
    class DummyEP:
        def __init__(self, name: str, obj):
            self.name = name
            self._obj = obj

        def load(self):
            return self._obj

    if group != "sqlsaber.tools":
        return []

    def factory(registry: ToolRegistry):
        registry.register(DummyToolAlt)
        return [DummyToolAlt]

    return [
        DummyEP("class", DummyTool),
        DummyEP("factory", factory),
    ]


def test_discover_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()

    monkeypatch.setattr(
        "sqlsaber.tools.registry._select_entry_points", _fake_entry_points
    )

    loaded = discover_plugins(registry)

    assert sorted(loaded) == ["class", "factory"]
    assert set(registry.list_tools()) == {"dummy_tool", "dummy_tool_alt"}
