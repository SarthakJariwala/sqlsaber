"""Tests for multi-database SQLSaberAgent: tool schemas and prompt routing."""

from __future__ import annotations

import inspect

import pytest

from sqlsaber.agents.pydantic_ai_agent import (
    SQLSaberAgent,
    _wrap_add_db_name,
    _wrap_strip_db_name,
)
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.tools.sql_tools import ExecuteSQLTool, ListTablesTool


def _registry(*names: str) -> DatabaseRegistry:
    return DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name=n,
                connection=SQLiteConnection("sqlite:///:memory:"),
                description=None,
                excluded_schemas=[],
            )
            for n in names
        ]
    )


def _tool_def(agent: SQLSaberAgent, tool_name: str):
    return agent.agent._function_toolset.tools[tool_name].tool_def


class TestWrappers:
    def test_strip_removes_db_name_from_signature(self):
        tool = ListTablesTool()
        wrapper = _wrap_strip_db_name(tool)
        sig = inspect.signature(wrapper)
        assert "db_name" not in sig.parameters

    def test_strip_calls_raw_execute_without_db_name(self):
        tool = ListTablesTool()
        wrapper = _wrap_strip_db_name(tool)

        captured: dict[str, object] = {}

        async def fake_execute(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "{}"

        tool.execute = fake_execute  # type: ignore[method-assign]
        # Re-wrap after monkey-patch
        wrapper = _wrap_strip_db_name(tool)

        import asyncio

        asyncio.run(wrapper())
        assert "db_name" not in captured["kwargs"]

    def test_add_makes_db_name_required_literal(self):
        tool = ExecuteSQLTool()
        wrapper = _wrap_add_db_name(tool, ("prod", "staging"))
        sig = inspect.signature(wrapper)
        assert "db_name" in sig.parameters
        param = sig.parameters["db_name"]
        # Required => no default
        assert param.default is inspect.Parameter.empty
        # Annotation should be a Literal of the names
        import typing

        origin = typing.get_origin(param.annotation)
        args = typing.get_args(param.annotation)
        assert origin is typing.Literal
        assert set(args) == {"prod", "staging"}


class TestAgentSchema:
    def test_single_db_schema_excludes_db_name(self):
        registry = _registry("solo")
        agent = SQLSaberAgent(
            registry=registry,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        for tool_name in ("list_tables", "introspect_schema", "execute_sql"):
            schema = _tool_def(agent, tool_name).parameters_json_schema
            assert "db_name" not in schema.get("properties", {}), (
                f"{tool_name} should not expose db_name in single-DB mode"
            )

    def test_multi_db_schema_requires_db_name_as_enum(self):
        registry = _registry("prod", "staging")
        agent = SQLSaberAgent(
            registry=registry,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        for tool_name in (
            "list_tables",
            "introspect_schema",
            "execute_sql",
            "search_knowledge",
        ):
            schema = _tool_def(agent, tool_name).parameters_json_schema
            assert "db_name" in schema.get("properties", {})
            db_prop = schema["properties"]["db_name"]
            # Literal types render as `enum` in JSON schema
            assert db_prop.get("enum") == ["prod", "staging"]
            assert "db_name" in schema.get("required", [])

    def test_list_dbs_only_registered_in_multi_db_mode(self):
        single = SQLSaberAgent(
            registry=_registry("solo"),
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        assert "list_dbs" not in single.agent._function_toolset.tools

        multi = SQLSaberAgent(
            registry=_registry("a", "b"),
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        assert "list_dbs" in multi.agent._function_toolset.tools


class TestAgentPrompt:
    def test_multi_db_prompt_contains_catalog(self):
        registry = _registry("prod", "staging")
        agent = SQLSaberAgent(
            registry=registry,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        prompt = agent.system_prompt_text()
        assert "prod" in prompt
        assert "staging" in prompt
        assert "db_name" in prompt.lower()

    def test_single_db_prompt_unchanged(self):
        registry = _registry("solo")
        agent = SQLSaberAgent(
            registry=registry,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        prompt = agent.system_prompt_text()
        # Today's prompt starts with this opener.
        assert prompt.startswith("You are a helpful SQL assistant")
        assert "Connected databases:" not in prompt


class TestKnowledgeRouting:
    @pytest.mark.asyncio
    async def test_knowledge_tool_registry_is_attached_in_multi_db(self):
        registry = _registry("prod", "staging")
        agent = SQLSaberAgent(
            registry=registry,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="x",
        )
        tool = agent._tools["search_knowledge"]
        assert tool.registry is registry
