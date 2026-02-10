"""Tests for the tool registry."""

from unittest.mock import MagicMock

import pytest

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.tools import Tool, ToolRegistry, register_tool
from sqlsaber.tools.sql_tools import SQLTool


class MockTestTool1(Tool):
    """Test tool 1."""

    @property
    def name(self) -> str:
        return "test_tool_1"

    @property
    def description(self) -> str:
        return "Test tool 1"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return '{"result": "test1"}'


class MockTestTool2(Tool):
    """Test tool 2."""

    @property
    def name(self) -> str:
        return "test_tool_2"

    @property
    def description(self) -> str:
        return "Test tool 2"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return '{"result": "test2"}'


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving tools."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        tool = registry.get_tool("test_tool_1")
        assert tool.name == "test_tool_1"
        assert isinstance(tool, MockTestTool1)

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate tools raises error."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockTestTool1)

    def test_get_unknown_tool_raises_error(self):
        """Test that getting unknown tool raises error."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_tool("unknown_tool")

    def test_unregister_tool(self):
        """Test unregistering tools."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        # Verify tool exists
        tool = registry.get_tool("test_tool_1")
        assert tool is not None

        # Unregister
        registry.unregister("test_tool_1")

        # Verify tool is gone
        with pytest.raises(KeyError):
            registry.get_tool("test_tool_1")

    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)
        registry.register(MockTestTool2)

        # List all tools
        all_tools = registry.list_tools()
        assert len(all_tools) == 2
        assert "test_tool_1" in all_tools
        assert "test_tool_2" in all_tools

    def test_get_all_tools(self):
        """Test getting all tool instances."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)
        registry.register(MockTestTool2)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        assert any(tool.name == "test_tool_1" for tool in all_tools)
        assert any(tool.name == "test_tool_2" for tool in all_tools)

    def test_get_tool_returns_fresh_instances(self):
        """Test that get_tool returns a new instance each time for isolation."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        tool1 = registry.get_tool("test_tool_1")
        tool2 = registry.get_tool("test_tool_1")

        assert tool1 is not tool2
        assert type(tool1) is type(tool2)

    def test_create_tool_returns_fresh_instances(self):
        """Test that create_tool always returns a new instance."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        tool1 = registry.create_tool("test_tool_1")
        tool2 = registry.create_tool("test_tool_1")

        assert tool1 is not tool2
        assert isinstance(tool1, MockTestTool1)

    def test_get_tool_class(self):
        """Test that get_tool_class returns the registered class."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        cls = registry.get_tool_class("test_tool_1")
        assert cls is MockTestTool1

    def test_create_all_tools(self):
        """Test creating fresh instances of all registered tools."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)
        registry.register(MockTestTool2)

        tools = registry.create_all_tools()
        assert len(tools) == 2
        assert any(isinstance(t, MockTestTool1) for t in tools)
        assert any(isinstance(t, MockTestTool2) for t in tools)


class TestRegisterDecorator:
    """Test the @register_tool decorator."""

    def test_decorator_registers_tool(self):
        """Test that decorator registers tool with global registry."""
        # Import the global registry
        from sqlsaber.tools import tool_registry

        # Define a tool with decorator
        @register_tool
        class DecoratedTool(Tool):
            @property
            def name(self) -> str:
                return "decorated_tool_test"

            @property
            def description(self) -> str:
                return "A decorated tool"

            @property
            def input_schema(self) -> dict:
                return {"type": "object"}

            async def execute(self, **kwargs) -> str:
                return '{"result": "decorated"}'

        # Check it was registered in the global registry
        tool = tool_registry.get_tool("decorated_tool_test")
        assert tool.name == "decorated_tool_test"
        assert isinstance(tool, DecoratedTool)

        # Clean up
        tool_registry.unregister("decorated_tool_test")


class MockSQLTool(SQLTool):
    """A mock SQL tool for cross-talk testing."""

    @property
    def name(self) -> str:
        return "mock_sql_tool"

    async def execute(self, **kwargs) -> str:
        return '{"result": "mock"}'


class TestNoCrossTalk:
    """Regression tests: per-agent tool instances must not share mutable state."""

    def test_create_tool_instances_are_independent(self):
        """Two instances from create_tool must not share db or allow_dangerous."""
        registry = ToolRegistry()
        registry.register(MockSQLTool)

        tool_a = registry.create_tool("mock_sql_tool")
        tool_b = registry.create_tool("mock_sql_tool")

        assert isinstance(tool_a, SQLTool)
        assert isinstance(tool_b, SQLTool)

        db_a = MagicMock(spec=BaseDatabaseConnection)
        db_b = MagicMock(spec=BaseDatabaseConnection)
        schema_a = MagicMock(spec=SchemaManager)
        schema_b = MagicMock(spec=SchemaManager)

        tool_a.set_connection(db_a, schema_a)
        tool_a.allow_dangerous = True

        tool_b.set_connection(db_b, schema_b)
        tool_b.allow_dangerous = False

        assert tool_a.db is db_a
        assert tool_a.schema_manager is schema_a
        assert tool_a.allow_dangerous is True

        assert tool_b.db is db_b
        assert tool_b.schema_manager is schema_b
        assert tool_b.allow_dangerous is False

    def test_configuring_one_agent_tools_does_not_affect_another(self):
        """Simulate two agents creating tools from the same registry."""
        registry = ToolRegistry()
        registry.register(MockSQLTool)

        agent_a_tools = {
            name: registry.create_tool(name) for name in registry.list_tools()
        }
        agent_b_tools = {
            name: registry.create_tool(name) for name in registry.list_tools()
        }

        db_a = MagicMock(spec=BaseDatabaseConnection)
        db_b = MagicMock(spec=BaseDatabaseConnection)
        schema_a = MagicMock(spec=SchemaManager)
        schema_b = MagicMock(spec=SchemaManager)

        for tool in agent_a_tools.values():
            if isinstance(tool, SQLTool):
                tool.set_connection(db_a, schema_a)
                tool.allow_dangerous = True

        for tool in agent_b_tools.values():
            if isinstance(tool, SQLTool):
                tool.set_connection(db_b, schema_b)
                tool.allow_dangerous = False

        a_tool = agent_a_tools["mock_sql_tool"]
        b_tool = agent_b_tools["mock_sql_tool"]

        assert a_tool is not b_tool
        assert a_tool.db is db_a
        assert a_tool.allow_dangerous is True
        assert b_tool.db is db_b
        assert b_tool.allow_dangerous is False
