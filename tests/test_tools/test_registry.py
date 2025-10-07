"""Tests for the tool registry."""

import pytest

from sqlsaber.tools import Tool, ToolRegistry, register_tool


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

    def test_singleton_pattern(self):
        """Test that tools are singletons within registry."""
        registry = ToolRegistry()
        registry.register(MockTestTool1)

        tool1 = registry.get_tool("test_tool_1")
        tool2 = registry.get_tool("test_tool_1")

        # Should be the same instance
        assert tool1 is tool2


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
