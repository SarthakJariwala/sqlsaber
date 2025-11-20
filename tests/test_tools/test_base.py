"""Tests for the base tool class."""

import pytest

from sqlsaber.tools.base import Tool


class MockTool(Tool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    async def execute(self, **kwargs) -> str:
        message = kwargs.get("message", "")
        return f'{{"result": "{message}"}}'


class TestBaseTool:
    """Test the base Tool class."""

    def test_tool_properties(self):
        """Test tool properties."""
        tool = MockTool()
        assert tool.name == "mock_tool"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test tool execution."""
        tool = MockTool()
        result = await tool.execute(message="Hello, World!")
        assert result == '{"result": "Hello, World!"}'
