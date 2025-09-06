"""Tests for dynamic instruction system."""

from sqlsaber.tools import Tool, ToolCategory, ToolRegistry, WorkflowPosition
from sqlsaber.tools.instructions import InstructionBuilder


class MockDiscoveryTool(Tool):
    """Mock discovery tool for testing."""

    @property
    def name(self) -> str:
        return "mock_discovery"

    @property
    def description(self) -> str:
        return "Mock discovery tool for testing"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SQL

    def get_usage_instructions(self) -> str | None:
        return "Use this tool first to discover things"

    def get_priority(self) -> int:
        return 10

    def get_workflow_position(self) -> WorkflowPosition:
        return WorkflowPosition.DISCOVERY

    async def execute(self, **kwargs) -> str:
        return '{"result": "discovery"}'


class MockAnalysisTool(Tool):
    """Mock analysis tool for testing."""

    @property
    def name(self) -> str:
        return "mock_analysis"

    @property
    def description(self) -> str:
        return "Mock analysis tool for testing"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SQL

    def get_usage_instructions(self) -> str | None:
        return "Use this tool to analyze data"

    def get_priority(self) -> int:
        return 20

    def get_workflow_position(self) -> WorkflowPosition:
        return WorkflowPosition.ANALYSIS

    async def execute(self, **kwargs) -> str:
        return '{"result": "analysis"}'


class TestInstructionBuilder:
    """Test the InstructionBuilder class."""

    def test_init(self):
        """Test InstructionBuilder initialization."""
        registry = ToolRegistry()
        builder = InstructionBuilder(registry)
        assert builder.registry is registry

    def test_build_instructions_empty_registry(self):
        """Test building instructions with empty registry."""
        registry = ToolRegistry()
        builder = InstructionBuilder(registry)

        instructions = builder.build_instructions("PostgreSQL")

        # Should still include base instructions
        assert "PostgreSQL database" in instructions
        assert "Your responsibilities:" in instructions

    def test_build_instructions_with_tools(self):
        """Test building instructions with tools."""
        registry = ToolRegistry()
        registry.register(MockDiscoveryTool)
        registry.register(MockAnalysisTool)

        builder = InstructionBuilder(registry)
        instructions = builder.build_instructions("PostgreSQL")

        # Should include base instructions
        assert "PostgreSQL database" in instructions
        assert "Your responsibilities:" in instructions

        # Should include workflow instructions
        assert "IMPORTANT - Tool Usage Strategy:" in instructions
        assert "Use this tool first to discover things" in instructions
        assert "Use this tool to analyze data" in instructions

    def test_sort_tools_by_workflow(self):
        """Test tool sorting by workflow position and priority."""
        registry = ToolRegistry()
        registry.register(MockDiscoveryTool)  # discovery, priority 10
        registry.register(MockAnalysisTool)  # analysis, priority 20

        builder = InstructionBuilder(registry)
        tools = registry.get_all_tools()
        sorted_tools = builder._sort_tools_by_workflow(tools)

        # Should be sorted by workflow position first, then priority
        assert sorted_tools[0].name == "mock_discovery"  # discovery, 10
        assert sorted_tools[1].name == "mock_analysis"  # analysis, 20

    def test_build_mcp_instructions(self):
        """Test building MCP-specific instructions."""
        registry = ToolRegistry()
        registry.register(MockDiscoveryTool)
        registry.register(MockAnalysisTool)

        builder = InstructionBuilder(registry)
        instructions = builder.build_mcp_instructions()

        # Should include MCP-specific content
        assert "This server provides helpful resources" in instructions
        assert "Get all databases using `get_databases()`" in instructions
        assert "`mock_discovery()` to mock discovery tool for testing" in instructions
        assert "`mock_analysis()` to mock analysis tool for testing" in instructions

        # Should include guidelines
        assert "Guidelines:" in instructions
        assert "Use proper JOIN syntax" in instructions

    def test_workflow_instructions_ordering(self):
        """Test that workflow instructions are ordered correctly."""
        registry = ToolRegistry()

        # Register in reverse order to test sorting
        registry.register(MockAnalysisTool)
        registry.register(MockDiscoveryTool)

        builder = InstructionBuilder(registry)
        instructions = builder.build_instructions("SQLite")

        # Find the workflow section
        lines = instructions.split("\n")
        workflow_start = -1
        for i, line in enumerate(lines):
            if "Tool Usage Strategy:" in line:
                workflow_start = i
                break

        assert workflow_start >= 0, "Workflow section not found"

        # Check ordering in workflow section
        workflow_section = "\n".join(lines[workflow_start : workflow_start + 5])

        # Discovery should come first (step 1)
        assert "1. Use this tool first to discover things" in workflow_section
        # Analysis should come second (step 2)
        assert "2. Use this tool to analyze data" in workflow_section

    def test_category_filtering(self):
        """Test filtering tools by category."""
        registry = ToolRegistry()
        registry.register(MockDiscoveryTool)  # sql category
        registry.register(MockAnalysisTool)  # sql category

        builder = InstructionBuilder(registry)

        # Filter for SQL tools only
        sql_instructions = builder.build_instructions(
            "PostgreSQL", category=ToolCategory.SQL
        )
        assert "Use this tool first to discover things" in sql_instructions
        assert "Use this tool to analyze data" in sql_instructions

    def test_base_instructions_flag(self):
        """Test the include_base_instructions flag."""
        registry = ToolRegistry()
        registry.register(MockDiscoveryTool)

        builder = InstructionBuilder(registry)

        # With base instructions
        with_base = builder.build_instructions(
            "PostgreSQL", include_base_instructions=True
        )
        assert "Your responsibilities:" in with_base

        # Without base instructions
        without_base = builder.build_instructions(
            "PostgreSQL", include_base_instructions=False
        )
        assert "Your responsibilities:" not in without_base
        assert (
            "Tool Usage Strategy:" in without_base
        )  # Should still have tool instructions
