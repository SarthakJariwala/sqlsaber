"""Dynamic instruction builder for tools."""

from .base import Tool
from .enums import ToolCategory, WorkflowPosition
from .registry import ToolRegistry


class InstructionBuilder:
    """Builds dynamic instructions based on available tools."""

    def __init__(self, tool_registry: ToolRegistry):
        """Initialize with a tool registry."""
        self.registry = tool_registry

    def build_instructions(
        self,
        db_type: str = "database",
        category: str | ToolCategory | None = None,
        include_base_instructions: bool = True,
    ) -> str:
        """Build dynamic instructions from available tools.

        Args:
            db_type: Type of database (PostgreSQL, MySQL, SQLite, etc.)
            category: Optional category to filter tools by (string or ToolCategory enum)
            include_base_instructions: Whether to include base SQL assistant instructions

        Returns:
            Complete instruction string for LLM
        """
        # Get available tools
        tools = self.registry.get_all_tools(category)

        if not tools:
            return self._get_base_instructions(db_type)

        # Sort tools by priority and workflow position
        sorted_tools = self._sort_tools_by_workflow(tools)

        # Build instruction components
        instructions_parts = []

        if include_base_instructions:
            instructions_parts.append(self._get_base_instructions(db_type))

        # Add tool-specific workflow guidance
        workflow_instructions = self._build_workflow_instructions(sorted_tools)
        if workflow_instructions:
            instructions_parts.append(workflow_instructions)

        # Add tool descriptions and guidelines
        tool_guidelines = self._build_tool_guidelines(sorted_tools)
        if tool_guidelines:
            instructions_parts.append(tool_guidelines)

        # Add general guidelines
        general_guidelines = self._build_general_guidelines(sorted_tools)
        if general_guidelines:
            instructions_parts.append(general_guidelines)

        return "\n\n".join(instructions_parts)

    def _get_base_instructions(self, db_type: str) -> str:
        """Get base SQL assistant instructions."""
        return f"""You are also a helpful SQL assistant that helps users query their {db_type} database.

Your responsibilities:
1. Understand user's natural language requests, think and convert them to SQL
2. Use the provided tools efficiently to explore database schema
3. Generate appropriate SQL queries
4. Execute queries safely - queries that modify the database are not allowed
5. Format and explain results clearly"""

    def _sort_tools_by_workflow(self, tools: list[Tool]) -> list[Tool]:
        """Sort tools by priority and workflow position."""
        # Define workflow position ordering
        position_order = {
            WorkflowPosition.DISCOVERY: 1,
            WorkflowPosition.ANALYSIS: 2,
            WorkflowPosition.EXECUTION: 3,
            WorkflowPosition.OTHER: 4,
        }

        return sorted(
            tools,
            key=lambda tool: (
                position_order.get(tool.get_workflow_position(), 4),
                tool.get_priority(),
                tool.name,
            ),
        )

    def _build_workflow_instructions(self, sorted_tools: list[Tool]) -> str:
        """Build workflow-based instructions."""
        # Group tools by workflow position
        workflow_groups = {}
        for tool in sorted_tools:
            position = tool.get_workflow_position()
            if position not in workflow_groups:
                workflow_groups[position] = []
            workflow_groups[position].append(tool)

        # Build workflow instructions
        instructions = ["IMPORTANT - Tool Usage Strategy:"]
        step = 1

        # Add discovery tools first
        if WorkflowPosition.DISCOVERY in workflow_groups:
            discovery_tools = workflow_groups[WorkflowPosition.DISCOVERY]
            for tool in discovery_tools:
                usage = tool.get_usage_instructions()
                if usage:
                    instructions.append(f"{step}. {usage}")
                else:
                    instructions.append(
                        f"{step}. Use '{tool.name}' to {tool.description.lower()}"
                    )
                step += 1

        # Add analysis tools
        if WorkflowPosition.ANALYSIS in workflow_groups:
            analysis_tools = workflow_groups[WorkflowPosition.ANALYSIS]
            for tool in analysis_tools:
                usage = tool.get_usage_instructions()
                if usage:
                    instructions.append(f"{step}. {usage}")
                else:
                    instructions.append(
                        f"{step}. Use '{tool.name}' to {tool.description.lower()}"
                    )
                step += 1

        # Add execution tools
        if WorkflowPosition.EXECUTION in workflow_groups:
            execution_tools = workflow_groups[WorkflowPosition.EXECUTION]
            for tool in execution_tools:
                usage = tool.get_usage_instructions()
                if usage:
                    instructions.append(f"{step}. {usage}")
                else:
                    instructions.append(
                        f"{step}. Use '{tool.name}' to {tool.description.lower()}"
                    )
                step += 1

        return "\n".join(instructions) if len(instructions) > 1 else ""

    def _build_tool_guidelines(self, sorted_tools: list[Tool]) -> str:
        """Build tool-specific guidelines."""
        guidelines = []

        for tool in sorted_tools:
            usage = tool.get_usage_instructions()
            if usage and not self._is_usage_in_workflow(usage):
                guidelines.append(f"- {tool.name}: {usage}")

        if guidelines:
            return "Tool-Specific Guidelines:\n" + "\n".join(guidelines)
        return ""

    def _build_general_guidelines(self, sorted_tools: list[Tool]) -> str:
        """Build general usage guidelines."""
        guidelines = [
            "Guidelines:",
            "- Use proper JOIN syntax and avoid cartesian products",
            "- Include appropriate WHERE clauses to limit results",
            "- Explain what the query does in simple terms",
            "- Handle errors gracefully and suggest fixes",
            "- Be security conscious - use parameterized queries when needed",
        ]

        # Add category-specific guidelines
        categories = {tool.category for tool in sorted_tools}

        if ToolCategory.SQL in categories:
            guidelines.extend(
                [
                    "- Timestamp columns must be converted to text when you write queries",
                    "- Use table patterns like 'sample%' or '%experiment%' to filter related tables",
                ]
            )

        return "\n".join(guidelines)

    def _is_usage_in_workflow(self, usage: str) -> bool:
        """Check if usage instruction is already covered in workflow section."""
        # Simple heuristic - if usage starts with workflow words, it's probably in workflow
        workflow_words = ["always start", "first", "use this", "begin with", "start by"]
        usage_lower = usage.lower()
        return any(word in usage_lower for word in workflow_words)

    def build_mcp_instructions(self) -> str:
        """Build instructions specifically for MCP server."""
        instructions = [
            "This server provides helpful resources and tools that will help you address users queries on their database.",
            "",
        ]

        # Add database discovery
        instructions.append("- Get all databases using `get_databases()`")

        # Add tool-specific instructions
        sql_tools = self.registry.get_all_tools(category=ToolCategory.SQL)
        sorted_tools = self._sort_tools_by_workflow(sql_tools)

        for tool in sorted_tools:
            instructions.append(f"- Call `{tool.name}()` to {tool.description.lower()}")

        # Add workflow guidelines
        instructions.extend(["", "Guidelines:"])

        workflow_instructions = self._build_workflow_instructions(sorted_tools)
        if workflow_instructions:
            # Extract just the numbered steps without the "IMPORTANT" header
            lines = workflow_instructions.split("\n")[1:]  # Skip header
            for line in lines:
                if line.strip():
                    # Convert numbered steps to bullet points
                    if line.strip()[0].isdigit():
                        instructions.append(f"- {line.strip()[3:]}")  # Remove "X. "

        # Add general guidelines
        instructions.extend(
            [
                "- Use proper JOIN syntax and avoid cartesian products",
                "- Include appropriate WHERE clauses to limit results",
                "- Handle errors gracefully and suggest fixes",
            ]
        )

        return "\n".join(instructions)
