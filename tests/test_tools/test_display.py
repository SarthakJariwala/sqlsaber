"""Tests for tool display specifications and rendering."""

from io import StringIO

from sqlsaber.cli.display import DisplayManager
from sqlsaber.theme.manager import create_console
from sqlsaber.tools import Tool
from sqlsaber.tools.display import (
    ColumnDef,
    DisplayMetadata,
    ExecutingConfig,
    FieldMappings,
    ResultConfig,
    SpecRenderer,
    TableConfig,
    ToolDisplaySpec,
)
from sqlsaber.tools.registry import tool_registry


class TestSpecRenderer:
    def _make_console(self):
        buffer = StringIO()
        console = create_console(file=buffer, width=120, legacy_windows=False)
        return console, buffer

    def test_render_executing_interpolation_and_args(self):
        console, buffer = self._make_console()
        renderer = SpecRenderer()
        spec = ToolDisplaySpec(
            executing=ExecutingConfig(
                message="Running {action} on {table}",
                icon="✅",
                show_args=["limit"],
            ),
            metadata=DisplayMetadata(display_name="Test"),
        )

        renderer.render_executing(
            console,
            "test_tool",
            {"action": "scan", "table": "users", "limit": 10},
            spec,
        )

        output = buffer.getvalue()
        assert "**✅ Running scan on users**" in output
        assert "**limit**: 10" in output

    def test_render_result_table_markdown(self):
        console, buffer = self._make_console()
        renderer = SpecRenderer()
        spec = ToolDisplaySpec(
            result=ResultConfig(
                format="table",
                title="Tables ({total} total)",
                fields=FieldMappings(items="tables"),
                table=TableConfig(
                    columns=[
                        ColumnDef(field="schema", header="Schema"),
                        ColumnDef(field="name", header="Name"),
                    ]
                ),
            )
        )

        renderer.render_result(
            console,
            "list_tables",
            {"total": 1, "tables": [{"schema": "public", "name": "users"}]},
            spec,
        )

        output = buffer.getvalue()
        assert "Tables (1 total)" in output
        assert "Schema" in output
        assert "public" in output
        assert "users" in output

    def test_render_result_code_block(self):
        console, buffer = self._make_console()
        renderer = SpecRenderer()
        spec = ToolDisplaySpec(
            result=ResultConfig(
                format="code",
                code_language="python",
                fields=FieldMappings(output="stdout"),
            )
        )

        renderer.render_result(
            console,
            "run_python",
            {"stdout": "print('hi')"},
            spec,
        )

        output = buffer.getvalue()
        assert "```python" in output
        assert "print('hi')" in output

    def test_render_result_error(self):
        console, buffer = self._make_console()
        renderer = SpecRenderer()
        spec = ToolDisplaySpec(result=ResultConfig(format="json"))

        renderer.render_result(console, "tool", {"error": "boom"}, spec)

        output = buffer.getvalue()
        assert "**Error:** boom" in output


class TestDisplayManagerResolution:
    def _make_console(self):
        buffer = StringIO()
        console = create_console(file=buffer, width=120, legacy_windows=False)
        return console, buffer

    def test_render_result_override_takes_precedence(self):
        class OverrideTool(Tool):
            @property
            def name(self) -> str:
                return "override_tool"

            async def execute(self, **kwargs) -> str:
                return "{}"

            def render_result(self, console, result: object) -> bool:
                console.print("override handled")
                return True

        tool_registry.register(OverrideTool)
        try:
            console, buffer = self._make_console()
            display = DisplayManager(console)
            display.show_tool_result("override_tool", {"result": 1})
            assert "override handled" in buffer.getvalue()
        finally:
            tool_registry.unregister("override_tool")

    def test_render_result_spec_used_when_no_override(self):
        class SpecTool(Tool):
            display_spec = ToolDisplaySpec(
                result=ResultConfig(format="panel", title="Spec Output")
            )

            @property
            def name(self) -> str:
                return "spec_tool"

            async def execute(self, **kwargs) -> str:
                return "{}"

        tool_registry.register(SpecTool)
        try:
            console, buffer = self._make_console()
            display = DisplayManager(console)
            display.show_tool_result("spec_tool", {"output": "hello"})
            output = buffer.getvalue()
            assert "Spec Output" in output
            assert "hello" in output
        finally:
            tool_registry.unregister("spec_tool")

    def test_render_result_fallback_for_unknown_tool(self):
        console, buffer = self._make_console()
        display = DisplayManager(console)
        display.show_tool_result("missing_tool", {"value": 42})
        output = buffer.getvalue()
        assert "```json" in output
        assert "value" in output

    def test_render_ask_database_executing_uses_readable_summary(self):
        console, buffer = self._make_console()
        display = DisplayManager(console)

        display.show_tool_executing(
            "ask_database",
            {"database_id": "orders", "question": "How many orders?"},
        )

        output = buffer.getvalue()
        assert "Asking database orders" in output
        assert "Question:" in output
        assert "How many orders?" in output
        assert "```json" not in output
        assert '"database_id"' not in output

    def test_render_ask_database_result_uses_matching_tool_call_question(self):
        console, buffer = self._make_console()
        display = DisplayManager(console)
        display.show_tool_executing(
            "ask_database",
            {"database_id": "spock-data", "question": "Question for spock"},
            tool_call_id="call-spock",
        )
        display.show_tool_executing(
            "ask_database",
            {"database_id": "ocean-local", "question": "Question for ocean"},
            tool_call_id="call-ocean",
        )

        display.show_tool_result(
            "ask_database",
            "Database: spock-data (id: spock-data)\n"
            "Child thread ID: thread-spock\n\n"
            "## Answer\nSpock answer.",
            tool_call_id="call-spock",
        )
        display.show_tool_result(
            "ask_database",
            "Database: ocean-local (id: ocean-local)\n"
            "Child thread ID: thread-ocean\n\n"
            "## Answer\nOcean answer.",
            tool_call_id="call-ocean",
        )

        output = buffer.getvalue()
        spock_question_index = output.index("Question for spock")
        spock_answer_index = output.index("Spock answer.")
        ocean_question_index = output.index("Question for ocean")
        ocean_answer_index = output.index("Ocean answer.")
        assert spock_question_index < spock_answer_index
        assert ocean_question_index < ocean_answer_index
        assert output.count("Question for spock") == 2
        assert output.count("Question for ocean") == 2

    def test_render_ask_database_result_as_bounded_markdown_panel(self):
        console, buffer = self._make_console()
        display = DisplayManager(console)
        display.show_tool_executing(
            "ask_database",
            {"database_id": "orders", "question": "How many orders?"},
        )

        display.show_tool_result(
            "ask_database",
            (
                "Database: Orders (id: orders)\n"
                "Child thread ID: thread-1\n\n"
                "## Answer\n"
                "There are 42 orders.\n\n"
                "## SQL\n"
                "```sql\nSELECT count(*) FROM orders;\n```"
            ),
        )

        output = buffer.getvalue()
        assert "Subagent answer: Orders" in output
        assert "orders" in output
        assert "thread-1" in output
        assert "Question" in output
        assert "How many orders?" in output
        assert "There are 42 orders." in output
        assert "SELECT count(*) FROM orders;" in output
        assert "Database: Orders" not in output
        assert "Child thread ID:" not in output
        assert "```" not in output

    def test_render_connected_databases_as_readable_list(self):
        console, buffer = self._make_console()
        display = DisplayManager(console)

        display.show_tool_result(
            "list_connected_databases",
            [
                {
                    "id": "orders",
                    "name": "Orders",
                    "type": "sqlite",
                    "description": "Order records",
                }
            ],
        )

        output = buffer.getvalue()
        assert "Connected databases" in output
        assert "orders" in output
        assert "Orders" in output
        assert "sqlite" in output
        assert "```json" not in output

    def test_render_result_html_from_spec(self):
        class HtmlTool(Tool):
            display_spec = ToolDisplaySpec(
                result=ResultConfig(
                    format="table",
                    fields=FieldMappings(items="rows"),
                    table=TableConfig(columns=[ColumnDef(field="name", header="Name")]),
                )
            )

            @property
            def name(self) -> str:
                return "html_tool"

            async def execute(self, **kwargs) -> str:
                return "{}"

        tool_registry.register(HtmlTool)
        try:
            console, _ = self._make_console()
            display = DisplayManager(console)
            html = display.render_tool_result_html(
                "html_tool", {"rows": [{"name": "alpha"}]}
            )
            assert "<table" in html
            assert "alpha" in html
        finally:
            tool_registry.unregister("html_tool")
