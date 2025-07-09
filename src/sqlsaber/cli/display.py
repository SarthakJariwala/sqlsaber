"""Display utilities for the CLI interface."""

import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


class DisplayManager:
    """Manages display formatting and output for the CLI."""

    def __init__(self, console: Console):
        self.console = console

    def _create_table(
        self,
        columns: list,
        header_style: str = "bold blue",
        title: str | None = None,
    ) -> Table:
        """Create a Rich table with specified columns."""
        table = Table(show_header=True, header_style=header_style, title=title)
        for col in columns:
            if isinstance(col, dict):
                table.add_column(
                    col["name"], style=col.get("style"), justify=col.get("justify")
                )
            else:
                table.add_column(col)
        return table

    def show_tool_executing(self, tool_name: str, tool_input: dict):
        """Display tool execution details."""
        self.console.print(f"\n[yellow]🔧 Using tool: {tool_name}[/yellow]")
        if tool_name == "list_tables":
            self.console.print("[dim]  → Discovering available tables[/dim]")
        elif tool_name == "introspect_schema":
            pattern = tool_input.get("table_pattern", "all tables")
            self.console.print(f"[dim]  → Examining schema for: {pattern}[/dim]")
        elif tool_name == "execute_sql":
            query = tool_input.get("query", "")
            self.console.print("\n[bold green]Executing SQL:[/bold green]")
            self.show_newline()
            syntax = Syntax(query, "sql")
            self.console.print(syntax)

    def show_text_stream(self, text: str):
        """Display streaming text."""
        if text is not None:  # Extra safety check
            self.console.print(text, end="", markup=False)

    def show_query_results(self, results: list):
        """Display query results in a formatted table."""
        if not results:
            return

        self.console.print(
            f"\n[bold magenta]Results ({len(results)} rows):[/bold magenta]"
        )

        # Create table with columns from first result
        all_columns = list(results[0].keys())
        display_columns = all_columns[:15]  # Limit to first 15 columns

        # Show warning if columns were truncated
        if len(all_columns) > 15:
            self.console.print(
                f"[yellow]Note: Showing first 15 of {len(all_columns)} columns[/yellow]"
            )

        table = self._create_table(display_columns)

        # Add rows (show first 20 rows)
        for row in results[:20]:
            table.add_row(*[str(row[key]) for key in display_columns])

        self.console.print(table)

        if len(results) > 20:
            self.console.print(
                f"[yellow]... and {len(results) - 20} more rows[/yellow]"
            )

    def show_error(self, error_message: str):
        """Display error message."""
        self.console.print(f"\n[bold red]Error:[/bold red] {error_message}")

    def show_processing(self, message: str):
        """Display processing message."""
        self.console.print()  # Add newline
        return self.console.status(
            f"[yellow]{message}[/yellow]", spinner="bouncingBall"
        )

    def show_newline(self):
        """Display a newline for spacing."""
        self.console.print()

    def show_table_list(self, tables_data: str):
        """Display the results from list_tables tool."""
        try:
            data = json.loads(tables_data)

            # Handle error case
            if "error" in data:
                self.show_error(data["error"])
                return

            tables = data.get("tables", [])
            total_tables = data.get("total_tables", 0)

            if not tables:
                self.console.print("[yellow]No tables found in the database.[/yellow]")
                return

            self.console.print(
                f"\n[bold green]Database Tables ({total_tables} total):[/bold green]"
            )

            # Create a rich table for displaying table information
            columns = [
                {"name": "Schema", "style": "cyan"},
                {"name": "Table Name", "style": "white"},
                {"name": "Type", "style": "yellow"},
            ]
            table = self._create_table(columns)

            # Add rows
            for table_info in tables:
                schema = table_info.get("schema", "")
                name = table_info.get("name", "")
                table_type = table_info.get("type", "")

                table.add_row(schema, name, table_type)

            self.console.print(table)

        except json.JSONDecodeError:
            self.show_error("Failed to parse table list data")
        except Exception as e:
            self.show_error(f"Error displaying table list: {str(e)}")

    def show_schema_info(self, schema_data: str):
        """Display the results from introspect_schema tool."""
        try:
            data = json.loads(schema_data)

            # Handle error case
            if "error" in data:
                self.show_error(data["error"])
                return

            if not data:
                self.console.print("[yellow]No schema information found.[/yellow]")
                return

            self.console.print(
                f"\n[bold green]Schema Information ({len(data)} tables):[/bold green]"
            )

            # Display each table's schema
            for table_name, table_info in data.items():
                self.console.print(f"\n[bold cyan]Table: {table_name}[/bold cyan]")

                # Show columns
                table_columns = table_info.get("columns", {})
                if table_columns:
                    # Create a table for columns
                    columns = [
                        {"name": "Column Name", "style": "white"},
                        {"name": "Type", "style": "yellow"},
                        {"name": "Nullable", "style": "cyan"},
                        {"name": "Default", "style": "dim"},
                    ]
                    col_table = self._create_table(columns, title="Columns")

                    for col_name, col_info in table_columns.items():
                        nullable = "✓" if col_info.get("nullable", False) else "✗"
                        default = (
                            str(col_info.get("default", ""))
                            if col_info.get("default")
                            else ""
                        )
                        col_table.add_row(
                            col_name, col_info.get("type", ""), nullable, default
                        )

                    self.console.print(col_table)

                # Show primary keys
                primary_keys = table_info.get("primary_keys", [])
                if primary_keys:
                    self.console.print(
                        f"[bold yellow]Primary Keys:[/bold yellow] {', '.join(primary_keys)}"
                    )

                # Show foreign keys
                foreign_keys = table_info.get("foreign_keys", [])
                if foreign_keys:
                    self.console.print("[bold magenta]Foreign Keys:[/bold magenta]")
                    for fk in foreign_keys:
                        self.console.print(f"  • {fk}")

        except json.JSONDecodeError:
            self.show_error("Failed to parse schema data")
        except Exception as e:
            self.show_error(f"Error displaying schema information: {str(e)}")

    def show_plot(self, plot_data: dict):
        """Display plot information and status."""
        try:
            # Parse the result if it's a string
            if isinstance(plot_data.get("result"), str):
                result = json.loads(plot_data["result"])
            else:
                result = plot_data.get("result", {})

            # Check if there was an error
            if "error" in result:
                self.show_error(f"Plot error: {result['error']}")
                return

            # If plot was successful, show plot info
            if result.get("success") and result.get("plot_rendered"):
                plot_info = result.get("plot_info", {})
                self.console.print(
                    f"\n[bold green]✓ Plot rendered:[/bold green] {plot_info.get('title', 'Plot')}"
                )
                self.console.print(
                    f"[dim]  Type: {plot_info.get('type', 'unknown')}, "
                    f"Data points: {plot_info.get('data_points', 0)}[/dim]"
                )

        except json.JSONDecodeError:
            self.show_error("Failed to parse plot result")
        except Exception as e:
            self.show_error(f"Error displaying plot: {str(e)}")

    def show_markdown_response(self, content: list):
        """Display the assistant's response as rich markdown in a panel."""
        if not content:
            return

        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)

        # Join all text parts and display as markdown in a panel
        full_text = "".join(text_parts).strip()
        if full_text:
            self.console.print()  # Add spacing before panel
            markdown = Markdown(full_text)
            panel = Panel.fit(markdown, border_style="green")
            self.console.print(panel)
            self.console.print()  # Add spacing after panel
