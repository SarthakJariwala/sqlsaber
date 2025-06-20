"""Display utilities for the CLI interface."""

import json

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table


class DisplayManager:
    """Manages display formatting and output for the CLI."""

    def __init__(self, console: Console):
        self.console = console

    def show_query_header(self, user_query: str):
        """Display the query header."""
        self.console.print(f"\n[bold blue]Query:[/bold blue] {user_query}")

    def show_tool_started(self, tool_name: str):
        """Display tool started message."""
        self.console.print(f"\n[yellow]🔧 Using tool: {tool_name}[/yellow]")

    def show_tool_executing(self, tool_name: str, tool_input: dict):
        """Display tool execution details."""
        if tool_name == "list_tables":
            self.console.print("[dim]  → Discovering available tables[/dim]")
        elif tool_name == "introspect_schema":
            pattern = tool_input.get("table_pattern", "all tables")
            self.console.print(f"[dim]  → Examining schema for: {pattern}[/dim]")
        elif tool_name == "execute_sql":
            query = tool_input.get("query", "")
            self.console.print("\n[bold green]Executing SQL:[/bold green]")
            syntax = Syntax(query, "sql", theme="monokai", line_numbers=True)
            self.console.print(syntax)

    def show_explanation_start(self):
        """Display explanation header."""
        self.console.print("\n[bold cyan]Explanation:[/bold cyan] ", end="")

    def show_text_stream(self, text: str):
        """Display streaming text."""
        self.console.print(text, end="", markup=False)

    def show_query_results(self, results: list):
        """Display query results in a formatted table."""
        if not results:
            return

        self.console.print(
            f"\n[bold magenta]Results ({len(results)} rows):[/bold magenta]"
        )

        # Create a rich table
        table = Table(show_header=True, header_style="bold blue")

        # Add columns
        for key in results[0].keys():
            table.add_column(key)

        # Add rows (show first 20 rows)
        for row in results[:20]:
            table.add_row(*[str(row[key]) for key in results[0].keys()])

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
        return self.console.status(f"[yellow]🧠 {message}[/yellow]")

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
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Schema", style="cyan")
            table.add_column("Table Name", style="white")
            table.add_column("Type", style="yellow")
            table.add_column("Row Count", justify="right", style="magenta")

            # Add rows
            for table_info in tables:
                schema = table_info.get("schema", "")
                name = table_info.get("name", "")
                table_type = table_info.get("type", "")
                row_count = table_info.get("row_count", 0)

                # Format row count with commas for readability
                formatted_count = f"{row_count:,}" if row_count else "0"

                table.add_row(schema, name, table_type, formatted_count)

            self.console.print(table)

        except json.JSONDecodeError:
            self.show_error("Failed to parse table list data")
        except Exception as e:
            self.show_error(f"Error displaying table list: {str(e)}")
