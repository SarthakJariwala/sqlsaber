"""Memory management CLI commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.memory.manager import MemoryManager

# Global instances for CLI commands
console = Console()
config_manager = DatabaseConfigManager()
memory_manager = MemoryManager()

# Create the memory management CLI app
memory_app = typer.Typer(
    name="memory",
    help="Manage database-specific memories",
    add_completion=True,
)


def _get_database_name(database: Optional[str] = None) -> str:
    """Get the database name to use, either specified or default."""
    if database:
        db_config = config_manager.get_database(database)
        if not db_config:
            console.print(
                f"[bold red]Error:[/bold red] Database connection '{database}' not found."
            )
            raise typer.Exit(1)
        return database
    else:
        db_config = config_manager.get_default_database()
        if not db_config:
            console.print(
                "[bold red]Error:[/bold red] No database connections configured."
            )
            console.print("Use 'sqlsaber db add <name>' to add a database connection.")
            raise typer.Exit(1)
        return db_config.name


@memory_app.command("add")
def add_memory(
    content: str = typer.Argument(..., help="Memory content to add"),
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """Add a new memory for the specified database."""
    database_name = _get_database_name(database)

    try:
        memory = memory_manager.add_memory(database_name, content)
        console.print(f"[green]✓ Memory added for database '{database_name}'[/green]")
        console.print(f"[dim]Memory ID:[/dim] {memory.id}")
        console.print(f"[dim]Content:[/dim] {memory.content}")
    except Exception as e:
        console.print(f"[bold red]Error adding memory:[/bold red] {e}")
        raise typer.Exit(1)


@memory_app.command("list")
def list_memories(
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """List all memories for the specified database."""
    database_name = _get_database_name(database)

    memories = memory_manager.get_memories(database_name)

    if not memories:
        console.print(
            f"[yellow]No memories found for database '{database_name}'[/yellow]"
        )
        console.print("Use 'sqlsaber memory add \"<content>\"' to add memories")
        return

    table = Table(title=f"Memories for Database: {database_name}")
    table.add_column("ID", style="cyan", width=36)
    table.add_column("Content", style="white")
    table.add_column("Created", style="dim")

    for memory in memories:
        # Truncate content if it's too long for display
        display_content = memory.content
        if len(display_content) > 80:
            display_content = display_content[:77] + "..."

        table.add_row(memory.id, display_content, memory.formatted_timestamp())

    console.print(table)
    console.print(f"\n[dim]Total memories: {len(memories)}[/dim]")


@memory_app.command("show")
def show_memory(
    memory_id: str = typer.Argument(..., help="Memory ID to show"),
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """Show the full content of a specific memory."""
    database_name = _get_database_name(database)

    memory = memory_manager.get_memory_by_id(database_name, memory_id)

    if not memory:
        console.print(
            f"[bold red]Error:[/bold red] Memory with ID '{memory_id}' not found for database '{database_name}'"
        )
        raise typer.Exit(1)

    console.print(f"[bold]Memory ID:[/bold] {memory.id}")
    console.print(f"[bold]Database:[/bold] {memory.database}")
    console.print(f"[bold]Created:[/bold] {memory.formatted_timestamp()}")
    console.print("[bold]Content:[/bold]")
    console.print(f"{memory.content}")


@memory_app.command("remove")
def remove_memory(
    memory_id: str = typer.Argument(..., help="Memory ID to remove"),
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """Remove a specific memory by ID."""
    database_name = _get_database_name(database)

    # First check if memory exists
    memory = memory_manager.get_memory_by_id(database_name, memory_id)
    if not memory:
        console.print(
            f"[bold red]Error:[/bold red] Memory with ID '{memory_id}' not found for database '{database_name}'"
        )
        raise typer.Exit(1)

    # Show memory content before removal
    console.print("[yellow]Removing memory:[/yellow]")
    console.print(f"[dim]Content:[/dim] {memory.content}")

    if memory_manager.remove_memory(database_name, memory_id):
        console.print(
            f"[green]✓ Memory removed from database '{database_name}'[/green]"
        )
    else:
        console.print(
            f"[bold red]Error:[/bold red] Failed to remove memory '{memory_id}'"
        )
        raise typer.Exit(1)


@memory_app.command("clear")
def clear_memories(
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """Clear all memories for the specified database."""
    database_name = _get_database_name(database)

    # Count memories first
    memories_count = len(memory_manager.get_memories(database_name))

    if memories_count == 0:
        console.print(
            f"[yellow]No memories to clear for database '{database_name}'[/yellow]"
        )
        return

    if not force:
        # Show confirmation
        console.print(
            f"[yellow]About to clear {memories_count} memories for database '{database_name}'[/yellow]"
        )
        confirm = typer.confirm("Are you sure you want to proceed?")
        if not confirm:
            console.print("Operation cancelled")
            return

    cleared_count = memory_manager.clear_memories(database_name)
    console.print(
        f"[green]✓ Cleared {cleared_count} memories for database '{database_name}'[/green]"
    )


@memory_app.command("summary")
def memory_summary(
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """Show memory summary for the specified database."""
    database_name = _get_database_name(database)

    summary = memory_manager.get_memories_summary(database_name)

    console.print(f"[bold]Memory Summary for Database: {summary['database']}[/bold]")
    console.print(f"[dim]Total memories:[/dim] {summary['total_memories']}")

    if summary["total_memories"] > 0:
        console.print("\n[bold]Recent memories:[/bold]")
        for memory in summary["memories"][-5:]:  # Show last 5 memories
            console.print(f"[dim]{memory['timestamp']}[/dim] - {memory['content']}")


def create_memory_app() -> typer.Typer:
    """Return the memory management CLI app."""
    return memory_app
