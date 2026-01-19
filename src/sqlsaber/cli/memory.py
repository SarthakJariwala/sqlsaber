"""Memory management CLI commands."""

import sys
from typing import Annotated

import cyclopts
import questionary
from rich.table import Table

from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.theme.manager import create_console

# Global instances for CLI commands
console = create_console()
config_manager = DatabaseConfigManager()
memory_manager = MemoryManager()
logger = get_logger(__name__)

# Create the memory management CLI app
memory_app = cyclopts.App(
    name="memory",
    help="Manage database-specific memories",
)


def _get_database_name(database: str | None = None) -> str:
    """Get the database name to use, either specified or default."""
    if database:
        db_config = config_manager.get_database(database)
        if not db_config:
            console.print(
                f"[bold error]Error:[/bold error] Database connection '{database}' not found."
            )
            logger.error("memory.db.not_found", database=database)
            sys.exit(1)
        return database
    db_config = config_manager.get_default_database()
    if db_config is None:
        console.print(
            "[bold error]Error:[/bold error] No database connections configured."
        )
        console.print("Use 'sqlsaber db add <name>' to add a database connection.")
        logger.error("memory.db.none_configured")
        raise SystemExit(1)
    return db_config.name


@memory_app.command
def add(
    content: Annotated[str, cyclopts.Parameter(help="Memory content to add")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Add a new memory for the specified database."""
    database_name = _get_database_name(database)
    logger.info("memory.add.start", database=database_name)

    try:
        memory = memory_manager.add_memory(database_name, content)
        console.print(
            f"[success]✓ Memory added for database '{database_name}'[/success]"
        )
        console.print(f"[dim]Memory ID:[/dim] {memory.id}")
        console.print(f"[dim]Content:[/dim] {memory.content}")
        logger.info("memory.add.success", database=database_name, id=memory.id)
    except Exception as e:
        console.print(f"[bold error]Error adding memory:[/bold error] {e}")
        logger.exception("memory.add.error", database=database_name, error=str(e))
        sys.exit(1)


@memory_app.command
def list(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """List all memories for the specified database."""
    database_name = _get_database_name(database)
    logger.info("memory.list.start", database=database_name)

    memories = memory_manager.get_memories(database_name)

    if not memories:
        console.print(
            f"[warning]No memories found for database '{database_name}'[/warning]"
        )
        console.print("Use 'sqlsaber memory add \"<content>\"' to add memories")
        logger.info("memory.list.empty", database=database_name)
        return

    table = Table(title=f"Memories for Database: {database_name}")
    table.add_column("ID", style="info", width=36)
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
    logger.info("memory.list.complete", database=database_name, count=len(memories))


@memory_app.command
def show(
    memory_id: Annotated[str, cyclopts.Parameter(help="Memory ID to show")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Show the full content of a specific memory."""
    database_name = _get_database_name(database)
    logger.info("memory.show.start", database=database_name, id=memory_id)

    memory = memory_manager.get_memory_by_id(database_name, memory_id)

    if memory is None:
        console.print(
            f"[bold error]Error:[/bold error] Memory with ID '{memory_id}' not found for database '{database_name}'"
        )
        logger.error("memory.show.not_found", database=database_name, id=memory_id)
        raise SystemExit(1)

    console.print(f"[bold]Memory ID:[/bold] {memory.id}")
    console.print(f"[bold]Database:[/bold] {database_name}")
    console.print(f"[bold]Created:[/bold] {memory.formatted_timestamp()}")
    console.print("[bold]Content:[/bold]")
    console.print(f"{memory.content}")


@memory_app.command
def remove(
    memory_id: Annotated[str, cyclopts.Parameter(help="Memory ID to remove")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Remove a specific memory by ID."""
    database_name = _get_database_name(database)
    logger.info("memory.remove.start", database=database_name, id=memory_id)

    # First check if memory exists
    memory = memory_manager.get_memory_by_id(database_name, memory_id)
    if memory is None:
        console.print(
            f"[bold error]Error:[/bold error] Memory with ID '{memory_id}' not found for database '{database_name}'"
        )
        logger.error("memory.remove.not_found", database=database_name, id=memory_id)
        raise SystemExit(1)

    # Show memory content before removal
    console.print("[warning]Removing memory:[/warning]")
    console.print(f"[dim]Content:[/dim] {memory.content}")

    if memory_manager.remove_memory(database_name, memory_id):
        console.print(
            f"[success]✓ Memory removed from database '{database_name}'[/success]"
        )
        logger.info("memory.remove.success", database=database_name, id=memory_id)
    else:
        console.print(
            f"[bold error]Error:[/bold error] Failed to remove memory '{memory_id}'"
        )
        logger.error("memory.remove.failed", database=database_name, id=memory_id)
        sys.exit(1)


@memory_app.command
def clear(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
    force: Annotated[
        bool,
        cyclopts.Parameter(
            ["--force", "-f"],
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Clear all memories for the specified database."""
    database_name = _get_database_name(database)
    logger.info("memory.clear.start", database=database_name, force=bool(force))

    # Count memories first
    memories_count = len(memory_manager.get_memories(database_name))

    if memories_count == 0:
        console.print(
            f"[warning]No memories to clear for database '{database_name}'[/warning]"
        )
        logger.info("memory.clear.nothing", database=database_name)
        return

    if not force:
        # Show confirmation
        console.print(
            f"[warning]About to clear {memories_count} memories for database '{database_name}'[/warning]"
        )

        if not questionary.confirm("Are you sure you want to proceed?").ask():
            console.print("Operation cancelled")
            logger.info("memory.clear.cancelled", database=database_name)
            return

    cleared_count = memory_manager.clear_memories(database_name)
    console.print(
        f"[success]✓ Cleared {cleared_count} memories for database '{database_name}'[/success]"
    )
    logger.info("memory.clear.success", database=database_name, deleted=cleared_count)


@memory_app.command
def summary(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Show memory summary for the specified database."""
    database_name = _get_database_name(database)
    logger.info("memory.summary.start", database=database_name)

    summary = memory_manager.get_memories_summary(database_name)

    console.print(f"[bold]Memory Summary for Database: {summary['database']}[/bold]")
    console.print(f"[dim]Total memories:[/dim] {summary['total_memories']}")

    if summary["total_memories"] > 0:
        console.print("\n[bold]Recent memories:[/bold]")
        for memory in summary["memories"][-5:]:  # Show last 5 memories
            console.print(f"[dim]{memory['timestamp']}[/dim] - {memory['content']}")
    logger.info(
        "memory.summary.complete",
        database=database_name,
        total=summary["total_memories"],
    )


def create_memory_app() -> cyclopts.App:
    """Return the memory management CLI app."""
    return memory_app
