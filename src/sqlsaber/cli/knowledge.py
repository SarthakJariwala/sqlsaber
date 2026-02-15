"""Knowledge management CLI commands."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Annotated, TypeVar

import cyclopts
import questionary
from rich.table import Table

from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.theme.manager import create_console

console = create_console()
config_manager = DatabaseConfigManager()
logger = get_logger(__name__)
knowledge_app = cyclopts.App(
    name="knowledge",
    help="Manage database-specific knowledge entries",
)
_knowledge_manager: KnowledgeManager | None = None
T = TypeVar("T")


def _manager() -> KnowledgeManager:
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager()
    return _knowledge_manager


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _format_timestamp(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _get_database_name(database: str | None = None) -> str:
    if database:
        db_config = config_manager.get_database(database)
        if not db_config:
            console.print(
                f"[bold error]Error:[/bold error] Database connection '{database}' not found."
            )
            logger.error("knowledge.db.not_found", database=database)
            raise SystemExit(1)
        return database

    db_config = config_manager.get_default_database()
    if db_config is None:
        console.print(
            "[bold error]Error:[/bold error] No database connections configured."
        )
        console.print("Use 'sqlsaber db add <name>' to add a database connection.")
        logger.error("knowledge.db.none_configured")
        raise SystemExit(1)
    return db_config.name


@knowledge_app.command
def add(
    name: Annotated[str, cyclopts.Parameter(help="Knowledge entry name")],
    description: Annotated[str, cyclopts.Parameter(help="Knowledge description")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
    sql: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--sql"],
            help="Optional SQL query or pattern",
        ),
    ] = None,
    source: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--source"],
            help="Optional source reference (wiki, URL, etc.)",
        ),
    ] = None,
):
    """Add knowledge for the specified database."""
    database_name = _get_database_name(database)
    logger.info("knowledge.add.start", database=database_name, source=source)

    try:
        entry = _run(
            _manager().add_knowledge(
                database_name=database_name,
                name=name,
                description=description,
                sql=sql,
                source=source,
            )
        )
    except Exception as exc:
        console.print(f"[bold error]Error adding knowledge:[/bold error] {exc}")
        logger.exception("knowledge.add.error", database=database_name, error=str(exc))
        raise SystemExit(1)

    console.print(
        f"[success]✓ Knowledge entry added for database '{database_name}'[/success]"
    )
    console.print(f"[dim]ID:[/dim] {entry.id}")
    console.print(f"[dim]Name:[/dim] {entry.name}")
    logger.info("knowledge.add.success", database=database_name, id=entry.id)


@knowledge_app.command
def list(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """List all knowledge entries for the specified database."""
    database_name = _get_database_name(database)
    logger.info("knowledge.list.start", database=database_name)

    entries = _run(_manager().list_knowledge(database_name))
    if not entries:
        console.print(
            f"[warning]No knowledge entries found for database '{database_name}'[/warning]"
        )
        console.print(
            'Use \'sqlsaber knowledge add "<name>" "<description>"\' to add entries'
        )
        logger.info("knowledge.list.empty", database=database_name)
        return

    table = Table(title=f"Knowledge Entries for Database: {database_name}")
    table.add_column("ID", style="info", width=36)
    table.add_column("Name", style="column.name")
    table.add_column("Description", style="white")
    table.add_column("Updated", style="dim")

    for entry in entries:
        table.add_row(
            entry.id,
            entry.name,
            _truncate(entry.description, 100),
            _format_timestamp(entry.updated_at),
        )

    console.print(table)
    console.print(f"\n[dim]Total entries: {len(entries)}[/dim]")
    logger.info("knowledge.list.complete", database=database_name, count=len(entries))


@knowledge_app.command
def show(
    entry_id: Annotated[str, cyclopts.Parameter(help="Knowledge entry ID")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Show a full knowledge entry."""
    database_name = _get_database_name(database)
    logger.info("knowledge.show.start", database=database_name, id=entry_id)

    entry = _run(_manager().get_knowledge(database_name, entry_id))
    if entry is None:
        console.print(
            f"[bold error]Error:[/bold error] Knowledge entry '{entry_id}' not found for database '{database_name}'"
        )
        logger.error("knowledge.show.not_found", database=database_name, id=entry_id)
        raise SystemExit(1)

    console.print(f"[bold]ID:[/bold] {entry.id}")
    console.print(f"[bold]Database:[/bold] {database_name}")
    console.print(f"[bold]Name:[/bold] {entry.name}")
    console.print(f"[bold]Created:[/bold] {_format_timestamp(entry.created_at)}")
    console.print(f"[bold]Updated:[/bold] {_format_timestamp(entry.updated_at)}")
    console.print("[bold]Description:[/bold]")
    console.print(entry.description)
    if entry.sql:
        console.print("\n[bold]SQL:[/bold]")
        console.print(entry.sql)
    if entry.source:
        console.print(f"\n[bold]Source:[/bold] {entry.source}")


@knowledge_app.command
def search(
    query: Annotated[str, cyclopts.Parameter(help="Search query")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
    limit: Annotated[
        int,
        cyclopts.Parameter(
            ["--limit"],
            help="Maximum number of entries to return",
        ),
    ] = 10,
):
    """Search knowledge entries for the specified database."""
    database_name = _get_database_name(database)
    logger.info("knowledge.search.start", database=database_name, limit=limit)

    entries = _run(_manager().search_knowledge(database_name, query, limit=limit))
    if not entries:
        console.print(
            f"[warning]No knowledge entries matched '{query}' for database '{database_name}'[/warning]"
        )
        logger.info("knowledge.search.empty", database=database_name)
        return

    table = Table(title=f"Knowledge Search Results ({len(entries)} matches)")
    table.add_column("ID", style="info", width=36)
    table.add_column("Name", style="column.name")
    table.add_column("Description", style="white")
    table.add_column("Source", style="dim")

    for entry in entries:
        table.add_row(
            entry.id,
            entry.name,
            _truncate(entry.description, 120),
            entry.source or "",
        )

    console.print(table)
    logger.info("knowledge.search.complete", database=database_name, count=len(entries))


@knowledge_app.command
def remove(
    entry_id: Annotated[str, cyclopts.Parameter(help="Knowledge entry ID")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Remove a specific knowledge entry by ID."""
    database_name = _get_database_name(database)
    logger.info("knowledge.remove.start", database=database_name, id=entry_id)

    entry = _run(_manager().get_knowledge(database_name, entry_id))
    if entry is None:
        console.print(
            f"[bold error]Error:[/bold error] Knowledge entry '{entry_id}' not found for database '{database_name}'"
        )
        logger.error("knowledge.remove.not_found", database=database_name, id=entry_id)
        raise SystemExit(1)

    if _run(_manager().remove_knowledge(database_name, entry_id)):
        console.print(
            f"[success]✓ Knowledge entry removed from database '{database_name}'[/success]"
        )
        logger.info("knowledge.remove.success", database=database_name, id=entry_id)
        return

    console.print(
        f"[bold error]Error:[/bold error] Failed to remove knowledge entry '{entry_id}'"
    )
    logger.error("knowledge.remove.failed", database=database_name, id=entry_id)
    raise SystemExit(1)


@knowledge_app.command
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
    """Clear all knowledge entries for the specified database."""
    database_name = _get_database_name(database)
    logger.info("knowledge.clear.start", database=database_name, force=bool(force))

    entries = _run(_manager().list_knowledge(database_name))
    if not entries:
        console.print(
            f"[warning]No knowledge entries to clear for database '{database_name}'[/warning]"
        )
        logger.info("knowledge.clear.nothing", database=database_name)
        return

    if not force:
        console.print(
            f"[warning]About to clear {len(entries)} knowledge entries for database '{database_name}'[/warning]"
        )
        if not questionary.confirm("Are you sure you want to proceed?").ask():
            console.print("Operation cancelled")
            logger.info("knowledge.clear.cancelled", database=database_name)
            return

    cleared_count = _run(_manager().clear_knowledge(database_name))
    console.print(
        f"[success]✓ Cleared {cleared_count} knowledge entries for database '{database_name}'[/success]"
    )
    logger.info(
        "knowledge.clear.success", database=database_name, deleted=cleared_count
    )


def create_knowledge_app() -> cyclopts.App:
    """Return the knowledge management CLI app."""
    return knowledge_app
