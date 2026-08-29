"""Knowledge management CLI commands."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Annotated, TypeVar

import cyclopts

from sqlsaber.cli.output import fail, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.render import blocks as b

config_manager = DatabaseConfigManager()
logger = get_logger(__name__)
knowledge_app = cyclopts.App(
    name="knowledge",
    help="Manage database-specific knowledge entries",
    help_epilogue=(
        'Examples:\n\nsaber knowledge list\n\nsaber knowledge search "shipped revenue"'
    ),
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
            logger.error("knowledge.db.not_found", database=database)
            fail(
                f"database connection '{database}' not found.\n"
                "  List connections with: saber db list"
            )
        return database

    db_config = config_manager.get_default_database()
    if db_config is None:
        logger.error("knowledge.db.none_configured")
        fail("no database connections configured.\n  Add one with: saber db add <name>")
    return db_config.name


@knowledge_app.command(
    help_epilogue=(
        "Examples:\n\n"
        'saber knowledge add "Revenue KPI" "Recognized shipped revenue"\n\n'
        'saber knowledge add "Revenue KPI" "Recognized shipped revenue" --database analytics --source finance-wiki'
    )
)
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
    """Add knowledge for the specified database.

    Examples:
        saber knowledge add "Revenue KPI" "Recognized shipped revenue"
        saber knowledge add "Revenue KPI" "Recognized shipped revenue" --database analytics --source finance-wiki
    """
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
        logger.exception("knowledge.add.error", database=database_name, error=str(exc))
        fail(f"Error adding knowledge: {exc}")

    out(
        b.success(f"Knowledge entry added for database '{database_name}'"),
        b.key_values({"ID": entry.id, "Name": entry.name}),
    )
    logger.info("knowledge.add.success", database=database_name, id=entry.id)


@knowledge_app.command(
    help_epilogue=(
        "Examples:\n\nsaber knowledge list\n\nsaber knowledge list --database analytics"
    )
)
def list(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """List all knowledge entries for the specified database.

    Examples:
        saber knowledge list
        saber knowledge list --database analytics
    """
    database_name = _get_database_name(database)
    logger.info("knowledge.list.start", database=database_name)

    entries = _run(_manager().list_knowledge(database_name))
    if not entries:
        out(
            b.warn(f"No knowledge entries found for database '{database_name}'"),
            b.md(
                'Use \'sqlsaber knowledge add "<name>" "<description>"\' to add entries'
            ),
        )
        logger.info("knowledge.list.empty", database=database_name)
        return

    out(
        b.table(
            [
                {
                    "id": entry.id,
                    "name": entry.name,
                    "description": _truncate(entry.description, 100),
                    "updated": _format_timestamp(entry.updated_at),
                }
                for entry in entries
            ],
            columns=(
                b.Column(field="id", header="ID", role="info"),
                b.Column(field="name", header="Name"),
                b.Column(field="description", header="Description"),
                b.Column(field="updated", header="Updated", role="muted"),
            ),
            caption=f"Knowledge Entries for Database: {database_name}",
            max_rows=1000,
        ),
        b.md(f"Total entries: {len(entries)}", role="muted"),
    )
    logger.info("knowledge.list.complete", database=database_name, count=len(entries))


@knowledge_app.command(
    help_epilogue="Example:\n\nsaber knowledge show ENTRY_ID --database analytics"
)
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
    """Show a full knowledge entry.

    Example:
        saber knowledge show ENTRY_ID --database analytics
    """
    database_name = _get_database_name(database)
    logger.info("knowledge.show.start", database=database_name, id=entry_id)

    entry = _run(_manager().get_knowledge(database_name, entry_id))
    if entry is None:
        logger.error("knowledge.show.not_found", database=database_name, id=entry_id)
        fail(
            f"knowledge entry '{entry_id}' not found for database '{database_name}'.\n"
            f"  List entries with: saber knowledge list --database {database_name}"
        )

    pairs = [
        ("ID", entry.id),
        ("Database", database_name),
        ("Name", entry.name),
        ("Created", _format_timestamp(entry.created_at)),
        ("Updated", _format_timestamp(entry.updated_at)),
    ]
    if entry.source:
        pairs.append(("Source", entry.source))
    out(b.key_values(pairs), b.md(f"**Description:**\n\n{entry.description}"))
    if entry.sql:
        out(b.md("**SQL:**"), b.code(entry.sql, "sql"))


@knowledge_app.command(
    help_epilogue=(
        "Examples:\n\n"
        'saber knowledge search "shipped revenue"\n\n'
        'saber knowledge search "shipped revenue" --database analytics --limit 5'
    )
)
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
    """Search knowledge entries for the specified database.

    Examples:
        saber knowledge search "shipped revenue"
        saber knowledge search "shipped revenue" --database analytics --limit 5
    """
    database_name = _get_database_name(database)
    logger.info("knowledge.search.start", database=database_name, limit=limit)

    entries = _run(_manager().search_knowledge(database_name, query, limit=limit))
    if not entries:
        out(
            b.warn(
                f"No knowledge entries matched '{query}' for database '{database_name}'"
            )
        )
        logger.info("knowledge.search.empty", database=database_name)
        return

    out(
        b.table(
            [
                {
                    "id": entry.id,
                    "name": entry.name,
                    "description": _truncate(entry.description, 120),
                    "source": entry.source or "",
                }
                for entry in entries
            ],
            columns=(
                b.Column(field="id", header="ID", role="info"),
                b.Column(field="name", header="Name"),
                b.Column(field="description", header="Description"),
                b.Column(field="source", header="Source", role="muted"),
            ),
            caption=f"Knowledge Search Results ({len(entries)} matches)",
            max_rows=1000,
        )
    )
    logger.info("knowledge.search.complete", database=database_name, count=len(entries))


@knowledge_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber knowledge remove ENTRY_ID\n\n"
        "saber knowledge remove ENTRY_ID --database analytics --yes"
    )
)
def remove(
    entry_id: Annotated[str, cyclopts.Parameter(help="Knowledge entry ID")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
):
    """Remove a specific knowledge entry by ID.

    Examples:
        saber knowledge remove ENTRY_ID
        saber knowledge remove ENTRY_ID --database analytics --yes
    """
    database_name = _get_database_name(database)
    logger.info("knowledge.remove.start", database=database_name, id=entry_id)

    entry = _run(_manager().get_knowledge(database_name, entry_id))
    if entry is None:
        logger.error("knowledge.remove.not_found", database=database_name, id=entry_id)
        fail(
            f"knowledge entry '{entry_id}' not found for database '{database_name}'.\n"
            f"  List entries with: saber knowledge list --database {database_name}"
        )

    if not confirm_action(
        yes=yes,
        prompt=f"Remove knowledge entry '{entry.name}'?",
        non_interactive_command=(
            f"saber knowledge remove {entry_id} --database {database_name} --yes"
        ),
    ):
        out(b.warn("Operation cancelled"))
        logger.info("knowledge.remove.cancelled", database=database_name, id=entry_id)
        return

    if _run(_manager().remove_knowledge(database_name, entry_id)):
        out(b.success(f"Knowledge entry removed from database '{database_name}'"))
        logger.info("knowledge.remove.success", database=database_name, id=entry_id)
        return

    logger.error("knowledge.remove.failed", database=database_name, id=entry_id)
    fail(f"failed to remove knowledge entry '{entry_id}'.")


@knowledge_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber knowledge clear --database analytics\n\n"
        "saber knowledge clear --database analytics --yes"
    )
)
def clear(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
    yes: Annotated[
        bool,
        cyclopts.Parameter(
            ["--yes"],
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Clear all knowledge entries for the specified database.

    Examples:
        saber knowledge clear --database analytics
        saber knowledge clear --database analytics --yes
    """
    database_name = _get_database_name(database)
    logger.info(
        "knowledge.clear.start", database=database_name, confirmation_skipped=bool(yes)
    )

    entries = _run(_manager().list_knowledge(database_name))
    if not entries:
        out(b.warn(f"No knowledge entries to clear for database '{database_name}'"))
        logger.info("knowledge.clear.nothing", database=database_name)
        return

    if not yes:
        out(
            b.warn(
                f"About to clear {len(entries)} knowledge entries for database '{database_name}'"
            )
        )
    if not confirm_action(
        yes=yes,
        prompt="Clear all knowledge entries?",
        non_interactive_command=(
            f"saber knowledge clear --database {database_name} --yes"
        ),
    ):
        out(b.warn("Operation cancelled"))
        logger.info("knowledge.clear.cancelled", database=database_name)
        return

    cleared_count = _run(_manager().clear_knowledge(database_name))
    out(
        b.success(
            f"Cleared {cleared_count} knowledge entries for database '{database_name}'"
        )
    )
    logger.info(
        "knowledge.clear.success", database=database_name, deleted=cleared_count
    )


def create_knowledge_app() -> cyclopts.App:
    """Return the knowledge management CLI app."""
    return knowledge_app
