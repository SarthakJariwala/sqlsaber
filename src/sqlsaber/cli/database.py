"""Database management CLI commands."""

import asyncio
import getpass
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
import questionary
from rich.table import Table

from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console

type SchemaList = list[str]

# Global instances for CLI commands
console = create_console()
config_manager = DatabaseConfigManager()
logger = get_logger(__name__)

# Create the database management CLI app
db_app = cyclopts.App(
    name="db",
    help="Manage database connections",
)


def _normalize_schema_list(raw_schemas: SchemaList) -> SchemaList:
    """Deduplicate schemas while preserving order and case."""
    schemas: SchemaList = []
    seen: set[str] = set()
    for schema in raw_schemas:
        item = schema.strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        schemas.append(item)
    return schemas


def _parse_schema_list(raw: str | None) -> SchemaList:
    """Parse comma-separated schema list into cleaned list."""
    if not raw:
        return []
    return _normalize_schema_list(raw.split(","))


@db_app.command
def add(
    name: Annotated[str, cyclopts.Parameter(help="Name for the database connection")],
    type: Annotated[
        str,
        cyclopts.Parameter(
            ["--type", "-t"],
            help="Database type (postgresql, mysql, sqlite, duckdb)",
        ),
    ] = "postgresql",
    host: Annotated[
        str | None,
        cyclopts.Parameter(["--host", "-h"], help="Database host"),
    ] = None,
    port: Annotated[
        int | None,
        cyclopts.Parameter(["--port", "-p"], help="Database port"),
    ] = None,
    database: Annotated[
        str | None,
        cyclopts.Parameter(["--database", "--db"], help="Database name"),
    ] = None,
    username: Annotated[
        str | None,
        cyclopts.Parameter(["--username", "-u"], help="Username"),
    ] = None,
    ssl_mode: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--ssl-mode"],
            help="SSL mode (disable, allow, prefer, require, verify-ca, verify-full for PostgreSQL; DISABLED, PREFERRED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY for MySQL)",
        ),
    ] = None,
    ssl_ca: Annotated[
        str | None,
        cyclopts.Parameter(["--ssl-ca"], help="SSL CA certificate file path"),
    ] = None,
    ssl_cert: Annotated[
        str | None,
        cyclopts.Parameter(["--ssl-cert"], help="SSL client certificate file path"),
    ] = None,
    ssl_key: Annotated[
        str | None,
        cyclopts.Parameter(["--ssl-key"], help="SSL client private key file path"),
    ] = None,
    exclude_schemas: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--exclude-schemas"],
            help="Comma-separated list of schemas to exclude from introspection",
        ),
    ] = None,
    interactive: Annotated[
        bool,
        cyclopts.Parameter(
            ["--interactive", "--no-interactive"],
            help="Use interactive mode",
        ),
    ] = True,
) -> None:
    """Add a new database connection."""
    logger.info(
        "db.add.start",
        name=name,
        type=type,
        interactive=bool(interactive),
        has_password=False,
    )

    if interactive:
        # Interactive mode - prompt for all required fields
        from sqlsaber.application.db_setup import collect_db_input
        from sqlsaber.application.prompts import AsyncPrompter

        console.print(f"[bold]Adding database connection: {name}[/bold]")

        async def collect_input():
            prompter = AsyncPrompter()
            return await collect_db_input(
                prompter=prompter, name=name, db_type=type, include_ssl=True
            )

        db_input = asyncio.run(collect_input())

        if db_input is None:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("db.add.cancelled")
            return

        # Extract values from db_input
        type = db_input.type
        host = db_input.host
        port = db_input.port
        database = db_input.database
        username = db_input.username
        password = db_input.password
        ssl_mode = db_input.ssl_mode
        ssl_ca = db_input.ssl_ca
        ssl_cert = db_input.ssl_cert
        ssl_key = db_input.ssl_key
        exclude_schema_list = _normalize_schema_list(db_input.exclude_schemas)
    else:
        # Non-interactive mode - use provided values or defaults
        if type == "sqlite":
            if not database:
                console.print(
                    "[bold error]Error:[/bold error] Database file path is required for SQLite"
                )
                logger.error("db.add.missing_path", db_type="sqlite")
                sys.exit(1)
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        elif type == "duckdb":
            if database is None:
                console.print(
                    "[bold error]Error:[/bold error] Database file path is required for DuckDB"
                )
                logger.error("db.add.missing_path", db_type="duckdb")
                raise SystemExit(1)
            database = str(Path(database).expanduser().resolve())
            host = "localhost"
            port = 0
            username = "duckdb"
            password = ""
        else:
            if not all([host, database, username]):
                console.print(
                    "[bold error]Error:[/bold error] Host, database, and username are required"
                )
                logger.error("db.add.missing_fields")
                sys.exit(1)

            if port is None:
                port = 5432 if type == "postgresql" else 3306

            password = (
                getpass.getpass("Password (stored in your OS keychain): ")
                if questionary.confirm("Enter password?").ask()
                else ""
            )
        exclude_schema_list = _parse_schema_list(exclude_schemas)

    # Create database config
    # At this point, all required values should be set
    assert database is not None, "Database should be set by now"
    if type != "sqlite":
        assert host is not None, "Host should be set by now"
        assert port is not None, "Port should be set by now"
        assert username is not None, "Username should be set by now"

    db_config = DatabaseConfig(
        name=name,
        type=type,
        host=host,
        port=port,
        database=database,
        username=username,
        ssl_mode=ssl_mode,
        ssl_ca=ssl_ca,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
        exclude_schemas=exclude_schema_list,
    )

    try:
        # Add the configuration
        config_manager.add_database(db_config, password if password else None)
        console.print(
            f"[success]Successfully added database connection '{name}'[/success]"
        )
        logger.info("db.add.success", name=name, type=type)

        # Set as default if it's the first one
        if len(config_manager.list_databases()) == 1:
            console.print(f"[blue]Set '{name}' as default database[/blue]")
            logger.info("db.default.set", name=name)

    except Exception as e:
        logger.exception("db.add.error", name=name, error=str(e))
        console.print(f"[bold error]Error adding database:[/bold error] {e}")
        sys.exit(1)


@db_app.command(name="list")
def list_databases() -> None:
    """List all configured database connections."""
    logger.info("db.list.start")
    databases = config_manager.list_databases()
    default_name = config_manager.get_default_name()

    if not databases:
        console.print("[warning]No database connections configured[/warning]")
        console.print("Use 'sqlsaber db add <name>' to add a database connection")
        logger.info("db.list.empty")
        return

    table = Table(title="Database Connections")
    table.add_column("Name", style="info")
    table.add_column("Type", style="accent")
    table.add_column("Host", style="success")
    table.add_column("Port", style="warning")
    table.add_column("Database", style="info")
    table.add_column("Username", style="info")
    table.add_column("Excluded Schemas", style="muted")
    table.add_column("SSL", style="success")
    table.add_column("Default", style="error")

    for db in databases:
        is_default = "✓" if db.name == default_name else ""

        # Format SSL status
        ssl_status = ""
        if db.ssl_mode:
            ssl_status = db.ssl_mode
            if db.ssl_ca or db.ssl_cert:
                ssl_status += " (certs)"
        else:
            ssl_status = "disabled" if db.type not in {"sqlite", "duckdb"} else "N/A"

        table.add_row(
            db.name,
            db.type,
            db.host,
            str(db.port) if db.port else "",
            db.database,
            db.username,
            ", ".join(db.exclude_schemas) if db.exclude_schemas else "",
            ssl_status,
            is_default,
        )

    console.print(table)
    logger.info("db.list.complete", count=len(databases))


@db_app.command
def exclude(
    name: Annotated[
        str,
        cyclopts.Parameter(help="Name of the database connection to update"),
    ],
    set_schemas: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--set"],
            help="Replace excluded schemas with this comma-separated list",
        ),
    ] = None,
    add_schemas: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--add"],
            help="Add comma-separated schemas to the existing exclude list",
        ),
    ] = None,
    remove_schemas: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--remove"],
            help="Remove comma-separated schemas from the existing exclude list",
        ),
    ] = None,
    clear: Annotated[
        bool,
        cyclopts.Parameter(
            ["--clear", "--no-clear"],
            help="Clear all excluded schemas",
        ),
    ] = False,
) -> None:
    """Update excluded schemas for a database connection."""
    logger.info(
        "db.exclude.start",
        name=name,
        set=bool(set_schemas),
        add=bool(add_schemas),
        remove=bool(remove_schemas),
        clear=clear,
    )
    db_config = config_manager.get_database(name)
    if db_config is None:
        console.print(
            f"[bold error]Error: Database connection '{name}' not found[/bold error]"
        )
        logger.error("db.exclude.not_found", name=name)
        raise SystemExit(1)

    actions_selected = sum(
        bool(flag)
        for flag in [
            set_schemas is not None,
            add_schemas is not None,
            remove_schemas is not None,
            clear,
        ]
    )
    if actions_selected > 1:
        console.print(
            "[bold error]Error: Specify only one of --set, --add, --remove, or --clear[/bold error]"
        )
        logger.error("db.exclude.multiple_actions", name=name)
        sys.exit(1)

    current = [*(db_config.exclude_schemas or [])]

    if clear:
        updated = []
    elif set_schemas is not None:
        updated = _parse_schema_list(set_schemas)
    elif add_schemas is not None:
        additions = _parse_schema_list(add_schemas)
        updated = [*current]
        current_set = set(current)
        for schema in additions:
            if schema not in current_set:
                updated.append(schema)
                current_set.add(schema)
    elif remove_schemas is not None:
        removals = set(_parse_schema_list(remove_schemas))
        updated = [schema for schema in current if schema not in removals]
    else:
        console.print(
            "[info]Update excluded schemas for "
            f"[primary]{name}[/primary] (leave blank to clear)[/info]"
        )
        default_value = ", ".join(current)
        response = questionary.text(
            "Schemas to exclude (comma separated):", default=default_value
        ).ask()
        if response is None:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("db.exclude.cancelled", name=name)
            return
        updated = _parse_schema_list(response)

    db_config.exclude_schemas = _normalize_schema_list(updated)
    config_manager.update_database(db_config)

    console.print(
        f"[success]Updated excluded schemas for '{name}':[/success] "
        f"{', '.join(db_config.exclude_schemas) if db_config.exclude_schemas else '(none)'}"
    )
    logger.info("db.exclude.success", name=name, count=len(db_config.exclude_schemas))


@db_app.command
def remove(
    name: Annotated[
        str, cyclopts.Parameter(help="Name of the database connection to remove")
    ],
) -> None:
    """Remove a database connection."""
    logger.info("db.remove.start", name=name)
    if not config_manager.get_database(name):
        console.print(
            f"[bold error]Error: Database connection '{name}' not found[/bold error]"
        )
        logger.error("db.remove.not_found", name=name)
        sys.exit(1)

    if questionary.confirm(
        f"Are you sure you want to remove database connection '{name}'?"
    ).ask():
        if config_manager.remove_database(name):
            console.print(
                f"[success]Successfully removed database connection '{name}'[/success]"
            )
            logger.info("db.remove.success", name=name)
        else:
            console.print(
                f"[bold error]Error: Failed to remove database connection '{name}'[/bold error]"
            )
            logger.error("db.remove.failed", name=name)
            sys.exit(1)
    else:
        console.print("[warning]Operation cancelled[/warning]")
        logger.info("db.remove.cancelled", name=name)


@db_app.command
def set_default(
    name: Annotated[
        str,
        cyclopts.Parameter(help="Name of the database connection to set as default"),
    ],
) -> None:
    """Set the default database connection."""
    logger.info("db.default.start", name=name)
    if not config_manager.get_database(name):
        console.print(
            f"[bold error]Error: Database connection '{name}' not found[/bold error]"
        )
        logger.error("db.default.not_found", name=name)
        sys.exit(1)

    if config_manager.set_default_database(name):
        console.print(
            f"[success]Successfully set '{name}' as default database[/success]"
        )
        logger.info("db.default.success", name=name)
    else:
        console.print(
            f"[bold error]Error: Failed to set '{name}' as default[/bold error]"
        )
        logger.error("db.default.failed", name=name)
        sys.exit(1)


@db_app.command
def test(
    name: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Name of the database connection to test (uses default if not specified)",
        ),
    ] = None,
) -> None:
    """Test a database connection."""
    logger.info("db.test.start")

    async def test_connection():
        # Lazy import to keep CLI startup fast
        from sqlsaber.database import DatabaseConnection

        if name:
            db_config = config_manager.get_database(name)
            if db_config is None:
                console.print(
                    f"[bold error]Error: Database connection '{name}' not found[/bold error]"
                )
                logger.error("db.test.not_found", name=name)
                raise SystemExit(1)
        else:
            db_config = config_manager.get_default_database()
            if db_config is None:
                console.print(
                    "[bold error]Error: No default database configured[/bold error]"
                )
                console.print(
                    "Use 'sqlsaber db add <name>' to add a database connection"
                )
                logger.error("db.test.no_default")
                raise SystemExit(1)

        console.print(f"[blue]Testing connection to '{db_config.name}'...[/blue]")

        try:
            connection_string = db_config.to_connection_string()
            db_conn = DatabaseConnection(
                connection_string, excluded_schemas=db_config.exclude_schemas
            )

            # Try to connect and run a simple query
            await db_conn.execute_query("SELECT 1 as test")
            await db_conn.close()

            console.print(
                f"[success]✓ Connection to '{db_config.name}' successful[/success]"
            )
            logger.info("db.test.success", name=db_config.name)

        except Exception as e:
            logger.exception(
                "db.test.failed",
                name=(
                    db_config.name if "db_config" in locals() and db_config else name
                ),
                error=str(e),
            )
            console.print(f"[bold error]✗ Connection failed: {e}[/bold error]")
            sys.exit(1)

    asyncio.run(test_connection())


def create_db_app() -> cyclopts.App:
    """Return the database management CLI app."""
    return db_app
