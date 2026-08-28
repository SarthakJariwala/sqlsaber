"""Database management CLI commands."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
import questionary
from rich.table import Table

from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console

type SchemaList = list[str]

# Global instances for CLI commands
console = create_console()
error_console = create_console(stderr=True)
config_manager = DatabaseConfigManager()
logger = get_logger(__name__)

# Create the database management CLI app
db_app = cyclopts.App(
    name="db",
    help="Manage database connections",
    help_epilogue=(
        "Examples:\n\n"
        "saber db list\n\n"
        "saber db add analytics\n\n"
        "saber db test analytics"
    ),
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


@db_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber db add analytics\n\n"
        "saber db add local --type sqlite --database ./local.db --no-interactive\n\n"
        "printf '%s' \"$DB_PASSWORD\" | saber db add analytics --no-interactive --host HOST --database DB --username USER --password-stdin"
    )
)
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
    description: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--description"],
            help=(
                "Short human-readable description of this connection. Shown to "
                "the agent in multi-database sessions to help it pick the right DB."
            ),
        ),
    ] = None,
    interactive: Annotated[
        bool,
        cyclopts.Parameter(
            ["--interactive"],
            help="Use interactive mode",
        ),
    ] = True,
    password_stdin: Annotated[
        bool,
        cyclopts.Parameter(
            ["--password-stdin"],
            help="Read the database password from stdin (requires --no-interactive)",
        ),
    ] = False,
) -> None:
    """Add a new database connection.

    Examples:
        saber db add analytics
        saber db add local --type sqlite --database ./local.db --no-interactive
        printf '%s' "$DB_PASSWORD" | saber db add analytics --no-interactive --type postgresql --host db.example.com --database analytics --username agent --password-stdin
    """
    logger.info(
        "db.add.start",
        name=name,
        type=type,
        interactive=bool(interactive),
        has_password=False,
    )

    supported_types = {"postgresql", "mysql", "sqlite", "duckdb"}
    if type not in supported_types:
        error_console.print(
            f"[error]Error: unsupported database type '{type}'.[/error]\n"
            "  Choose from: postgresql, mysql, sqlite, duckdb\n"
            "  Example: saber db add analytics --type postgresql"
        )
        raise SystemExit(2)
    if interactive and password_stdin:
        error_console.print(
            "[error]Error: --password-stdin requires --no-interactive.[/error]\n"
            "  Example: printf '%s' \"$DB_PASSWORD\" | saber db add analytics "
            "--no-interactive --host HOST --database DB --username USER "
            "--password-stdin"
        )
        raise SystemExit(2)

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
                error_console.print(
                    "[error]Error: database file path is required for SQLite.[/error]\n"
                    "  Example: saber db add local --no-interactive --type sqlite "
                    "--database ./local.db"
                )
                logger.error("db.add.missing_path", db_type="sqlite")
                raise SystemExit(2)
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        elif type == "duckdb":
            if database is None:
                error_console.print(
                    "[error]Error: database file path is required for DuckDB.[/error]\n"
                    "  Example: saber db add warehouse --no-interactive --type duckdb "
                    "--database ./warehouse.duckdb"
                )
                logger.error("db.add.missing_path", db_type="duckdb")
                raise SystemExit(2)
            database = str(Path(database).expanduser().resolve())
            host = "localhost"
            port = 0
            username = "duckdb"
            password = ""
        else:
            if not all([host, database, username]):
                error_console.print(
                    "[error]Error: --host, --database, and --username are required "
                    "in non-interactive mode.[/error]\n"
                    "  Example: saber db add analytics --no-interactive "
                    "--host HOST --database DB --username USER"
                )
                logger.error("db.add.missing_fields")
                raise SystemExit(2)

            if port is None:
                port = 5432 if type == "postgresql" else 3306

            if password_stdin:
                if sys.stdin.isatty():
                    error_console.print(
                        "[error]Error: --password-stdin requires piped stdin.[/error]\n"
                        "  Example: printf '%s' \"$DB_PASSWORD\" | saber db add "
                        "analytics --no-interactive --host HOST --database DB "
                        "--username USER --password-stdin"
                    )
                    raise SystemExit(2)
                password = sys.stdin.read().rstrip("\r\n")
                if not password:
                    error_console.print(
                        "[error]Error: --password-stdin received an empty password.[/error]"
                    )
                    raise SystemExit(2)
            else:
                password = ""
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
        description=description,
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
        error_console.print(f"[error]Error adding database:[/error] {e}")
        sys.exit(1)


@db_app.command(name="list", help_epilogue="Example:\n\nsaber db list")
def list_databases() -> None:
    """List all configured database connections.

    Example:
        saber db list
    """
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


@db_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber db exclude analytics --add audit,temp\n\n"
        "saber db exclude analytics --clear"
    )
)
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
    """Update excluded schemas for a database connection.

    Examples:
        saber db exclude analytics --add audit,temp
        saber db exclude analytics --clear
    """
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
        error_console.print(
            f"[error]Error: database connection '{name}' not found.[/error]\n"
            "  List connections with: saber db list"
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
        error_console.print(
            "[error]Error: specify only one of --set, --add, --remove, or --clear.[/error]\n"
            "  Example: saber db exclude analytics --add audit,temp"
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


@db_app.command(
    help_epilogue=(
        "Examples:\n\nsaber db remove analytics\n\nsaber db remove analytics --yes"
    )
)
def remove(
    name: Annotated[
        str, cyclopts.Parameter(help="Name of the database connection to remove")
    ],
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Remove a database connection.

    Examples:
        saber db remove analytics
        saber db remove analytics --yes
    """
    logger.info("db.remove.start", name=name)
    if not config_manager.get_database(name):
        error_console.print(
            f"[error]Error: database connection '{name}' not found.[/error]\n"
            "  List connections with: saber db list"
        )
        logger.error("db.remove.not_found", name=name)
        sys.exit(1)

    if confirm_action(
        yes=yes,
        prompt=f"Remove database connection '{name}'?",
        non_interactive_command=f"saber db remove {name} --yes",
        error_console=error_console,
    ):
        if config_manager.remove_database(name):
            console.print(
                f"[success]Successfully removed database connection '{name}'[/success]"
            )
            logger.info("db.remove.success", name=name)
        else:
            error_console.print(
                f"[error]Error: failed to remove database connection '{name}'.[/error]"
            )
            logger.error("db.remove.failed", name=name)
            sys.exit(1)
    else:
        console.print("[warning]Operation cancelled[/warning]")
        logger.info("db.remove.cancelled", name=name)


@db_app.command(help_epilogue="Example:\n\nsaber db set-default analytics")
def set_default(
    name: Annotated[
        str,
        cyclopts.Parameter(help="Name of the database connection to set as default"),
    ],
) -> None:
    """Set the default database connection.

    Example:
        saber db set-default analytics
    """
    logger.info("db.default.start", name=name)
    if not config_manager.get_database(name):
        error_console.print(
            f"[error]Error: database connection '{name}' not found.[/error]\n"
            "  List connections with: saber db list"
        )
        logger.error("db.default.not_found", name=name)
        sys.exit(1)

    if config_manager.set_default_database(name):
        console.print(
            f"[success]Successfully set '{name}' as default database[/success]"
        )
        logger.info("db.default.success", name=name)
    else:
        error_console.print(f"[error]Error: failed to set '{name}' as default.[/error]")
        logger.error("db.default.failed", name=name)
        sys.exit(1)


@db_app.command(help_epilogue=("Examples:\n\nsaber db test\n\nsaber db test analytics"))
def test(
    name: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Name of the database connection to test (uses default if not specified)",
        ),
    ] = None,
) -> None:
    """Test a database connection.

    Examples:
        saber db test
        saber db test analytics
    """
    logger.info("db.test.start")

    async def test_connection():
        # Lazy import to keep CLI startup fast
        from sqlsaber.database import DatabaseConnection

        if name:
            db_config = config_manager.get_database(name)
            if db_config is None:
                error_console.print(
                    f"[error]Error: database connection '{name}' not found.[/error]\n"
                    "  List connections with: saber db list"
                )
                logger.error("db.test.not_found", name=name)
                raise SystemExit(1)
        else:
            db_config = config_manager.get_default_database()
            if db_config is None:
                error_console.print(
                    "[error]Error: no default database configured.[/error]\n"
                    "  Add one with: saber db add <name>"
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
            error_console.print(f"[error]Connection failed: {e}[/error]")
            sys.exit(1)

    asyncio.run(test_connection())


def create_db_app() -> cyclopts.App:
    """Return the database management CLI app."""
    return db_app
