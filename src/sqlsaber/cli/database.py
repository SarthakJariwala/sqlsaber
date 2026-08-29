"""Database management CLI commands."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import cyclopts

from sqlsaber.cli.output import fail, fail_usage, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b
from sqlsaber.render import cli_out, form_session

type SchemaList = list[str]

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
        fail_usage(
            f"unsupported database type '{type}'.\n"
            "  Choose from: postgresql, mysql, sqlite, duckdb\n"
            "  Example: saber db add analytics --type postgresql"
        )
    if interactive and password_stdin:
        fail_usage(
            "--password-stdin requires --no-interactive.\n"
            "  Example: printf '%s' \"$DB_PASSWORD\" | saber db add analytics "
            "--no-interactive --host HOST --database DB --username USER "
            "--password-stdin"
        )

    if interactive:
        from sqlsaber.application.db_setup import collect_db_input
        from sqlsaber.application.prompts import AsyncPrompter

        out(b.md(f"**Adding database connection: {name}**"))

        async def collect_input():
            prompter = AsyncPrompter()
            with form_session(cli_out()):
                return await collect_db_input(
                    prompter=prompter, name=name, db_type=type, include_ssl=True
                )

        db_input = asyncio.run(collect_input())

        if db_input is None:
            out(b.warn("Operation cancelled"))
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
                logger.error("db.add.missing_path", db_type="sqlite")
                fail_usage(
                    "database file path is required for SQLite.\n"
                    "  Example: saber db add local --no-interactive --type sqlite "
                    "--database ./local.db"
                )
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        elif type == "duckdb":
            if database is None:
                logger.error("db.add.missing_path", db_type="duckdb")
                fail_usage(
                    "database file path is required for DuckDB.\n"
                    "  Example: saber db add warehouse --no-interactive --type duckdb "
                    "--database ./warehouse.duckdb"
                )
            database = str(Path(database).expanduser().resolve())
            host = "localhost"
            port = 0
            username = "duckdb"
            password = ""
        else:
            if not all([host, database, username]):
                logger.error("db.add.missing_fields")
                fail_usage(
                    "--host, --database, and --username are required "
                    "in non-interactive mode.\n"
                    "  Example: saber db add analytics --no-interactive "
                    "--host HOST --database DB --username USER"
                )

            if port is None:
                port = 5432 if type == "postgresql" else 3306

            if password_stdin:
                if sys.stdin.isatty():
                    fail_usage(
                        "--password-stdin requires piped stdin.\n"
                        "  Example: printf '%s' \"$DB_PASSWORD\" | saber db add "
                        "analytics --no-interactive --host HOST --database DB "
                        "--username USER --password-stdin"
                    )
                password = sys.stdin.read().rstrip("\r\n")
                if not password:
                    fail_usage("--password-stdin received an empty password.")
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
        config_manager.add_database(db_config, password if password else None)
        out(b.success(f"Successfully added database connection '{name}'"))
        logger.info("db.add.success", name=name, type=type)

        if len(config_manager.list_databases()) == 1:
            out(b.md(f"Set '{name}' as default database"))
            logger.info("db.default.set", name=name)

    except Exception as e:
        logger.exception("db.add.error", name=name, error=str(e))
        fail(f"Error adding database: {e}")


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
        out(
            b.warn("No database connections configured"),
            b.md("Use 'sqlsaber db add <name>' to add a database connection"),
        )
        logger.info("db.list.empty")
        return

    rows: list[dict[str, str]] = []
    for db in databases:
        is_default = "✓" if db.name == default_name else ""
        if db.ssl_mode:
            ssl_status = db.ssl_mode
            if db.ssl_ca or db.ssl_cert:
                ssl_status += " (certs)"
        else:
            ssl_status = "disabled" if db.type not in {"sqlite", "duckdb"} else "N/A"
        rows.append(
            {
                "name": db.name,
                "type": db.type,
                "host": db.host or "",
                "port": str(db.port) if db.port else "",
                "database": db.database or "",
                "username": db.username or "",
                "excluded": ", ".join(db.exclude_schemas) if db.exclude_schemas else "",
                "ssl": ssl_status,
                "default": is_default,
            }
        )

    out(
        b.table(
            rows,
            columns=(
                b.Column("name", "Name", role="info"),
                b.Column("type", "Type", role="accent"),
                b.Column("host", "Host", role="success"),
                b.Column("port", "Port", role="warning"),
                b.Column("database", "Database", role="info"),
                b.Column("username", "Username", role="info"),
                b.Column("excluded", "Excluded Schemas", role="muted"),
                b.Column("ssl", "SSL", role="success"),
                b.Column("default", "Default", role="error"),
            ),
            caption="Database Connections",
            max_rows=1000,
        )
    )
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
        logger.error("db.exclude.not_found", name=name)
        fail(
            f"database connection '{name}' not found.\n"
            "  List connections with: saber db list"
        )

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
        logger.error("db.exclude.multiple_actions", name=name)
        fail(
            "specify only one of --set, --add, --remove, or --clear.\n"
            "  Example: saber db exclude analytics --add audit,temp"
        )

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
        from sqlsaber.application.prompts import AsyncPrompter

        out(b.md(f"Update excluded schemas for **{name}** (leave blank to clear)"))
        default_value = ", ".join(current)
        response = asyncio.run(
            AsyncPrompter().text(
                "Schemas to exclude (comma separated):", default=default_value
            )
        )
        if response is None:
            out(b.warn("Operation cancelled"))
            logger.info("db.exclude.cancelled", name=name)
            return
        updated = _parse_schema_list(response)

    db_config.exclude_schemas = _normalize_schema_list(updated)
    config_manager.update_database(db_config)

    schemas = (
        ", ".join(db_config.exclude_schemas) if db_config.exclude_schemas else "(none)"
    )
    out(b.success(f"Updated excluded schemas for '{name}': {schemas}"))
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
        logger.error("db.remove.not_found", name=name)
        fail(
            f"database connection '{name}' not found.\n"
            "  List connections with: saber db list"
        )

    if confirm_action(
        yes=yes,
        prompt=f"Remove database connection '{name}'?",
        non_interactive_command=f"saber db remove {name} --yes",
    ):
        if config_manager.remove_database(name):
            out(b.success(f"Successfully removed database connection '{name}'"))
            logger.info("db.remove.success", name=name)
        else:
            logger.error("db.remove.failed", name=name)
            fail(f"failed to remove database connection '{name}'.")
    else:
        out(b.warn("Operation cancelled"))
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
        logger.error("db.default.not_found", name=name)
        fail(
            f"database connection '{name}' not found.\n"
            "  List connections with: saber db list"
        )

    if config_manager.set_default_database(name):
        out(b.success(f"Successfully set '{name}' as default database"))
        logger.info("db.default.success", name=name)
    else:
        logger.error("db.default.failed", name=name)
        fail(f"failed to set '{name}' as default.")


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
                logger.error("db.test.not_found", name=name)
                fail(
                    f"database connection '{name}' not found.\n"
                    "  List connections with: saber db list"
                )
        else:
            db_config = config_manager.get_default_database()
            if db_config is None:
                logger.error("db.test.no_default")
                fail(
                    "no default database configured.\n"
                    "  Add one with: saber db add <name>"
                )

        out(b.md(f"Testing connection to '{db_config.name}'..."))

        try:
            connection_string = db_config.to_connection_string()
            db_conn = DatabaseConnection(
                connection_string, excluded_schemas=db_config.exclude_schemas
            )

            await db_conn.execute_query("SELECT 1 as test")
            await db_conn.close()

            out(b.success(f"Connection to '{db_config.name}' successful"))
            logger.info("db.test.success", name=db_config.name)

        except Exception as e:
            logger.exception(
                "db.test.failed",
                name=(
                    db_config.name if "db_config" in locals() and db_config else name
                ),
                error=str(e),
            )
            fail(f"Connection failed: {e}")

    asyncio.run(test_connection())


def create_db_app() -> cyclopts.App:
    """Return the database management CLI app."""
    return db_app
