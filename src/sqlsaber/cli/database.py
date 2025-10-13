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

# Global instances for CLI commands
console = create_console()
config_manager = DatabaseConfigManager()
logger = get_logger(__name__)

# Create the database management CLI app
db_app = cyclopts.App(
    name="db",
    help="Manage database connections",
)


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
    interactive: Annotated[
        bool,
        cyclopts.Parameter(
            ["--interactive", "--no-interactive"],
            help="Use interactive mode",
        ),
    ] = True,
):
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
            if not database:
                console.print(
                    "[bold error]Error:[/bold error] Database file path is required for DuckDB"
                )
                logger.error("db.add.missing_path", db_type="duckdb")
                sys.exit(1)
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
    )

    try:
        # Add the configuration
        config_manager.add_database(db_config, password if password else None)
        console.print(f"[green]Successfully added database connection '{name}'[/green]")
        logger.info("db.add.success", name=name, type=type)

        # Set as default if it's the first one
        if len(config_manager.list_databases()) == 1:
            console.print(f"[blue]Set '{name}' as default database[/blue]")
            logger.info("db.default.set", name=name)

    except Exception as e:
        logger.exception("db.add.error", name=name, error=str(e))
        console.print(f"[bold error]Error adding database:[/bold error] {e}")
        sys.exit(1)


@db_app.command
def list():
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
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="accent")
    table.add_column("Host", style="success")
    table.add_column("Port", style="warning")
    table.add_column("Database", style="info")
    table.add_column("Username", style="info")
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
            ssl_status,
            is_default,
        )

    console.print(table)
    logger.info("db.list.complete", count=len(databases))


@db_app.command
def remove(
    name: Annotated[
        str, cyclopts.Parameter(help="Name of the database connection to remove")
    ],
):
    """Remove a database connection."""
    logger.info("db.remove.start", name=name)
    if not config_manager.get_database(name):
        console.print(
            f"[bold error]Error:[/bold error] Database connection '{name}' not found"
        )
        logger.error("db.remove.not_found", name=name)
        sys.exit(1)

    if questionary.confirm(
        f"Are you sure you want to remove database connection '{name}'?"
    ).ask():
        if config_manager.remove_database(name):
            console.print(
                f"[green]Successfully removed database connection '{name}'[/green]"
            )
            logger.info("db.remove.success", name=name)
        else:
            console.print(
                f"[bold error]Error:[/bold error] Failed to remove database connection '{name}'"
            )
            logger.error("db.remove.failed", name=name)
            sys.exit(1)
    else:
        console.print("Operation cancelled")
        logger.info("db.remove.cancelled", name=name)


@db_app.command
def set_default(
    name: Annotated[
        str,
        cyclopts.Parameter(help="Name of the database connection to set as default"),
    ],
):
    """Set the default database connection."""
    logger.info("db.default.start", name=name)
    if not config_manager.get_database(name):
        console.print(
            f"[bold error]Error:[/bold error] Database connection '{name}' not found"
        )
        logger.error("db.default.not_found", name=name)
        sys.exit(1)

    if config_manager.set_default_database(name):
        console.print(f"[green]Successfully set '{name}' as default database[/green]")
        logger.info("db.default.success", name=name)
    else:
        console.print(
            f"[bold error]Error:[/bold error] Failed to set '{name}' as default"
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
):
    """Test a database connection."""
    logger.info("db.test.start")

    async def test_connection():
        # Lazy import to keep CLI startup fast
        from sqlsaber.database import DatabaseConnection

        if name:
            db_config = config_manager.get_database(name)
            if not db_config:
                console.print(
                    f"[bold error]Error:[/bold error] Database connection '{name}' not found"
                )
                logger.error("db.test.not_found", name=name)
                sys.exit(1)
        else:
            db_config = config_manager.get_default_database()
            if not db_config:
                console.print(
                    "[bold error]Error:[/bold error] No default database configured"
                )
                console.print(
                    "Use 'sqlsaber db add <name>' to add a database connection"
                )
                logger.error("db.test.no_default")
                sys.exit(1)

        console.print(f"[blue]Testing connection to '{db_config.name}'...[/blue]")

        try:
            connection_string = db_config.to_connection_string()
            db_conn = DatabaseConnection(connection_string)

            # Try to connect and run a simple query
            await db_conn.execute_query("SELECT 1 as test")
            await db_conn.close()

            console.print(
                f"[green]✓ Connection to '{db_config.name}' successful[/green]"
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
            console.print(f"[bold error]✗ Connection failed:[/bold error] {e}")
            sys.exit(1)

    asyncio.run(test_connection())


def create_db_app() -> cyclopts.App:
    """Return the database management CLI app."""
    return db_app
