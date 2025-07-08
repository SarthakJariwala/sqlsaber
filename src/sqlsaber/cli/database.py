"""Database management CLI commands."""

import asyncio
import getpass
import sys
from pathlib import Path
from typing import Annotated

import questionary
import cyclopts
from rich.console import Console
from rich.table import Table

from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.database.connection import DatabaseConnection

# Global instances for CLI commands
console = Console()
config_manager = DatabaseConfigManager()

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
            help="Database type (postgresql, mysql, sqlite)",
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

    if interactive:
        # Interactive mode - prompt for all required fields
        console.print(f"[bold]Adding database connection: {name}[/bold]")

        # Database type
        if not type or type == "postgresql":
            type = questionary.select(
                "Database type:",
                choices=["postgresql", "mysql", "sqlite"],
                default="postgresql",
            ).ask()

        if type == "sqlite":
            # SQLite only needs database path
            database = database or questionary.path("Database file path:").ask()
            database = str(Path(database).expanduser().resolve())
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        else:
            # PostgreSQL/MySQL need connection details
            host = host or questionary.text("Host:", default="localhost").ask()

            default_port = 5432 if type == "postgresql" else 3306
            port = port or int(
                questionary.text("Port:", default=str(default_port)).ask()
            )

            database = database or questionary.text("Database name:").ask()
            username = username or questionary.text("Username:").ask()

            # Ask for password
            password = getpass.getpass("Password (stored in your OS keychain): ")

            # Ask for SSL configuration
            if questionary.confirm("Configure SSL/TLS settings?", default=False).ask():
                if type == "postgresql":
                    ssl_mode = (
                        ssl_mode
                        or questionary.select(
                            "SSL mode for PostgreSQL:",
                            choices=[
                                "disable",
                                "allow",
                                "prefer",
                                "require",
                                "verify-ca",
                                "verify-full",
                            ],
                            default="prefer",
                        ).ask()
                    )
                elif type == "mysql":
                    ssl_mode = (
                        ssl_mode
                        or questionary.select(
                            "SSL mode for MySQL:",
                            choices=[
                                "DISABLED",
                                "PREFERRED",
                                "REQUIRED",
                                "VERIFY_CA",
                                "VERIFY_IDENTITY",
                            ],
                            default="PREFERRED",
                        ).ask()
                    )

                if ssl_mode and ssl_mode not in ["disable", "DISABLED"]:
                    if questionary.confirm(
                        "Specify SSL certificate files?", default=False
                    ).ask():
                        ssl_ca = (
                            ssl_ca or questionary.path("SSL CA certificate file:").ask()
                        )
                        if questionary.confirm(
                            "Specify client certificate?", default=False
                        ).ask():
                            ssl_cert = (
                                ssl_cert
                                or questionary.path(
                                    "SSL client certificate file:"
                                ).ask()
                            )
                            ssl_key = (
                                ssl_key
                                or questionary.path(
                                    "SSL client private key file:"
                                ).ask()
                            )
    else:
        # Non-interactive mode - use provided values or defaults
        if type == "sqlite":
            if not database:
                console.print(
                    "[bold red]Error:[/bold red] Database file path is required for SQLite"
                )
                sys.exit(1)
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        else:
            if not all([host, database, username]):
                console.print(
                    "[bold red]Error:[/bold red] Host, database, and username are required"
                )
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

        # Set as default if it's the first one
        if len(config_manager.list_databases()) == 1:
            console.print(f"[blue]Set '{name}' as default database[/blue]")

    except Exception as e:
        console.print(f"[bold red]Error adding database:[/bold red] {e}")
        sys.exit(1)


@db_app.command
def list():
    """List all configured database connections."""
    databases = config_manager.list_databases()
    default_name = config_manager.get_default_name()

    if not databases:
        console.print("[yellow]No database connections configured[/yellow]")
        console.print("Use 'sqlsaber db add <name>' to add a database connection")
        return

    table = Table(title="Database Connections")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Host", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Database", style="blue")
    table.add_column("Username", style="white")
    table.add_column("SSL", style="bright_green")
    table.add_column("Default", style="bold red")

    for db in databases:
        is_default = "✓" if db.name == default_name else ""

        # Format SSL status
        ssl_status = ""
        if db.ssl_mode:
            ssl_status = db.ssl_mode
            if db.ssl_ca or db.ssl_cert:
                ssl_status += " (certs)"
        else:
            ssl_status = "disabled" if db.type != "sqlite" else "N/A"

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


@db_app.command
def remove(
    name: Annotated[
        str, cyclopts.Parameter(help="Name of the database connection to remove")
    ],
):
    """Remove a database connection."""
    if not config_manager.get_database(name):
        console.print(
            f"[bold red]Error:[/bold red] Database connection '{name}' not found"
        )
        sys.exit(1)

    if questionary.confirm(
        f"Are you sure you want to remove database connection '{name}'?"
    ).ask():
        if config_manager.remove_database(name):
            console.print(
                f"[green]Successfully removed database connection '{name}'[/green]"
            )
        else:
            console.print(
                f"[bold red]Error:[/bold red] Failed to remove database connection '{name}'"
            )
            sys.exit(1)
    else:
        console.print("Operation cancelled")


@db_app.command
def set_default(
    name: Annotated[
        str,
        cyclopts.Parameter(help="Name of the database connection to set as default"),
    ],
):
    """Set the default database connection."""
    if not config_manager.get_database(name):
        console.print(
            f"[bold red]Error:[/bold red] Database connection '{name}' not found"
        )
        sys.exit(1)

    if config_manager.set_default_database(name):
        console.print(f"[green]Successfully set '{name}' as default database[/green]")
    else:
        console.print(f"[bold red]Error:[/bold red] Failed to set '{name}' as default")
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

    async def test_connection():
        if name:
            db_config = config_manager.get_database(name)
            if not db_config:
                console.print(
                    f"[bold red]Error:[/bold red] Database connection '{name}' not found"
                )
                sys.exit(1)
        else:
            db_config = config_manager.get_default_database()
            if not db_config:
                console.print(
                    "[bold red]Error:[/bold red] No default database configured"
                )
                console.print(
                    "Use 'sqlsaber db add <name>' to add a database connection"
                )
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

        except Exception as e:
            console.print(f"[bold red]✗ Connection failed:[/bold red] {e}")
            sys.exit(1)

    asyncio.run(test_connection())


def create_db_app() -> cyclopts.App:
    """Return the database management CLI app."""
    return db_app
