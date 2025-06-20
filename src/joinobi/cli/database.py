"""Database management CLI commands."""

import asyncio
import getpass
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from joinobi.config.database import DatabaseConfig, DatabaseConfigManager
from joinobi.database.connection import DatabaseConnection

# Global instances for CLI commands
console = Console()
config_manager = DatabaseConfigManager()

# Create the database management CLI app
db_app = typer.Typer(
    name="db",
    help="Manage database connections",
    add_completion=True,
)


@db_app.command("add")
def add_database(
    name: str = typer.Argument(..., help="Name for the database connection"),
    type: str = typer.Option(
        "postgresql",
        "--type",
        "-t",
        help="Database type (postgresql, mysql, sqlite)",
    ),
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Database host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Database port"),
    database: Optional[str] = typer.Option(
        None, "--database", "--db", help="Database name"
    ),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Username"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Use interactive mode"
    ),
):
    """Add a new database connection."""

    if interactive:
        # Interactive mode - prompt for all required fields
        console.print(f"[bold]Adding database connection: {name}[/bold]")

        # Database type
        if not type or type == "postgresql":
            type = Prompt.ask(
                "Database type",
                choices=["postgresql", "mysql", "sqlite"],
                default="postgresql",
            )

        if type == "sqlite":
            # SQLite only needs database path
            database = database or Prompt.ask("Database file path")
            host = "localhost"
            port = 0
            username = "sqlite"
        else:
            # PostgreSQL/MySQL need connection details
            host = host or Prompt.ask("Host", default="localhost")

            default_port = 5432 if type == "postgresql" else 3306
            port = port or int(Prompt.ask("Port", default=str(default_port)))

            database = database or Prompt.ask("Database name")
            username = username or Prompt.ask("Username")

            # Ask for password
            password = getpass.getpass("Password (optional, will be stored securely): ")
    else:
        # Non-interactive mode - use provided values or defaults
        if type == "sqlite":
            if not database:
                console.print(
                    "[bold red]Error:[/bold red] Database file path is required for SQLite"
                )
                raise typer.Exit(1)
            host = "localhost"
            port = 0
            username = "sqlite"
            password = ""
        else:
            if not all([host, database, username]):
                console.print(
                    "[bold red]Error:[/bold red] Host, database, and username are required"
                )
                raise typer.Exit(1)

            if port is None:
                port = 5432 if type == "postgresql" else 3306

            password = (
                getpass.getpass("Password (optional): ")
                if typer.confirm("Enter password?")
                else ""
            )

    # Create database config
    db_config = DatabaseConfig(
        name=name,
        type=type,
        host=host,
        port=port,
        database=database,
        username=username,
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
        raise typer.Exit(1)


@db_app.command("list")
def list_databases():
    """List all configured database connections."""
    databases = config_manager.list_databases()
    default_name = config_manager.get_default_name()

    if not databases:
        console.print("[yellow]No database connections configured[/yellow]")
        console.print("Use 'joinobi db add <name>' to add a database connection")
        return

    table = Table(title="Database Connections")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Host", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Database", style="blue")
    table.add_column("Username", style="white")
    table.add_column("Default", style="bold red")

    for db in databases:
        is_default = "✓" if db.name == default_name else ""
        table.add_row(
            db.name,
            db.type,
            db.host,
            str(db.port) if db.port else "",
            db.database,
            db.username,
            is_default,
        )

    console.print(table)


@db_app.command("remove")
def remove_database(
    name: str = typer.Argument(..., help="Name of the database connection to remove"),
):
    """Remove a database connection."""
    if not config_manager.get_database(name):
        console.print(
            f"[bold red]Error:[/bold red] Database connection '{name}' not found"
        )
        raise typer.Exit(1)

    if Confirm.ask(f"Are you sure you want to remove database connection '{name}'?"):
        if config_manager.remove_database(name):
            console.print(
                f"[green]Successfully removed database connection '{name}'[/green]"
            )
        else:
            console.print(
                f"[bold red]Error:[/bold red] Failed to remove database connection '{name}'"
            )
            raise typer.Exit(1)
    else:
        console.print("Operation cancelled")


@db_app.command("set-default")
def set_default_database(
    name: str = typer.Argument(
        ..., help="Name of the database connection to set as default"
    ),
):
    """Set the default database connection."""
    if not config_manager.get_database(name):
        console.print(
            f"[bold red]Error:[/bold red] Database connection '{name}' not found"
        )
        raise typer.Exit(1)

    if config_manager.set_default_database(name):
        console.print(f"[green]Successfully set '{name}' as default database[/green]")
    else:
        console.print(f"[bold red]Error:[/bold red] Failed to set '{name}' as default")
        raise typer.Exit(1)


@db_app.command("test")
def test_database(
    name: Optional[str] = typer.Argument(
        None,
        help="Name of the database connection to test (uses default if not specified)",
    ),
):
    """Test a database connection."""

    async def test_connection():
        if name:
            db_config = config_manager.get_database(name)
            if not db_config:
                console.print(
                    f"[bold red]Error:[/bold red] Database connection '{name}' not found"
                )
                raise typer.Exit(1)
        else:
            db_config = config_manager.get_default_database()
            if not db_config:
                console.print(
                    "[bold red]Error:[/bold red] No default database configured"
                )
                console.print(
                    "Use 'joinobi db add <name>' to add a database connection"
                )
                raise typer.Exit(1)

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
            raise typer.Exit(1)

    asyncio.run(test_connection())


def create_db_app() -> typer.Typer:
    """Return the database management CLI app."""
    return db_app
