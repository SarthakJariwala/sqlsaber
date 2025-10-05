"""Shared database setup logic for onboarding and CLI."""

import getpass
from dataclasses import dataclass
from pathlib import Path

from sqlsaber.application.prompts import Prompter
from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.theme.manager import create_console

console = create_console()


@dataclass
class DatabaseInput:
    """Input data for database configuration."""

    name: str
    type: str
    host: str
    port: int
    database: str
    username: str
    password: str | None
    ssl_mode: str | None = None
    ssl_ca: str | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None


async def collect_db_input(
    prompter: Prompter,
    name: str,
    db_type: str = "postgresql",
    include_ssl: bool = True,
) -> DatabaseInput | None:
    """Collect database connection details interactively.

    Args:
        prompter: Prompter instance for interaction
        name: Database connection name
        db_type: Initial database type (can be changed via prompt)
        include_ssl: Whether to prompt for SSL configuration

    Returns:
        DatabaseInput with collected values or None if cancelled
    """
    # Ask for database type
    db_type = await prompter.select(
        "Database type:",
        choices=["postgresql", "mysql", "sqlite", "duckdb"],
        default=db_type,
    )

    if db_type is None:
        return None

    # Handle file-based databases
    if db_type in {"sqlite", "duckdb"}:
        database_path = await prompter.path(
            f"{db_type.upper()} file path:", only_directories=False
        )

        if database_path is None:
            return None

        database = str(Path(database_path).expanduser().resolve())
        host = "localhost"
        port = 0
        username = db_type
        password = ""
        ssl_mode = None
        ssl_ca = None
        ssl_cert = None
        ssl_key = None

    else:
        # PostgreSQL/MySQL need connection details
        host = await prompter.text("Host:", default="localhost")
        if host is None:
            return None

        default_port = 5432 if db_type == "postgresql" else 3306
        port_str = await prompter.text("Port:", default=str(default_port))
        if port_str is None:
            return None

        try:
            port = int(port_str)
        except ValueError:
            console.print("[error]Invalid port number. Using default.[/error]")
            port = default_port

        database = await prompter.text("Database name:")
        if database is None:
            return None

        username = await prompter.text("Username:")
        if username is None:
            return None

        password = getpass.getpass("Password (stored in your OS keychain): ")

        ssl_mode = None
        ssl_ca = None
        ssl_cert = None
        ssl_key = None

        # Ask for SSL configuration if enabled
        if include_ssl:
            configure_ssl = await prompter.confirm(
                "Configure SSL/TLS settings?", default=False
            )
            if configure_ssl:
                if db_type == "postgresql":
                    ssl_mode = await prompter.select(
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
                    )
                elif db_type == "mysql":
                    ssl_mode = await prompter.select(
                        "SSL mode for MySQL:",
                        choices=[
                            "DISABLED",
                            "PREFERRED",
                            "REQUIRED",
                            "VERIFY_CA",
                            "VERIFY_IDENTITY",
                        ],
                        default="PREFERRED",
                    )

                if ssl_mode and ssl_mode not in ["disable", "DISABLED"]:
                    specify_certs = await prompter.confirm(
                        "Specify SSL certificate files?", default=False
                    )
                    if specify_certs:
                        ssl_ca = await prompter.path("SSL CA certificate file:")
                        specify_client = await prompter.confirm(
                            "Specify client certificate?", default=False
                        )
                        if specify_client:
                            ssl_cert = await prompter.path(
                                "SSL client certificate file:"
                            )
                            ssl_key = await prompter.path(
                                "SSL client private key file:"
                            )

    return DatabaseInput(
        name=name,
        type=db_type,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        ssl_mode=ssl_mode,
        ssl_ca=ssl_ca,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
    )


def build_config(db_input: DatabaseInput) -> DatabaseConfig:
    """Build DatabaseConfig from DatabaseInput."""
    return DatabaseConfig(
        name=db_input.name,
        type=db_input.type,
        host=db_input.host,
        port=db_input.port,
        database=db_input.database,
        username=db_input.username,
        ssl_mode=db_input.ssl_mode,
        ssl_ca=db_input.ssl_ca,
        ssl_cert=db_input.ssl_cert,
        ssl_key=db_input.ssl_key,
    )


async def test_connection(config: DatabaseConfig, password: str | None) -> bool:
    """Test database connection.

    Args:
        config: DatabaseConfig to test
        password: Password for connection (not stored in config yet)

    Returns:
        True if connection successful, False otherwise
    """
    from sqlsaber.database import DatabaseConnection

    try:
        connection_string = config.to_connection_string()
        db_conn = DatabaseConnection(connection_string)
        await db_conn.execute_query("SELECT 1 as test")
        await db_conn.close()
        return True
    except Exception as e:
        console.print(f"[bold error]Connection failed:[/bold error] {e}", style="error")
        return False


def save_database(
    config_manager: DatabaseConfigManager, config: DatabaseConfig, password: str | None
) -> None:
    """Save database configuration.

    Args:
        config_manager: DatabaseConfigManager instance
        config: DatabaseConfig to save
        password: Password to store in keyring (if provided)
    """
    config_manager.add_database(config, password if password else None)
