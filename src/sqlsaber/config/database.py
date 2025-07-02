"""Database configuration management."""

import json
import os
import platform
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import keyring
import platformdirs


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    name: str
    type: str  # postgresql, mysql, sqlite, csv
    host: str | None
    port: int | None
    database: str
    username: str | None
    password: str | None = None
    ssl_mode: str | None = None
    ssl_ca: str | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None
    schema: str | None = None

    def to_connection_string(self) -> str:
        """Convert config to database connection string."""
        password = self.password or self._get_password_from_keyring()

        if self.type == "postgresql":
            if not all([self.host, self.port, self.username]):
                raise ValueError("Host, port, and username are required for PostgreSQL")

            # Build base connection string
            if password:
                encoded_password = quote_plus(password)
                base_url = f"postgresql://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            else:
                base_url = f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"

            # Add SSL parameters
            ssl_params = []
            if self.ssl_mode:
                ssl_params.append(f"sslmode={self.ssl_mode}")
            if self.ssl_ca:
                ssl_params.append(f"sslrootcert={quote_plus(self.ssl_ca)}")
            if self.ssl_cert:
                ssl_params.append(f"sslcert={quote_plus(self.ssl_cert)}")
            if self.ssl_key:
                ssl_params.append(f"sslkey={quote_plus(self.ssl_key)}")

            if ssl_params:
                return f"{base_url}?{'&'.join(ssl_params)}"
            return base_url

        elif self.type == "mysql":
            if not all([self.host, self.port, self.username]):
                raise ValueError("Host, port, and username are required for MySQL")

            # Build base connection string
            if password:
                encoded_password = quote_plus(password)
                base_url = f"mysql://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            else:
                base_url = (
                    f"mysql://{self.username}@{self.host}:{self.port}/{self.database}"
                )

            # Add SSL parameters
            ssl_params = []
            if self.ssl_mode:
                ssl_params.append(f"ssl_mode={self.ssl_mode}")
            if self.ssl_ca:
                ssl_params.append(f"ssl_ca={quote_plus(self.ssl_ca)}")
            if self.ssl_cert:
                ssl_params.append(f"ssl_cert={quote_plus(self.ssl_cert)}")
            if self.ssl_key:
                ssl_params.append(f"ssl_key={quote_plus(self.ssl_key)}")

            if ssl_params:
                return f"{base_url}?{'&'.join(ssl_params)}"
            return base_url

        elif self.type == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.type == "csv":
            # For CSV files, database field contains the file path
            base_url = f"csv:///{self.database}"

            # Add CSV-specific parameters if they exist in schema field
            if self.schema:
                # Schema field can contain CSV options in JSON format
                try:
                    csv_options = json.loads(self.schema)
                    params = []
                    if "delimiter" in csv_options:
                        params.append(f"delimiter={csv_options['delimiter']}")
                    if "encoding" in csv_options:
                        params.append(f"encoding={csv_options['encoding']}")
                    if "header" in csv_options:
                        params.append(f"header={str(csv_options['header']).lower()}")

                    if params:
                        return f"{base_url}?{'&'.join(params)}"
                except (json.JSONDecodeError, KeyError):
                    pass
            return base_url
        else:
            raise ValueError(f"Unsupported database type: {self.type}")

    def _get_password_from_keyring(self) -> str | None:
        """Get password from OS keyring."""
        try:
            return keyring.get_password("sqlsaber", f"{self.name}_{self.username}")
        except Exception:
            return None

    def store_password_in_keyring(self, password: str) -> None:
        """Store password in OS keyring."""
        keyring.set_password("sqlsaber", f"{self.name}_{self.username}", password)

    def delete_password_from_keyring(self) -> None:
        """Delete password from OS keyring."""
        try:
            keyring.delete_password("sqlsaber", f"{self.name}_{self.username}")
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "ssl_mode": self.ssl_mode,
            "ssl_ca": self.ssl_ca,
            "ssl_cert": self.ssl_cert,
            "ssl_key": self.ssl_key,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            host=data["host"],
            port=data["port"],
            database=data["database"],
            username=data["username"],
            ssl_mode=data.get("ssl_mode"),
            ssl_ca=data.get("ssl_ca"),
            ssl_cert=data.get("ssl_cert"),
            ssl_key=data.get("ssl_key"),
            schema=data.get("schema"),
        )


class DatabaseConfigManager:
    """Manages database configurations."""

    def __init__(self):
        self.config_dir = Path(platformdirs.user_config_dir("sqlsaber", "sqlsaber"))
        self.config_file = self.config_dir / "database_config.json"
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists with proper permissions."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._set_secure_permissions(self.config_dir, is_directory=True)

    def _set_secure_permissions(self, path: Path, is_directory: bool = False) -> None:
        """Set secure permissions cross-platform."""
        try:
            if platform.system() == "Windows":
                # On Windows, rely on NTFS permissions and avoid chmod
                # The default permissions are usually sufficient for user-only access
                return
            else:
                # Unix-like systems (Linux, macOS)
                if is_directory:
                    os.chmod(
                        path, stat.S_IRWXU
                    )  # 0o700 - owner read/write/execute only
                else:
                    os.chmod(
                        path, stat.S_IRUSR | stat.S_IWUSR
                    )  # 0o600 - owner read/write only
        except (OSError, PermissionError):
            # If we can't set permissions, continue anyway
            # The directory/file creation should still work
            pass

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {"default": None, "connections": {}}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"default": None, "connections": {}}

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Set secure permissions cross-platform
        self._set_secure_permissions(self.config_file, is_directory=False)

    def add_database(
        self, db_config: DatabaseConfig, password: str | None = None
    ) -> None:
        """Add a database configuration."""
        config = self._load_config()

        # Check if database with this name already exists
        if db_config.name in config["connections"]:
            raise ValueError(f"Database '{db_config.name}' already exists")

        # Store password in keyring if provided
        if password:
            db_config.store_password_in_keyring(password)

        # Add to config
        config["connections"][db_config.name] = db_config.to_dict()

        # Set as default if it's the first one
        if not config["default"]:
            config["default"] = db_config.name

        self._save_config(config)

    def get_database(self, name: str) -> DatabaseConfig | None:
        """Get a database configuration by name."""
        config = self._load_config()

        if name not in config["connections"]:
            return None

        return DatabaseConfig.from_dict(config["connections"][name])

    def get_default_database(self) -> DatabaseConfig | None:
        """Get the default database configuration."""
        config = self._load_config()

        default_name = config.get("default")
        if not default_name:
            return None

        return self.get_database(default_name)

    def list_databases(self) -> list[DatabaseConfig]:
        """List all database configurations."""
        config = self._load_config()

        databases = []
        for name, db_data in config["connections"].items():
            databases.append(DatabaseConfig.from_dict(db_data))

        return databases

    def remove_database(self, name: str) -> bool:
        """Remove a database configuration."""
        config = self._load_config()

        if name not in config["connections"]:
            return False

        # Remove password from keyring
        db_config = DatabaseConfig.from_dict(config["connections"][name])
        db_config.delete_password_from_keyring()

        # Remove from config
        del config["connections"][name]

        # Update default if this was the default
        if config["default"] == name:
            remaining_connections = list(config["connections"].keys())
            config["default"] = (
                remaining_connections[0] if remaining_connections else None
            )

        self._save_config(config)
        return True

    def set_default_database(self, name: str) -> bool:
        """Set the default database."""
        config = self._load_config()

        if name not in config["connections"]:
            return False

        config["default"] = name
        self._save_config(config)
        return True

    def has_databases(self) -> bool:
        """Check if any databases are configured."""
        config = self._load_config()
        return len(config["connections"]) > 0

    def get_default_name(self) -> str | None:
        """Get the name of the default database."""
        config = self._load_config()
        return config.get("default")
