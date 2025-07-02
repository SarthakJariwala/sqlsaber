"""Authentication configuration management for SQLSaber."""

import json
import os
import platform
import stat
from enum import Enum
from pathlib import Path
from typing import Any

import platformdirs


class AuthMethod(Enum):
    """Authentication methods available in SQLSaber."""

    API_KEY = "api_key"
    CLAUDE_PRO = "claude_pro"


class AuthConfigManager:
    """Manages authentication configuration persistence."""

    def __init__(self):
        self.config_dir = Path(platformdirs.user_config_dir("sqlsaber", "sqlsaber"))
        self.config_file = self.config_dir / "auth_config.json"
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists with proper permissions."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._set_secure_permissions(self.config_dir, is_directory=True)

    def _set_secure_permissions(self, path: Path, is_directory: bool = False) -> None:
        """Set secure permissions cross-platform."""
        try:
            if platform.system() == "Windows":
                return
            else:
                if is_directory:
                    os.chmod(path, stat.S_IRWXU)  # 0o700
                else:
                    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except (OSError, PermissionError):
            pass

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {"auth_method": None}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"auth_method": None}

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        self._set_secure_permissions(self.config_file, is_directory=False)

    def get_auth_method(self) -> AuthMethod | None:
        """Get the configured authentication method."""
        config = self._load_config()
        auth_method_str = config.get("auth_method")

        if auth_method_str is None:
            return None

        try:
            return AuthMethod(auth_method_str)
        except ValueError:
            return None

    def set_auth_method(self, auth_method: AuthMethod) -> None:
        """Set the authentication method."""
        config = self._load_config()
        config["auth_method"] = auth_method.value
        self._save_config(config)

    def has_auth_configured(self) -> bool:
        """Check if authentication method is configured."""
        return self.get_auth_method() is not None
