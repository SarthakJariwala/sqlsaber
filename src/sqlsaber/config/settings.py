"""Configuration management for SQLSaber SQL Agent."""

import json
import os
import platform
import stat
from pathlib import Path
from typing import Any

import platformdirs

from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager


class ModelConfigManager:
    """Manages model configuration persistence."""

    DEFAULT_MODEL = "anthropic:claude-sonnet-4-20250514"

    def __init__(self):
        self.config_dir = Path(platformdirs.user_config_dir("sqlsaber", "sqlsaber"))
        self.config_file = self.config_dir / "model_config.json"
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
            if is_directory:
                os.chmod(path, stat.S_IRWXU)  # 0o700
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except (OSError, PermissionError):
            pass

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {
                "model": self.DEFAULT_MODEL,
                "thinking_enabled": False,
            }

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            if "model" not in config:
                config["model"] = self.DEFAULT_MODEL
            if "thinking_enabled" not in config:
                config["thinking_enabled"] = False
            return config
        except (json.JSONDecodeError, IOError):
            return {
                "model": self.DEFAULT_MODEL,
                "thinking_enabled": False,
            }

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        self._set_secure_permissions(self.config_file, is_directory=False)

    def get_model(self) -> str:
        """Get the configured model."""
        config = self._load_config()
        return config.get("model", self.DEFAULT_MODEL)

    def set_model(self, model: str) -> None:
        """Set the model configuration."""
        config = self._load_config()
        config["model"] = model
        self._save_config(config)

    def get_thinking_enabled(self) -> bool:
        """Get whether thinking is enabled."""
        config = self._load_config()
        return config.get("thinking_enabled", False)

    def set_thinking_enabled(self, enabled: bool) -> None:
        """Set whether thinking is enabled."""
        config = self._load_config()
        config["thinking_enabled"] = enabled
        self._save_config(config)


class ModelConfig:
    """Configuration specific to the model."""

    def __init__(self):
        self._manager = ModelConfigManager()

    @property
    def name(self) -> str:
        """Get the configured model name."""
        return self._manager.get_model()

    @name.setter
    def name(self, value: str) -> None:
        """Set the model name."""
        self._manager.set_model(value)

    @property
    def thinking_enabled(self) -> bool:
        """Get whether thinking is enabled."""
        return self._manager.get_thinking_enabled()

    @thinking_enabled.setter
    def thinking_enabled(self, value: bool) -> None:
        """Set whether thinking is enabled."""
        self._manager.set_thinking_enabled(value)


class AuthConfig:
    """Configuration specific to authentication."""

    def __init__(self):
        self._api_key_manager = APIKeyManager()

    def get_api_key(self, model_name: str) -> str | None:
        """Get API key for the model provider using cascading logic."""
        model = model_name or ""
        provider_key = providers.provider_from_model(model)
        if provider_key in set(providers.all_keys()):
            return self._api_key_manager.get_api_key(provider_key)  # type: ignore[arg-type]
        return None

    def validate(self, model_name: str) -> None:
        """Validate authentication for the given model.

        On success, this hydrates the provider's expected environment variable (if
        missing) so downstream SDKs can pick it up.
        """
        model = model_name or ""
        provider_key = providers.provider_from_model(model)
        env_var = providers.env_var_name(provider_key or "") if provider_key else None

        if not env_var:
            return

        api_key = self.get_api_key(model_name)
        if not api_key:
            provider_name = provider_key.capitalize() if provider_key else "Provider"
            raise ValueError(f"{provider_name} API key not found.")

        if not os.getenv(env_var):
            os.environ[env_var] = api_key


class Config:
    """Configuration class for SQLSaber."""

    def __init__(self):
        self.model = ModelConfig()
        self.auth = AuthConfig()

    @property
    def model_name(self) -> str:
        """Backwards compatibility wrapper for model name."""
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Backwards compatibility wrapper for model name setter."""
        self.model.name = value

    @property
    def thinking_enabled(self) -> bool:
        """Backwards compatibility wrapper for thinking_enabled."""
        return self.model.thinking_enabled

    @property
    def api_key(self) -> str | None:
        """Backwards compatibility wrapper for api_key."""
        return self.auth.get_api_key(self.model.name)

    def set_model(self, model: str) -> None:
        """Set the model and update configuration."""
        self.model.name = model

    def validate(self) -> None:
        """Validate that necessary configuration is present."""
        self.auth.validate(self.model.name)
