"""Configuration management for SQLSaber SQL Agent."""

import json
import os
import platform
import stat
from enum import Enum
from pathlib import Path
from typing import Any

import platformdirs

from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager


class ThinkingLevel(str, Enum):
    """Thinking levels that map to provider-specific configurations."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

    @classmethod
    def from_string(cls, value: str) -> "ThinkingLevel":
        """Convert a string to a ThinkingLevel, defaulting to MEDIUM if invalid or 'off'."""
        normalized = value.lower()
        if normalized == "off":
            return cls.MEDIUM
        try:
            return cls(normalized)
        except ValueError:
            return cls.MEDIUM


class ModelConfigManager:
    """Manages model configuration persistence.

    Supports both v1 and v2 config formats with automatic migration.

    v1 format:
        {"model": "...", "thinking_enabled": bool}

    v2 format:
        {"version": 2, "model": "...", "thinking": {"enabled": bool, "level": "medium"}}
    """

    DEFAULT_MODEL = "anthropic:claude-opus-4-5"
    DEFAULT_THINKING_LEVEL = ThinkingLevel.MEDIUM
    CONFIG_VERSION = 2

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

    def _migrate_v1_to_v2(self, config: dict[str, Any]) -> dict[str, Any]:
        """Migrate v1 config format to v2.

        v1: {"model": "...", "thinking_enabled": bool}
        v2: {"version": 2, "model": "...", "thinking": {"enabled": bool, "level": "medium"}}
        """
        thinking_enabled = config.get("thinking_enabled", False)
        return {
            "version": self.CONFIG_VERSION,
            "model": config.get("model", self.DEFAULT_MODEL),
            "thinking": {
                "enabled": thinking_enabled,
                "level": self.DEFAULT_THINKING_LEVEL.value,
            },
        }

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file, migrating v1 to v2 if needed."""
        if not self.config_file.exists():
            return {
                "version": self.CONFIG_VERSION,
                "model": self.DEFAULT_MODEL,
                "thinking": {
                    "enabled": False,
                    "level": self.DEFAULT_THINKING_LEVEL.value,
                },
            }

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Check if this is v1 format (no version key or version < 2)
            if config.get("version", 1) < 2:
                config = self._migrate_v1_to_v2(config)
                # Save migrated config
                self._save_config(config)
                return config

            # Ensure all required fields exist in v2
            if "model" not in config:
                config["model"] = self.DEFAULT_MODEL
            if "thinking" not in config:
                config["thinking"] = {
                    "enabled": False,
                    "level": self.DEFAULT_THINKING_LEVEL.value,
                }
            else:
                if "enabled" not in config["thinking"]:
                    config["thinking"]["enabled"] = False
                if "level" not in config["thinking"]:
                    config["thinking"]["level"] = self.DEFAULT_THINKING_LEVEL.value
            thinking_level = config["thinking"].get(
                "level", self.DEFAULT_THINKING_LEVEL.value
            )
            if isinstance(thinking_level, str) and thinking_level.lower() == "off":
                config["thinking"]["enabled"] = False
                config["thinking"]["level"] = self.DEFAULT_THINKING_LEVEL.value
                try:
                    self._save_config(config)
                except OSError:
                    pass

            return config
        except (json.JSONDecodeError, IOError):
            return {
                "version": self.CONFIG_VERSION,
                "model": self.DEFAULT_MODEL,
                "thinking": {
                    "enabled": False,
                    "level": self.DEFAULT_THINKING_LEVEL.value,
                },
            }

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file."""
        # Ensure version is set
        config["version"] = self.CONFIG_VERSION
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
        return config.get("thinking", {}).get("enabled", False)

    def set_thinking_enabled(self, enabled: bool) -> None:
        """Set whether thinking is enabled."""
        config = self._load_config()
        if "thinking" not in config:
            config["thinking"] = {
                "enabled": enabled,
                "level": self.DEFAULT_THINKING_LEVEL.value,
            }
        else:
            config["thinking"]["enabled"] = enabled
        self._save_config(config)

    def get_thinking_level(self) -> ThinkingLevel:
        """Get the configured thinking level."""
        config = self._load_config()
        level_str = config.get("thinking", {}).get(
            "level", self.DEFAULT_THINKING_LEVEL.value
        )
        return ThinkingLevel.from_string(level_str)

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking level."""
        config = self._load_config()
        if "thinking" not in config:
            config["thinking"] = {"enabled": False, "level": level.value}
        else:
            config["thinking"]["level"] = level.value
        self._save_config(config)

    def set_thinking(self, enabled: bool, level: ThinkingLevel) -> None:
        """Set both thinking enabled state and level."""
        config = self._load_config()
        config["thinking"] = {"enabled": enabled, "level": level.value}
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

    @property
    def thinking_level(self) -> ThinkingLevel:
        """Get the configured thinking level."""
        return self._manager.get_thinking_level()

    @thinking_level.setter
    def thinking_level(self, value: ThinkingLevel) -> None:
        """Set the thinking level."""
        self._manager.set_thinking_level(value)

    def set_thinking(self, enabled: bool, level: ThinkingLevel) -> None:
        """Set both thinking enabled state and level atomically."""
        self._manager.set_thinking(enabled, level)


class AuthConfig:
    """Configuration specific to authentication."""

    def __init__(self):
        self._api_key_manager = APIKeyManager()

    def get_api_key(self, model_name: str) -> str | None:
        """Get API key for the model provider using cascading logic."""
        model = model_name or ""
        provider_key = providers.provider_from_model(model)
        if provider_key in set(providers.all_keys()):
            return self._api_key_manager.get_api_key(provider_key)
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
