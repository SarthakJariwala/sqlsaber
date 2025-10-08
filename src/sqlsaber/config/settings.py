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
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow


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
            return {
                "model": self.DEFAULT_MODEL,
                "thinking_enabled": False,
            }

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                # Ensure we have a model set
                if "model" not in config:
                    config["model"] = self.DEFAULT_MODEL
                # Set defaults for thinking if not present
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


class Config:
    """Configuration class for SQLSaber."""

    def __init__(self):
        self.model_config_manager = ModelConfigManager()
        self.model_name = self.model_config_manager.get_model()
        self.api_key = None
        self.api_key_manager = APIKeyManager()
        self.auth_config_manager = AuthConfigManager()

        # Thinking configuration
        self.thinking_enabled = self.model_config_manager.get_thinking_enabled()

        # Authentication method (API key or Anthropic OAuth)
        self.auth_method = self.auth_config_manager.get_auth_method()

        # Optional Anthropic OAuth access token (only relevant for provider=='anthropic')
        if self.auth_method == AuthMethod.CLAUDE_PRO and self.model_name.startswith(
            "anthropic"
        ):
            self.oauth_token = self.get_oauth_access_token()
        else:
            self.api_key = self._get_api_key()
            # self.oauth_token = None

    def _get_api_key(self) -> str | None:
        """Get API key for the model provider using cascading logic."""
        model = self.model_name or ""
        prov = providers.provider_from_model(model)
        if prov in set(providers.all_keys()):
            return self.api_key_manager.get_api_key(prov)  # type: ignore[arg-type]
        return None

    def set_model(self, model: str) -> None:
        """Set the model and update configuration."""
        self.model_config_manager.set_model(model)
        self.model_name = model

    def get_oauth_access_token(self) -> str | None:
        """Return a valid Anthropic OAuth access token if configured, else None.

        Uses the stored refresh token (if present) to refresh as needed.
        Only relevant when provider is 'anthropic'.
        """
        if not self.model_name.startswith("anthropic"):
            return None
        try:
            flow = AnthropicOAuthFlow()
            token = flow.refresh_token_if_needed()
            return token.access_token if token else None
        except Exception:
            return None

    def validate(self):
        """Validate that necessary configuration is present.

        Also ensure provider env var is set from keyring if needed for API-key flows.
        """
        model = self.model_name or ""
        provider_key = providers.provider_from_model(model)
        env_var = providers.env_var_name(provider_key or "") if provider_key else None
        if env_var:
            # Anthropic special-case: allow OAuth in lieu of API key only when explicitly configured
            if (
                provider_key == "anthropic"
                and self.auth_method == AuthMethod.CLAUDE_PRO
                and self.oauth_token
            ):
                return
            # If we don't have a key resolved from env/keyring, raise
            if not self.api_key:
                raise ValueError(f"{provider_key.capitalize()} API key not found.")
            # Hydrate env var for downstream SDKs if missing
            if not os.getenv(env_var):
                os.environ[env_var] = self.api_key
