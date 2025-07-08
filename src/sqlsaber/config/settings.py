"""Configuration management for SQLSaber SQL Agent."""

import json
import os
import platform
import stat
from pathlib import Path
from typing import Any

import platformdirs

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
            return {"model": self.DEFAULT_MODEL}

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                # Ensure we have a model set
                if "model" not in config:
                    config["model"] = self.DEFAULT_MODEL
                return config
        except (json.JSONDecodeError, IOError):
            return {"model": self.DEFAULT_MODEL}

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


class Config:
    """Configuration class for SQLSaber."""

    def __init__(self):
        self.model_config_manager = ModelConfigManager()
        self.model_name = self.model_config_manager.get_model()
        self.api_key_manager = APIKeyManager()
        self.auth_config_manager = AuthConfigManager()
        self.oauth_flow = AnthropicOAuthFlow()

        # Get authentication credentials based on configured method
        self.auth_method = self.auth_config_manager.get_auth_method()
        self.api_key = None
        self.oauth_token = None

        if self.auth_method == AuthMethod.CLAUDE_PRO:
            # Try to get OAuth token and refresh if needed
            try:
                token = self.oauth_flow.refresh_token_if_needed()
                if token:
                    self.oauth_token = token.access_token
            except Exception:
                # OAuth token unavailable, will need to re-authenticate
                pass
        else:
            # Use API key authentication (default or explicitly configured)
            self.api_key = self._get_api_key()

    def _get_api_key(self) -> str | None:
        """Get API key for the model provider using cascading logic."""
        model = self.model_name
        if model.startswith("anthropic:"):
            return self.api_key_manager.get_api_key("anthropic")

    def set_model(self, model: str) -> None:
        """Set the model and update configuration."""
        self.model_config_manager.set_model(model)
        self.model_name = model

    def validate(self):
        """Validate that necessary configuration is present."""
        # 1. Claude-Pro flow → require OAuth token only
        if self.auth_method == AuthMethod.CLAUDE_PRO:
            if not self.oauth_token:
                raise ValueError(
                    "OAuth token not available. Run 'saber auth setup' to authenticate with Claude Pro."
                )
            return  # OAuth path satisfied – nothing more to check

        # 2. Default / API-key flow → require API key
        if not self.api_key:
            raise ValueError("Anthropic API key not found.")
