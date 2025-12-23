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
        self._auth_manager = AuthConfigManager()
        self._api_key_manager = APIKeyManager()

    @property
    def method(self) -> AuthMethod | None:
        """Get the configured authentication method."""
        return self._auth_manager.get_auth_method()

    def get_api_key(self, model_name: str) -> str | None:
        """Get API key for the model provider using cascading logic."""
        model = model_name or ""
        prov = providers.provider_from_model(model)
        if prov in set(providers.all_keys()):
            return self._api_key_manager.get_api_key(prov)  # type: ignore[arg-type]
        return None

    def get_oauth_token(self, model_name: str) -> str | None:
        """Return a valid Anthropic OAuth access token if configured, else None."""
        if not model_name.startswith("anthropic"):
            return None

        # Only check/refresh token if the method is explicitly CLAUDE_PRO
        if self.method != AuthMethod.CLAUDE_PRO:
            return None

        try:
            flow = AnthropicOAuthFlow()
            token = flow.refresh_token_if_needed()
            return token.access_token if token else None
        except Exception:
            return None

    def validate(self, model_name: str) -> None:
        """Validate authentication for the given model."""
        model = model_name or ""
        provider_key = providers.provider_from_model(model)
        env_var = providers.env_var_name(provider_key or "") if provider_key else None

        if env_var:
            # Anthropic special-case: allow OAuth in lieu of API key
            if (
                provider_key == "anthropic"
                and self.method == AuthMethod.CLAUDE_PRO
                and self.get_oauth_token(model_name)
            ):
                return

            # If we don't have a key resolved from env/keyring, raise
            api_key = self.get_api_key(model_name)
            if not api_key:
                provider_name = (
                    provider_key.capitalize() if provider_key else "Provider"
                )
                raise ValueError(f"{provider_name} API key not found.")

            # Hydrate env var for downstream SDKs if missing
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

    @property
    def oauth_token(self) -> str | None:
        """Backwards compatibility wrapper for oauth_token."""
        return self.auth.get_oauth_token(self.model.name)

    def set_model(self, model: str) -> None:
        """Set the model and update configuration."""
        self.model.name = model

    def validate(self) -> None:
        """Validate that necessary configuration is present."""
        self.auth.validate(self.model.name)
