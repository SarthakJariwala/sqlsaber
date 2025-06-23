"""Configuration management for SQLSaber SQL Agent."""

import json
import os
import platform
import stat
from pathlib import Path
from typing import Any, Dict, Optional

import platformdirs

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
            else:
                if is_directory:
                    os.chmod(path, stat.S_IRWXU)  # 0o700
                else:
                    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except (OSError, PermissionError):
            pass

    def _load_config(self) -> Dict[str, Any]:
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

    def _save_config(self, config: Dict[str, Any]) -> None:
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
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> Optional[str]:
        """Get API key for the model provider using cascading logic."""
        model = self.model_name

        if model.startswith("openai:"):
            return self.api_key_manager.get_api_key("openai")
        elif model.startswith("anthropic:"):
            return self.api_key_manager.get_api_key("anthropic")
        else:
            # For other providers, use generic key
            return self.api_key_manager.get_api_key("generic")

    def set_model(self, model: str) -> None:
        """Set the model and update configuration."""
        self.model_config_manager.set_model(model)
        self.model_name = model
        # Update API key for new model
        self.api_key = self._get_api_key()

    def validate(self):
        """Validate that necessary configuration is present."""
        if not self.api_key:
            model = self.model_name
            provider = "generic"
            if model.startswith("openai:"):
                provider = "OpenAI"
            elif model.startswith("anthropic:"):
                provider = "Anthropic"

            raise ValueError(f"{provider} API key not found.")
