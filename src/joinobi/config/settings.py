"""Configuration management for JoinObi SQL Agent."""

import os
from typing import Optional

from .api_keys import APIKeyManager


class Config:
    """Configuration class for JoinObi."""

    def __init__(self):
        self.model_name = self._get_model_name()
        self.api_key_manager = APIKeyManager()
        self.api_key = self._get_api_key()
        self.database_url = self._get_database_url()

    def _get_model_name(self) -> str:
        """Get the model name from environment or use default."""
        return os.getenv("JOINOBI_MODEL", "anthropic:claude-sonnet-4-0")

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

    def _get_database_url(self) -> Optional[str]:
        """Get database URL from environment."""
        return os.getenv("DATABASE_URL")

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
