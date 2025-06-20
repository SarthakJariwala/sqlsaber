"""Configuration management for JoinObi SQL Agent."""

import os
from typing import Optional


class Config:
    """Configuration class for JoinObi."""

    def __init__(self):
        self.model_name = self._get_model_name()
        self.api_key = self._get_api_key()
        self.database_url = self._get_database_url()

    def _get_model_name(self) -> str:
        """Get the model name from environment or use default."""
        return os.getenv("JOINOBI_MODEL", "anthropic:claude-sonnet-4-0")

    def _get_api_key(self) -> Optional[str]:
        """Get API key for the model provider."""
        model = self.model_name

        if model.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            # For other providers, check generic key
            return os.getenv("AI_API_KEY")

    def _get_database_url(self) -> Optional[str]:
        """Get database URL from environment."""
        return os.getenv("DATABASE_URL")

    def validate(self):
        """Validate that necessary configuration is present."""
        if not self.api_key:
            model = self.model_name
            if model.startswith("openai:"):
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
                )
            elif model.startswith("anthropic:"):
                raise ValueError(
                    "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable."
                )
            else:
                raise ValueError(
                    f"API key not found for model {model}. Please set appropriate API key."
                )


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Convenience function for backward compatibility
def get_api_key() -> Optional[str]:
    """Get API key for the model provider."""
    return get_config().api_key
