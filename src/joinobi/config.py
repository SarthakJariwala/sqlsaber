"""Configuration for JoinObi SQL Agent."""

import os
from typing import Optional


def get_model_name() -> str:
    """Get the model name from environment or use default."""
    return os.getenv("JOINOBI_MODEL", "anthropic:claude-sonnet-4-0")


def get_api_key() -> Optional[str]:
    """Get API key for the model provider."""
    model = get_model_name()

    if model.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    else:
        # For other providers, check generic key
        return os.getenv("AI_API_KEY")


def validate_config():
    """Validate that necessary configuration is present."""
    api_key = get_api_key()
    if not api_key:
        model = get_model_name()
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
