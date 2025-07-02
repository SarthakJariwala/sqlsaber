"""API Key management for SQLSaber."""

import getpass
import os

import keyring
from rich.console import Console

console = Console()


class APIKeyManager:
    """Manages API keys with cascading retrieval: env var -> keyring -> prompt."""

    def __init__(self):
        self.service_prefix = "sqlsaber"

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for the specified provider using cascading logic."""
        env_var_name = self._get_env_var_name(provider)
        service_name = self._get_service_name(provider)

        # 1. Check environment variable first
        api_key = os.getenv(env_var_name)
        if api_key:
            console.print(f"Using {env_var_name} from environment", style="dim")
            return api_key

        # 2. Check keyring storage
        try:
            api_key = keyring.get_password(service_name, provider)
            if api_key:
                console.print(
                    f"Using stored {provider} API key from keyring", style="dim"
                )
                return api_key
        except Exception as e:
            # Keyring access failed, continue to prompt
            console.print(f"Keyring access failed: {e}", style="dim yellow")

        # 3. Prompt user for API key
        return self._prompt_and_store_key(provider, env_var_name, service_name)

    def _get_env_var_name(self, provider: str) -> str:
        """Get the expected environment variable name for a provider."""
        if provider == "openai":
            return "OPENAI_API_KEY"
        elif provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        else:
            return "AI_API_KEY"

    def _get_service_name(self, provider: str) -> str:
        """Get the keyring service name for a provider."""
        return f"{self.service_prefix}-{provider}-api-key"

    def _prompt_and_store_key(
        self, provider: str, env_var_name: str, service_name: str
    ) -> str | None:
        """Prompt user for API key and store it in keyring."""
        try:
            console.print(
                f"\n{provider.title()} API key not found in environment or keyring."
            )
            console.print("You can either:")
            console.print(f"  1. Set the {env_var_name} environment variable")
            console.print(
                "  2. Enter it now to securely store using your operating system's credentials store"
            )

            api_key = getpass.getpass(
                f"\nEnter your {provider.title()} API key (or press Enter to skip): "
            )

            if not api_key.strip():
                console.print(
                    "No API key provided. Some functionality may not work.",
                    style="yellow",
                )
                return None

            # Store in keyring for future use
            try:
                keyring.set_password(service_name, provider, api_key.strip())
                console.print("API key stored securely for future use", style="green")
            except Exception as e:
                console.print(
                    f"Warning: Could not store API key in keyring: {e}", style="yellow"
                )
                console.print(
                    "You may need to enter it again next time", style="yellow"
                )

            return api_key.strip()

        except KeyboardInterrupt:
            console.print("\nOperation cancelled", style="yellow")
            return None
        except Exception as e:
            console.print(f"Error prompting for API key: {e}", style="red")
            return None
