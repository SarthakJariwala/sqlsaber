"""API Key management for SQLSaber."""

import getpass
import os

import keyring
import keyring.errors

from sqlsaber.config import providers
from sqlsaber.theme.manager import create_console

console = create_console()


class APIKeyManager:
    """Manages API keys with cascading retrieval: env var -> keyring -> prompt."""

    def __init__(self):
        self.service_prefix = "sqlsaber"

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for the specified provider using cascading logic."""
        env_var_name = self.get_env_var_name(provider)
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
                console.print(f"Using stored {provider} API key", style="dim")
                return api_key
        except Exception as e:
            # Keyring access failed, continue to prompt
            console.print(f"Keyring access failed: {e}", style="warning")

        # 3. Prompt user for API key
        return self._prompt_and_store_key(provider, env_var_name, service_name)

    def has_stored_api_key(self, provider: str) -> bool:
        """Check if an API key is stored for the provider."""
        service_name = self._get_service_name(provider)
        try:
            return keyring.get_password(service_name, provider) is not None
        except Exception:
            return False

    def delete_api_key(self, provider: str) -> bool:
        """Remove stored API key for the provider."""
        service_name = self._get_service_name(provider)
        try:
            keyring.delete_password(service_name, provider)
            return True
        except keyring.errors.PasswordDeleteError:
            return True
        except Exception as e:
            console.print(
                f"Warning: Could not remove API key: {e}",
                style="warning",
            )
            return False

    def get_env_var_name(self, provider: str) -> str:
        """Get the expected environment variable name for a provider."""
        # Normalize aliases to canonical provider keys
        key = providers.canonical(provider) or provider
        return providers.env_var_name(key)

    def _get_service_name(self, provider: str) -> str:
        """Get the keyring service name for a provider."""
        return f"{self.service_prefix}-{provider}-api-key"

    def _prompt_and_store_key(
        self, provider: str, env_var_name: str, service_name: str
    ) -> str | None:
        """Prompt user for API key and store it in keyring."""
        try:
            console.print(
                f"\n{provider.title()} API key not found in environment or your OS's credentials store."
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
                    style="warning",
                )
                return None

            # Store in keyring for future use
            try:
                keyring.set_password(service_name, provider, api_key.strip())
                console.print("API key stored securely for future use", style="green")
            except Exception as e:
                console.print(
                    f"Warning: Could not store API key in your operating system's credentials store: {e}",
                    style="warning",
                )
                console.print(
                    "You may need to enter it again next time", style="warning"
                )

            return api_key.strip()

        except KeyboardInterrupt:
            console.print("\nOperation cancelled", style="warning")
            return None
        except Exception as e:
            console.print(f"Error prompting for API key: {e}", style="red")
            return None
