"""API Key management for SQLSaber."""

import getpass
import os

import keyring
import keyring.errors

from sqlsaber.config import providers
from sqlsaber.render import blocks as b
from sqlsaber.render import cli_err, cli_out


class APIKeyManager:
    """Manages API keys with cascading retrieval: env var -> keyring -> prompt."""

    def __init__(self):
        self.service_prefix = "sqlsaber"

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for the specified provider using cascading logic."""
        env_var_name = self.get_env_var_name(provider)
        service_name = self._get_service_name(provider)

        api_key = os.getenv(env_var_name)
        if api_key:
            return api_key

        try:
            api_key = keyring.get_password(service_name, provider)
            if api_key:
                return api_key
        except Exception as e:
            cli_err().emit(b.warn(f"Keyring access failed: {e}"))

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
            cli_err().emit(b.warn(f"Could not remove API key: {e}"))
            return False

    def get_env_var_name(self, provider: str) -> str:
        """Get the expected environment variable name for a provider."""
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
            cli_out().emit(
                b.md(
                    f"{provider.title()} API key not found in environment or your OS's credentials store.\n\n"
                    "You can either:\n"
                    f"  1. Set the `{env_var_name}` environment variable\n"
                    "  2. Enter it now to securely store using your operating system's credentials store"
                )
            )

            api_key = getpass.getpass(
                f"\nEnter your {provider.title()} API key (or press Enter to skip): "
            )

            if not api_key.strip():
                cli_out().emit(
                    b.warn("No API key provided. Some functionality may not work.")
                )
                return None

            try:
                keyring.set_password(service_name, provider, api_key.strip())
                cli_out().emit(b.success("API key stored securely for future use"))
            except Exception as e:
                cli_out().emit(
                    b.warn(
                        f"Could not store API key in your operating system's credentials store: {e}"
                    ),
                    b.warn("You may need to enter it again next time"),
                )

            return api_key.strip()

        except KeyboardInterrupt:
            cli_out().emit(b.warn("Operation cancelled"))
            return None
        except Exception as e:
            cli_err().emit(b.error(f"Error prompting for API key: {e}"))
            return None
