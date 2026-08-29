"""Shared auth setup logic for onboarding and CLI."""

import os

from sqlsaber.cli.prompts import Prompter
from sqlsaber.cli.output import err, out
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.render import blocks as b


async def select_provider(prompter: Prompter, default: str = "anthropic") -> str | None:
    """Interactive provider selection.

    Args:
        prompter: Prompter instance for interaction
        default: Default provider to select

    Returns:
        Selected provider name or None if cancelled
    """
    provider = await prompter.select(
        "Select AI provider:", choices=providers.all_keys(), default=default
    )
    return provider


async def configure_api_key(
    provider: str, api_key_manager: APIKeyManager, auth_manager: AuthConfigManager
) -> bool:
    """Configure API key for a provider.

    Args:
        provider: Provider name
        api_key_manager: APIKeyManager instance
        auth_manager: AuthConfigManager instance

    Returns:
        True if API key configured successfully, False otherwise
    """
    api_key = api_key_manager.get_api_key(provider)

    if api_key:
        auth_manager.set_auth_method(AuthMethod.API_KEY)
        return True

    return False


async def setup_auth(
    prompter: Prompter,
    auth_manager: AuthConfigManager,
    api_key_manager: APIKeyManager,
    default_provider: str = "anthropic",
) -> tuple[bool, str | None]:
    """Interactive authentication setup.

    Args:
        prompter: Prompter instance for interaction
        auth_manager: AuthConfigManager instance
        api_key_manager: APIKeyManager instance
        default_provider: Default provider to select

    Returns:
        Tuple of (success: bool, provider: str | None)
    """
    provider = await select_provider(prompter, default=default_provider)

    if provider is None:
        return False, None

    env_var = api_key_manager.get_env_var_name(provider)
    api_key_in_env = bool(os.getenv(env_var))
    api_key_in_keyring = api_key_manager.has_stored_api_key(provider)

    if api_key_in_env or api_key_in_keyring:
        parts: list[str] = []
        if api_key_in_keyring:
            parts.append("stored API key")
        if api_key_in_env:
            parts.append(f"{env_var} environment variable")
        summary = ", ".join(parts)
        out(b.md(f"Existing authentication found for {provider}: {summary}"))

    if api_key_in_keyring:
        reset_api_key = await prompter.confirm(
            f"{provider.title()} API key is stored in your keyring. Reset before continuing?",
            default=False,
        )
        if not reset_api_key:
            out(b.warn("No changes made to stored API key credentials."))
            return True, None
        if not api_key_manager.delete_api_key(provider):
            err(b.error("Failed to remove existing API key credentials."))
            return False, None
        out(b.md(f"{provider.title()} API key removed from keyring."))
        api_key_in_keyring = False

    if api_key_in_env:
        out(
            b.md(
                f"{env_var} is set in your environment. Update it there if you need a new value."
            )
        )

    out(
        b.md(
            f"To use {provider.title()}, you need an API key.\n"
            f"You can set the {env_var} environment variable,\n"
            "or enter it now to store securely in your OS keychain."
        )
    )

    api_key_configured = await configure_api_key(
        provider, api_key_manager, auth_manager
    )

    if api_key_configured:
        out(b.success(f"{provider.title()} API key configured successfully!"))
        return True, provider

    out(b.warn("No API key provided."))
    return False, None
