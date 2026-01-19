"""Shared auth setup logic for onboarding and CLI."""

import os

from sqlsaber.application.prompts import Prompter
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.theme.manager import create_console

console = create_console()


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
    # Get API key (cascades env -> keyring -> prompt)
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
        console.print(
            f"[info]Existing authentication found for {provider}: {summary}[/info]"
        )

    # API key flow
    if api_key_in_keyring:
        reset_api_key = await prompter.confirm(
            f"{provider.title()} API key is stored in your keyring. Reset before continuing?",
            default=False,
        )
        if not reset_api_key:
            console.print(
                "[warning]No changes made to stored API key credentials.[/warning]"
            )
            return True, None
        if not api_key_manager.delete_api_key(provider):
            console.print(
                "[error]Failed to remove existing API key credentials.[/error]"
            )
            return False, None
        console.print(
            f"[muted]{provider.title()} API key removed from keyring.[/muted]"
        )
        api_key_in_keyring = False

    if api_key_in_env:
        console.print(
            f"[muted]{env_var} is set in your environment. Update it there if you need a new value.[/muted]"
        )

    console.print()
    console.print(f"[dim]To use {provider.title()}, you need an API key.[/dim]")
    console.print(f"[dim]You can set the {env_var} environment variable,[/dim]")
    console.print("[dim]or enter it now to store securely in your OS keychain.[/dim]")
    console.print()

    api_key_configured = await configure_api_key(
        provider, api_key_manager, auth_manager
    )

    if api_key_configured:
        console.print(
            f"[success]✓ {provider.title()} API key configured successfully![/success]"
        )
        return True, provider

    console.print("[warning]No API key provided.[/warning]")
    return False, None
