"""Shared auth setup logic for onboarding and CLI."""

import asyncio
import os

from questionary import Choice

from sqlsaber.application.prompts import Prompter
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow
from sqlsaber.config.oauth_tokens import OAuthTokenManager
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


async def configure_oauth_anthropic(
    auth_manager: AuthConfigManager, run_in_thread: bool = False
) -> bool:
    """Configure Anthropic OAuth.

    Args:
        auth_manager: AuthConfigManager instance
        run_in_thread: Whether to run OAuth flow in a separate thread (for onboarding)

    Returns:
        True if OAuth configured successfully, False otherwise
    """
    flow = AnthropicOAuthFlow()

    if run_in_thread:
        # Run in thread to avoid event loop conflicts (onboarding)
        oauth_success = await asyncio.to_thread(flow.authenticate)
    else:
        # Run directly (CLI)
        oauth_success = flow.authenticate()

    if oauth_success:
        auth_manager.set_auth_method(AuthMethod.CLAUDE_PRO)
        return True

    return False


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
    allow_oauth: bool = True,
    default_provider: str = "anthropic",
    run_oauth_in_thread: bool = False,
) -> tuple[bool, str | None]:
    """Interactive authentication setup.

    Args:
        prompter: Prompter instance for interaction
        auth_manager: AuthConfigManager instance
        api_key_manager: APIKeyManager instance
        allow_oauth: Whether to offer OAuth option for Anthropic
        default_provider: Default provider to select
        run_oauth_in_thread: Whether to run OAuth in thread (for onboarding)

    Returns:
        Tuple of (success: bool, provider: str | None)
    """
    oauth_manager = OAuthTokenManager()

    provider = await select_provider(prompter, default=default_provider)

    if provider is None:
        return False, None

    env_var = api_key_manager.get_env_var_name(provider)
    api_key_in_env = bool(os.getenv(env_var))
    api_key_in_keyring = api_key_manager.has_stored_api_key(provider)
    has_oauth = (
        oauth_manager.has_oauth_token("anthropic")
        if provider == "anthropic" and allow_oauth
        else False
    )

    if api_key_in_env or api_key_in_keyring or has_oauth:
        parts: list[str] = []
        if api_key_in_keyring:
            parts.append("stored API key")
        if api_key_in_env:
            parts.append(f"{env_var} environment variable")
        if has_oauth:
            parts.append("OAuth token")
        summary = ", ".join(parts)
        console.print(
            f"[info]Existing authentication found for {provider}: {summary}[/info]"
        )

    # For Anthropic, offer OAuth or API key
    if provider == "anthropic" and allow_oauth:
        api_key_label = "API Key"
        if api_key_in_keyring or api_key_in_env:
            api_key_label += " [configured]"
        oauth_label = "Claude Pro/Max (OAuth)"
        if has_oauth:
            oauth_label += " [configured]"

        method_choice = await prompter.select(
            "Authentication method:",
            choices=[
                Choice(api_key_label, value=AuthMethod.API_KEY),
                Choice(oauth_label, value=AuthMethod.CLAUDE_PRO),
            ],
        )

        if method_choice is None:
            return False, None

        if method_choice == AuthMethod.CLAUDE_PRO:
            if has_oauth:
                reset = await prompter.confirm(
                    "Anthropic OAuth is already configured. Reset before continuing?",
                    default=False,
                )
                if not reset:
                    console.print(
                        "[warning]No changes made to Anthropic OAuth credentials.[/warning]"
                    )
                    return True, None

                removal_success = oauth_manager.remove_oauth_token("anthropic")
                if not removal_success:
                    console.print(
                        "[error]Failed to remove existing Anthropic OAuth credentials.[/error]"
                    )
                    return False, None

                current_method = auth_manager.get_auth_method()
                if current_method == AuthMethod.CLAUDE_PRO:
                    auth_manager.clear_auth_method()

            console.print()
            oauth_success = await configure_oauth_anthropic(
                auth_manager, run_in_thread=run_oauth_in_thread
            )
            if oauth_success:
                console.print(
                    "[green]✓ Anthropic OAuth configured successfully![/green]"
                )
                return True, provider

            console.print("[error]✗ Anthropic OAuth setup failed.[/error]")
            return False, None

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

    # Configure API key
    api_key_configured = await configure_api_key(
        provider, api_key_manager, auth_manager
    )

    if api_key_configured:
        console.print(
            f"[green]✓ {provider.title()} API key configured successfully![/green]"
        )
        return True, provider
    else:
        console.print("[warning]No API key provided.[/warning]")
        return False, None
