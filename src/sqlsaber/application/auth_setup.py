"""Shared auth setup logic for onboarding and CLI."""

import asyncio

from questionary import Choice

from sqlsaber.application.prompts import Prompter
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow
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
    # Check if auth is already configured
    if auth_manager.has_auth_configured():
        console.print("[success]✓ Authentication already configured![/success]")
        return True, None

    # Select provider
    provider = await select_provider(prompter, default=default_provider)

    if provider is None:
        return False, None

    # For Anthropic, offer OAuth or API key
    if provider == "anthropic" and allow_oauth:
        method_choice = await prompter.select(
            "Authentication method:",
            choices=[
                Choice("API Key", value=AuthMethod.API_KEY),
                Choice("Claude Pro/Max (OAuth)", value=AuthMethod.CLAUDE_PRO),
            ],
        )

        if method_choice is None:
            return False, None

        if method_choice == AuthMethod.CLAUDE_PRO:
            console.print()
            oauth_success = await configure_oauth_anthropic(
                auth_manager, run_in_thread=run_oauth_in_thread
            )
            if oauth_success:
                console.print(
                    "[green]✓ Anthropic OAuth configured successfully![/green]"
                )
                return True, provider
            else:
                console.print("[error]✗ Anthropic OAuth setup failed.[/error]")
                return False, None

    # API key flow
    env_var = api_key_manager.get_env_var_name(provider)

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
