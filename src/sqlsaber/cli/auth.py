"""Authentication CLI commands."""

import os

import cyclopts
import keyring
import keyring.errors
import questionary

from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_tokens import OAuthTokenManager
from sqlsaber.theme.manager import create_console
from sqlsaber.config.logging import get_logger

# Global instances for CLI commands
console = create_console()
config_manager = AuthConfigManager()
logger = get_logger(__name__)

# Create the authentication management CLI app
auth_app = cyclopts.App(
    name="auth",
    help="Manage authentication configuration",
)


@auth_app.command
def setup():
    """Configure authentication for SQLsaber (API keys and Anthropic OAuth)."""
    import asyncio

    from sqlsaber.application.auth_setup import setup_auth
    from sqlsaber.application.prompts import AsyncPrompter

    console.print("\n[bold]SQLsaber Authentication Setup[/bold]\n")

    async def run_setup():
        prompter = AsyncPrompter()
        api_key_manager = APIKeyManager()
        success, provider = await setup_auth(
            prompter=prompter,
            auth_manager=config_manager,
            api_key_manager=api_key_manager,
            allow_oauth=True,
            default_provider="anthropic",
            run_oauth_in_thread=True,  # questionary's sync prompts need their own event loop
        )
        return success, provider

    logger.info("auth.setup.start")
    success, provider = asyncio.run(run_setup())
    logger.info("auth.setup.complete", success=bool(success), provider=str(provider))

    if not success:
        console.print("\n[warning]No authentication configured.[/warning]")

    console.print(
        "\nYou can change this anytime by running [cyan]saber auth setup[/cyan] again."
    )


@auth_app.command
def status():
    """Show current authentication configuration and provider key status."""
    logger.info("auth.status.start")
    auth_method = config_manager.get_auth_method()

    console.print("\n[bold blue]Authentication Status[/bold blue]")

    if auth_method is None:
        console.print("[warning]No authentication method configured[/warning]")
        console.print(
            "Run [primary]saber auth setup[/primary] to configure authentication."
        )
        logger.info("auth.status.none_configured")
        return

    # Show configured method summary
    if auth_method == AuthMethod.CLAUDE_PRO:
        console.print("[green]✓ Anthropic Claude Pro/Max (OAuth) configured[/green]\n")
    else:
        console.print("[green]✓ API Key authentication configured[/green]\n")

    # Show per-provider status without prompting
    api_key_manager = APIKeyManager()
    for provider in providers.all_keys():
        if provider == "anthropic":
            # Include OAuth status
            if OAuthTokenManager().has_oauth_token("anthropic"):
                console.print("> anthropic (oauth): [green]configured[/green]")
        env_var = api_key_manager.get_env_var_name(provider)
        service = api_key_manager._get_service_name(provider)
        from_env = bool(os.getenv(env_var))
        from_keyring = bool(keyring.get_password(service, provider))
        if from_env:
            console.print(f"> {provider}: configured via {env_var}")
        elif from_keyring:
            console.print(f"> {provider}: [green]configured[/green]")
        else:
            console.print(f"> {provider}: [warning]not configured[/warning]")
    logger.info("auth.status.complete", method=str(auth_method))


@auth_app.command
def reset():
    """Reset credentials for a selected provider (API key and/or OAuth)."""
    console.print("\n[bold]SQLsaber Authentication Reset[/bold]\n")

    # Choose provider to reset (mirrors setup)
    provider = questionary.select(
        "Select provider to reset:",
        choices=providers.all_keys(),
    ).ask()

    if provider is None:
        console.print("[warning]Reset cancelled.[/warning]")
        logger.info("auth.reset.cancelled_no_provider")
        return

    api_key_manager = APIKeyManager()
    service = api_key_manager._get_service_name(provider)

    # Determine what exists in keyring
    api_key_present = bool(keyring.get_password(service, provider))
    oauth_present = (
        OAuthTokenManager().has_oauth_token("anthropic")
        if provider == "anthropic"
        else False
    )

    if not api_key_present and not oauth_present:
        console.print(
            f"[warning]No stored credentials found for {provider}. Nothing to reset.[/warning]"
        )
        logger.info("auth.reset.nothing_to_reset", provider=provider)
        return

    # Build confirmation message
    to_remove: list[str] = []
    if oauth_present:
        to_remove.append("Anthropic OAuth token")
    if api_key_present:
        to_remove.append(f"{provider.title()} API key")

    summary = ", ".join(to_remove)
    confirmed = questionary.confirm(
        f"Remove the following for {provider}: {summary}?",
        default=False,
    ).ask()

    if not confirmed:
        console.print("Reset cancelled.")
        logger.info("auth.reset.cancelled_confirm", provider=provider)
        return

    # Perform deletions
    if oauth_present:
        OAuthTokenManager().remove_oauth_token("anthropic")
    if api_key_present:
        try:
            keyring.delete_password(service, provider)
            console.print(f"Removed {provider} API key from keyring", style="green")
            logger.info("auth.reset.api_key_removed", provider=provider)
        except keyring.errors.PasswordDeleteError:
            # Already absent; treat as success
            pass
        except Exception as e:
            console.print(f"Warning: Could not remove API key: {e}", style="warning")
            logger.warning(
                "auth.reset.api_key_remove_failed", provider=provider, error=str(e)
            )

    # Optionally clear global auth method if removing Anthropic OAuth configuration
    if provider == "anthropic" and oauth_present:
        current_method = config_manager.get_auth_method()
        if current_method == AuthMethod.CLAUDE_PRO:
            also_clear = questionary.confirm(
                "Anthropic OAuth was removed. Also unset the global auth method?",
                default=False,
            ).ask()
            if also_clear:
                config = config_manager._load_config()
                config["auth_method"] = None
                config_manager._save_config(config)
                console.print("Global auth method unset.", style="green")
                logger.info("auth.reset.global_method_unset")

    console.print("\n[success]✓ Reset complete.[/success]")
    logger.info("auth.reset.complete", provider=provider)
    console.print(
        "Environment variables are not modified by this command.", style="dim"
    )


def create_auth_app() -> cyclopts.App:
    """Return the authentication management CLI app."""
    return auth_app
