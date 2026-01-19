"""Authentication CLI commands."""

import os

import cyclopts
import keyring
import keyring.errors
import questionary

from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console

console = create_console()
config_manager = AuthConfigManager()
logger = get_logger(__name__)

auth_app = cyclopts.App(
    name="auth",
    help="Manage authentication configuration",
)


@auth_app.command
def setup():
    """Configure authentication for SQLsaber (API keys)."""
    import asyncio

    from sqlsaber.application.auth_setup import setup_auth
    from sqlsaber.application.prompts import AsyncPrompter

    console.print("\n[bold]SQLsaber Authentication Setup[/bold]\n")

    async def run_setup():
        prompter = AsyncPrompter()
        api_key_manager = APIKeyManager()
        return await setup_auth(
            prompter=prompter,
            auth_manager=config_manager,
            api_key_manager=api_key_manager,
            default_provider="anthropic",
        )

    logger.info("auth.setup.start")
    success, provider = asyncio.run(run_setup())
    logger.info("auth.setup.complete", success=bool(success), provider=str(provider))

    if not success:
        console.print("\n[warning]No authentication configured.[/warning]")

    console.print(
        "\nYou can change this anytime by running [info]saber auth setup[/info] again."
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

    console.print("[success]✓ API Key authentication configured[/success]\n")

    api_key_manager = APIKeyManager()
    for provider in providers.all_keys():
        env_var = api_key_manager.get_env_var_name(provider)
        service = api_key_manager._get_service_name(provider)
        from_env = bool(os.getenv(env_var))
        from_keyring = bool(keyring.get_password(service, provider))
        if from_env:
            console.print(f"> {provider}: configured via {env_var}")
        elif from_keyring:
            console.print(f"> {provider}: [success]configured[/success]")
        else:
            console.print(f"> {provider}: [warning]not configured[/warning]")

    logger.info("auth.status.complete", method=str(auth_method))


@auth_app.command
def reset():
    """Reset stored API key credentials for a selected provider."""
    console.print("\n[bold]SQLsaber Authentication Reset[/bold]\n")

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

    api_key_present = bool(keyring.get_password(service, provider))

    if not api_key_present:
        console.print(
            f"[warning]No stored credentials found for {provider}. Nothing to reset.[/warning]"
        )
        logger.info("auth.reset.nothing_to_reset", provider=provider)
        return

    confirmed = questionary.confirm(
        f"Remove the stored {provider.title()} API key from your keyring?",
        default=False,
    ).ask()

    if not confirmed:
        console.print("Reset cancelled.")
        logger.info("auth.reset.cancelled_confirm", provider=provider)
        return

    try:
        keyring.delete_password(service, provider)
        console.print(f"Removed {provider} API key from keyring", style="green")
        logger.info("auth.reset.api_key_removed", provider=provider)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as e:
        console.print(f"Warning: Could not remove API key: {e}", style="warning")
        logger.warning(
            "auth.reset.api_key_remove_failed", provider=provider, error=str(e)
        )

    console.print("\n[success]✓ Reset complete.[/success]")
    logger.info("auth.reset.complete", provider=provider)
    console.print(
        "Environment variables are not modified by this command.", style="dim"
    )


def create_auth_app() -> cyclopts.App:
    """Return the authentication management CLI app."""
    return auth_app
