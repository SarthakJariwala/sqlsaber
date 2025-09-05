"""Authentication CLI commands."""

import os

import cyclopts
import keyring
import questionary
from rich.console import Console

from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow
from sqlsaber.config.oauth_tokens import OAuthTokenManager

# Global instances for CLI commands
console = Console()
config_manager = AuthConfigManager()

# Create the authentication management CLI app
auth_app = cyclopts.App(
    name="auth",
    help="Manage authentication configuration",
)


@auth_app.command
def setup():
    """Configure authentication for SQLsaber (API keys and Anthropic OAuth)."""
    console.print("\n[bold]SQLsaber Authentication Setup[/bold]\n")

    provider = questionary.select(
        "Select provider to configure:",
        choices=providers.all_keys(),
    ).ask()

    if provider is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return

    if provider == "anthropic":
        # Let user choose API key or OAuth
        method_choice = questionary.select(
            "Select Anthropic authentication method:",
            choices=[
                {"name": "API key", "value": AuthMethod.API_KEY},
                {"name": "Claude Pro/Max (OAuth)", "value": AuthMethod.CLAUDE_PRO},
            ],
        ).ask()

        if method_choice == AuthMethod.CLAUDE_PRO:
            flow = AnthropicOAuthFlow()
            if flow.authenticate():
                config_manager.set_auth_method(AuthMethod.CLAUDE_PRO)
                console.print(
                    "\n[bold green]✓ Anthropic OAuth configured successfully![/bold green]"
                )
            else:
                console.print("\n[red]✗ Anthropic OAuth setup failed.[/red]")
            console.print(
                "You can change this anytime by running [cyan]saber auth setup[/cyan] again."
            )
            return

    # API key flow (all providers + Anthropic when selected above)
    api_key_manager = APIKeyManager()
    env_var = api_key_manager._get_env_var_name(provider)
    console.print("\nTo configure your API key, you can either:")
    console.print(f"• Set the {env_var} environment variable")
    console.print("• Let SQLsaber prompt you for the key when needed (stored securely)")

    # Fetch/store key (cascades env -> keyring -> prompt)
    api_key = api_key_manager.get_api_key(provider)
    if api_key:
        config_manager.set_auth_method(AuthMethod.API_KEY)
        console.print(
            f"\n[bold green]✓ {provider.title()} API key configured successfully![/bold green]"
        )
    else:
        console.print("\n[yellow]No API key configured.[/yellow]")

    console.print(
        "You can change this anytime by running [cyan]saber auth setup[/cyan] again."
    )


@auth_app.command
def status():
    """Show current authentication configuration and provider key status."""
    auth_method = config_manager.get_auth_method()

    console.print("\n[bold blue]Authentication Status[/bold blue]")

    if auth_method is None:
        console.print("[yellow]No authentication method configured[/yellow]")
        console.print("Run [cyan]saber auth setup[/cyan] to configure authentication.")
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
        env_var = api_key_manager._get_env_var_name(provider)
        service = api_key_manager._get_service_name(provider)
        from_env = bool(os.getenv(env_var))
        from_keyring = bool(keyring.get_password(service, provider))
        if from_env:
            console.print(f"> {provider}: configured via {env_var}")
        elif from_keyring:
            console.print(f"> {provider}: [green]configured[/green]")
        else:
            console.print(f"> {provider}: [yellow]not configured[/yellow]")


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
        console.print("[yellow]Reset cancelled.[/yellow]")
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
            f"[yellow]No stored credentials found for {provider}. Nothing to reset.[/yellow]"
        )
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
        return

    # Perform deletions
    if oauth_present:
        OAuthTokenManager().remove_oauth_token("anthropic")
    if api_key_present:
        try:
            keyring.delete_password(service, provider)
            console.print(f"Removed {provider} API key from keyring", style="green")
        except keyring.errors.PasswordDeleteError:
            # Already absent; treat as success
            pass
        except Exception as e:
            console.print(f"Warning: Could not remove API key: {e}", style="yellow")

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

    console.print("\n[bold green]✓ Reset complete.[/bold green]")
    console.print(
        "Environment variables are not modified by this command.", style="dim"
    )


def create_auth_app() -> cyclopts.App:
    """Return the authentication management CLI app."""
    return auth_app
