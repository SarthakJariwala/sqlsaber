"""Authentication CLI commands."""

import os

import cyclopts
import keyring
import questionary
from rich.console import Console

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
        choices=[
            "anthropic",
            "openai",
            "google",
            "groq",
            "mistral",
            "cohere",
            "huggingface",
        ],
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
    providers = [
        "anthropic",
        "openai",
        "google",
        "groq",
        "mistral",
        "cohere",
        "huggingface",
    ]
    for provider in providers:
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
    """Reset authentication configuration."""
    if not config_manager.has_auth_configured():
        console.print("[yellow]No authentication configuration to reset.[/yellow]")
        return

    current_method = config_manager.get_auth_method()
    method_name = (
        "API Key"
        if current_method == AuthMethod.API_KEY
        else "Claude Pro/Max (OAuth)"
        if current_method == AuthMethod.CLAUDE_PRO
        else "Unknown"
    )

    if questionary.confirm(
        f"Are you sure you want to reset the current authentication method ({method_name})?",
        default=False,
    ).ask():
        # If OAuth, remove stored token
        if current_method == AuthMethod.CLAUDE_PRO:
            OAuthTokenManager().remove_oauth_token("anthropic")

        # Clear the auth config by setting it to None
        config = config_manager._load_config()
        config["auth_method"] = None
        config_manager._save_config(config)
        console.print("[green]Authentication configuration reset.[/green]")
        console.print(
            "Run [cyan]saber auth setup[/cyan] to configure authentication again."
        )
    else:
        console.print("Reset cancelled.")


def create_auth_app() -> cyclopts.App:
    """Return the authentication management CLI app."""
    return auth_app
