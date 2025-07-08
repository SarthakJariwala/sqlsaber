"""Authentication CLI commands."""

import questionary
import cyclopts
from rich.console import Console

from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow

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
    """Configure authentication method for SQLSaber."""
    console.print("\n[bold]SQLSaber Authentication Setup[/bold]\n")

    # Use questionary for selection
    auth_choice = questionary.select(
        "Choose your authentication method:",
        choices=[
            questionary.Choice(
                title="Anthropic API Key",
                value=AuthMethod.API_KEY,
                description="You can create one by visiting https://console.anthropic.com",
            ),
            questionary.Choice(
                title="Claude Pro or Max Subscription",
                value=AuthMethod.CLAUDE_PRO,
                description="This does not require creating an API Key, but requires a subscription at https://claude.ai",
            ),
        ],
    ).ask()

    if auth_choice is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return

    # Handle auth method setup
    if auth_choice == AuthMethod.API_KEY:
        console.print("\nTo configure your API key, you can either:")
        console.print("• Set the ANTHROPIC_API_KEY environment variable")
        console.print(
            "• Let SQLsaber prompt you for the key when needed (stored securely)"
        )

        config_manager.set_auth_method(auth_choice)
        console.print("\n[bold green]Authentication method saved![/bold green]")

    elif auth_choice == AuthMethod.CLAUDE_PRO:
        oauth_flow = AnthropicOAuthFlow()
        try:
            success = oauth_flow.authenticate()
            if success:
                config_manager.set_auth_method(auth_choice)
                console.print(
                    "\n[bold green]Authentication setup complete![/bold green]"
                )
            else:
                console.print(
                    "\n[yellow]OAuth authentication failed. Please try again.[/yellow]"
                )
                return
        except Exception as e:
            console.print(f"\n[red]Authentication setup failed: {str(e)}[/red]")
            return

    console.print(
        "You can change this anytime by running [cyan]saber auth setup[/cyan] again."
    )


@auth_app.command
def status():
    """Show current authentication configuration."""
    auth_method = config_manager.get_auth_method()

    console.print("\n[bold blue]Authentication Status[/bold blue]")

    if auth_method is None:
        console.print("[yellow]No authentication method configured[/yellow]")
        console.print("Run [cyan]saber auth setup[/cyan] to configure authentication.")
    else:
        if auth_method == AuthMethod.API_KEY:
            console.print("[green]✓ API Key authentication configured[/green]")
            console.print("Using Anthropic API key for authentication")
        elif auth_method == AuthMethod.CLAUDE_PRO:
            console.print("[green]✓ Claude Pro/Max subscription configured[/green]")

            # Check OAuth token status
            oauth_flow = AnthropicOAuthFlow()
            if oauth_flow.has_valid_authentication():
                console.print("OAuth token is valid and ready to use")
            else:
                console.print("[yellow]OAuth token missing or expired[/yellow]")


@auth_app.command
def reset():
    """Reset authentication configuration."""
    if not config_manager.has_auth_configured():
        console.print("[yellow]No authentication configuration to reset.[/yellow]")
        return

    current_method = config_manager.get_auth_method()
    method_name = (
        "API Key" if current_method == AuthMethod.API_KEY else "Claude Pro/Max"
    )

    if questionary.confirm(
        f"Are you sure you want to reset the current authentication method ({method_name})?",
        default=False,
    ).ask():
        # If Claude Pro, also remove OAuth tokens
        if current_method == AuthMethod.CLAUDE_PRO:
            oauth_flow = AnthropicOAuthFlow()
            oauth_flow.remove_authentication()

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
