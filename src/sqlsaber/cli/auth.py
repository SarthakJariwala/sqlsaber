"""Authentication CLI commands."""

import questionary
import typer
from rich.console import Console

from sqlsaber.config.auth import AuthConfigManager, AuthMethod

# Global instances for CLI commands
console = Console()
config_manager = AuthConfigManager()

# Create the authentication management CLI app
auth_app = typer.Typer(
    name="auth",
    help="Manage authentication configuration",
    add_completion=False,
)


@auth_app.command("setup")
def setup_auth():
    """Configure authentication method for SQLSaber."""
    console.print("\n[bold]SQLSaber Authentication Setup[/bold]\n")

    # Use questionary for selection
    auth_choice = questionary.select(
        "Choose your authentication method:",
        choices=[
            questionary.Choice(
                title="API Key - Use your own Anthropic API key",
                value=AuthMethod.API_KEY,
            ),
            questionary.Choice(
                title="Claude Pro/Max Subscription - Use your Claude subscription",
                value=AuthMethod.CLAUDE_PRO,
            ),
        ],
    ).ask()

    if auth_choice is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return

    # Show selected method and details
    if auth_choice == AuthMethod.API_KEY:
        console.print("\n[green]✓[/green] API Key authentication selected")
        console.print("\nTo configure your API key, you can either:")
        console.print("• Set the ANTHROPIC_API_KEY environment variable")
        console.print(
            "• Let SQLSaber prompt you for the key when needed (stored securely)"
        )
    elif auth_choice == AuthMethod.CLAUDE_PRO:
        console.print(
            "\n[green]✓[/green] Claude Pro/Max subscription authentication selected"
        )
        console.print(
            "\nYou'll be prompted to authenticate via your web browser when needed."
        )

    # Save the configuration
    config_manager.set_auth_method(auth_choice)
    console.print("\n[bold green]Authentication method saved![/bold green]")
    console.print(
        "You can change this anytime by running [cyan]saber auth setup[/cyan] again."
    )


@auth_app.command("status")
def show_auth_status():
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
            console.print("Using Claude subscription for authentication")


@auth_app.command("reset")
def reset_auth():
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


def create_auth_app() -> typer.Typer:
    """Return the authentication management CLI app."""
    return auth_app
