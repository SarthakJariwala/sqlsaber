"""Interactive onboarding flow for first-time SQLSaber users."""

import sys

from rich.panel import Panel

from sqlsaber.cli.models import ModelManager
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.theme.manager import create_console

console = create_console()


def needs_onboarding(database_arg: str | list[str] | None = None) -> bool:
    """Check if user needs onboarding.

    Onboarding is needed if:
    - No database is configured AND no database connection string provided via CLI
    """
    # If user provided a database argument, skip onboarding
    if database_arg:
        return False

    # Check if databases are configured
    db_manager = DatabaseConfigManager()
    has_db = db_manager.has_databases()

    return not has_db


def welcome_screen() -> None:
    """Display welcome screen to new users."""
    banner = """[primary]
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
           ▀▀
    [/primary]"""

    console.print(Panel.fit(banner, style="primary"))
    console.print()

    welcome_message = """
[bold]Welcome to SQLsaber! 🎉[/bold]

SQLsaber is an agentic SQL assistant that lets you query your database using natural language.

Let's get you set up in just a few steps.
    """

    console.print(
        Panel.fit(welcome_message.strip(), border_style="primary", padding=(1, 2))
    )
    console.print()


async def setup_database_guided() -> str | None:
    """Guide user through database setup.

    Returns the name of the configured database or None if cancelled.
    """
    from sqlsaber.application.db_setup import (
        build_config,
        collect_db_input,
        save_database,
        test_connection,
    )
    from sqlsaber.application.prompts import AsyncPrompter

    console.print("[heading]Step 1 of 2: Database Connection[/heading]")
    console.print()

    try:
        # Ask for connection name
        prompter = AsyncPrompter()
        name = await prompter.text(
            "What would you like to name this connection?",
            default="mydb",
            validate=lambda x: bool(x.strip()) or "Name cannot be empty",
        )

        if name is None:
            return None

        name = name.strip()

        # Check if name already exists
        db_manager = DatabaseConfigManager()
        if db_manager.get_database(name):
            console.print(
                f"[warning]Database connection '{name}' already exists.[/warning]"
            )
            return name

        # Collect database input (simplified - no SSL in onboarding)
        db_input = await collect_db_input(
            prompter=prompter, name=name, db_type="postgresql", include_ssl=False
        )

        if db_input is None:
            return None

        # Build config
        db_config = build_config(db_input)

        # Test the connection
        console.print(f"[muted]Testing connection to '{name}'...[/muted]")
        connection_success = await test_connection(db_config, db_input.password)

        if not connection_success:
            retry = await prompter.confirm(
                "Would you like to try again with different settings?", default=True
            )
            if retry:
                return await setup_database_guided()
            else:
                console.print(
                    "[warning]You can add a database later using 'saber db add'[/warning]"
                )
                return None

        # Save the configuration
        try:
            save_database(db_manager, db_config, db_input.password)
            console.print(f"[success]✓ Connection to '{name}' successful![/success]")
            console.print()
            return name
        except Exception as e:
            console.print(f"[error]Error saving database:[/error] {e}")
            return None

    except KeyboardInterrupt:
        console.print("\n[warning]Setup cancelled.[/warning]")
        return None
    except Exception as e:
        console.print(f"[error]Unexpected error:[/error] {e}")
        return None


async def select_model_for_provider(provider: str) -> str | None:
    """Fetch and let user select a model for the given provider.

    Returns the selected model ID or None if cancelled/failed.
    """
    from sqlsaber.application.model_selection import choose_model, fetch_models
    from sqlsaber.application.prompts import AsyncPrompter

    try:
        console.print()
        console.print(f"[muted]Fetching available {provider.title()} models...[/muted]")

        model_manager = ModelManager()
        models = await fetch_models(model_manager, providers=[provider])

        if not models:
            console.print(
                f"[warning]Could not fetch models for {provider}. Using default.[/warning]"
            )
            # Use provider-specific default or fallback to Anthropic
            default_model_id = ModelManager.RECOMMENDED_MODELS.get(
                provider, ModelManager.DEFAULT_MODEL
            )
            # Format it properly if we have a recommended model for this provider
            if provider in ModelManager.RECOMMENDED_MODELS:
                return f"{provider}:{ModelManager.RECOMMENDED_MODELS[provider]}"
            return default_model_id

        prompter = AsyncPrompter()
        console.print()
        selected_model = await choose_model(
            prompter, models, restrict_provider=provider, use_search_filter=True
        )

        return selected_model

    except KeyboardInterrupt:
        console.print("\n[warning]Model selection cancelled.[/warning]")
        return None
    except Exception as e:
        console.print(f"[warning]Error selecting model: {e}. Using default.[/warning]")
        # Fallback to provider default
        if provider in ModelManager.RECOMMENDED_MODELS:
            return f"{provider}:{ModelManager.RECOMMENDED_MODELS[provider]}"
        return ModelManager.DEFAULT_MODEL


async def setup_auth_guided() -> tuple[bool, str | None]:
    """Guide user through auth setup.

    Returns tuple of (success: bool, selected_model: str | None).
    """
    from sqlsaber.application.auth_setup import setup_auth
    from sqlsaber.application.prompts import AsyncPrompter

    console.print("[primary]Step 2 of 2: Authentication[/primary]")
    console.print()

    try:
        # Run auth setup
        prompter = AsyncPrompter()
        auth_manager = AuthConfigManager()
        api_key_manager = APIKeyManager()

        success, provider = await setup_auth(
            prompter=prompter,
            auth_manager=auth_manager,
            api_key_manager=api_key_manager,
            default_provider="anthropic",
        )

        if not success:
            console.print(
                "[warning]You can set it up later using 'saber auth setup'[/warning]"
            )
            console.print()
            return False, None

        # If auth configured but we don't know the provider (already configured case)
        if provider is None:
            console.print()
            return True, None

        # Select model for this provider
        selected_model = await select_model_for_provider(provider)
        if selected_model:
            model_manager = ModelManager()
            model_manager.set_model(selected_model)
            console.print(f"[success]✓ Model set to: {selected_model}[/success]")
        console.print()
        return True, selected_model

    except KeyboardInterrupt:
        console.print("\n[warning]Setup cancelled.[/warning]")
        console.print()
        return False, None
    except Exception as e:
        console.print(f"[error]Unexpected error:[/error] {e}")
        console.print()
        return False, None


def success_screen(
    database_name: str | None, auth_configured: bool, model_name: str | None = None
) -> None:
    """Display success screen after onboarding."""

    console.print("[success]You're all set! 🚀[/success]")
    console.print()

    if database_name and auth_configured:
        console.print(
            f"[success]✓ Database '{database_name}' connected and ready to use[/success]"
        )
        console.print("[success]✓ Authentication configured[/success]")
        if model_name:
            console.print(f"[success]✓ Model: {model_name}[/success]")
    elif database_name:
        console.print(
            f"[success]✓ Database '{database_name}' connected and ready to use[/success]"
        )
        console.print(
            "[warning]⚠ AI authentication not configured - you'll be prompted when needed[/warning]"
        )
    elif auth_configured:
        console.print("[success]✓ AI authentication configured[/success]")
        if model_name:
            console.print(f"[success]✓ Model: {model_name}[/success]")
        console.print(
            "[warning]⚠ No database configured - you'll need to provide one via -d flag[/warning]"
        )

    console.print()
    console.print("[muted]Starting interactive session...[/muted]")
    console.print()


async def run_onboarding() -> bool:
    """Run the complete onboarding flow.

    Returns True if onboarding completed successfully (at least database configured),
    False if user cancelled or onboarding failed.
    """
    try:
        # Welcome screen
        welcome_screen()

        # Database setup
        database_name = await setup_database_guided()

        # If user cancelled database setup, exit
        if database_name is None:
            console.print("[warning]Database setup is required to continue.[/warning]")
            console.print(
                "[muted]You can also provide a connection string using: saber -d <connection-string>[/muted]"
            )
            return False

        # Auth setup
        auth_configured, model_name = await setup_auth_guided()

        # Show success screen
        success_screen(database_name, auth_configured, model_name)

        return True

    except KeyboardInterrupt:
        console.print("\n[warning]Onboarding cancelled.[/warning]")
        console.print(
            "[muted]You can run setup commands manually:[/muted]\n"
            "[muted]  - saber db add <name>  # Add database connection[/muted]\n"
            "[muted]  - saber auth setup     # Configure authentication[/muted]"
        )
        sys.exit(0)
    except Exception as e:
        console.print(f"[error]Onboarding failed:[/error] {e}")
        return False
