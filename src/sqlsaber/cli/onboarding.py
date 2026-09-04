"""Interactive onboarding flow for first-time SQLSaber users."""

import sys

from sqlsaber.cli.output import err, out
from sqlsaber.render import blocks as b

BANNER = """
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
           ▀▀
""".strip()


def needs_onboarding(database_arg: str | list[str] | None = None) -> bool:
    """Check if user needs onboarding.

    Onboarding is needed if:
    - No database is configured AND no database connection string provided via CLI

    """

    if database_arg:
        return False

    from sqlsaber.config.database import DatabaseConfigManager

    db_manager = DatabaseConfigManager()
    has_db = db_manager.has_databases()

    return not has_db


def welcome_screen() -> None:
    """Display welcome screen to new users."""

    out(
        b.panel((b.md(f"```\n{BANNER}\n```"),), role="primary"),
        b.panel(
            (
                b.md(
                    "**Welcome to SQLsaber!**\n\n"
                    "SQLsaber is an agentic SQL assistant that lets you query your "
                    "database using natural language.\n\n"
                    "Let's get you set up in just a few steps."
                ),
            ),
            role="primary",
        ),
    )


async def setup_database_guided() -> str | None:
    """Guide user through database setup.

    Returns the name of the configured database or None if cancelled.

    """

    from sqlsaber.cli.prompts import AsyncPrompter
    from sqlsaber.cli.workflows.db_setup import (
        build_config,
        collect_db_input,
        save_database,
        test_connection,
    )
    from sqlsaber.config.database import DatabaseConfigManager

    out(b.md("**Step 1 of 2: Database Connection**", role="primary"))

    try:
        prompter = AsyncPrompter()
        name = await prompter.text(
            "What would you like to name this connection?",
            default="mydb",
            validate=lambda x: bool(x.strip()) or "Name cannot be empty",
        )

        if name is None:
            return None

        name = name.strip()

        db_manager = DatabaseConfigManager()
        if db_manager.get_database(name):
            out(b.warn(f"Database connection '{name}' already exists."))
            return name

        db_input = await collect_db_input(
            prompter=prompter, name=name, db_type="postgresql", include_ssl=False
        )

        if db_input is None:
            return None

        db_config = build_config(db_input)

        out(b.md(f"Testing connection to '{name}'...", role="muted"))
        connection_success = await test_connection(db_config, db_input.password)

        if not connection_success:
            retry = await prompter.confirm(
                "Would you like to try again with different settings?", default=True
            )
            if retry:
                return await setup_database_guided()
            out(b.warn("You can add a database later using 'saber db add'"))
            return None

        try:
            save_database(db_manager, db_config, db_input.password)
            out(b.success(f"Connection to '{name}' successful"))
            return name
        except Exception as e:
            err(b.error(f"Error saving database: {e}"))
            return None

    except KeyboardInterrupt:
        out(b.warn("Setup cancelled."))
        return None
    except Exception as e:
        err(b.error(f"Unexpected error: {e}"))
        return None


async def select_model_for_provider(provider: str) -> str | None:
    """Fetch and let user select a model for the given provider.

    Returns the selected model ID or None if cancelled/failed.

    """

    from sqlsaber.cli.models import ModelManager
    from sqlsaber.cli.prompts import AsyncPrompter
    from sqlsaber.cli.workflows.model_selection import choose_model, fetch_models

    try:
        out(b.md(f"Fetching available {provider.title()} models...", role="muted"))

        model_manager = ModelManager()
        models = await fetch_models(model_manager, providers=[provider])

        if not models:
            out(b.warn(f"Could not fetch models for {provider}. Using default."))
            return (
                ModelManager.recommended_model_id(provider)
                or ModelManager.DEFAULT_MODEL
            )

        prompter = AsyncPrompter()
        selected_model = await choose_model(
            prompter, models, restrict_provider=provider, use_search_filter=True
        )

        return selected_model

    except KeyboardInterrupt:
        out(b.warn("Model selection cancelled."))
        return None
    except Exception as e:
        out(b.warn(f"Error selecting model: {e}. Using default."))
        return ModelManager.recommended_model_id(provider) or ModelManager.DEFAULT_MODEL


async def setup_auth_guided() -> tuple[bool, str | None]:
    """Guide user through auth setup.

    Returns tuple of (success: bool, selected_model: str | None).

    """

    from sqlsaber.cli.models import ModelManager
    from sqlsaber.cli.prompts import AsyncPrompter
    from sqlsaber.cli.workflows.auth_setup import setup_auth
    from sqlsaber.config.api_keys import APIKeyManager
    from sqlsaber.config.auth import AuthConfigManager

    out(b.md("**Step 2 of 2: Authentication**", role="primary"))

    try:
        prompter = AsyncPrompter()
        auth_manager = AuthConfigManager()
        api_key_manager = APIKeyManager()

        configured, provider = await setup_auth(
            prompter=prompter,
            auth_manager=auth_manager,
            api_key_manager=api_key_manager,
        )

        if not configured:
            out(b.warn("You can set it up later using 'saber auth setup'"))
            return False, None

        if provider is None:
            return True, None

        selected_model = await select_model_for_provider(provider)
        if selected_model:
            model_manager = ModelManager()
            model_manager.set_model(selected_model)
            out(b.success(f"Model set to: {selected_model}"))
        return True, selected_model

    except KeyboardInterrupt:
        out(b.warn("Setup cancelled."))
        return False, None
    except Exception as e:
        err(b.error(f"Unexpected error: {e}"))
        return False, None


def success_screen(
    database_name: str | None, auth_configured: bool, model_name: str | None = None
) -> None:
    """Display success screen after onboarding."""

    notes: list[b.Block] = [b.success("You're all set!")]

    if database_name and auth_configured:
        notes.append(
            b.success(f"Database '{database_name}' connected and ready to use")
        )
        notes.append(b.success("Authentication configured"))
        if model_name:
            notes.append(b.success(f"Model: {model_name}"))
    elif database_name:
        notes.append(
            b.success(f"Database '{database_name}' connected and ready to use")
        )
        notes.append(
            b.warn("AI authentication not configured - you'll be prompted when needed")
        )
    elif auth_configured:
        notes.append(b.success("AI authentication configured"))
        if model_name:
            notes.append(b.success(f"Model: {model_name}"))
        notes.append(
            b.warn("No database configured - you'll need to provide one via -d flag")
        )

    notes.append(b.md("Starting interactive session...", role="muted"))
    out(*notes)


async def run_onboarding() -> bool:
    """Run the complete onboarding flow.

    Returns True if onboarding completed successfully (at least database configured),
    False if user cancelled or onboarding failed.

    """

    try:
        welcome_screen()

        database_name = await setup_database_guided()

        if database_name is None:
            out(
                b.warn("Database setup is required to continue."),
                b.md(
                    "You can also provide a connection string using: "
                    "`saber -d <connection-string>`",
                    role="muted",
                ),
            )
            return False

        auth_configured, model_name = await setup_auth_guided()

        success_screen(database_name, auth_configured, model_name)

        return True

    except KeyboardInterrupt:
        out(
            b.warn("Onboarding cancelled."),
            b.md(
                "You can run setup commands manually:\n"
                "  - `saber db add <name>`  # Add database connection\n"
                "  - `saber auth setup`     # Configure authentication",
                role="muted",
            ),
        )
        sys.exit(0)
    except Exception as e:
        err(b.error(f"Onboarding failed: {e}"))
        return False
