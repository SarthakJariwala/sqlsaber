"""Model management CLI commands."""

import asyncio
import sys

import cyclopts
import httpx
import questionary
from rich.table import Table

from sqlsaber.config import providers
from sqlsaber.config.settings import Config
from sqlsaber.theme.manager import create_console
from sqlsaber.config.logging import get_logger

# Global instances for CLI commands
console = create_console()
logger = get_logger(__name__)

# Create the model management CLI app
models_app = cyclopts.App(
    name="models",
    help="Select and manage models",
)


class ModelManager:
    """Manages AI model configuration and fetching."""

    DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"
    MODELS_API_URL = "https://models.dev/api.json"
    # Providers come from central registry
    SUPPORTED_PROVIDERS = providers.all_keys()

    RECOMMENDED_MODELS = {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-5",
        "google": "gemini-2.5-pro",
        "groq": "llama-3-3-70b-versatile",
        "mistral": "mistral-large-latest",
        "cohere": "command-r-plus",
    }

    async def fetch_available_models(
        self, providers: list[str] | None = None
    ) -> list[dict]:
        """Fetch available models across providers from models.dev API.

        Returns list of dicts with keys: id (provider:model_id), provider, name, description, context_length, knowledge.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.MODELS_API_URL)
                response.raise_for_status()
                data = response.json()

                providers = providers or self.SUPPORTED_PROVIDERS
                results: list[dict] = []

                for provider in providers:
                    prov_data = data.get(provider, {})
                    models_obj = (
                        prov_data.get("models") or prov_data.get("Models") or {}
                    )
                    if not isinstance(models_obj, dict):
                        continue
                    for model_id, model_info in models_obj.items():
                        formatted_id = f"{provider}:{model_id}"
                        # cost
                        cost_info = (
                            model_info.get("cost", {})
                            if isinstance(model_info, dict)
                            else {}
                        )
                        cost_display = ""
                        if isinstance(cost_info, dict) and cost_info:
                            input_cost = cost_info.get("input", 0)
                            output_cost = cost_info.get("output", 0)
                            cost_display = f"${input_cost}/{output_cost} per 1M tokens"
                        # context
                        limit_info = (
                            model_info.get("limit", {})
                            if isinstance(model_info, dict)
                            else {}
                        )
                        context_length = (
                            limit_info.get("context", 0)
                            if isinstance(limit_info, dict)
                            else 0
                        )

                        results.append(
                            {
                                "id": formatted_id,
                                "provider": provider,
                                "name": model_info.get("name", model_id)
                                if isinstance(model_info, dict)
                                else model_id,
                                "description": cost_display,
                                "context_length": context_length,
                                "knowledge": model_info.get("knowledge", "")
                                if isinstance(model_info, dict)
                                else "",
                            }
                        )

                # Sort by provider then by name
                results.sort(key=lambda x: (x["provider"], x["name"]))
                logger.info("models.fetch.success", count=len(results))
                return results
        except Exception as e:
            console.print(f"[error]Error fetching models: {e}[/error]")
            logger.warning("models.fetch.error", error=str(e))
            return []

    def get_current_model(self) -> str:
        """Get the currently configured model."""
        config = Config()
        return config.model_name

    def set_model(self, model_id: str) -> bool:
        """Set the current model."""
        try:
            config = Config()
            config.set_model(model_id)
            logger.info("models.set.success", model=model_id)
            return True
        except Exception as e:
            console.print(f"[error]Error setting model: {e}[/error]")
            logger.error("models.set.error", model=model_id, error=str(e))
            return False

    def reset_model(self) -> bool:
        """Reset to default model."""
        return self.set_model(self.DEFAULT_MODEL)


model_manager = ModelManager()


@models_app.command
def list():
    """List available AI models."""
    logger.info("models.list.start")

    async def fetch_and_display():
        console.print("[blue]Fetching available models...[/blue]")
        models = await model_manager.fetch_available_models()

        if not models:
            console.print(
                "[warning]No models available or failed to fetch models[/warning]"
            )
            logger.info("models.list.empty")
            return

        table = Table(title="Available Models")
        table.add_column("Provider", style="magenta")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="success")
        table.add_column("Description", style="info")
        table.add_column("Context", style="warning", justify="right")
        table.add_column("Current", style="accent", justify="center")

        current_model = model_manager.get_current_model()

        for model in models:
            is_current = "✓" if model["id"] == current_model else ""
            context_str = (
                f"{model['context_length']:,}" if model["context_length"] else "N/A"
            )

            # Truncate description if too long
            description = (
                model["description"][:50] + "..."
                if len(model["description"]) > 50
                else model["description"]
            )

            table.add_row(
                model.get("provider", "-"),
                model["id"],
                model["name"],
                description,
                context_str,
                is_current,
            )

        console.print(table)
        console.print(f"\n[dim]Current model: {current_model}[/dim]")
        logger.info("models.list.complete", current=current_model, count=len(models))

    asyncio.run(fetch_and_display())


@models_app.command
def set():
    """Set the AI model to use."""
    logger.info("models.set.start")

    async def interactive_set():
        from sqlsaber.application.model_selection import choose_model, fetch_models
        from sqlsaber.application.prompts import AsyncPrompter

        console.print("[blue]Fetching available models...[/blue]")
        models = await fetch_models(model_manager)

        if not models:
            console.print("[error]Failed to fetch models. Cannot set model.[/error]")
            logger.error("models.set.no_models")
            sys.exit(1)

        prompter = AsyncPrompter()
        selected_model = await choose_model(
            prompter, models, restrict_provider=None, use_search_filter=True
        )

        if selected_model:
            if model_manager.set_model(selected_model):
                console.print(f"[green]✓ Model set to: {selected_model}[/green]")
                logger.info("models.set.done", model=selected_model)
            else:
                console.print("[error]✗ Failed to set model[/error]")
                logger.error("models.set.failed", model=selected_model)
                sys.exit(1)
        else:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("models.set.cancelled")

    asyncio.run(interactive_set())


@models_app.command
def current():
    """Show the currently configured model."""
    current = model_manager.get_current_model()
    console.print(f"Current model: [cyan]{current}[/cyan]")
    logger.info("models.current", model=current)


@models_app.command
def reset():
    """Reset to the default model."""
    logger.info("models.reset.start")

    async def interactive_reset():
        if await questionary.confirm(
            f"Reset to default model ({ModelManager.DEFAULT_MODEL})?"
        ).ask_async():
            if model_manager.reset_model():
                console.print(
                    f"[green]✓ Model reset to default: {ModelManager.DEFAULT_MODEL}[/green]"
                )
                logger.info("models.reset.done", model=ModelManager.DEFAULT_MODEL)
            else:
                console.print("[error]✗ Failed to reset model[/error]")
                logger.error("models.reset.failed")
                sys.exit(1)
        else:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("models.reset.cancelled")

    asyncio.run(interactive_reset())


def create_models_app() -> cyclopts.App:
    """Return the model management CLI app."""
    return models_app
