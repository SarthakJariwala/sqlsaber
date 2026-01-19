"""Model management CLI commands."""

import asyncio
import sys
from collections.abc import Sequence
from typing import Any, TypedDict

import cyclopts
import httpx
import questionary
from questionary import Choice
from rich.table import Table

from sqlsaber.config import providers
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.theme.manager import create_console

# Global instances for CLI commands
console = create_console()
logger = get_logger(__name__)

# Create the model management CLI app
models_app = cyclopts.App(
    name="models",
    help="Select and manage models",
)


class FetchedModel(TypedDict):
    """Structure for fetched model information."""

    id: str
    provider: str
    name: str
    description: str
    context_length: int
    knowledge: str


class ModelManager:
    """Manages AI model configuration and fetching."""

    DEFAULT_MODEL: str = "anthropic:claude-sonnet-4-5-20250929"
    MODELS_API_URL: str = "https://models.dev/api.json"
    SUPPORTED_PROVIDERS: Sequence[str] = providers.all_keys()

    RECOMMENDED_MODELS: dict[str, str] = {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-5",
        "google": "gemini-2.5-pro",
        "groq": "llama-3-3-70b-versatile",
        "mistral": "mistral-large-latest",
        "cohere": "command-r-plus",
    }

    async def fetch_available_models(
        self, providers: Sequence[str] | None = None
    ) -> list[FetchedModel]:
        """Fetch available models across providers from models.dev API.

        Returns list of dicts with keys: id (provider:model_id), provider, name,
        description, context_length, knowledge.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.MODELS_API_URL)
                response.raise_for_status()
                data: dict[str, Any] = response.json()

                selected_providers = providers or self.SUPPORTED_PROVIDERS
                results: list[FetchedModel] = []

                for provider in selected_providers:
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

                        name = (
                            model_info.get("name", model_id)
                            if isinstance(model_info, dict)
                            else model_id
                        )
                        knowledge = (
                            model_info.get("knowledge", "")
                            if isinstance(model_info, dict)
                            else ""
                        )

                        results.append(
                            FetchedModel(
                                id=formatted_id,
                                provider=provider,
                                name=name,
                                description=cost_display,
                                context_length=context_length,
                                knowledge=knowledge,
                            )
                        )

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


@models_app.command(name="list")
def list_models() -> None:
    """List available AI models."""
    logger.info("models.list.start")

    async def fetch_and_display() -> None:
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
        table.add_column("ID", style="info")
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


def _get_thinking_level_choices() -> list[Choice]:
    """Build thinking level choices for interactive selection."""
    return [
        Choice(
            "medium (Recommended - balanced cost/quality)", value=ThinkingLevel.MEDIUM
        ),
        Choice("low (faster, cheaper)", value=ThinkingLevel.LOW),
        Choice("high (deeper reasoning)", value=ThinkingLevel.HIGH),
        Choice("maximum (complex problems, highest cost)", value=ThinkingLevel.MAXIMUM),
        Choice("minimal (quick responses)", value=ThinkingLevel.MINIMAL),
        Choice("off (disable extended thinking)", value="off"),
    ]


async def _prompt_thinking_level(prompter: Any) -> tuple[bool, ThinkingLevel]:
    """Prompt user to configure thinking level.

    Returns:
        Tuple of (thinking_enabled, thinking_level)
    """
    configure = await prompter.confirm(
        "Configure thinking mode?",
        default=True,
    )

    if not configure:
        config = Config()
        return config.model.thinking_enabled, config.model.thinking_level

    level = await prompter.select(
        "Select thinking level:",
        choices=_get_thinking_level_choices(),
        use_search_filter=False,
    )

    if level is None:
        return False, ThinkingLevel.MEDIUM

    if isinstance(level, str) and level == "off":
        config = Config()
        return False, config.model.thinking_level

    return True, level


@models_app.command(name="set")
def set_model_command() -> None:
    """Set the AI model to use."""
    logger.info("models.set.start")

    async def interactive_set() -> None:
        from sqlsaber.application.model_selection import choose_model, fetch_models
        from sqlsaber.application.prompts import AsyncPrompter

        console.print("[blue]Fetching available models...[/blue]")
        models = await fetch_models(model_manager)

        if not models:
            console.print("[error]Failed to fetch models. Cannot set model.[/error]")
            logger.error("models.set.no_models")
            sys.exit(1)

        prompter = AsyncPrompter()
        selected_model: str | None = await choose_model(
            prompter, models, restrict_provider=None, use_search_filter=True
        )

        if selected_model:
            if model_manager.set_model(selected_model):
                console.print(f"[success]✓ Model set to: {selected_model}[/success]")
                logger.info("models.set.done", model=selected_model)

                # Prompt for thinking level configuration
                thinking_enabled, thinking_level = await _prompt_thinking_level(
                    prompter
                )
                config = Config()
                config.model.set_thinking(thinking_enabled, thinking_level)

                if thinking_enabled:
                    console.print(
                        f"[success]✓ Thinking: {thinking_level.value}[/success]"
                    )
                else:
                    console.print("[success]✓ Thinking: disabled[/success]")
                logger.info(
                    "models.set.thinking",
                    enabled=thinking_enabled,
                    level=thinking_level.value,
                )
            else:
                console.print("[error]✗ Failed to set model[/error]")
                logger.error("models.set.failed", model=selected_model)
                sys.exit(1)
        else:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("models.set.cancelled")

    asyncio.run(interactive_set())


@models_app.command(name="current")
def current_model() -> None:
    """Show the currently configured model and thinking settings."""
    current = model_manager.get_current_model()
    config = Config()
    thinking_enabled = config.model.thinking_enabled
    thinking_level = config.model.thinking_level

    console.print(f"Current model: [info]{current}[/info]")

    if thinking_enabled:
        console.print(
            f"Thinking: [success]enabled[/success] ([info]{thinking_level.value}[/info])"
        )
    else:
        console.print("Thinking: [dim]disabled[/dim]")

    logger.info(
        "models.current",
        model=current,
        thinking_enabled=thinking_enabled,
        thinking_level=thinking_level.value,
    )


@models_app.command(name="reset")
def reset_model_command() -> None:
    """Reset to the default model."""
    logger.info("models.reset.start")

    async def interactive_reset() -> None:
        if await questionary.confirm(
            f"Reset to default model ({ModelManager.DEFAULT_MODEL})?"
        ).ask_async():
            if model_manager.reset_model():
                console.print(
                    f"[success]✓ Model reset to default: {ModelManager.DEFAULT_MODEL}[/success]"
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
