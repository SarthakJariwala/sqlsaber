"""Model management CLI commands."""

import asyncio
import sys

import httpx
import questionary
import cyclopts
from rich.console import Console
from rich.table import Table

from sqlsaber.config.settings import Config

# Global instances for CLI commands
console = Console()

# Create the model management CLI app
models_app = cyclopts.App(
    name="models",
    help="Select and manage models",
)


class ModelManager:
    """Manages AI model configuration and fetching."""

    DEFAULT_MODEL = "anthropic:claude-sonnet-4-20250514"
    MODELS_API_URL = "https://models.dev/api.json"

    async def fetch_available_models(self) -> list[dict]:
        """Fetch available models from models.dev API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.MODELS_API_URL)
                response.raise_for_status()
                data = response.json()

                # Filter for Anthropic models only
                anthropic_models = []
                anthropic_data = data.get("anthropic", {})

                if "models" in anthropic_data:
                    for model_id, model_info in anthropic_data["models"].items():
                        # Convert to our format (anthropic:model-name)
                        formatted_id = f"anthropic:{model_id}"

                        # Extract cost information for display
                        cost_info = model_info.get("cost", {})
                        cost_display = ""
                        if cost_info:
                            input_cost = cost_info.get("input", 0)
                            output_cost = cost_info.get("output", 0)
                            cost_display = f"${input_cost}/{output_cost} per 1M tokens"

                        # Extract context length
                        limit_info = model_info.get("limit", {})
                        context_length = limit_info.get("context", 0)

                        anthropic_models.append(
                            {
                                "id": formatted_id,
                                "name": model_info.get("name", model_id),
                                "description": cost_display,
                                "context_length": context_length,
                                "knowledge": model_info.get("knowledge", ""),
                            }
                        )

                # Sort by name for better display
                anthropic_models.sort(key=lambda x: x["name"])
                return anthropic_models
        except Exception as e:
            console.print(f"[red]Error fetching models: {e}[/red]")
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
            return True
        except Exception as e:
            console.print(f"[red]Error setting model: {e}[/red]")
            return False

    def reset_model(self) -> bool:
        """Reset to default model."""
        return self.set_model(self.DEFAULT_MODEL)


model_manager = ModelManager()


@models_app.command
def list():
    """List available AI models."""

    async def fetch_and_display():
        console.print("[blue]Fetching available models...[/blue]")
        models = await model_manager.fetch_available_models()

        if not models:
            console.print(
                "[yellow]No models available or failed to fetch models[/yellow]"
            )
            return

        table = Table(title="Available Anthropic Models")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Context", style="yellow", justify="right")
        table.add_column("Current", style="bold red", justify="center")

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
                model["id"],
                model["name"],
                description,
                context_str,
                is_current,
            )

        console.print(table)
        console.print(f"\n[dim]Current model: {current_model}[/dim]")

    asyncio.run(fetch_and_display())


@models_app.command
def set():
    """Set the AI model to use."""

    async def interactive_set():
        console.print("[blue]Fetching available models...[/blue]")
        models = await model_manager.fetch_available_models()

        if not models:
            console.print("[red]Failed to fetch models. Cannot set model.[/red]")
            sys.exit(1)

        # Create choices for questionary
        choices = []
        for model in models:
            # Format: "ID - Name (Description)"
            choice_text = f"{model['id']} - {model['name']}"
            if model["description"]:
                choice_text += f" ({model['description'][:50]}{'...' if len(model['description']) > 50 else ''})"

            choices.append({"name": choice_text, "value": model["id"]})

        # Get current model to set as default
        current_model = model_manager.get_current_model()
        default_index = 0
        for i, choice in enumerate(choices):
            if choice["value"] == current_model:
                default_index = i
                break

        selected_model = await questionary.select(
            "Select a model:",
            choices=choices,
            use_shortcuts=True,
            use_search_filter=True,
            use_jk_keys=False,  # Disable j/k keys when using search filter
            default=choices[default_index] if choices else None,
        ).ask_async()

        if selected_model:
            if model_manager.set_model(selected_model):
                console.print(f"[green]✓ Model set to: {selected_model}[/green]")
            else:
                console.print("[red]✗ Failed to set model[/red]")
                sys.exit(1)
        else:
            console.print("[yellow]Operation cancelled[/yellow]")

    asyncio.run(interactive_set())


@models_app.command
def current():
    """Show the currently configured model."""
    current = model_manager.get_current_model()
    console.print(f"Current model: [cyan]{current}[/cyan]")


@models_app.command
def reset():
    """Reset to the default model."""

    async def interactive_reset():
        if await questionary.confirm(
            f"Reset to default model ({ModelManager.DEFAULT_MODEL})?"
        ).ask_async():
            if model_manager.reset_model():
                console.print(
                    f"[green]✓ Model reset to default: {ModelManager.DEFAULT_MODEL}[/green]"
                )
            else:
                console.print("[red]✗ Failed to reset model[/red]")
                sys.exit(1)
        else:
            console.print("[yellow]Operation cancelled[/yellow]")

    asyncio.run(interactive_reset())


def create_models_app() -> cyclopts.App:
    """Return the model management CLI app."""
    return models_app
