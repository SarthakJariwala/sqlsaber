"""Shared model selection logic for onboarding and CLI."""

from questionary import Choice

from sqlsaber.application.prompts import Prompter
from sqlsaber.cli.models import FetchedModel, ModelManager
from sqlsaber.theme.manager import create_console

console = create_console()


async def fetch_models(
    model_manager: ModelManager, providers: list[str] | None = None
) -> list[FetchedModel]:
    """Fetch available models from models.dev API."""
    return await model_manager.fetch_available_models(providers=providers)


async def choose_model(
    prompter: Prompter,
    models: list[FetchedModel],
    restrict_provider: str | None = None,
    use_search_filter: bool = True,
) -> str | None:
    """Interactive model selection with recommended models prioritized.

    Args:
        prompter: Prompter instance for interaction
        models: List of model dicts from fetch_models
        restrict_provider: If set, only show models from this provider and use provider-specific recommendation
        use_search_filter: Enable search filter for large lists

    Returns:
        Selected model ID (provider:model_id) or None if cancelled
    """
    if not models:
        console.print("[warning]No models available[/warning]")
        return None

    # Filter by provider if restricted
    if restrict_provider:
        models = [m for m in models if m.get("provider") == restrict_provider]
        if not models:
            console.print(
                f"[warning]No models available for {restrict_provider}[/warning]"
            )
            return None

    # Get recommended model for the provider
    recommended_id = None
    if restrict_provider and restrict_provider in ModelManager.RECOMMENDED_MODELS:
        recommended_id = ModelManager.RECOMMENDED_MODELS[restrict_provider]

    # Build choices
    choices = []
    recommended_index = 0

    for i, model in enumerate(models):
        model_id_without_provider = model["id"].split(":", 1)[1]
        is_recommended = recommended_id == model_id_without_provider

        choice_text = model["name"]
        if is_recommended:
            choice_text += " (Recommended)"
            recommended_index = i
        elif model["description"]:
            desc_short = model["description"][:40]
            choice_text += (
                f" ({desc_short}...)"
                if len(model["description"]) > 40
                else f" ({desc_short})"
            )

        choices.append(Choice(choice_text, value=model["id"]))

    # Move recommended model to top if it exists
    if recommended_index > 0:
        choices.insert(0, choices.pop(recommended_index))

    # Prompt user
    selected_model = await prompter.select(
        "Select a model:",
        choices=choices,
        use_search_filter=use_search_filter,
    )

    if selected_model:
        return selected_model

    # User cancelled, return recommended or first available
    if recommended_id and restrict_provider:
        return f"{restrict_provider}:{recommended_id}"
    return models[0]["id"] if models else None


def set_model(model_manager: ModelManager, model_id: str) -> bool:
    """Set the current model."""
    return model_manager.set_model(model_id)
