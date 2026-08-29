"""Model management CLI commands."""

import asyncio
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

import cyclopts
import httpx

from sqlsaber.application.prompts import Choice
from sqlsaber.cli.output import err, fail, fail_usage, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config import providers
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import SUBAGENT_KEYS, Config, ThinkingLevel
from sqlsaber.render import blocks as b

logger = get_logger(__name__)

models_app = cyclopts.App(
    name="models",
    help="Select and manage models",
    help_epilogue=(
        "Examples:\n\n"
        "saber models current\n\n"
        "saber models set openai:gpt-5 --thinking-level medium"
    ),
)

AGENT_CHOICES: tuple[str, ...] = ("main", *SUBAGENT_KEYS)


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
            err(b.error(f"Error fetching models: {e}"))
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
            err(b.error(f"Error setting model: {e}"))
            logger.error("models.set.error", model=model_id, error=str(e))
            return False

    def reset_model(self) -> bool:
        """Reset to default model."""
        return self.set_model(self.DEFAULT_MODEL)


model_manager = ModelManager()


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if normalized not in AGENT_CHOICES:
        options = ", ".join(AGENT_CHOICES)
        raise ValueError(f"Invalid agent '{agent}'. Choose from: {options}.")
    return normalized


@models_app.command(name="list", help_epilogue="Example:\n\nsaber models list")
def list_models() -> None:
    """List available AI models.

    Example:
        saber models list
    """
    logger.info("models.list.start")

    async def fetch_and_display() -> None:
        out(b.md("Fetching available models..."))
        models = await model_manager.fetch_available_models()

        if not models:
            logger.info("models.list.empty")
            fail(
                "no models were returned.\n"
                "  Check your network connection, then retry: saber models list"
            )

        current_model = model_manager.get_current_model()
        rows = []
        for model in models:
            description = (
                model["description"][:50] + "..."
                if len(model["description"]) > 50
                else model["description"]
            )
            rows.append(
                {
                    "provider": model.get("provider", "-"),
                    "id": model["id"],
                    "name": model["name"],
                    "description": description,
                    "context": (
                        f"{model['context_length']:,}"
                        if model["context_length"]
                        else "N/A"
                    ),
                    "current": "✓" if model["id"] == current_model else "",
                }
            )

        out(
            b.table(
                rows,
                columns=(
                    b.Column("provider", "Provider", role="accent"),
                    b.Column("id", "ID", role="info"),
                    b.Column("name", "Name", role="success"),
                    b.Column("description", "Description", role="info"),
                    b.Column("context", "Context", role="warning"),
                    b.Column("current", "Current", role="accent"),
                ),
                caption="Available Models",
                max_rows=1000,
            ),
            b.md(f"Current model: {current_model}", role="muted"),
        )
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


def _resolve_thinking_level(value: str) -> tuple[bool, ThinkingLevel]:
    """Resolve a CLI thinking value into enabled state and level."""
    normalized = value.strip().lower()
    if normalized == "off":
        return False, Config().model.thinking_level
    try:
        return True, ThinkingLevel(normalized)
    except ValueError:
        choices = "off, " + ", ".join(level.value for level in ThinkingLevel)
        raise ValueError(
            f"Invalid thinking level '{value}'. Choose from: {choices}."
        ) from None


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


@models_app.command(
    name="set",
    help_epilogue=(
        "Examples:\n\n"
        "saber models set\n\n"
        "saber models set openai:gpt-5 --thinking-level medium\n\n"
        "saber models set openai:gpt-5 --agent handoff"
    ),
)
def set_model_command(
    model: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Provider-prefixed model ID (omit to select interactively)",
        ),
    ] = None,
    agent: Annotated[
        str,
        cyclopts.Parameter(
            ["--agent"],
            help="Target agent (main, handoff, viz, notebook)",
        ),
    ] = "main",
    thinking_level: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--thinking-level"],
            help="Thinking level for the main model (off, minimal, low, medium, high, maximum)",
        ),
    ] = None,
) -> None:
    """Set the AI model to use.

    Examples:
        saber models set
        saber models set anthropic:claude-sonnet-4-5-20250929 --thinking-level medium
        saber models set openai:gpt-5 --agent handoff
    """
    logger.info("models.set.start")

    try:
        target_agent = _normalize_agent(agent)
    except ValueError as exc:
        fail_usage(f"{exc}\n  Example: saber models set openai:gpt-5 --agent handoff")

    if thinking_level is not None and target_agent != "main":
        fail_usage(
            "--thinking-level applies only to the main model.\n"
            "  Example: saber models set openai:gpt-5 --agent handoff"
        )

    resolved_thinking: tuple[bool, ThinkingLevel] | None = None
    if thinking_level is not None:
        try:
            resolved_thinking = _resolve_thinking_level(thinking_level)
        except ValueError as exc:
            fail_usage(
                f"{exc}\n"
                "  Example: saber models set openai:gpt-5 --thinking-level medium"
            )

    if model is not None:
        model = model.strip()
        provider_name, separator, model_name = model.partition(":")
        if (
            not separator
            or not model_name
            or providers.canonical(provider_name) is None
        ):
            provider_choices = ", ".join(providers.all_keys())
            fail_usage(
                "MODEL must use a supported PROVIDER:MODEL ID.\n"
                f"  Providers: {provider_choices}\n"
                "  Example: saber models set openai:gpt-5 --thinking-level medium"
            )
        if target_agent == "main":
            if not model_manager.set_model(model):
                raise SystemExit(1)
            out(b.success(f"Model set to: {model}"))
            if resolved_thinking is not None:
                thinking_enabled, level = resolved_thinking
                Config().model.set_thinking(thinking_enabled, level)
                thinking_status = level.value if thinking_enabled else "disabled"
                out(b.success(f"Thinking: {thinking_status}"))
        else:
            Config().model.set_subagent_model(target_agent, model)
            out(b.success(f"{target_agent.title()} model set to: {model}"))
        logger.info("models.set.done", model=model, agent=target_agent)
        return

    async def interactive_set() -> None:
        from sqlsaber.application.model_selection import choose_model, fetch_models
        from sqlsaber.application.prompts import AsyncPrompter

        out(b.md("Fetching available models..."))
        models = await fetch_models(model_manager)

        if not models:
            logger.error("models.set.no_models")
            fail(
                "failed to fetch models; cannot set a model.\n"
                "  Set one directly with: saber models set PROVIDER:MODEL"
            )

        prompter = AsyncPrompter()
        selected_model: str | None = await choose_model(
            prompter, models, restrict_provider=None, use_search_filter=True
        )

        if selected_model:
            if target_agent == "main":
                if model_manager.set_model(selected_model):
                    out(b.success(f"Model set to: {selected_model}"))
                    logger.info(
                        "models.set.done", model=selected_model, agent=target_agent
                    )

                    if resolved_thinking is None:
                        thinking_enabled, selected_level = await _prompt_thinking_level(
                            prompter
                        )
                    else:
                        thinking_enabled, selected_level = resolved_thinking
                    config = Config()
                    config.model.set_thinking(thinking_enabled, selected_level)

                    if thinking_enabled:
                        out(b.success(f"Thinking: {selected_level.value}"))
                    else:
                        out(b.success("Thinking: disabled"))
                    logger.info(
                        "models.set.thinking",
                        enabled=thinking_enabled,
                        level=selected_level.value,
                        agent=target_agent,
                    )
                else:
                    logger.error(
                        "models.set.failed", model=selected_model, agent=target_agent
                    )
                    fail("failed to set model.")
            else:
                config = Config()
                config.model.set_subagent_model(target_agent, selected_model)
                out(
                    b.success(
                        f"{target_agent.title()} model set to: {selected_model}"
                    )
                )
                logger.info(
                    "models.set.subagent",
                    model=selected_model,
                    agent=target_agent,
                )
        else:
            out(b.warn("Operation cancelled"))
            logger.info("models.set.cancelled", agent=target_agent)

    asyncio.run(interactive_set())


@models_app.command(
    name="current",
    help_epilogue=(
        "Examples:\n\nsaber models current\n\nsaber models current --agent handoff"
    ),
)
def current_model(
    agent: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--agent"],
            help="Show model for agent (main, handoff, viz, notebook)",
        ),
    ] = None,
) -> None:
    """Show the currently configured model and thinking settings.

    Examples:
        saber models current
        saber models current --agent handoff
    """
    current = model_manager.get_current_model()
    config = Config()
    thinking_enabled = config.model.thinking_enabled
    thinking_level = config.model.thinking_level

    if agent is not None:
        try:
            target_agent = _normalize_agent(agent)
        except ValueError as exc:
            fail_usage(f"{exc}\n  Example: saber models current --agent handoff")

        if target_agent == "main":
            pairs: list[tuple[str, str]] = [("Current model", current)]
            if thinking_enabled:
                pairs.append(("Thinking", f"enabled ({thinking_level.value})"))
            else:
                pairs.append(("Thinking", "disabled"))
            out(b.key_values(pairs))
        else:
            override = config.model.get_subagent_model(target_agent)
            effective_model = override or current
            pairs = [(f"{target_agent.title()} model", effective_model)]
            if override:
                pairs.append(("Override", override))
            else:
                pairs.append(("Override", "not set (uses main)"))
            pairs.append(("Main model", current))
            out(b.key_values(pairs))

        logger.info(
            "models.current",
            model=current,
            thinking_enabled=thinking_enabled,
            thinking_level=thinking_level.value,
            agent=target_agent,
        )
        return

    pairs = [("Current model", current)]
    if thinking_enabled:
        pairs.append(("Thinking", f"enabled ({thinking_level.value})"))
    else:
        pairs.append(("Thinking", "disabled"))
    out(b.key_values(pairs, caption="Subagent overrides"))
    override_rows = []
    subagents = config.model.get_subagent_models()
    for subagent in SUBAGENT_KEYS:
        override = subagents.get(subagent)
        override_rows.append(
            {
                "subagent": subagent,
                "model": override if override else "(uses main)",
            }
        )
    out(
        b.table(
            override_rows,
            columns=(
                b.Column("subagent", "Subagent"),
                b.Column("model", "Model"),
            ),
            max_rows=1000,
        )
    )

    logger.info(
        "models.current",
        model=current,
        thinking_enabled=thinking_enabled,
        thinking_level=thinking_level.value,
        agent="all",
    )


@models_app.command(
    name="reset",
    help_epilogue=(
        "Examples:\n\nsaber models reset\n\nsaber models reset --agent handoff --yes"
    ),
)
def reset_model_command(
    agent: Annotated[
        str,
        cyclopts.Parameter(
            ["--agent"],
            help="Reset model for agent (main, handoff, viz, notebook)",
        ),
    ] = "main",
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Reset to the default model.

    Examples:
        saber models reset
        saber models reset --agent handoff --yes
    """
    logger.info("models.reset.start")

    try:
        target_agent = _normalize_agent(agent)
    except ValueError as exc:
        fail_usage(f"{exc}\n  Example: saber models reset --agent handoff --yes")

    prompt = (
        f"Reset to default model ({ModelManager.DEFAULT_MODEL})?"
        if target_agent == "main"
        else f"Clear {target_agent} model override (use main model)?"
    )
    command = f"saber models reset --agent {target_agent} --yes"
    if not confirm_action(
        yes=yes,
        prompt=prompt,
        non_interactive_command=command,
    ):
        out(b.warn("Operation cancelled"))
        logger.info("models.reset.cancelled", agent=target_agent)
        return

    if target_agent == "main":
        if not model_manager.reset_model():
            logger.error("models.reset.failed", agent=target_agent)
            fail("failed to reset model.")
        out(b.success(f"Model reset to default: {ModelManager.DEFAULT_MODEL}"))
        logger.info(
            "models.reset.done",
            model=ModelManager.DEFAULT_MODEL,
            agent=target_agent,
        )
        return

    Config().model.set_subagent_model(target_agent, None)
    out(b.success(f"{target_agent.title()} model override cleared"))
    logger.info("models.reset.subagent", agent=target_agent)


def create_models_app() -> cyclopts.App:
    """Return the model management CLI app."""
    return models_app
