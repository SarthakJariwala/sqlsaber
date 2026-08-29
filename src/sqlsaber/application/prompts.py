"""Prompter abstraction over ``Surface.ask``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from sqlsaber.render.surface import AskChoice, AskConfirm, AskPath, AskSecret, AskText
from sqlsaber.render.surface import Choice as RenderChoice
from sqlsaber.render.surface import PromptUnavailable
from sqlsaber.render import cli_out


class Choice:
    """Selectable option. ``title`` is the label, ``value`` is returned."""

    def __init__(
        self,
        title: str,
        value: Any = None,
        description: str | None = None,
    ) -> None:
        self.title = title
        self.value = title if value is None else value
        self.description = description


class Prompter(ABC):
    """Abstract base class for interactive prompting."""

    @abstractmethod
    async def text(
        self,
        message: str,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
    ) -> str | None:
        """Prompt for text input."""

    @abstractmethod
    async def select(
        self,
        message: str,
        choices: Sequence[str | Choice | dict[str, Any]],
        default: Any = None,
        use_search_filter: bool = False,
        use_jk_keys: bool = True,
    ) -> Any:
        """Prompt for selection from choices."""

    @abstractmethod
    async def confirm(self, message: str, default: bool = False) -> bool | None:
        """Prompt for yes/no confirmation."""

    @abstractmethod
    async def path(self, message: str, only_directories: bool = False) -> str | None:
        """Prompt for file/directory path."""

    @abstractmethod
    async def secret(self, message: str) -> str | None:
        """Prompt for a secret via getpass."""


def _adapt_validate(
    validate: Callable[[str], bool | str] | None,
) -> Callable[[str], str | None] | None:
    if validate is None:
        return None

    def wrapped(text: str) -> str | None:
        result = validate(text)
        if result is True:
            return None
        if result is False:
            return "Invalid value"
        return str(result)

    return wrapped


def _to_choices(
    choices: Sequence[str | Choice | dict[str, Any]],
) -> list[RenderChoice[Any]]:
    converted: list[RenderChoice[Any]] = []
    for item in choices:
        if isinstance(item, str):
            converted.append(RenderChoice(label=item, value=item))
        elif isinstance(item, Choice):
            converted.append(
                RenderChoice(
                    label=item.title,
                    value=item.value,
                    description=item.description,
                )
            )
        elif isinstance(item, dict):
            label = str(
                item.get("name") or item.get("title") or item.get("label") or ""
            )
            converted.append(
                RenderChoice(
                    label=label,
                    value=item.get("value", label),
                    description=item.get("description"),
                )
            )
        else:
            converted.append(RenderChoice(label=str(item), value=item))
    return converted


class AsyncPrompter(Prompter):
    """Async prompter using ``cli_out().ask``."""

    async def text(
        self,
        message: str,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
    ) -> str | None:
        try:
            return await cli_out().ask(
                AskText(
                    message,
                    default=default,
                    validate=_adapt_validate(validate),
                )
            )
        except PromptUnavailable:
            return None

    async def select(
        self,
        message: str,
        choices: Sequence[str | Choice | dict[str, Any]],
        default: Any = None,
        use_search_filter: bool = True,
        use_jk_keys: bool = False,
    ) -> Any:
        del use_jk_keys
        try:
            return await cli_out().ask(
                AskChoice(
                    message,
                    choices=_to_choices(choices),
                    default=default,
                    searchable=use_search_filter,
                )
            )
        except PromptUnavailable:
            return None

    async def confirm(self, message: str, default: bool = False) -> bool | None:
        try:
            return await cli_out().ask(AskConfirm(message, default=default))
        except PromptUnavailable:
            return None

    async def path(self, message: str, only_directories: bool = False) -> str | None:
        try:
            return await cli_out().ask(
                AskPath(message, only_directories=only_directories)
            )
        except PromptUnavailable:
            return None

    async def secret(self, message: str) -> str | None:
        try:
            return await cli_out().ask(AskSecret(message))
        except PromptUnavailable:
            return None
