"""Prompter abstraction for sync/async questionary interactions."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import questionary
from questionary import Choice


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
        pass

    @abstractmethod
    async def select(
        self,
        message: str,
        choices: list[str] | list[Choice] | list[dict],
        default: Any = None,
        use_search_filter: bool = False,
        use_jk_keys: bool = True,
    ) -> Any:
        """Prompt for selection from choices."""
        pass

    @abstractmethod
    async def confirm(self, message: str, default: bool = False) -> bool | None:
        """Prompt for yes/no confirmation."""
        pass

    @abstractmethod
    async def path(self, message: str, only_directories: bool = False) -> str | None:
        """Prompt for file/directory path."""
        pass


class AsyncPrompter(Prompter):
    """Async prompter using questionary.ask_async() for onboarding."""

    async def text(
        self,
        message: str,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
    ) -> str | None:
        return await questionary.text(
            message, default=default, validate=validate
        ).ask_async()

    async def select(
        self,
        message: str,
        choices: list[str] | list[Choice] | list[dict],
        default: Any = None,
        use_search_filter: bool = True,
        use_jk_keys: bool = False,
    ) -> Any:
        return await questionary.select(
            message,
            choices=choices,
            default=default,
            use_search_filter=use_search_filter,
            use_jk_keys=use_jk_keys,
        ).ask_async()

    async def confirm(self, message: str, default: bool = False) -> bool | None:
        return await questionary.confirm(message, default=default).ask_async()

    async def path(self, message: str, only_directories: bool = False) -> str | None:
        return await questionary.path(
            message, only_directories=only_directories
        ).ask_async()


class SyncPrompter(Prompter):
    """Sync prompter using questionary.ask() for CLI commands."""

    async def text(
        self,
        message: str,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
    ) -> str | None:
        return questionary.text(message, default=default, validate=validate).ask()

    async def select(
        self,
        message: str,
        choices: list[str] | list[Choice] | list[dict],
        default: Any = None,
        use_search_filter: bool = True,
        use_jk_keys: bool = False,
    ) -> Any:
        return questionary.select(
            message,
            choices=choices,
            default=default,
            use_search_filter=use_search_filter,
            use_jk_keys=use_jk_keys,
        ).ask()

    async def confirm(self, message: str, default: bool = False) -> bool | None:
        return questionary.confirm(message, default=default).ask()

    async def path(self, message: str, only_directories: bool = False) -> str | None:
        return questionary.path(message, only_directories=only_directories).ask()
