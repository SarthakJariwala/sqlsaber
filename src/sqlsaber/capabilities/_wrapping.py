"""Signature and prompt helpers shared by SQLSaber capabilities."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic_ai import RunContext

from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.tools.base import Tool


def wrap_strip_db_name(tool: Tool) -> Callable[..., Awaitable[str]]:
    """Wrap a tool so its public signature no longer includes ``db_name``."""
    raw = tool.execute
    raw_sig = inspect.signature(raw)
    new_params = [p for name, p in raw_sig.parameters.items() if name != "db_name"]
    new_sig = raw_sig.replace(parameters=new_params)

    if tool.requires_ctx:

        async def wrapper(ctx: RunContext, *args, **kwargs) -> str:
            return await raw(ctx, *args, **kwargs)

    else:

        async def wrapper(*args, **kwargs) -> str:
            return await raw(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(raw, "__name__", tool.name)
    wrapper.__doc__ = raw.__doc__
    wrapper.__annotations__ = {
        key: value
        for key, value in getattr(raw, "__annotations__", {}).items()
        if key != "db_name"
    }
    return wrapper


def wrap_add_db_name(
    tool: Tool, names: tuple[str, ...]
) -> Callable[..., Awaitable[str]]:
    """Wrap a tool so its public schema requires ``db_name: Literal[...]``."""
    raw = tool.execute
    raw_sig = inspect.signature(raw)
    db_literal = Literal[names]  # type: ignore[valid-type]

    new_params = []
    for name, parameter in raw_sig.parameters.items():
        if name == "db_name":
            new_params.append(
                inspect.Parameter(
                    "db_name",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=db_literal,
                )
            )
        else:
            new_params.append(parameter)
    if not any(parameter.name == "db_name" for parameter in new_params):
        new_params.append(
            inspect.Parameter(
                "db_name",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=db_literal,
            )
        )
    new_sig = raw_sig.replace(parameters=new_params)

    if tool.requires_ctx:

        async def wrapper(ctx: RunContext, *args, db_name, **kwargs) -> str:
            return await raw(ctx, *args, db_name=db_name, **kwargs)

    else:

        async def wrapper(*args, db_name, **kwargs) -> str:
            return await raw(*args, db_name=db_name, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(raw, "__name__", tool.name)
    wrapper.__doc__ = docstring_with_db_name(raw.__doc__)
    wrapper.__annotations__ = {
        **getattr(raw, "__annotations__", {}),
        "db_name": db_literal,
    }
    return wrapper


_DB_NAME_DOC_TEXT = "db_name: which connected database to target."


def dedent_docstring(doc: str) -> str:
    """Dedent a docstring per PEP 257 while leaving its first line untouched."""
    lines = doc.splitlines()
    if not lines:
        return doc

    rest = lines[1:]
    indents = [len(line) - len(line.lstrip(" ")) for line in rest if line.strip()]
    if not indents:
        return doc.strip("\n")

    common = min(indents)
    dedented_rest = [line[common:] if line.strip() else line for line in rest]
    return "\n".join([lines[0].lstrip(), *dedented_rest]).strip("\n")


def docstring_with_db_name(doc: str | None) -> str:
    """Add a ``db_name`` entry to a tool docstring's existing Args section."""
    if not doc:
        return f"Args:\n    {_DB_NAME_DOC_TEXT}\n"

    dedented = dedent_docstring(doc)
    lines = dedented.splitlines()
    args_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("Args:")),
        None,
    )
    if args_index is None:
        return dedented + f"\n\nArgs:\n    {_DB_NAME_DOC_TEXT}\n"

    insert_at = args_index + 1
    while insert_at < len(lines) and (
        lines[insert_at].startswith(" ") or lines[insert_at] == ""
    ):
        insert_at += 1
    lines.insert(insert_at, f"    {_DB_NAME_DOC_TEXT}")
    return "\n".join(lines) + "\n"


def build_db_catalog(registry: DatabaseRegistry) -> str:
    """Render a database registry as a markdown bullet list."""
    return "\n".join(
        f"- {entry.name} ({entry.display_name}, dialect={entry.dialect}) "
        f"— {entry.description or 'no description'}"
        for entry in registry
    )


# Backwards-compatible private names for one migration cycle.
_wrap_strip_db_name = wrap_strip_db_name
_wrap_add_db_name = wrap_add_db_name
_dedent_docstring = dedent_docstring
_docstring_with_db_name = docstring_with_db_name
_build_db_catalog = build_db_catalog
