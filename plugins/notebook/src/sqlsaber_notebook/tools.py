"""Model-visible tools owned by the notebook analyst."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic_ai import BinaryContent, RunContext, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from .execution import NotebookExecutionError
from .rendering import render_snapshot_for_model
from .session import NotebookSession


async def edit_cell(
    ctx: RunContext[NotebookSession],
    contents: str,
    idx: int | None = None,
) -> ToolReturn:
    """Add or edit a code cell, then rerun the entire notebook with a fresh kernel.

    Args:
        contents: Complete Python source for the cell.
        idx: Existing zero-based cell index to replace. Omit to append a new cell.
    """

    session = ctx.deps
    async with session.run_lock:
        try:
            target = _coerce_idx(idx, len(session.cells))
        except ValueError as exc:
            markdown, images = render_snapshot_for_model(session)
            return _snapshot_return(
                f"Edit was not applied: {exc}", markdown, images, None
            )

        old_cells = list(session.cells)
        old_outputs = deepcopy(session.outputs)
        old_artifacts = session.artifacts
        if target is None:
            session.cells.append(contents)
            session.outputs.append([])
            target = len(session.cells) - 1
            acknowledgement = f"Appended cell #{target}."
        else:
            session.cells[target] = contents
            acknowledgement = f"Edited cell #{target}."

        try:
            await session.run_notebook()
        except NotebookExecutionError as exc:
            session.cells = old_cells
            session.outputs = old_outputs
            session.artifacts = old_artifacts
            acknowledgement = (
                "Edit was not applied because notebook execution failed: "
                f"{exc} (backend={exc.backend}, phase={exc.phase})"
            )
        except BaseException:
            session.cells = old_cells
            session.outputs = old_outputs
            session.artifacts = old_artifacts
            raise

        markdown, images = render_snapshot_for_model(session)
        return _snapshot_return(acknowledgement, markdown, images, target)


async def list_workspace(ctx: RunContext[NotebookSession]) -> str:
    """List immutable inputs, generated files, and available provenance metadata."""

    async with ctx.deps.run_lock:
        return await ctx.deps.list_workspace()


def analyst_toolset() -> FunctionToolset[NotebookSession]:
    toolset = FunctionToolset[NotebookSession](id="notebook-analyst")
    toolset.add_function(
        list_workspace,
        name="list_workspace",
        takes_ctx=True,
        sequential=True,
    )
    toolset.add_function(
        edit_cell,
        name="edit_cell",
        takes_ctx=True,
        sequential=True,
    )
    return toolset


def _coerce_idx(value: object, cell_count: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("cell index must be an integer, not a boolean")
    if not isinstance(value, int):
        raise ValueError("cell index must be an integer or null")
    if value < 0:
        raise ValueError("cell index cannot be negative")
    if value >= cell_count:
        raise ValueError(
            f"cell index {value} is out of range for {cell_count} existing cells; omit idx to append"
        )
    return value


def _snapshot_return(
    acknowledgement: str,
    markdown: str,
    images: list[bytes],
    cell: int | None,
) -> ToolReturn:
    content: list[Any] = [f"Current notebook state:\n\n{markdown}"]
    content.extend(
        BinaryContent(data=image, media_type="image/png") for image in images
    )
    return ToolReturn(
        return_value=acknowledgement,
        content=content,
        metadata={"cell": cell},
    )
