"""Managed SQLsaber capability for notebook data analysis."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from PIL import Image, UnidentifiedImageError
from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from rich.color import Color
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
)
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.tools.base import Tool
from sqlsaber.utils.json_utils import json_dumps

from ._shared import (
    MAX_DEFAULT_RESULTS,
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_TOTAL_BYTES,
)
from .analyst import analyze, supports_notebook_images
from .execution import (
    NotebookExecutionError,
    NotebookInput,
    NotebookLimitExceeded,
    resolve_notebook_backend,
    resolve_notebook_image,
)
from .rendering import limit_output, render_notebook_bytes
from .result import AnalysisResult, ManifestEntry, Workspace

_RESULT_FILE_PATTERN = re.compile(r"^result_[A-Za-z0-9._-]+\.json$")
_MAX_DISPLAY_RESULTS = 2
_MAX_PLOT_COLUMNS = 80
_MAX_PLOT_ROWS = 24


@dataclass(frozen=True, slots=True)
class _NotebookDisplay:
    markdown: str
    images: tuple[bytes, ...]


class AnalyzeDataTool(Tool):
    """Delegate multi-step analysis to the notebook analyst."""

    requires_ctx = True

    def __init__(self, context: PluginContext) -> None:
        super().__init__()
        self._context = context
        self._display_results: OrderedDict[str, _NotebookDisplay] = OrderedDict()

    @property
    def name(self) -> str:
        return "analyze_data"

    async def execute(
        self,
        ctx: RunContext,
        goal: str,
        files: list[str] | None = None,
    ) -> ToolReturn | str:
        """Hand a data-analysis goal to a notebook subagent.

        Use this after running SQL when the answer requires multi-step calculations,
        statistical analysis, data transformations, or plots.

        Args:
            goal: The question to answer and analysis to perform.
            files: Optional execute_sql result keys to analyze. When omitted, the
                newest bounded successful query results are included.
        """

        try:
            workspace = build_workspace_from_history(ctx, only=files)
            model_name, model, provider = self._context.resolve_subagent_model(
                "notebook",
                tool_name=self.name,
            )
            backend = resolve_notebook_backend()
            publisher = getattr(self._context, "artifact_publisher", None)
            result = await analyze(
                goal,
                workspace,
                model=model,
                model_provider=provider,
                backend=backend,
                image=resolve_notebook_image(),
                include_snapshot_images=supports_notebook_images(model_name, provider),
                collect_files=publisher is not None,
                parent_usage=ctx.usage,
            )
            markdown, notebook_images = render_notebook_bytes(result.notebook)
            display_images = _dedupe_images([*notebook_images, *result.images])
            self._remember_display(
                ctx.tool_call_id,
                _NotebookDisplay(markdown, tuple(display_images)),
            )
            metadata: dict[str, object] = {
                "backend": backend.name,
                "model": model_name,
                "provenance": result.provenance,
                "files": [item.name for item in workspace.files],
            }
            if publisher is not None:
                try:
                    publication = await publisher.publish(
                        _artifact_bundle(
                            result,
                            backend=backend.name,
                            model=model_name,
                        ),
                        context=ArtifactContext(
                            run_id=ctx.run_id,
                            conversation_id=ctx.conversation_id,
                            tool_call_id=ctx.tool_call_id,
                            metadata=ctx.metadata or {},
                        ),
                    )
                except Exception as exc:
                    failure_mode = getattr(
                        self._context, "artifact_failure_mode", "required"
                    )
                    if failure_mode == "required":
                        return _error_result(
                            f"Analysis completed but artifacts could not be "
                            f"published: {exc}",
                            backend=backend.name,
                            phase="artifact-publication",
                        )
                    metadata["artifact_error"] = limit_output(str(exc), 2_000)
                else:
                    metadata.update(publication.to_metadata())
            return ToolReturn(return_value=result.answer, metadata=metadata)
        except NotebookExecutionError as exc:
            return _error_result(
                str(exc),
                backend=exc.backend,
                phase=exc.phase,
            )
        except (TimeoutError, ValueError) as exc:
            return _error_result(str(exc))

    def render_executing(self, console: Console, args: dict) -> bool:
        goal = args.get("goal")
        if isinstance(goal, str) and goal.strip():
            console.print(f"[muted bold]Analyzing data:[/muted bold] {goal.strip()}")
        else:
            console.print("[muted bold]Analyzing data in notebook[/muted bold]")
        return True

    def render_result_event(
        self,
        console: Console,
        result: object,
        *,
        tool_call_id: str | None = None,
        metadata: object = None,
    ) -> bool:
        del metadata
        display = self._display_results.pop(tool_call_id or "", None)
        if display is None:
            return False

        console.print(Rule("Analysis notebook"))
        if display.markdown.strip():
            console.print(Markdown(display.markdown))
        else:
            console.print("[muted]No notebook cells were executed.[/muted]")

        for index, image in enumerate(display.images, start=1):
            _render_plot(console, image, index=index)

        answer = limit_output(str(result)).strip()
        if answer:
            console.print()
            console.print("[bold]Analysis result[/bold]")
            console.print(Markdown(answer))
        console.print(Rule())
        return True

    async def close(self) -> None:
        self._display_results.clear()

    def _remember_display(
        self, tool_call_id: str | None, display: _NotebookDisplay
    ) -> None:
        key = tool_call_id or ""
        self._display_results[key] = display
        self._display_results.move_to_end(key)
        while len(self._display_results) > _MAX_DISPLAY_RESULTS:
            self._display_results.popitem(last=False)


class Notebook(SqlSaberCapability):
    """Delegate multi-step data analysis to a notebook subagent."""

    id = "notebook"
    description = "Delegate multi-step data analysis to a notebook subagent."

    def __init__(self, context: PluginContext) -> None:
        self.tool = AnalyzeDataTool(context)
        self._toolset = FunctionToolset[Any](id=self.id)
        self._toolset.add_function(
            self.tool.execute,
            name=self.tool.name,
            takes_ctx=True,
        )

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self.tool.name: self.tool}

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def close(self) -> None:
        await self.tool.close()


def capability(
    context: PluginContext,
) -> AbstractCapability[Any] | Sequence[AbstractCapability[Any]]:
    """Always expose the installed plugin; backend checks happen on use."""

    return Notebook(context)


def _artifact_bundle(
    result: AnalysisResult,
    *,
    backend: str,
    model: str,
) -> ArtifactBundle:
    artifacts = [
        Artifact(
            name="analysis.ipynb",
            data=result.notebook,
            media_type="application/x-ipynb+json",
            kind="notebook",
        )
    ]
    artifacts.extend(
        Artifact(
            name=f"plots/plot_{index}.png",
            data=image,
            media_type="image/png",
            kind="image",
        )
        for index, image in enumerate(result.images, start=1)
    )
    artifacts.extend(
        Artifact(
            name=f"files/{artifact.name}",
            data=artifact.data,
            media_type=artifact.media_type,
            kind="file",
        )
        for artifact in result.files
    )
    return ArtifactBundle(
        kind="notebook-analysis",
        artifacts=tuple(artifacts),
        metadata={
            "backend": backend,
            "model": model,
            "provenance": result.provenance,
        },
    )


def build_workspace_from_history(
    ctx: RunContext,
    only: list[str] | None,
) -> Workspace:
    """Build a bounded workspace from successful execute_sql history."""

    queries_by_id: dict[str, str] = {}
    for message in ctx.messages:
        for part in getattr(message, "parts", []):
            if (
                getattr(part, "part_kind", "") not in ("tool-call", "builtin-tool-call")
                or getattr(part, "tool_name", "") != "execute_sql"
            ):
                continue
            tool_call_id = getattr(part, "tool_call_id", None)
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            args = _args_as_dict(part)
            query = args.get("query")
            if isinstance(query, str) and query.strip():
                queries_by_id[tool_call_id] = query

    requested = _normalize_requested_files(only)
    if requested is not None and len(requested) > MAX_WORKSPACE_FILES:
        raise NotebookLimitExceeded(
            f"Workspace has {len(requested)} files; maximum is {MAX_WORKSPACE_FILES}",
            backend="notebook",
            phase="input-validation",
        )

    eligible: OrderedDict[str, tuple[bytes, ManifestEntry]] = OrderedDict()
    eligible_bytes = 0
    done = False
    for message in reversed(ctx.messages):
        for part in reversed(getattr(message, "parts", [])):
            if (
                getattr(part, "part_kind", "")
                not in ("tool-return", "builtin-tool-return")
                or getattr(part, "tool_name", "") != "execute_sql"
            ):
                continue
            tool_call_id = getattr(part, "tool_call_id", None)
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            payload = _payload_as_dict(getattr(part, "content", None))
            if payload is None or payload.get("success") is not True:
                continue
            if "results" not in payload:
                continue
            key = payload.get("file")
            expected_key = f"result_{tool_call_id}.json"
            if (
                not isinstance(key, str)
                or key != expected_key
                or not _RESULT_FILE_PATTERN.fullmatch(key)
                or key in eligible
                or (requested is not None and key not in requested)
            ):
                continue
            data = json_dumps(payload).encode("utf-8")
            _validate_file_size(key, data)
            if eligible_bytes + len(data) > MAX_WORKSPACE_TOTAL_BYTES:
                if requested is not None:
                    raise NotebookLimitExceeded(
                        f"Workspace exceeds {MAX_WORKSPACE_TOTAL_BYTES} total bytes",
                        backend="notebook",
                        phase="input-validation",
                    )
                done = True
                break
            eligible[key] = (
                data,
                ManifestEntry(file=key, sql=queries_by_id.get(tool_call_id)),
            )
            eligible_bytes += len(data)
            done = (
                len(eligible) >= MAX_DEFAULT_RESULTS
                if requested is None
                else len(eligible) == len(requested)
            )
            if done:
                break
        if done:
            break

    if requested is not None:
        missing = [key for key in requested if key not in eligible]
        if missing:
            raise ValueError(
                "Requested SQL result files were not found: " + ", ".join(missing)
            )
        selected = [(key, *eligible[key]) for key in requested]
    else:
        selected = [(key, data, manifest) for key, (data, manifest) in eligible.items()]

    if not selected:
        raise ValueError(
            "No successful row-returning execute_sql results are available to analyze"
        )

    files = tuple(NotebookInput(key, data) for key, data, _ in selected)
    manifest = tuple(entry for _, _, entry in selected)
    _validate_workspace(files)
    return Workspace(files=files, manifest=manifest)


def _normalize_requested_files(files: list[str] | None) -> list[str] | None:
    if files is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for key in files:
        if not isinstance(key, str) or not _RESULT_FILE_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid SQL result file key: {key!r}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    if not normalized:
        raise ValueError("files must contain at least one SQL result key")
    return normalized


def _validate_workspace(files: tuple[NotebookInput, ...]) -> None:
    if len(files) > MAX_WORKSPACE_FILES:
        raise NotebookLimitExceeded(
            f"Workspace has {len(files)} files; maximum is {MAX_WORKSPACE_FILES}",
            backend="notebook",
            phase="input-validation",
        )
    total = 0
    for item in files:
        _validate_file_size(item.name, item.data)
        total += len(item.data)
    if total > MAX_WORKSPACE_TOTAL_BYTES:
        raise NotebookLimitExceeded(
            f"Workspace exceeds {MAX_WORKSPACE_TOTAL_BYTES} total bytes",
            backend="notebook",
            phase="input-validation",
        )


def _validate_file_size(key: str, data: bytes) -> None:
    if len(data) > MAX_WORKSPACE_FILE_BYTES:
        raise NotebookLimitExceeded(
            f"SQL result exceeds {MAX_WORKSPACE_FILE_BYTES} bytes: {key}",
            backend="notebook",
            phase="input-validation",
        )


def _args_as_dict(part: object) -> dict[str, Any]:
    method = getattr(part, "args_as_dict", None)
    if callable(method):
        try:
            value = method()
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    args = getattr(part, "args", None)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _payload_as_dict(content: object) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return cast(dict[str, Any], content)
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_result(
    message: str,
    *,
    backend: str | None = None,
    phase: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "error": limit_output(message or "Notebook analysis failed", 2_000)
    }
    if backend:
        payload["backend"] = backend
    if phase:
        payload["phase"] = phase
    return json_dumps(payload)


def _dedupe_images(images: list[bytes]) -> list[bytes]:
    selected: list[bytes] = []
    hashes: set[str] = set()
    for image in images:
        digest = hashlib.sha256(image).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        selected.append(image)
    return selected


def _render_plot(console: Console, image_data: bytes, *, index: int) -> None:
    if console.color_system is None:
        console.print(f"[Plot {index} omitted: terminal color is unavailable]")
        return
    try:
        with Image.open(io.BytesIO(image_data)) as source:
            source.load()
            image = source.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        console.print(f"[Plot {index} omitted: invalid PNG]")
        return

    max_columns = max(1, min(_MAX_PLOT_COLUMNS, console.width - 4))
    width = min(max_columns, image.width)
    pixel_height = max(2, round(image.height * width / max(1, image.width)))
    pixel_height = min(pixel_height, _MAX_PLOT_ROWS * 2)
    if pixel_height % 2:
        pixel_height += 1
    image.thumbnail((width, pixel_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    rgb = canvas.convert("RGB")

    console.print(f"[bold]Plot {index}[/bold]")
    for y in range(0, rgb.height, 2):
        line = Text()
        bottom_y = min(y + 1, rgb.height - 1)
        for x in range(rgb.width):
            top = cast(tuple[int, int, int], rgb.getpixel((x, y)))
            bottom = cast(tuple[int, int, int], rgb.getpixel((x, bottom_y)))
            line.append(
                "▀",
                style=Style(
                    color=Color.from_rgb(*top),
                    bgcolor=Color.from_rgb(*bottom),
                ),
            )
        console.print(line)
