"""SQLsaber-independent notebook analyst orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from functools import partial

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from ._shared import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    DEFAULT_EXECUTION_LIMITS,
    DEFAULT_OPERATION_TIMEOUT_SECONDS,
    MAX_GOAL_CHARS,
    MAX_MODEL_REQUESTS,
    MAX_SNAPSHOT_IMAGE_BYTES,
    MAX_SNAPSHOT_IMAGES,
)
from .execution import (
    ExecutionLimits,
    NotebookBackend,
    resolve_notebook_backend,
    resolve_notebook_image,
)
from .history import collapse_old_snapshots
from .prompts import ANALYST_SYSTEM_PROMPT, goal_prompt
from .rendering import extract_notebook_images, normalize_png_bytes
from .result import AnalysisResult, ArtifactRef, Workspace
from .session import NotebookSession
from .tools import analyst_toolset


def build_analyst_agent(
    model: Model | str,
    *,
    model_provider: str,
) -> Agent[NotebookSession, str]:
    provider = model_provider.strip().lower()
    return Agent(
        model,
        deps_type=NotebookSession,
        output_type=str,
        instructions=ANALYST_SYSTEM_PROMPT,
        model_settings=_cache_settings(provider),
        capabilities=[
            ProcessHistory(
                partial(
                    collapse_old_snapshots,
                    cache=provider == "anthropic",
                )
            )
        ],
        toolsets=[analyst_toolset()],
    )


async def analyze(
    goal: str,
    workspace: Workspace,
    *,
    model: Model | str,
    model_provider: str,
    backend: NotebookBackend | str | None = None,
    image: str | None = None,
    include_snapshot_images: bool = False,
    collect_files: bool = True,
    execution_limits: ExecutionLimits = DEFAULT_EXECUTION_LIMITS,
    usage_limits: UsageLimits | None = None,
    parent_usage: RunUsage | None = None,
) -> AnalysisResult:
    """Run one bounded notebook-analysis environment and always clean it up."""

    if not goal.strip():
        raise ValueError("Analysis goal cannot be empty")
    if len(goal) > MAX_GOAL_CHARS:
        raise ValueError(f"Analysis goal exceeds {MAX_GOAL_CHARS} characters")
    selected_backend = (
        backend
        if isinstance(backend, NotebookBackend)
        else resolve_notebook_backend(backend)
    )
    session = NotebookSession(
        workspace=workspace,
        backend=selected_backend,
        image=resolve_notebook_image(image),
        execution_limits=execution_limits,
        include_snapshot_images=include_snapshot_images,
    )
    agent = build_analyst_agent(model, model_provider=model_provider)
    child_usage = RunUsage()
    try:
        async with asyncio.timeout(DEFAULT_OPERATION_TIMEOUT_SECONDS):
            await session.ensure_environment()
            async with asyncio.timeout(DEFAULT_ANALYSIS_TIMEOUT_SECONDS):
                result = await agent.run(
                    goal_prompt(goal),
                    deps=session,
                    usage=child_usage,
                    usage_limits=usage_limits
                    or UsageLimits(request_limit=MAX_MODEL_REQUESTS),
                )
            images, files = await _harvest_artifacts(
                session,
                collect_files=collect_files,
            )
            return AnalysisResult(
                answer=result.output,
                notebook=session.notebook_bytes(),
                images=images,
                files=files,
                provenance=_infer_provenance(session),
            )
    finally:
        if parent_usage is not None:
            parent_usage.incr(child_usage)
        await _bounded_shielded_close(session)


def supports_notebook_images(model_name: str, model_provider: str) -> bool:
    """Conservative allowlist for analyst snapshot image input."""

    provider = model_provider.strip().lower()
    name = model_name.lower()
    if provider == "anthropic":
        return "claude" in name
    if provider == "google":
        return "gemini" in name
    if provider == "openai":
        return any(token in name for token in ("gpt-4", "gpt-5", "o3", "o4"))
    return False


def _cache_settings(provider: str) -> ModelSettings:
    if provider == "anthropic":
        return AnthropicModelSettings(
            parallel_tool_calls=False,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
        )
    return ModelSettings(parallel_tool_calls=False)


async def _harvest_artifacts(
    session: NotebookSession,
    *,
    collect_files: bool,
) -> tuple[list[bytes], list[ArtifactRef]]:
    images = extract_notebook_images(session.outputs)
    image_hashes = {hashlib.sha256(image).hexdigest() for image in images}
    image_bytes = sum(map(len, images))
    files: list[ArtifactRef] = []
    if session.environment is None:
        return images, files

    for artifact in session.artifacts:
        is_png = artifact.media_type == "image/png" or artifact.path.lower().endswith(
            ".png"
        )
        if not is_png and not collect_files:
            continue
        data = await session.environment.read_artifact(artifact)
        if is_png:
            normalized = normalize_png_bytes(data)
            if normalized is None:
                continue
            digest = hashlib.sha256(normalized).hexdigest()
            if digest in image_hashes or len(images) >= MAX_SNAPSHOT_IMAGES:
                continue
            if image_bytes + len(normalized) > MAX_SNAPSHOT_IMAGE_BYTES:
                continue
            image_hashes.add(digest)
            image_bytes += len(normalized)
            images.append(normalized)
        elif collect_files:
            files.append(
                ArtifactRef(
                    name=artifact.path,
                    data=data,
                    media_type=artifact.media_type or "application/octet-stream",
                )
            )
    return images, files


def _infer_provenance(session: NotebookSession) -> list[str]:
    provenance: list[str] = []
    for item in session.workspace.files:
        token = f"../inputs/{item.name}"
        if any(token in source or item.name in source for source in session.cells):
            provenance.append(f"input:{item.name}")
    provenance.extend(
        f"cell:{index}" for index, source in enumerate(session.cells) if source.strip()
    )
    return provenance


async def _bounded_shielded_close(session: NotebookSession) -> None:
    task = asyncio.create_task(session.close())
    with contextlib.suppress(Exception):
        async with asyncio.timeout(30):
            await asyncio.shield(task)
