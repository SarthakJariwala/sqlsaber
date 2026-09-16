"""Publish completed sandbox analyses through SQLsaber's artifact boundary."""

from __future__ import annotations

from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactPublication,
    ArtifactStore,
)

from .result import AnalysisResult


async def publish_analysis(
    result: AnalysisResult,
    *,
    store: ArtifactStore,
    context: ArtifactContext,
) -> ArtifactPublication:
    """Publish one immutable completed analysis without consulting live state."""

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
            name=f"files/{artifact.name}",
            data=artifact.data,
            media_type=artifact.media_type,
            kind="image" if artifact.media_type.startswith("image/") else "file",
        )
        for artifact in result.files
    )
    return await store.publish(
        ArtifactBundle(kind="sandbox-analysis", artifacts=tuple(artifacts)),
        context=context,
    )
