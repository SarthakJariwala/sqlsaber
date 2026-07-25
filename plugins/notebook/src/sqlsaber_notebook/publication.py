"""Publication and replay adapters for completed notebook analyses."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from sqlsaber.artifact_resolution import ResolvedArtifactPublication
from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactPublication,
    ArtifactStore,
    ArtifactUnavailable,
    StoredArtifact,
)

from ._shared import MAX_SNAPSHOT_IMAGE_BYTES, MAX_SNAPSHOT_IMAGES
from .rendering import render_notebook_bytes
from .result import AnalysisResult


@dataclass(frozen=True, slots=True)
class PublishedAnalysisDisplay:
    """Presentation-neutral reconstruction of one durable notebook analysis."""

    markdown: str
    images: tuple[bytes, ...]
    files: tuple[StoredArtifact, ...]
    unavailable: tuple[StoredArtifact, ...] = ()


async def publish_analysis(
    result: AnalysisResult,
    *,
    store: ArtifactStore,
    context: ArtifactContext,
) -> ArtifactPublication:
    """Publish a completed analysis using the canonical notebook artifact shape."""

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
    bundle = ArtifactBundle(
        kind="notebook-analysis",
        artifacts=tuple(artifacts),
        metadata={"provenance": result.provenance},
    )
    return await store.publish(bundle, context=context)


def display_from_publication(
    publication: ResolvedArtifactPublication,
) -> PublishedAnalysisDisplay:
    """Reconstruct a bounded notebook display from verified publication members."""

    reference = publication.reference
    if reference.publication_kind != "notebook-analysis":
        raise ArtifactUnavailable()

    notebooks: list[StoredArtifact] = []
    plots: list[tuple[int, StoredArtifact]] = []
    files: list[StoredArtifact] = []
    for descriptor in reference.artifacts:
        if descriptor.name == "analysis.ipynb":
            if (
                descriptor.kind != "notebook"
                or descriptor.media_type != "application/x-ipynb+json"
            ):
                raise ArtifactUnavailable()
            notebooks.append(descriptor)
            continue
        plot_match = re.fullmatch(r"plots/plot_([1-9][0-9]*)\.png", descriptor.name)
        if plot_match is not None:
            if descriptor.kind != "image" or descriptor.media_type != "image/png":
                raise ArtifactUnavailable()
            plots.append((int(plot_match.group(1)), descriptor))
            continue
        if descriptor.name.startswith("files/") and descriptor.kind == "file":
            files.append(descriptor)
            continue
        raise ArtifactUnavailable()

    plots.sort(key=lambda item: item[0])
    if len(notebooks) != 1 or [index for index, _ in plots] != list(
        range(1, len(plots) + 1)
    ):
        raise ArtifactUnavailable()

    loaded_by_id = {
        artifact.descriptor.id: artifact for artifact in publication.artifacts
    }
    notebook = loaded_by_id.get(notebooks[0].id)
    if notebook is None:
        raise ArtifactUnavailable()
    try:
        markdown, notebook_images = render_notebook_bytes(notebook.data)
    except ValueError as exc:
        raise ArtifactUnavailable() from exc

    images: list[bytes] = []
    digests: set[str] = set()
    image_bytes = 0
    for image in notebook_images:
        if (
            len(images) >= MAX_SNAPSHOT_IMAGES
            or image_bytes + len(image) > MAX_SNAPSHOT_IMAGE_BYTES
        ):
            break
        digests.add(hashlib.sha256(image).hexdigest())
        images.append(image)
        image_bytes += len(image)
    for _, descriptor in plots:
        loaded = loaded_by_id.get(descriptor.id)
        if loaded is None:
            continue
        digest = hashlib.sha256(loaded.data).hexdigest()
        if digest in digests:
            continue
        if (
            len(images) >= MAX_SNAPSHOT_IMAGES
            or image_bytes + len(loaded.data) > MAX_SNAPSHOT_IMAGE_BYTES
        ):
            break
        digests.add(digest)
        images.append(loaded.data)
        image_bytes += len(loaded.data)

    return PublishedAnalysisDisplay(
        markdown=markdown,
        images=tuple(images),
        files=tuple(files),
        unavailable=publication.unavailable,
    )
