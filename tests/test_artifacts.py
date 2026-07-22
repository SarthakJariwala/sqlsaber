from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from sqlsaber.api import SQLSaberResult
from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    FilesystemArtifactPublisher,
    InMemoryArtifactPublisher,
    artifacts_from_metadata,
)


def _bundle() -> ArtifactBundle:
    return ArtifactBundle(
        kind="notebook-analysis",
        artifacts=(
            Artifact(
                "analysis.ipynb",
                b"notebook",
                "application/x-ipynb+json",
                "notebook",
            ),
            Artifact("plots/plot_1.png", b"png", "image/png", "image"),
        ),
        metadata={"model": "test:model"},
    )


async def test_in_memory_publisher_returns_serializable_references() -> None:
    publisher = InMemoryArtifactPublisher()
    context = ArtifactContext(run_id="run-1", metadata={"tenant_id": "acme"})

    publication = await publisher.publish(_bundle(), context=context)

    assert publisher.publications[publication.id] == (_bundle(), context)
    assert [artifact.kind for artifact in publication.artifacts] == [
        "notebook",
        "image",
    ]
    assert publication.artifacts[0].uri.startswith("memory://")
    assert artifacts_from_metadata(publication.to_metadata()) == list(
        publication.artifacts
    )


async def test_filesystem_publisher_writes_atomic_bundle_and_manifest(tmp_path) -> None:
    publisher = FilesystemArtifactPublisher(
        tmp_path,
        namespace=lambda context: f"tenants/{context.metadata['tenant_id']}",
    )

    publication = await publisher.publish(
        _bundle(),
        context=ArtifactContext(
            run_id="run-1",
            conversation_id="conversation-1",
            tool_call_id="tool-1",
            metadata={"tenant_id": "acme"},
        ),
    )

    target = tmp_path / "tenants" / "acme" / publication.id
    assert (target / "artifacts" / "analysis.ipynb").read_bytes() == b"notebook"
    assert (target / "artifacts" / "plots" / "plot_1.png").read_bytes() == b"png"
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["run_id"] == "run-1"
    assert manifest["conversation_id"] == "conversation-1"
    assert [item["kind"] for item in manifest["artifacts"]] == [
        "notebook",
        "image",
    ]


async def test_filesystem_publisher_separates_artifact_named_manifest(tmp_path) -> None:
    publisher = FilesystemArtifactPublisher(tmp_path)
    artifact_data = b"user data"
    bundle = ArtifactBundle(
        kind="test",
        artifacts=(Artifact("manifest.json", artifact_data, "application/json"),),
    )

    publication = await publisher.publish(bundle, context=ArtifactContext())

    target = tmp_path / publication.id
    assert (target / "artifacts" / "manifest.json").read_bytes() == artifact_data
    publisher_manifest = json.loads((target / "manifest.json").read_text())
    stored = publication.artifacts[0]
    assert stored.size == len(artifact_data)
    assert stored.sha256 == hashlib.sha256(artifact_data).hexdigest()
    assert publisher_manifest["artifacts"][0] == stored.to_dict()


@pytest.mark.parametrize("name", ["../secret", ".", "./duplicate", "path//file"])
def test_artifact_rejects_unsafe_or_noncanonical_paths(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        Artifact(name, b"value", "text/plain")


class _RunResult:
    def __init__(self, metadata: dict[str, object]) -> None:
        self._messages = [
            ModelRequest(
                parts=[ToolReturnPart("analyze_data", "answer", metadata=metadata)]
            )
        ]

    def new_messages(self):
        return self._messages

    def all_messages(self):
        return self._messages

    def usage(self):
        return SimpleNamespace(requests=1)


def test_sqlsaber_result_exposes_published_artifacts() -> None:
    metadata = {
        "artifacts": [
            {
                "id": "analysis-1:notebook",
                "name": "analysis.ipynb",
                "media_type": "application/x-ipynb+json",
                "size": 8,
                "sha256": "digest",
                "uri": "s3://bucket/analysis.ipynb",
                "kind": "notebook",
            }
        ]
    }
    result = SQLSaberResult("answer", _RunResult(metadata))

    assert result.artifacts[0].uri == "s3://bucket/analysis.ipynb"
    assert result.artifacts[0].kind == "notebook"
