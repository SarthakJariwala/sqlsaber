from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from sqlsaber.api import SQLSaberResult
from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactPublicationError,
    ArtifactUnavailable,
    FilesystemArtifactStore,
    InMemoryArtifactStore,
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


async def test_in_memory_store_publishes_and_retrieves_exact_bytes() -> None:
    store = InMemoryArtifactStore()
    context = ArtifactContext(run_id="run-1", metadata={"tenant_id": "acme"})

    publication = await store.publish(_bundle(), context=context)
    loaded = await store.get(publication.artifacts[0].id, context=context)

    assert loaded.descriptor == publication.artifacts[0]
    assert loaded.data == b"notebook"


async def test_in_memory_store_returns_serializable_references() -> None:
    store = InMemoryArtifactStore()
    context = ArtifactContext(run_id="run-1", metadata={"tenant_id": "acme"})

    publication = await store.publish(_bundle(), context=context)

    assert [artifact.kind for artifact in publication.artifacts] == [
        "notebook",
        "image",
    ]
    assert publication.artifacts[0].uri.startswith("memory://")
    assert artifacts_from_metadata(publication.to_metadata()) == list(
        publication.artifacts
    )


async def test_filesystem_store_retrieves_after_restart(tmp_path) -> None:
    context = ArtifactContext(run_id="run-1", conversation_id="conversation-1")
    publication = await FilesystemArtifactStore(tmp_path).publish(
        _bundle(), context=context
    )

    loaded = await FilesystemArtifactStore(tmp_path).get(
        publication.artifacts[0].id,
        context=context,
    )

    assert loaded.descriptor == publication.artifacts[0]
    assert loaded.data == b"notebook"


async def test_filesystem_store_rejects_symlinked_root(tmp_path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(
        ArtifactPublicationError,
        match="Artifacts could not be published",
    ):
        await FilesystemArtifactStore(linked_root).publish(
            _bundle(), context=ArtifactContext()
        )


async def test_filesystem_store_rejects_changed_bytes(tmp_path) -> None:
    context = ArtifactContext()
    store = FilesystemArtifactStore(tmp_path)
    publication = await store.publish(_bundle(), context=context)
    artifact = publication.artifacts[0]
    Path(url2pathname(urlparse(artifact.uri).path)).write_bytes(b"changed")

    with pytest.raises(ArtifactUnavailable, match="Artifact is unavailable"):
        await store.get(artifact.id, context=context)


async def test_filesystem_store_rejects_symlinked_shard(tmp_path) -> None:
    context = ArtifactContext()
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await store.publish(_bundle(), context=context)
    shard = store.root / publication.id[3:5]
    moved_shard = tmp_path / "moved-shard"
    shard.rename(moved_shard)
    shard.symlink_to(moved_shard, target_is_directory=True)

    with pytest.raises(ArtifactUnavailable):
        await store.get(publication.artifacts[0].id, context=context)


async def test_filesystem_store_rejects_symlinked_artifact(tmp_path) -> None:
    context = ArtifactContext()
    store = FilesystemArtifactStore(tmp_path)
    publication = await store.publish(_bundle(), context=context)
    artifact = publication.artifacts[0]
    artifact_path = Path(url2pathname(urlparse(artifact.uri).path))
    replacement = tmp_path / "replacement.ipynb"
    replacement.write_bytes(b"notebook")
    artifact_path.unlink()
    artifact_path.symlink_to(replacement)

    with pytest.raises(ArtifactUnavailable):
        await store.get(artifact.id, context=context)


async def test_filesystem_store_rejects_malformed_descriptor(tmp_path) -> None:
    context = ArtifactContext()
    store = FilesystemArtifactStore(tmp_path)
    publication = await store.publish(_bundle(), context=context)
    manifest_path = tmp_path / publication.id[3:5] / publication.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["media_type"] = ""
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactUnavailable):
        await store.get(publication.artifacts[0].id, context=context)


async def test_filesystem_store_writes_allowlisted_context_to_manifest(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path)

    publication = await store.publish(
        _bundle(),
        context=ArtifactContext(
            run_id="run-1",
            conversation_id="conversation-1",
            tool_call_id="tool-1",
            metadata={"tenant_id": "acme"},
        ),
    )

    target = tmp_path / publication.id[3:5] / publication.id
    assert (target / "artifacts" / "analysis.ipynb").read_bytes() == b"notebook"
    assert (target / "artifacts" / "plots" / "plot_1.png").read_bytes() == b"png"
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["context"] == {
        "conversation_id": "conversation-1",
        "run_id": "run-1",
        "tool_call_id": "tool-1",
    }
    assert "tenant_id" not in json.dumps(manifest)
    assert [item["kind"] for item in manifest["artifacts"]] == [
        "notebook",
        "image",
    ]


async def test_filesystem_store_retrieves_artifact_named_manifest(tmp_path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    artifact_data = b"user data"
    bundle = ArtifactBundle(
        kind="test",
        artifacts=(Artifact("manifest.json", artifact_data, "application/json"),),
    )

    publication = await store.publish(bundle, context=ArtifactContext())
    loaded = await FilesystemArtifactStore(tmp_path).get(
        publication.artifacts[0].id,
        context=ArtifactContext(),
    )

    assert loaded.data == artifact_data
    assert loaded.descriptor.size == len(artifact_data)
    assert loaded.descriptor.sha256 == hashlib.sha256(artifact_data).hexdigest()


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
        "artifact_publication": {
            "id": "ap_11111111111111111111111111111111",
            "kind": "notebook-analysis",
            "artifacts": [
                {
                    "id": "ar_22222222222222222222222222222222",
                    "name": "analysis.ipynb",
                    "media_type": "application/x-ipynb+json",
                    "size": 8,
                    "sha256": hashlib.sha256(b"notebook").hexdigest(),
                    "uri": "s3://bucket/analysis.ipynb",
                    "kind": "notebook",
                }
            ],
        }
    }
    result = SQLSaberResult("answer", _RunResult(metadata))

    assert result.artifacts[0].uri == "s3://bucket/analysis.ipynb"
    assert result.artifacts[0].kind == "notebook"
