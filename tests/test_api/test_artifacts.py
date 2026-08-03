from __future__ import annotations

import pytest

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactUnavailable,
    InMemoryArtifactStore,
    LoadedArtifact,
)
from sqlsaber.config.settings import Config


class TenantArtifactStore:
    def __init__(self) -> None:
        self._store = InMemoryArtifactStore()
        self.contexts: list[ArtifactContext] = []

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ):
        return await self._store.publish(bundle, context=context)

    async def get(
        self,
        artifact_id: str,
        *,
        context: ArtifactContext,
    ) -> LoadedArtifact:
        self.contexts.append(context)
        if context.metadata.get("tenant_id") != "acme":
            raise ArtifactUnavailable()
        return await self._store.get(artifact_id, context=context)


async def test_sdk_retrieval_uses_current_application_context() -> None:
    store = TenantArtifactStore()
    publication = await store.publish(
        ArtifactBundle(
            kind="report",
            artifacts=(Artifact("report.csv", b"a,b\n1,2\n", "text/csv"),),
        ),
        context=ArtifactContext(metadata={"tenant_id": "acme"}),
    )
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=Config.in_memory(
                model_name="anthropic:claude-3-5-sonnet",
                api_keys={"anthropic": "test-key"},
            ),
            artifact_store=store,
        )
    )

    loaded = await saber.get_artifact(
        publication.artifacts[0],
        conversation_id="conversation-1",
        metadata={"tenant_id": "acme"},
    )

    assert loaded.data == b"a,b\n1,2\n"
    assert store.contexts[-1].conversation_id == "conversation-1"
    assert store.contexts[-1].metadata == {"tenant_id": "acme"}

    with pytest.raises(ArtifactUnavailable):
        await saber.get_artifact(
            publication.artifacts[0].id,
            metadata={"tenant_id": "other"},
        )
    await saber.close()
