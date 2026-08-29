"""CLI retention policy for durable query outputs."""

from __future__ import annotations

from sqlsaber.artifacts import FilesystemArtifactStore
from sqlsaber.config.logging import get_logger
from sqlsaber.query_results import FilesystemQueryResultStore
from sqlsaber.threads.storage import ThreadStorage

logger = get_logger(__name__)


async def run_cli_retention(
    thread_storage: ThreadStorage,
    artifact_store: FilesystemArtifactStore,
    query_result_store: FilesystemQueryResultStore,
) -> None:
    """Run rate-limited retention after a CLI conversation closes."""
    try:
        from sqlsaber.cli.query_result_gc import collect_cli_query_results

        await collect_cli_query_results(thread_storage, query_result_store)
    except Exception as exc:
        logger.warning("cli.retention.query_results_failed", error=str(exc))

    try:
        from sqlsaber.cli.artifact_gc import collect_cli_artifacts

        await collect_cli_artifacts(thread_storage, artifact_store)
    except Exception as exc:
        logger.warning("cli.retention.artifacts_failed", error=str(exc))
