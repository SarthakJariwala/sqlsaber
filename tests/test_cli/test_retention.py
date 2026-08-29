from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlsaber.cli import retention


@pytest.mark.asyncio
async def test_query_result_retention_failure_is_logged_and_artifacts_still_run(
    monkeypatch,
) -> None:
    query_results = AsyncMock(side_effect=RuntimeError("query result failure"))
    artifacts = AsyncMock()
    logger = MagicMock()
    monkeypatch.setattr(
        "sqlsaber.cli.query_result_gc.collect_cli_query_results", query_results
    )
    monkeypatch.setattr("sqlsaber.cli.artifact_gc.collect_cli_artifacts", artifacts)
    monkeypatch.setattr(retention, "logger", logger)
    thread_storage = MagicMock()
    artifact_store = MagicMock()
    query_result_store = MagicMock()

    await retention.run_cli_retention(
        thread_storage,
        artifact_store,
        query_result_store,
    )

    query_results.assert_awaited_once_with(thread_storage, query_result_store)
    artifacts.assert_awaited_once_with(thread_storage, artifact_store)
    logger.warning.assert_called_once_with(
        "cli.retention.query_results_failed", error="query result failure"
    )


@pytest.mark.asyncio
async def test_artifact_retention_failure_is_logged_and_suppressed(monkeypatch) -> None:
    query_results = AsyncMock()
    artifacts = AsyncMock(side_effect=RuntimeError("artifact failure"))
    logger = MagicMock()
    monkeypatch.setattr(
        "sqlsaber.cli.query_result_gc.collect_cli_query_results", query_results
    )
    monkeypatch.setattr("sqlsaber.cli.artifact_gc.collect_cli_artifacts", artifacts)
    monkeypatch.setattr(retention, "logger", logger)
    thread_storage = MagicMock()
    artifact_store = MagicMock()
    query_result_store = MagicMock()

    await retention.run_cli_retention(
        thread_storage,
        artifact_store,
        query_result_store,
    )

    query_results.assert_awaited_once_with(thread_storage, query_result_store)
    artifacts.assert_awaited_once_with(thread_storage, artifact_store)
    logger.warning.assert_called_once_with(
        "cli.retention.artifacts_failed", error="artifact failure"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_failure", ["query failure", "close failure"])
async def test_retention_failures_do_not_mask_primary_failure(
    monkeypatch,
    primary_failure: str,
) -> None:
    query_results = AsyncMock(side_effect=RuntimeError("query result retention"))
    artifacts = AsyncMock(side_effect=RuntimeError("artifact retention"))
    monkeypatch.setattr(
        "sqlsaber.cli.query_result_gc.collect_cli_query_results", query_results
    )
    monkeypatch.setattr("sqlsaber.cli.artifact_gc.collect_cli_artifacts", artifacts)
    monkeypatch.setattr(retention, "logger", MagicMock())

    async def fail_with_retention() -> None:
        try:
            raise RuntimeError(primary_failure)
        finally:
            await retention.run_cli_retention(
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

    with pytest.raises(RuntimeError, match=primary_failure):
        await fail_with_retention()

    query_results.assert_awaited_once()
    artifacts.assert_awaited_once()
