"""Opt-in provider contract: live kernel, ML weights, plots, and cleanup.

Run with SQLSABER_SANDBOX_INTEGRATION=1 and provider credentials. Select another
provider with SQLSABER_TEST_SANDBOX_PROVIDER. These tests create paid sandboxes.
"""

import asyncio
import hashlib
import os

import nbformat
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits
import pytest

from sqlsaber.artifacts import ArtifactContext, FilesystemArtifactStore
from sqlsaber_sandbox import SandboxConfig, SandboxSession, Workspace, publish_analysis

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("SQLSABER_SANDBOX_INTEGRATION") != "1",
        reason="live sandbox test requires explicit opt-in",
    ),
]


async def test_live_persistence_ml_and_artifacts(tmp_path):
    config = SandboxConfig(
        provider=os.getenv("SQLSABER_TEST_SANDBOX_PROVIDER", "e2b"),
        max_lifetime_seconds=600,
        cell_seconds=30,
    )
    cells = [
        "import json, numpy as np\nfrom pathlib import Path\nx=np.array(json.loads(Path('../inputs/x.json').read_text()))\ny=np.array(json.loads(Path('../inputs/y.json').read_text()))\nweights=np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]),y,rcond=None)[0]\ncounter=1\nweights.tolist()",
        "counter += 1\nassert counter == 2\nassert abs(weights @ [10,1] - 32) < 1e-9\nnp.savez('weights.npz', weights=weights)\nPath('binary.bin').write_bytes(bytes([0,255,13,128,42]))\nimport matplotlib.pyplot as plt\nplt.plot(x,y)\nplt.show()",
    ]
    if config.provider in {"docker", "microsandbox", "modal", "daytona"}:
        cells[0] += (
            "\nfrom scipy.linalg import lstsq"
            "\nfrom sklearn.linear_model import LinearRegression"
            "\nassert np.allclose(lstsq(np.column_stack([x, np.ones(len(x))]), y)[0], [3, 2])"
            "\nassert abs(LinearRegression().fit(x.reshape(-1, 1), y).predict([[10]])[0] - 32) < 1e-9"
        )
    calls = 0

    def analyst(messages, info):
        nonlocal calls
        step = calls
        calls += 1
        if step % 2 == 0:
            if step == 2:
                assert any(
                    isinstance(p, TextPart)
                    for m in messages
                    if isinstance(m, ModelResponse)
                    for p in m.parts
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_python",
                        {"code": cells[step // 2]},
                        tool_call_id=f"cell-{step}",
                    )
                ]
            )
        returned = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        assert returned[-1].content["status"] == "ok"
        return ModelResponse(parts=[TextPart("Linear fit predicts 32 at x=10")])

    # Only the model boundary is scripted. Native provisioning, the child tool
    # loop, kernel, transfers, export and cleanup all execute for real.
    session = SandboxSession(model=FunctionModel(analyst), config=config)
    async with session:
        first = await session.analyze(
            "Fit a trend from both input files",
            workspace=Workspace.from_files(
                [("x.json", b"[1,3,7]"), ("y.json", b"[5,11,23]")]
            ),
            usage_limits=UsageLimits(request_limit=2),
        )
        assert first.cells[-1].status == "ok"
        snapshot = await session.analyze(
            "Predict at x=10 using the existing fit, and save weights and a plot",
            usage_limits=UsageLimits(request_limit=2),
        )
        assert len(snapshot.cells) == 2
        assert snapshot.cells[-1].status == "ok"
        files = {item.name: item for item in snapshot.files}
        assert files["binary.bin"].data == bytes([0, 255, 13, 128, 42])
        assert files["weights.npz"].data.startswith(b"PK")
        assert any(
            item.media_type == "image/png" and item.data.startswith(b"\x89PNG")
            for item in files.values()
        )
        nbformat.validate(nbformat.reads(snapshot.notebook.decode(), as_version=4))
        # Kernel exceptions preserve partial state without replaying prior cells.
        error = await session.execute(
            "counter += 5\nraise ValueError('expected test error')"
        )
        assert error.status == "error"
        resumed = await session.execute("assert counter == 7\nprint('retained')")
        assert resumed.status == "ok"
        # Exercise the same controller's deduplication after a dropped acknowledgement.
        execution_id = "deduplicated-test"
        await session._execution.request(
            "execute", execution_id=execution_id, code="counter += 11"
        )
        await session._execution.request(
            "execute", execution_id=execution_id, code="counter += 11"
        )
        while (await session._execution.request("status", execution_id=execution_id))[
            "status"
        ] == "running":
            await asyncio.sleep(0.1)
        assert (await session.execute("assert counter == 18")).status == "ok"
    assert session.closed
    assert getattr(session._execution.backend, "_sandbox", None) is None
    # Publication consumes the exported snapshot after the provider is destroyed.
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        snapshot, store=store, context=ArtifactContext()
    )
    binary = next(
        item for item in publication.artifacts if item.name == "files/binary.bin"
    )
    loaded = await store.get(binary.id, context=ArtifactContext())
    assert loaded.data == bytes([0, 255, 13, 128, 42])
    assert binary.sha256 == hashlib.sha256(loaded.data).hexdigest()
