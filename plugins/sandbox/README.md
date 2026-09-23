# SQLsaber sandbox plugin

`sqlsaber-sandbox` runs Python analysis in a persistent local or remote sandbox.
Give the analysis agent a goal and input files. It runs code and returns findings,
plots, and generated files.

## Configure the CLI

Install the plugin with the extra for your provider. This example installs E2B:

```bash
uv tool install --with 'sqlsaber-sandbox[e2b]' sqlsaber
saber plugins setup sandbox
```

The setup command requires an explicit provider. SQLsaber does not select a cloud
provider from the credentials it finds. The command stores secrets in the OS
credential store and saves non-secret settings separately.

`E2B_API_KEY`, `DAYTONA_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and
`SPRITES_TOKEN` override saved credentials. `DAYTONA_API_URL` overrides the saved
Daytona endpoint. You can also set `SQLSABER_SANDBOX_PROVIDER` to select the
provider for the current process.

For Modal, leave both token fields blank to use Modal's native authentication.
Run `modal token new` first if Modal is not configured. Remote providers receive
the query results and files selected for analysis, and provider charges may apply.

## Start an SDK session

Select a provider with `SandboxConfig`. Pass input files through `Workspace`:

```python
from pathlib import Path

from pydantic_ai.usage import UsageLimits
from sqlsaber_sandbox import SandboxConfig, SandboxSession, Workspace

config = SandboxConfig(provider="e2b", idle_seconds=300, max_lifetime_seconds=900)

async with SandboxSession(
    model="anthropic:claude-sonnet-4-6",
    config=config,
) as session:
    first = await session.analyze(
        "Fit a trend and save a plot",
        workspace=Workspace.from_files([("sales.csv", Path("sales.csv").read_bytes())]),
        usage_limits=UsageLimits(request_limit=12),
    )
    follow_up = await session.analyze(
        "Use the same fitted model to explain the final month",
        usage_limits=UsageLimits(request_limit=6),
    )
```

Reuse the session for follow-up goals. Variables, installed packages, generated
files, and the analysis agent's history remain available until the session closes.
The `async with` block closes the sandbox on exit.

Pass `usage_limits` to limit model requests for each `analyze` call. To run Python
without a model request, call `await session.execute(code, workspace=...)`.

## Save analysis files

Ask the analysis agent to save deliverables in its working directory. The returned
`AnalysisResult` contains notebook bytes and generated files, including plots and
model weights. To export the current state without running another cell, call
`await session.snapshot()`.

Pass the result to your application's `ArtifactStore` to publish the files.
The result remains usable after the session closes:

```python
from sqlsaber.artifacts import ArtifactContext
from sqlsaber_sandbox import publish_analysis

publication = await publish_analysis(
    follow_up,
    store=artifact_store,
    context=ArtifactContext(conversation_id="conversation-123"),
)
```

## Add sandbox analysis to SQLsaber

Register the capability through `SQLSaberOptions.capabilities`. Pass your query
result store and artifact store to share SQL inputs and retain generated files:

```python
from functools import partial

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber_sandbox import SandboxConfig, capability

options = SQLSaberOptions(
    database="analytics",
    query_result_store=query_result_store,
    artifact_store=artifact_store,
    capabilities=[
        partial(
            capability,
            config=SandboxConfig(provider="e2b"),
            model_name="openai:gpt-5-mini",
            api_key="application-managed-key",
        ),
    ],
)

async with SQLSaber(options=options) as saber:
    result = await saber.query("Analyze recent revenue in a sandbox")
```

The main agent delegates goals through `analyze_in_sandbox` and receives the
answer and artifact references. Each completed analysis automatically publishes
its notebook and files to the configured artifact store. The analysis agent keeps
its code iterations in a separate conversation.

To continue an analysis, pass the returned `session_id` to `analyze_in_sandbox`.
`close_sandbox` retries pending publication before releasing the environment.
With the default required-publication policy, a storage failure leaves the session
open and returns an error. To retry publication without closing the session or
rerunning Python, call `publish_sandbox_artifacts`.

Capability shutdown also retries publication, but still attempts to release all
environments if storage fails. Required publication failures raise a cleanup error.

Pass `usage_limits` to `saber.query()` to share a model budget between the main
agent and the analysis agent. For non-SQL inputs, set
`SQLSaberOptions.workspace_input_resolver` to your application's attachment resolver.

For provider requirements, GPU options, resource limits, and session lifetime, see
the [sandbox SDK reference](https://sqlsaber.com/sdk/capabilities/#persistent-sandbox-analysis).
