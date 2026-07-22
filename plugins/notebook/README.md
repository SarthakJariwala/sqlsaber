# SQLsaber Notebook

Notebook-specific data-analysis subagent for SQLsaber.

Implemented components:

- provider-neutral notebook execution contract,
- hardened local Docker execution (default),
- explicit remote Modal Sandbox execution through the optional `modal` extra,
- fresh-kernel transactional notebook sessions,
- bounded notebook/image rendering and history collapse,
- `list_workspace` and `edit_cell` analyst tools,
- a Pydantic AI notebook analyst,
- a managed SQLsaber `analyze_data` capability, and
- the standalone `sqlsaber-notebook` CLI.

When installed with SQLsaber, the main agent can hand prior successful SQL results to
`analyze_data` for multi-step calculations, statistics, transformations, and plots.
The terminal displays the bounded executed notebook and plot previews before the main
agent's text response. Notebook bytes and images are not sent to the parent model.
Managed SDK applications can persist the notebook, plots, and generated files through
`SQLSaberOptions.artifact_publisher`; only the publisher's durable references are
stored in tool metadata and exposed through `SQLSaberResult.artifacts`.

The default balanced runtime targets larger EDA and classical ML: 4 CPUs, 8 GiB
memory, and up to 100 MiB per input/250 MiB total. SQLsaber does not cap model
requests, notebook cell count, the analyst loop, or the whole operation. Individual
cells retain a 10-minute timeout so a stuck computation can be diagnosed without
ending the overall analysis. These are fixed product defaults rather than CLI tuning
flags. Use an immutable custom image through `SQLSABER_NOTEBOOK_IMAGE` when
additional ML libraries are required.

## Managed SQLsaber usage

```bash
uv tool install --with sqlsaber-notebook sqlsaber
saber
```

Docker is the default local backend. Select Modal explicitly because query results
will be uploaded to a third party:

```bash
SQLSABER_NOTEBOOK_BACKEND=modal saber
```

Configure a dedicated analyst model with:

```bash
saber models set --agent notebook
```

For a web backend, inject an application-owned artifact publisher and pass tenant
scope as run metadata:

```python
from sqlsaber import FilesystemArtifactPublisher, SQLSaber, SQLSaberOptions

options = SQLSaberOptions(
    database="sqlite:///analytics.db",
    artifact_publisher=FilesystemArtifactPublisher("/private/artifacts"),
)

async with SQLSaber(options=options) as saber:
    result = await saber.query(
        "Analyze and plot revenue anomalies",
        conversation_id="conversation-123",
        metadata={"tenant_id": "acme"},
    )
    print(result.artifacts)
```

Implement the cloud-neutral `ArtifactPublisher` protocol to use S3, GCS, Azure
Blob Storage, or another bucket. Return stable object references rather than
expiring signed URLs.

## Standalone usage

```bash
uv run sqlsaber-notebook \
  --model anthropic:claude-sonnet-4-6 \
  --backend docker \
  --output analysis.ipynb \
  "Compare revenue by region and explain material anomalies" data.csv
```

Modal is never selected as an automatic fallback. Select it explicitly because local
files will be uploaded to Modal:

```bash
modal setup
SQLSABER_NOTEBOOK_BACKEND=modal uv run sqlsaber-notebook \
  --model anthropic:claude-sonnet-4-6 \
  "Analyze this dataset" data.csv
```

## Development

```bash
uv sync
uv run pytest plugins/notebook/tests -q
```

Run credentialed remote integration tests explicitly:

```bash
SQLSABER_RUN_MODAL_INTEGRATION=1 \
  uv run pytest plugins/notebook/tests/test_notebook_modal_integration.py -q
```
