# SQLsaber Notebook

Notebook-specific data-analysis subagent for SQLsaber.

Implemented components:

- provider-neutral notebook execution contract,
- hardened local Docker execution (default),
- explicit local microVM execution through the optional `microsandbox` extra,
- explicit remote Modal Sandbox execution through the optional `modal` extra,
- explicit remote Daytona execution through the pinned `daytona` extra,
- fresh-kernel transactional notebook sessions,
- bounded notebook/image rendering and history collapse,
- `list_workspace` and `edit_cell` analyst tools,
- a Pydantic AI notebook analyst,
- a managed SQLsaber `analyze_data` capability,
- reusable artifact publication and persisted notebook replay, and
- the standalone `sqlsaber-notebook` CLI.

When installed with SQLsaber, the main agent can hand prior successful SQL results to
`analyze_data` for multi-step calculations, statistics, transformations, and plots.
The terminal displays the bounded executed notebook and plot previews before the main
agent's text response. Notebook bytes and images are not sent to the parent model.
Managed SDK applications can persist the notebook, plots, and generated files through
`SQLSaberOptions.artifact_store`; only the store's durable references are stored
in tool metadata and exposed through `SQLSaberResult.artifacts`.

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

Docker is the default local backend. Microsandbox is an opt-in local backend that
runs notebook code in a hardware-isolated Linux microVM without host bind mounts:

```bash
uv tool install --with 'sqlsaber-notebook[microsandbox]' sqlsaber
SQLSABER_NOTEBOOK_BACKEND=microsandbox saber
```

Microsandbox 0.6 is beta. It supports Apple Silicon macOS, Linux x86_64/ARM64 with
usable KVM, and preview Windows x86_64/ARM64 hosts with Windows Hypervisor Platform.
Intel macOS is not supported. Its OCI cache under `~/.microsandbox` is separate from
Docker, so the first image preparation can be large and slow. For private or
overridden registry images, authenticate them with Microsandbox's registry login
support or set `SQLSABER_NOTEBOOK_IMAGE` to an immutable digest in an accessible
registry.
Guest networking is disabled, the restricted security profile is requested, and
notebook processes receive a process-level PID rlimit. Microsandbox runs locally;
query results are not uploaded to a third-party sandbox service.

Select Modal explicitly because query results will be uploaded to a third party:

```bash
SQLSABER_NOTEBOOK_BACKEND=modal saber
```

Daytona is also an explicit remote backend. The deployed legacy control plane requires
exactly `daytona==0.143.0`, which is installed by the extra:

```bash
uv tool install --with 'sqlsaber-notebook[daytona]' sqlsaber
export DAYTONA_API_KEY=...
export DAYTONA_API_URL=https://your-daytona.example/api
SQLSABER_NOTEBOOK_BACKEND=daytona saber
```

SQL query results and selected local files are uploaded to the configured Daytona
service. SQLsaber derives a minimal `USER root` control image from the exact configured
`SQLSABER_NOTEBOOK_IMAGE` parent, protects inputs as root, and executes notebooks as
`jovyan`. The sandbox requests blocked outbound networking and ephemeral deletion.
Daytona 0.143.0 has no hard age-based TTL: its 24-hour setting is inactivity-based.
Deployments requiring a strict maximum resource age must run a label-based reaper for
sandboxes labeled `application=sqlsaber,purpose=notebook`.

Backend isolation differs by provider:

| Backend | Location | Guest network | CPU/memory units | PID limit | Abandonment cleanup |
| --- | --- | --- | --- | --- | --- |
| Docker | Local | Docker `none` | Fractional CPU / MiB | Enforced | Per-run container removal |
| Microsandbox | Local microVM | Disabled | Whole CPU / MiB | Process rlimit | 24-hour max duration |
| Modal | Remote | Blocked | Fractional CPU / MiB | Not exposed | 24-hour platform lifetime |
| Daytona | Remote | Provider block requested | Whole CPU / GiB | Not exposed | Ephemeral 24-hour inactivity stop; no hard TTL |

Daytona image derivation can make the first cold start slower. Network denial, root
input ownership, and deletion are verified by credentialed tests, but do not assume
PID-limit or complete isolation parity across providers.

Backends never fall back automatically after selection or failure.

Configure a dedicated analyst model with:

```bash
saber models set --agent notebook
```

For a web backend, inject an application-owned artifact store and pass tenant scope
as run metadata:

```python
from sqlsaber import FilesystemArtifactStore, SQLSaber, SQLSaberOptions

options = SQLSaberOptions(
    database="sqlite:///analytics.db",
    artifact_store=FilesystemArtifactStore("/private/artifacts"),
)

async with SQLSaber(options=options) as saber:
    result = await saber.query(
        "Analyze and plot revenue anomalies",
        conversation_id="conversation-123",
        metadata={"tenant_id": "acme"},
    )
    print(result.artifacts)
```

Implement the cloud-neutral `ArtifactStore` protocol to use a private database plus
S3, GCS, Azure Blob Storage, or another bucket. Authorize `get()` from current run
metadata and return stable private object references rather than expiring signed
URLs.

Managed applications can also configure
`SQLSaberOptions.workspace_input_resolver` to expose authorized private inputs to
`analyze_data`. The model-visible argument is `attachment_refs`, never raw bytes,
paths, URLs, bucket names, or object keys. The resolver receives only run,
conversation, tool-call, and application metadata context and returns ordered
`WorkspaceFile` values. The host owns authorization and must reject invented,
expired, cross-tenant, or out-of-history references; SQLsaber validates filenames,
collisions, immutable bytes, MIME/provenance metadata, and aggregate limits before
starting a notebook. When no resolver is configured, `attachment_refs` is omitted
from the tool schema and existing SQL-only behavior is unchanged.

SQL result files are staged first in their selected order, followed by resolver
outputs in resolver order. The 50-file, 100-MiB-per-file, and 250-MiB-total limits
apply to the combined workspace. Filenames are capped at 255 UTF-8 bytes.
`manifest.json` is reserved, has a separate 1-MiB limit, and records each resolved
file's media type and structured string provenance.

## Direct embedded usage

Analysis and publication are separate operations. This keeps the analyst independent
of SQLsaber storage while giving embedded callers the same canonical publication as
the managed capability:

```python
from sqlsaber import ArtifactContext, FilesystemArtifactStore
from sqlsaber_notebook import Workspace, WorkspaceFile, analyze, publish_analysis

workspace = Workspace.from_files([
    ("sales.csv", sales_csv_bytes),  # Backwards-compatible tuple form
    WorkspaceFile(
        "preview.jpeg",
        preview_bytes,
        media_type="image/jpeg",
        provenance={"attachment_id": "attachment-1"},
    ),
])
result = await analyze(
    "Plot monthly revenue and explain anomalies",
    workspace,
    model="anthropic:claude-sonnet-4-6",
    model_provider="anthropic",
    collect_files=True,
)
publication = await publish_analysis(
    result,
    store=FilesystemArtifactStore("/private/artifacts"),
    context=ArtifactContext(
        conversation_id="conversation-123",
        metadata={"tenant_id": "acme"},
    ),
)
```

`WorkspaceFile` is provider-neutral: embedded callers supply their own trusted
bytes, while the standalone CLI continues to accept explicit local paths. Staged
images are files, not initial multimodal child-model content; the analyst can display
them and inspect the existing bounded PNG snapshots. The tuple form of
`Workspace.from_files` remains supported.

`publish_analysis` writes `analysis.ipynb`, ordered `plots/plot_<n>.png` members,
and bounded generated files below `files/`. It forwards the supplied context to the
application-owned store and raises if publication fails.

## Standalone usage

```bash
uv run sqlsaber-notebook \
  --model anthropic:claude-sonnet-4-6 \
  --backend docker \
  --output analysis.ipynb \
  "Compare revenue by region and explain material anomalies" data.csv
```

Standalone mode writes the explicit `--output` notebook and, when needed, a sibling
`<output-stem>_artifacts` directory. It does not use SQLsaber conversation storage
or its user-data artifact directory.

Remote backends are never selected as automatic fallbacks. Select one explicitly
because local files will be uploaded to that provider:

```bash
modal setup
SQLSABER_NOTEBOOK_BACKEND=modal uv run sqlsaber-notebook \
  --model anthropic:claude-sonnet-4-6 \
  "Analyze this dataset" data.csv

DAYTONA_API_KEY=... DAYTONA_API_URL=https://your-daytona.example/api \
  SQLSABER_NOTEBOOK_BACKEND=daytona uv run sqlsaber-notebook \
  --model anthropic:claude-sonnet-4-6 \
  "Analyze this dataset" data.csv
```

## Development

```bash
uv sync
uv run pytest plugins/notebook/tests -q
```

Run live backend integration tests explicitly:

```bash
SQLSABER_RUN_DOCKER_INTEGRATION=1 \
  uv run pytest plugins/notebook/tests/test_notebook_docker_integration.py -q

SQLSABER_RUN_MICROSANDBOX_INTEGRATION=1 \
  uv run --project plugins/notebook pytest \
  plugins/notebook/tests/test_notebook_microsandbox_integration.py -q

SQLSABER_RUN_MODAL_INTEGRATION=1 \
  uv run pytest plugins/notebook/tests/test_notebook_modal_integration.py -q

SQLSABER_RUN_DAYTONA_INTEGRATION=1 \
  uv run pytest plugins/notebook/tests/test_notebook_daytona_integration.py -q
```
