# SQLsaber Notebook

Notebook-specific data-analysis subagent for SQLsaber.

Implemented components:

- provider-neutral notebook execution contract,
- hardened local Docker execution (default),
- explicit remote Modal Sandbox execution through the optional `modal` extra,
- fresh-kernel transactional notebook sessions,
- bounded notebook/image rendering and history collapse,
- `list_workspace` and `edit_cell` analyst tools,
- a Pydantic AI notebook analyst, and
- the standalone `sqlsaber-notebook` CLI.

The managed SQLsaber `analyze_data` capability is added in the next integration phase.

The default balanced runtime targets larger EDA and classical ML: 4 CPUs, 8 GiB
memory, and up to 100 MiB per input/250 MiB total. SQLsaber does not cap model
requests, notebook cell count, the analyst loop, or the whole operation. Individual
cells retain a 10-minute timeout so a stuck computation can be diagnosed without
ending the overall analysis. These are fixed product defaults rather than CLI tuning
flags. Use an immutable custom image through `SQLSABER_NOTEBOOK_IMAGE` when
additional ML libraries are required.

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
