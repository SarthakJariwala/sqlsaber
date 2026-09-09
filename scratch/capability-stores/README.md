# Capability store prototypes (throwaway)

These scripts compare ways to give a host pydantic-ai `Agent` the same query-result store and retrieve path that `SQLSaber` exposes.

The production API is now an explicit `SQLSaberOptions.capabilities` list plus a required `PluginContext.query_result_store`. `SQLSaber` does not call `discover_capabilities`. The CLI passes `load_capability_factories()`. `Sandbox` and `VizTool` no longer allocate a private store.

They are not package code. Run them with:

```bash
uv run python scratch/capability-stores/compare.py
uv run python scratch/capability-stores/tutorial_agent.py
uv run python scratch/capability-stores/plugin_share.py
```

## Decision

How should a host `Agent` share one `QueryResultStore` across `SqlTools`, notebook, viz, and sandbox, and load complete rows after `agent.run()`?

## Observed output

`compare.py` seeded 250 wide customer rows and drove `execute_sql` then a stand-in `peek_sql_results` tool.

| variant | host rows | peek rows | missing | model truncated | stores shared |
| --- | --- | --- | --- | --- | --- |
| split default stores | 250 | 0 | 1 | True | False |
| A thread the store | 250 | 250 | 0 | True | True |
| B `SqlSaberKit` | 250 | 250 | 0 | True | True |
| C `CapabilityRun.adopt` | 250 | 250 | 0 | True | True |

Variant B also constructed installed `Notebook` and `Visualization` plugins with the same store id. Sandbox was loaded by entry point and returned no capability because no sandbox provider was configured.

`tutorial_agent.py` printed `preview_row_count 165` and `stored_row_count 250`.

## Recommendation

Use A for a host that only attaches `SqlTools`. Pass `query_result_store` into the constructor and read descriptors with `query_result_references_from_messages`.

Use B when notebook, viz, or sandbox must see those rows. Own the store, `SqlTools`, and `PluginContext` on one session object. Put `get_query_result` on that object. That is C grafted onto B.

Do not put the store on `ctx.deps`. `SqlTools` is dependency-agnostic.

Do not construct `Sandbox()` or a second `InMemoryQueryResultStore()` beside `SqlTools`. That is the split row in the table.
