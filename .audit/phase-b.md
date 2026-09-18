# Phase B workflow (draft, pending explainer + why)

Riskiest unknown first. Architect/arena owns unit 0.

0. Design sketch (architect/arena). Competing shapes for nested plugin model ownership. Must honor grounding constraints. Handoff stays core.
1. Verification harness first. Extend `scripts/verify_plugin_settings.py` and pytest for current behavior as baseline (old `--agent viz|notebook` still works). Capture `saber models current` and `saber plugins show notebook`.
2. Data shape. Optional model on each plugin runtime config. `Setting(name="model", kind="text")` on notebook/sandbox/viz declarations. Viz gets a settings entry.
3. Resolution. One function. Session `tool_overrides[tool_name]` wins. Then capability config model. Then migrated leftover `subagents` key only if we keep a compatibility read. Then main model. Viz and sandbox must use this path, not a fresh `Config()`.
4. CLI. `plugins setup|set|unset|show` round-trip the field. `saber models set --agent` accepts only `main` and `handoff`. Invalid plugin names point at `saber plugins setup NAME`.
5. Migrate or delete saved `subagents.viz` / `subagents.notebook`. Prefer one-time copy into plugin_config then drop the keys.
6. SDK docs. Show `partial(capability, config=NotebookConfig(model=...))`. Keep `tool_overrides` as session override.
7. Prove. pytest, `scripts/verify_plugin_settings.py`, verify-sqlsaber plugins + models features, `time uv run saber --help`.
