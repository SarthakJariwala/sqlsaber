# Frame: saber-tui as the only renderer

## Predicate (done when all are true)

1. `uv run python scripts/assert_no_legacy_renderers.py` exits 0.
2. `pyproject.toml` does not list `rich` or `questionary` as direct dependencies.
3. Ruff `TID251` (or equivalent) bans `rich` and `questionary` imports in `src/` and `plugins/`.
4. Interactive `saber` still runs on `ChatApp` / saber-tui. `ChatConsole` no longer subclasses `rich.console.Console`. `append_rich` is gone.
5. Non-interactive TTY output is saber-tui `Component.render` lines. Piped output is unstyled markdown, not Rich markup.
6. Prompts (onboarding, db/auth/theme/models, `confirm_action`) use saber-tui `Input` / `SelectList`, not questionary.
7. `env -u FORCE_COLOR uv run python -m pytest` is green.
8. `time uv run saber --help` stays under 0.5s.
9. `control-sqlsaber` doctor plus database-connections, knowledge-base, and interactive-session recipes still match their stable handles. One-shot query is driven when a provider key is present.

## Scope

About 29 production modules and 11 test modules still import Rich, Questionary, or `create_console`. Call sites number in the hundreds (`console.print` alone is 30+ files). Plugins in scope: `sqlsaber-notebook`, `sqlsaber-viz`. `sqlsaber-sandbox` does not import Rich.

Out of scope:

- Replacing cyclopts `--help` / `--version` (cyclopts 4.11 requires Rich).
- structlog debug logs on stderr.
- HTML thread export internals that never print to a terminal.
- Adding a Table widget to saber-tui itself.

## Rigor

High. The CLI is the product. The change is a one-way door for every user-visible line of output. Gates are scripts and tests, not eyeballing.

## Constraints that already exist

- saber-tui 0.6.0 is a `Component` library (`render(width) -> list[str]`). It has Markdown, Box, Text, Image, SelectList, Input, Editor, Loader. It has no Console, Table, Panel, Live, or Questionary clone.
- Interactive chat already uses native Markdown streaming. One-shot query still uses Rich `Live`.
- Tools already have a TUI path (`render_result_tui`) and a Rich path (`render_result`). Collapse to one.
- Piped / agent-friendly CLI currently emits markdown (`**Error:**`, github tables). That contract stays.
- Destructive commands require `--yes` when stdin is not a TTY.

## Riskiest unknown

How to host questionary-equivalent prompts on saber-tui without a second UI toolkit, including nested prompts during an already-running ChatApp, and how to stream one-shot markdown without Rich Live while keeping pipe-friendly output.
