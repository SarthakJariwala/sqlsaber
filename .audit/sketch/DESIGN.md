# Blocks and Surfaces (synthesized)

Base: arena candidate-4. Cross-judge score 23 vs 20/20/14. See
`/tmp/arena-saber-tui-render/JUDGE.md`.

## Problem

SQLsaber prints through three languages: Rich markup, saber-tui components, and
HTML. Most producers branch on `console.is_terminal`. The 2026 migration left a
Rich bridge (`ChatConsole`, `append_rich`, `RichCapture`) so the third language
can be spoken inside the second. One logical output has four implementations.
Interactive and one-shot stream presenters have drifted.

Constraints: saber-tui 0.6.0 is a `Component` library with no Console/Live/ask;
its Markdown renders GFM tables as box-drawing. Piped stdout stays unstyled
markdown (`**Error:**`, GitHub tables, `- **key**: value`). Destructive commands
still need `--yes` off a TTY. `--help` stays under 0.5s. `saber_tui.components`
costs ~262ms and must not load on the help path.

## Caller's view

Producers build frozen `Block` values and hand them to a `Surface`. Nobody asks
whether they are on a TTY.

```python
from sqlsaber.render import AskConfirm, blocks as b, cli_err, cli_out

out, err = cli_out(), cli_err()
out.emit(b.key_values({"Connected to": f"{db_name} ({db_type})", "Model": model_name}))
if allow_dangerous:
    out.emit(b.warn(DANGEROUS_MODE_WARNING, label="DANGEROUS MODE ENABLED"))

run = await AgentStreamPresenter(out, tools=ToolRenderer(registry)).run(...)
```

Tools return `Sequence[Block]`. No console, no `is_terminal`, no `*_tui`, no
`*_html`. HTML export calls `html_of(renderer.result(...))`.

Prompts are asked of the same surface:

```python
if not await out.ask(AskConfirm("Reset theme?", assume_yes=yes,
                                 unavailable_hint="saber theme reset --yes")):
    return
```

## Shape

`Block = Md | Note | Code | Table | KeyValues | Image | Ansi | Panel`

A case exists when the three serializers must treat it differently. `Role` is
metadata on a block, never markup inside its text.

`markdown_source(block)` is the only place a text-bearing block becomes
characters. TTY builds a saber-tui `Markdown` from that string. Pipes write
`md_of`. Test: `md_of([block]) == markdown_source(block)` and the Markdown
component was constructed with that source.

**Grafts applied**

1. Constructor validation (C2). `Md`/`Note`/alt text reject ANSI and C0/C1
   controls. `Image` and `Ansi` require ANSI-free fallback markdown.
2. `ToolRenderContext` (C2). Frozen context instead of growing kwargs on
   `ToolRenderer.result`.
3. Live-region ordering (C1). Any open stream/status is finalized before
   `emit`. Repaint height is clamped to the screen.
4. One TUI per form session (C1). Sequential onboarding asks share a host.
   `AskSecret` still uses `getpass`.
5. `reset_io` (C3). Test seam rebinds `cli_out`/`cli_err`.
6. Lazy-import list (C3). Documented on `render/__init__.py`.

**Policy vs the C4 sketch**

- `KeyValues` serializes as `- **key**: value` lists, not GFM tables. The
  one-formatter invariant still holds: TTY markdown-renders that list.
- Session summary stays stdout on a TTY and is skipped on pipes.
- `Surface.width` is not public. Terminal sizing stays inside the sink.
- No raw-mode `TUI` for one-shot queries. `TerminalSurface` reprints in place.
- No `styled: bool`. No str-only select values. No silent non-TTY prompt.

`Surface` has four members: `emit`, `stream`, `status`, `ask`. Three
implementations: `ChatSurface`, `TerminalSurface`, `PlainSurface`. Chosen once
in `cli_out`/`cli_err`.

One `AgentStreamPresenter` replaces `StreamingQueryHandler`,
`TUIStreamingQueryHandler`, and `BufferedStreamingHandler`.

## Tradeoffs

Non-interactive TTY appearance changes (GFM via Markdown instead of Rich
Table). We keep `getpass` because saber-tui Input has no mask. `theme.json`
values like `"white on blue"` keep the foreground and drop the background.

## Rejected

Console facade. Tools returning saber-tui Components. Dual
`format_result` + `render_result_tui`. `ChatConsole`. C2's one-shot TUI.
C3's `DeadHost` and str-only `SelectOption`. C1's public `styled` bit.
