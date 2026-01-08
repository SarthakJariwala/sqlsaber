"""HTML export utilities for thread transcripts."""

import json
from html import escape

from pydantic_ai.messages import ModelMessage

from sqlsaber.threads.storage import Thread


def build_turn_slices(all_msgs: list[ModelMessage]) -> list[tuple[int, int]]:
    """Return (start, end) slices for turns, grouped by user-prompt messages."""
    user_indices: list[int] = []
    for idx, message in enumerate(all_msgs):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                user_indices.append(idx)
                break

    slices: list[tuple[int, int]] = []
    if user_indices:
        for i, start_idx in enumerate(user_indices):
            end_idx = (
                user_indices[i + 1] if i + 1 < len(user_indices) else len(all_msgs)
            )
            slices.append((start_idx, end_idx))
    else:
        slices = [(0, len(all_msgs))]
    return slices


def _html_escape_multimodal(content: object) -> str:
    """Flatten and HTML-escape multimodal user content (strings + JSON dumps)."""
    if isinstance(content, str):
        return escape(content)
    if isinstance(content, list):
        parts: list[str] = []
        for seg in content:
            if isinstance(seg, str):
                parts.append(seg)
            else:
                try:
                    parts.append(json.dumps(seg, ensure_ascii=False))
                except Exception:
                    parts.append(str(seg))
        text = "\n".join(s for s in parts if s)
        return escape(text)
    return escape(str(content))


def _render_sql_results_table(content_str: str, sql_query: str | None = None) -> str:
    """Render SQL execute results as an HTML table with the query."""
    parts: list[str] = []

    if sql_query:
        parts.append(
            f'<div class="sql-query"><pre><code class="language-sql">{escape(sql_query)}</code></pre></div>'
        )

    try:
        data = json.loads(content_str)
        if isinstance(data, dict) and data.get("success") and data.get("results"):
            results = data["results"]
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    headers = list(first.keys())
                    rows_html = []
                    for row in results[:100]:  # Limit to 100 rows
                        cells = "".join(
                            f"<td>{escape(str(row.get(h, '')))}</td>" for h in headers
                        )
                        rows_html.append(f"<tr>{cells}</tr>")
                    header_html = "".join(f"<th>{escape(h)}</th>" for h in headers)
                    count_note = (
                        f'<p class="result-count">{len(results)} row(s) returned</p>'
                    )
                    if len(results) > 100:
                        count_note = f'<p class="result-count">Showing 100 of {len(results)} rows</p>'
                    parts.append(f"""
{count_note}
<div class="table-wrapper">
<table class="sql-results">
<thead><tr>{header_html}</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</div>""")
                    return "".join(parts)
            elif isinstance(results, list) and not results:
                parts.append('<p class="result-count">0 rows returned</p>')
                return "".join(parts)
        if isinstance(data, dict) and "error" in data:
            error = escape(str(data.get("error", "")))
            parts.append(
                f'<div class="sql-error"><strong>Error:</strong> {error}</div>'
            )
            return "".join(parts)
    except (json.JSONDecodeError, TypeError):
        pass

    parts.append(f"<pre><code>{escape(content_str)}</code></pre>")
    return "".join(parts)


def _render_list_tables_html(content_str: str) -> str:
    """Render list_tables results as an HTML table."""
    try:
        data = json.loads(content_str)

        if isinstance(data, dict) and "error" in data:
            return f'<div class="sql-error"><strong>Error:</strong> {escape(str(data["error"]))}</div>'

        if isinstance(data, list):
            if not data:
                return '<p class="result-count">No tables found in the database.</p>'
            if isinstance(data[0], str):
                rows_html = [
                    f"<tr><td>—</td><td>{escape(t)}</td><td>—</td></tr>" for t in data
                ]
                return f"""
<p class="result-count">{len(data)} table(s) found</p>
<div class="table-wrapper">
<table class="sql-results">
<thead><tr><th>Schema</th><th>Table Name</th><th>Type</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</div>"""

        if isinstance(data, dict):
            tables = data.get("tables", [])
            total = data.get("total_tables", len(tables))

            if not tables:
                return '<p class="result-count">No tables found in the database.</p>'

            rows_html = []
            for t in tables:
                schema = escape(str(t.get("schema", "")))
                name = escape(str(t.get("name", "")))
                ttype = escape(str(t.get("type", "")))
                rows_html.append(
                    f"<tr><td>{schema}</td><td>{name}</td><td>{ttype}</td></tr>"
                )

            return f"""
<p class="result-count">{total} table(s) found</p>
<div class="table-wrapper">
<table class="sql-results">
<thead><tr><th>Schema</th><th>Table Name</th><th>Type</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</div>"""

        return f"<pre><code>{escape(content_str)}</code></pre>"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return f"<pre><code>{escape(content_str)}</code></pre>"


def _render_introspect_schema_html(content_str: str) -> str:
    """Render introspect_schema results as formatted HTML tables."""
    try:
        data = json.loads(content_str)
        if "error" in data:
            return f'<div class="sql-error"><strong>Error:</strong> {escape(str(data["error"]))}</div>'

        if not data:
            return '<p class="result-count">No schema information found.</p>'

        parts: list[str] = []
        parts.append(f'<p class="result-count">{len(data)} table(s) introspected</p>')

        for table_name, table_info in data.items():
            parts.append('<div class="schema-table">')
            parts.append(f'<h4 class="table-name">{escape(table_name)}</h4>')

            if table_info.get("comment"):
                parts.append(
                    f'<p class="table-comment">{escape(str(table_info["comment"]))}</p>'
                )

            columns = table_info.get("columns", {})
            if columns:
                rows = []
                for col_name, col_info in columns.items():
                    nullable = "✓" if col_info.get("nullable") else "✗"
                    default = escape(str(col_info.get("default") or "—"))
                    comment = escape(str(col_info.get("comment") or ""))
                    rows.append(
                        f"<tr><td>{escape(col_name)}</td><td>{escape(str(col_info.get('type', '')))}</td>"
                        f"<td>{nullable}</td><td>{default}</td><td>{comment}</td></tr>"
                    )
                parts.append("""
<div class="table-wrapper">
<table class="sql-results schema-columns">
<thead><tr><th>Column</th><th>Type</th><th>Nullable</th><th>Default</th><th>Comment</th></tr></thead>
<tbody>""")
                parts.append("".join(rows))
                parts.append("</tbody></table></div>")

            pks = table_info.get("primary_keys", [])
            if pks:
                parts.append(
                    f'<p class="key-info"><strong>Primary Keys:</strong> {escape(", ".join(pks))}</p>'
                )

            fks = table_info.get("foreign_keys", [])
            if fks:
                fk_list = "".join(f"<li>{escape(fk)}</li>" for fk in fks)
                parts.append(
                    f'<p class="key-info"><strong>Foreign Keys:</strong></p><ul class="key-list">{fk_list}</ul>'
                )

            parts.append("</div>")

        return "".join(parts)
    except (json.JSONDecodeError, TypeError):
        return f"<pre><code>{escape(content_str)}</code></pre>"


def _human_readable_time(timestamp: float | None) -> str:
    """Format a timestamp as a human-readable string."""
    import time

    if not timestamp:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def render_thread_html(thread: Thread, all_msgs: list[ModelMessage]) -> str:
    """Generate a standalone HTML document for a thread transcript."""
    title = thread.title or f"SQLsaber thread {thread.id}"
    escaped_title = escape(title)

    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <link id="hljs-theme" rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    (function() {{
      const saved = localStorage.getItem('sqlsaber-theme');
      if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
    }})();
  </script>
  <style>
    :root {{
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-tertiary: #21262d;
      --border-color: #30363d;
      --text-primary: #e6edf3;
      --text-secondary: #8b949e;
      --text-muted: #6e7681;
      --accent-blue: #58a6ff;
      --accent-green: #3fb950;
      --accent-amber: #d29922;
      --accent-red: #f85149;
      --accent-purple: #a371f7;
      --font-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
    }}
    [data-theme="light"] {{
      --bg-primary: #ffffff;
      --bg-secondary: #f6f8fa;
      --bg-tertiary: #eaeef2;
      --border-color: #d0d7de;
      --text-primary: #1f2328;
      --text-secondary: #59636e;
      --text-muted: #8c959f;
      --accent-blue: #0969da;
      --accent-green: #1a7f37;
      --accent-amber: #9a6700;
      --accent-red: #d1242f;
      --accent-purple: #8250df;
    }}
    * {{ box-sizing: border-box; }}
    html {{ font-size: 15px; }}
    body {{
      font-family: var(--font-sans);
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
      margin: 0;
      padding: 2rem 1rem;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
    }}
    header {{
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }}
    .logo {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }}
    .logo svg {{
      width: 28px;
      height: 28px;
    }}
    .logo span {{
      font-family: var(--font-mono);
      font-weight: 600;
      font-size: 0.85rem;
      color: var(--text-secondary);
      letter-spacing: 0.02em;
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      margin: 0 0 0.75rem 0;
      color: var(--text-primary);
    }}
    .thread-id {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-muted);
      background: var(--bg-tertiary);
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      display: inline-block;
      margin-bottom: 1rem;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
    }}
    .meta-item {{
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.75rem 1rem;
    }}
    .meta-label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.25rem;
    }}
    .meta-value {{
      font-size: 0.9rem;
      color: var(--text-primary);
      font-weight: 500;
    }}
    .turn {{
      margin-bottom: 1.5rem;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-color);
    }}
    .turn-header {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem 1rem;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .turn.user .turn-header {{
      background: linear-gradient(135deg, rgba(88,166,255,0.15), rgba(88,166,255,0.05));
      color: var(--accent-blue);
      border-bottom: 1px solid rgba(88,166,255,0.2);
    }}
    .turn.assistant .turn-header {{
      background: linear-gradient(135deg, rgba(63,185,80,0.15), rgba(63,185,80,0.05));
      color: var(--accent-green);
      border-bottom: 1px solid rgba(63,185,80,0.2);
    }}
    .turn-content {{
      padding: 1rem 1.25rem;
      background: var(--bg-secondary);
    }}
    .turn.user .turn-content {{
      font-size: 1rem;
      white-space: pre-wrap;
    }}
    .markdown-content {{
      font-size: 0.95rem;
    }}
    .markdown-content h1, .markdown-content h2, .markdown-content h3 {{
      color: var(--text-primary);
      margin-top: 1.5rem;
      margin-bottom: 0.75rem;
    }}
    .markdown-content h1 {{ font-size: 1.4rem; }}
    .markdown-content h2 {{ font-size: 1.2rem; }}
    .markdown-content h3 {{ font-size: 1.05rem; }}
    .markdown-content p {{ margin: 0.75rem 0; }}
    .markdown-content ul, .markdown-content ol {{
      margin: 0.75rem 0;
      padding-left: 1.5rem;
    }}
    .markdown-content li {{ margin: 0.25rem 0; }}
    .markdown-content code {{
      font-family: var(--font-mono);
      background: var(--bg-tertiary);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.85em;
    }}
    .markdown-content pre {{
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 1rem;
      overflow-x: auto;
      margin: 1rem 0;
    }}
    .markdown-content pre code {{
      background: none;
      padding: 0;
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    .markdown-content a {{
      color: var(--accent-blue);
      text-decoration: none;
    }}
    .markdown-content a:hover {{ text-decoration: underline; }}
    .markdown-content blockquote {{
      border-left: 3px solid var(--accent-purple);
      margin: 1rem 0;
      padding-left: 1rem;
      color: var(--text-secondary);
    }}
    .tool-section {{
      margin-top: 1rem;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      overflow: hidden;
    }}
    .tool-header {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.6rem 1rem;
      background: var(--bg-tertiary);
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--text-secondary);
      border: none;
      width: 100%;
      text-align: left;
    }}
    .tool-header:hover {{ background: var(--bg-primary); }}
    .tool-icon {{
      color: var(--accent-amber);
    }}
    .tool-name {{
      font-family: var(--font-mono);
      color: var(--accent-amber);
      font-weight: 500;
    }}
    .tool-body {{
      padding: 1rem;
      background: var(--bg-secondary);
      border-top: 1px solid var(--border-color);
    }}
    details.tool > summary {{
      list-style: none;
    }}
    details.tool > summary::-webkit-details-marker {{
      display: none;
    }}
    details.tool > summary::before {{
      content: '▸';
      margin-right: 0.5rem;
      transition: transform 0.2s;
      display: inline-block;
    }}
    details.tool[open] > summary::before {{
      transform: rotate(90deg);
    }}
    .table-wrapper {{
      overflow-x: auto;
      margin: 0.5rem 0;
    }}
    .sql-results {{
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 0.8rem;
    }}
    .sql-results th {{
      background: var(--bg-tertiary);
      color: var(--accent-blue);
      font-weight: 600;
      text-align: left;
      padding: 0.6rem 0.75rem;
      border-bottom: 2px solid var(--accent-blue);
      white-space: nowrap;
    }}
    .sql-results td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-primary);
    }}
    .sql-results tr:hover td {{
      background: var(--bg-tertiary);
    }}
    .sql-results tr:last-child td {{
      border-bottom: none;
    }}
    .result-count {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin: 0 0 0.5rem 0;
      font-family: var(--font-mono);
    }}
    .sql-error {{
      background: rgba(248, 81, 73, 0.1);
      border: 1px solid rgba(248, 81, 73, 0.3);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      color: var(--accent-red);
      font-size: 0.9rem;
    }}
    .sql-query {{
      margin-bottom: 0.75rem;
    }}
    .sql-query pre {{
      margin: 0;
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.75rem 1rem;
    }}
    .sql-query code {{
      font-family: var(--font-mono);
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    .schema-table {{
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border-color);
    }}
    .schema-table:last-child {{
      border-bottom: none;
      margin-bottom: 0;
    }}
    .table-name {{
      color: var(--accent-purple);
      font-family: var(--font-mono);
      font-size: 1rem;
      margin: 0 0 0.5rem 0;
    }}
    .table-comment {{
      color: var(--text-secondary);
      font-size: 0.85rem;
      margin: 0 0 0.75rem 0;
      font-style: italic;
    }}
    .key-info {{
      font-size: 0.8rem;
      color: var(--text-secondary);
      margin: 0.5rem 0 0 0;
    }}
    .key-info strong {{
      color: var(--accent-amber);
    }}
    .key-list {{
      margin: 0.25rem 0 0 1.25rem;
      padding: 0;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }}
    .key-list li {{
      margin: 0.15rem 0;
    }}
    .empty-state {{
      text-align: center;
      padding: 3rem;
      color: var(--text-muted);
    }}
    footer {{
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border-color);
      text-align: center;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
    footer a {{ color: var(--accent-blue); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}
    .header-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }}
    .theme-toggle {{
      background: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.5rem;
      cursor: pointer;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s, color 0.2s;
    }}
    .theme-toggle:hover {{
      background: var(--bg-secondary);
      color: var(--text-primary);
    }}
    .theme-toggle svg {{
      width: 18px;
      height: 18px;
    }}
    .icon-sun {{ display: none; }}
    .icon-moon {{ display: block; }}
    [data-theme="light"] .icon-sun {{ display: block; }}
    [data-theme="light"] .icon-moon {{ display: none; }}
    @media (max-width: 600px) {{
      html {{ font-size: 14px; }}
      body {{ padding: 1rem 0.75rem; }}
      .meta-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
"""

    meta_html = f"""
<body>
<div class="container">
  <header>
    <div class="header-row">
      <div class="logo">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
        <span>SQLsaber</span>
      </div>
      <button class="theme-toggle" id="theme-toggle" aria-label="Toggle theme">
        <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
        <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      </button>
    </div>
    <h1>{escaped_title}</h1>
    <code class="thread-id">{escape(thread.id)}</code>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="meta-label">Database</div>
        <div class="meta-value">{escape(thread.database_name or "—")}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Model</div>
        <div class="meta-value">{escape(thread.model_name or "—")}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Created</div>
        <div class="meta-value">{escape(_human_readable_time(thread.created_at))}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Last Activity</div>
        <div class="meta-value">{escape(_human_readable_time(thread.last_activity_at))}</div>
      </div>
    </div>
  </header>
  <main>
"""

    body_parts: list[str] = []
    slices = build_turn_slices(all_msgs)

    for start_idx, end_idx in slices:
        if not (0 <= start_idx < len(all_msgs)):
            continue

        user_msg = all_msgs[start_idx]
        user_text = ""
        for part in getattr(user_msg, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                content = getattr(part, "content", None)
                user_text = _html_escape_multimodal(content)
                break

        body_parts.append(
            f"""<section class="turn user">
  <div class="turn-header">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>
    User
  </div>
  <div class="turn-content">{user_text or "(no content)"}</div>
</section>
"""
        )

        tool_calls: dict[str, dict[str, str]] = {}
        for i in range(start_idx + 1, end_idx):
            msg = all_msgs[i]
            for part in getattr(msg, "parts", []):
                kind = getattr(part, "part_kind", "")
                if kind in ("tool-call", "builtin-tool-call"):
                    call_id = getattr(part, "tool_call_id", None)
                    tool_name = str(getattr(part, "tool_name", ""))
                    args = getattr(part, "args", {})
                    if call_id and tool_name:
                        args_dict: dict = {}
                        if isinstance(args, dict):
                            args_dict = args
                        elif isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                                if isinstance(parsed, dict):
                                    args_dict = parsed
                            except Exception:
                                pass
                        sql_query = None
                        if tool_name == "execute_sql":
                            sql_query = args_dict.get("sql") or args_dict.get("query")
                        tool_calls[call_id] = {
                            "name": tool_name,
                            "sql": sql_query or "",
                        }

        for i in range(start_idx + 1, end_idx):
            msg = all_msgs[i]
            assistant_text_blocks: list[str] = []
            tool_details_html: list[str] = []

            for part in getattr(msg, "parts", []):
                kind = getattr(part, "part_kind", "")

                if kind == "text":
                    text = getattr(part, "content", "")
                    if isinstance(text, str) and text.strip():
                        assistant_text_blocks.append(escape(text))

                elif kind in ("tool-return", "builtin-tool-return"):
                    tool_name = str(getattr(part, "tool_name", "tool"))
                    call_id = getattr(part, "tool_call_id", None)
                    content = getattr(part, "content", None)
                    if isinstance(content, (dict, list)):
                        content_str = json.dumps(content, ensure_ascii=False, indent=2)
                    elif isinstance(content, str):
                        content_str = content
                    else:
                        content_str = json.dumps(
                            {"return_value": str(content)}, ensure_ascii=False, indent=2
                        )

                    sql_query = (
                        tool_calls.get(call_id, {}).get("sql") if call_id else None
                    )

                    if tool_name == "execute_sql":
                        result_html = _render_sql_results_table(content_str, sql_query)
                    elif tool_name == "list_tables":
                        result_html = _render_list_tables_html(content_str)
                    elif tool_name == "introspect_schema":
                        result_html = _render_introspect_schema_html(content_str)
                    else:
                        result_html = f"<pre><code>{escape(content_str)}</code></pre>"

                    tool_details_html.append(
                        f"""<details class="tool" open>
  <summary class="tool-header">
    <span class="tool-icon">⚡</span>
    <span class="tool-name">{escape(tool_name)}</span>
  </summary>
  <div class="tool-body">{result_html}</div>
</details>"""
                    )

            if assistant_text_blocks or tool_details_html:
                markdown_content = "\n\n".join(assistant_text_blocks)
                tool_html = "\n".join(tool_details_html)

                body_parts.append(
                    f"""<section class="turn assistant">
  <div class="turn-header">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2a10 10 0 1 0 10 10H12V2z"/>
      <path d="M12 2a7 7 0 0 1 7 7h-7V2z"/>
    </svg>
    Assistant
  </div>
  <div class="turn-content">
    <div class="markdown-content" data-markdown>{markdown_content}</div>
    {tool_html}
  </div>
</section>
"""
                )

    transcript_html = (
        "".join(body_parts)
        or '    <div class="empty-state"><p>No messages in this thread.</p></div>\n'
    )

    tail = """  </main>
  <footer>
    Exported from <a href="https://github.com/SarthakJariwala/sqlsaber">SQLsaber</a>
  </footer>
</div>
<script>
document.querySelectorAll('[data-markdown]').forEach(el => {
  const raw = el.textContent;
  el.innerHTML = marked.parse(raw);
});
hljs.highlightAll();

// Theme toggle
const toggle = document.getElementById('theme-toggle');
const hljsTheme = document.getElementById('hljs-theme');
function setTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    hljsTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
  } else {
    document.documentElement.removeAttribute('data-theme');
    hljsTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css';
  }
  localStorage.setItem('sqlsaber-theme', theme);
}
toggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  setTheme(current === 'light' ? 'dark' : 'light');
});
// Apply saved theme on load for hljs
if (localStorage.getItem('sqlsaber-theme') === 'light') {
  hljsTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
}
</script>
</body>
</html>
"""

    return head + meta_html + transcript_html + tail
