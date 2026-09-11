---
title: Conversation threads
description: List, show, resume, export, and prune saved SQLsaber conversations.
---

SQLsaber saves each conversation as a thread on your machine. Resume a thread to continue the same analysis.

Complete SQL rows live in a private `query-results` directory. Notebooks, plots, and generated files live in a private `artifacts` directory. Thread messages keep short previews and references, not the full bytes. `show`, export, resume, visualization, sandbox, and notebook reload those files. Terminal and HTML views may still show a bounded table.

Resume never reopens an old notebook kernel. If a notebook is missing or invalid, replay lists the artifacts and notes that they are unavailable. If a query result or artifact is missing or corrupt, replay marks it unavailable. It does not rewrite the saved descriptor.

Prune deletes old threads, then removes unreferenced query results and artifacts older than a 24-hour creation-time grace period. Normal saves run that cleanup at most once a day.

## List threads

```bash
saber threads list
```

## Show a transcript

```bash
saber threads show bb7b4d72
```

## List thread artifacts

List publication IDs, artifact names, kinds, sizes, and local URIs without replaying the full transcript:

```bash
saber threads artifacts bb7b4d72
```

## Resume a thread

```bash
saber threads resume bb7b4d72
```

This loads the saved messages, reconnects to the databases stored on the thread, and uses the model you have configured now. You continue in interactive mode.

For one follow-up without the prompt, pass the thread to the root command:

```bash
saber --thread bb7b4d72 "Now compare that with last quarter"
```

This keeps the same thread ID and the stored database. Pass `-d DATABASE` to override the database.

Automatic resume requires every database on the thread to be a saved connection. If the thread used a connection string or file path, pass `-d` again. See [Multiple databases](/guides/multi-database/#resume-a-multi-database-thread).

## Export a thread as HTML

```bash
saber threads export bb7b4d72
saber threads export bb7b4d72 --output analysis.html
```

With no `--output`, SQLsaber writes `thread-<id>.html` in the current directory.

## Preview prune, then delete old threads

```bash
saber threads prune --days 30 --dry-run
```

To delete, confirm in a terminal, or pass `--yes` in automation:

```bash
saber threads prune --days 30 --yes
```

## Share a transcript

```bash
saber threads show abc123 > analysis_report.md
```

See `saber threads --help`, or [Commands](/reference/commands/#saber-threads).

## Next steps

1. [Run a query](/guides/queries/)
2. [Choose a model](/guides/models/)
3. [Command reference](/reference/commands/)
