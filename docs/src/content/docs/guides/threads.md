---
title: Conversation Threads
description: "Save, resume, and manage conversation threads in SQLsaber. Continue past database analysis sessions with full context preserved."
---

SQLsaber automatically saves your conversations locally so that you can view, resume, and manage them.

Threads allow you to pick up where you left off and track your analytical work over time.

Complete SQL row results are stored separately under SQLsaber's private user-data
`query-results` directory; thread messages retain only stable bounded model
projections and opaque descriptors. Show, export, resume, visualization, sandbox,
and notebook paths hydrate complete data from that store without rewriting thread
history. Terminal and HTML views may still intentionally render a bounded table.

Capability artifacts are stored separately under the private user-data `artifacts`
directory. Thread messages retain only publication references, not notebook, image,
or generated-file bytes. For retained notebook analyses, show and resume reconstruct
bounded notebook cells and available plots and list generated-file locations. If the
notebook is missing or invalid, replay falls back to the generic artifact listing
with an availability notice. Resume is read-only: it never reopens an old notebook
kernel.

Thread retention is the source of truth for both CLI query-result and artifact
retention. After pruning, SQLsaber removes unreferenced entries older than a 24-hour
creation-time safety grace period. Normal saves run maintenance at most daily. If a
query result or artifact is missing or corrupt, replay marks it unavailable rather
than trusting or rewriting the historical descriptor.

### Show All Threads

View all your conversation threads:

```bash
saber threads list
```

### Show Full Conversation

View the complete transcript of a thread:

```bash
saber threads show bb7b4d72
```

### List Thread Artifacts

List publication IDs, artifact names, kinds, sizes, and local URIs without replaying
the full transcript:

```bash
saber threads artifacts bb7b4d72
```

### Continue Previous Thread

Resume an existing conversation thread:

```bash
saber threads resume bb7b4d72
```

This:
- Loads the full conversation context
- Connects to the same database used in the original thread
- Uses the currently configured model
- Allows you to continue where you left off in interactive mode

For one non-interactive follow-up, pass the thread to the root query command:

```bash
saber --thread bb7b4d72 "Now compare that with last quarter"
```

This loads the saved message history, keeps the same thread ID, and uses the
stored configured database. Pass `-d DATABASE` to override it.

### Prune Old Threads Safely

Preview cleanup before deleting anything:

```bash
saber threads prune --days 30 --dry-run
```

Run the deletion interactively, or make the intent explicit in automation:

```bash
saber threads prune --days 30 --yes
```

### Sharing Threads

```bash
# Review what you analyzed
saber threads show abc123 > analysis_report.md

# Share the conversation transcript with colleagues
cat analysis_report.md
```

### Getting Help

Check thread commands and options:

```bash
saber threads --help
saber threads list --help
saber threads resume --help
saber threads artifacts --help
saber threads prune --help
```

### What's Next?

Now that you understand conversation threads:

1. [Learn advanced querying techniques](/guides/queries)
2. [Explore model selection](/guides/models) for different thread purposes
3. [Review the command reference](/reference/commands) for all thread options
