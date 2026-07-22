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

Thread retention is the source of truth for CLI result retention. After pruning,
SQLsaber removes unreferenced entries older than a 24-hour safety grace period.
Normal saves run the same maintenance at most daily. If an entry was deleted or is
corrupt, replay shows the bounded preview with a “complete result unavailable”
notice rather than presenting the preview as complete.

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

### Continue Previous Thread

Resume an existing conversation thread:

```bash
saber threads resume bb7b4d72
```

This:
- Loads the full conversation context
- Connects to the same database used in the original thread
- Uses the same model from the original conversation
- Allows you to continue where you left off in interactive mode


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
```

### What's Next?

Now that you understand conversation threads:

1. [Learn advanced querying techniques](/guides/queries)
2. [Explore model selection](/guides/models) for different thread purposes
3. [Review the command reference](/reference/commands) for all thread options
