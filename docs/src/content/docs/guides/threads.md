---
title: Conversation Threads
description: Manage and resume your SQLsaber conversations
---

SQLsaber automatically saves your conversations locally so that you can view, resume, and manage them.

Threads allow you to pick up where you left off and track your analytical work over time.

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

1. [Set up memory](/guides/memory) for persistent context across threads
2. [Learn advanced querying techniques](/guides/queries)
3. [Explore model selection](/guides/models) for different thread purposes
4. [Review the command reference](/reference/commands) for all thread options
