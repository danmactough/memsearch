---
allowed-tools: Bash(bash:*)
description: Summarize current or all Claude Code sessions into memsearch memory
---

Run the memsearch session summarizer exactly once:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/memsearch-summarize.sh" $ARGUMENTS
```

Use no argument for the current session. Use `all` when the user runs `/memsearch-summarize all`. Use `-n`/`--dry-run` to print the exact AI prompt without writing memory.
Report the command output only.
