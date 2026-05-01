---
name: memsearch-summarize
description: Summarize Codex session history into memsearch memory. Use when the user types `/memsearch-summarize`, `/memsearch-summarize all`, `$memsearch-summarize`, or asks to manually save/summarize the current or all Codex sessions into memsearch.
---

# memsearch Summarize

Run the plugin script exactly once from the installed plugin directory.

For current session:

```bash
bash "__INSTALL_DIR__/scripts/memsearch-summarize.sh"
```

For all sessions:

```bash
bash "__INSTALL_DIR__/scripts/memsearch-summarize.sh" all
```

For dry-run output, add `-n` or `--dry-run` to print the exact AI prompt without writing memory.

Report the command output only. Do not summarize the conversation yourself.
