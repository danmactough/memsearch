#!/usr/bin/env bash
# Manual slash-command entrypoint for summarizing Claude Code sessions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PLUGIN_ROOT/hooks/common.sh" < /dev/null

if [ "$#" -eq 1 ] && [ -z "${1:-}" ]; then
  set --
fi

python3 "$SCRIPT_DIR/session_summarizer.py" \
  --agent claude \
  --project-dir "$_PROJECT_DIR" \
  --memory-dir "$MEMORY_DIR" \
  --collection "$COLLECTION_NAME" \
  --plugin-root "$PLUGIN_ROOT" \
  "$@"
