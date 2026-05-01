#!/usr/bin/env bash
# Manual slash-command entrypoint for summarizing Codex sessions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export MEMSEARCH_SKIP_HOOK_STDIN="${MEMSEARCH_SKIP_HOOK_STDIN:-1}"
source "$PLUGIN_ROOT/hooks/common.sh"

if [ "$#" -eq 1 ] && [ -z "${1:-}" ]; then
  set --
fi

python3 "$SCRIPT_DIR/session_summarizer.py" \
  --agent codex \
  --project-dir "$PROJECT_DIR" \
  --memory-dir "$MEMORY_DIR" \
  --collection "$COLLECTION_NAME" \
  --plugin-root "$PLUGIN_ROOT" \
  "$@"
