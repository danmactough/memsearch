#!/usr/bin/env bash
# UserPromptSubmit hook: lightweight hint reminding Codex about the memory-recall skill.
# The actual search + expand is handled by the memory-recall skill.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Skip short prompts (greetings, single words, etc.)
PROMPT=$(_json_val "$INPUT" "prompt" "")
if [ -z "$PROMPT" ] || [ "${#PROMPT}" -lt 10 ]; then
  echo '{}'
  exit 0
fi

# Need memsearch available
if ! memsearch_available; then
  echo '{}'
  exit 0
fi

if [[ "$PROMPT" == /memsearch-summarize* ]]; then
  msg="[memsearch] Summarize requested — use the memsearch-summarize skill to save session memory"
  json_msg=$(_json_encode_str "$msg")
  echo "{\"systemMessage\": $json_msg}"
  exit 0
fi

echo '{"systemMessage": "[memsearch] Memory available"}'
