#!/usr/bin/env bash
# PreToolUse/Bash hook: warns (never refuses) before a `git commit` that stages
# files outside docs/ without docs/STATUS.md -- CLAUDE.md requires STATUS.md to
# update in the same commit as the work it describes, or it drifts.
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"

printf '%s' "$cmd" | grep -qE '\bgit\s+commit\b' || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

staged="$(files_for_commit "$cmd")"
[ -z "$staged" ] && exit 0

# STATUS.md already staged, or nothing outside docs/ changed -> nothing to say.
printf '%s\n' "$staged" | grep -q '^docs/STATUS\.md$' && exit 0
printf '%s\n' "$staged" | grep -qv '^docs/' || exit 0

msg="docs/STATUS.md is not staged, but other files are. CLAUDE.md requires STATUS.md to update in the same commit as the work it describes -- confirm this commit doesn't need a STATUS.md update (a task done, a verification result, or a new decision) before proceeding."
jq -n --arg msg "$msg" '{systemMessage: $msg, hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "allow", permissionDecisionReason: $msg}}'
exit 0
