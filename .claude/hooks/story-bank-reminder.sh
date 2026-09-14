#!/usr/bin/env bash
# PreToolUse/Bash hook: on a commit staging docs/STATUS.md (this repo's "task
# closed out" marker), remind that the story-bank gist may need the story.
# Rationale and incident history: almanac's CLAUDE.md, `.claude/` tooling
# section -- the same trigger failed to self-fire across roughly a dozen
# sessions on a prior project, which is why this exists as a hook rather
# than being left to memory.
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"

printf '%s' "$cmd" | grep -qE '\bgit\s+commit\b' || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

staged="$(files_for_commit "$cmd")"
printf '%s\n' "$staged" | grep -q '^docs/STATUS\.md$' || exit 0

gist_id="$(cat .claude/story-bank-gist-id 2>/dev/null || true)"
target="the dispatch interview story bank gist"
[ -n "$gist_id" ] && target="gist $gist_id"

msg="A task is closing out (docs/STATUS.md staged). Did anything here earn a story-bank entry -- a debugging saga root-caused, a decision made and defended, a scope call, a measured result that contradicted an assumption? If so add it to $target now, in STAR format, while the detail is fresh. This trigger is known to be unreliable when left to memory, which is why it is a hook."
jq -n --arg msg "$msg" '{systemMessage: $msg, hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "allow", permissionDecisionReason: $msg}}'
exit 0
