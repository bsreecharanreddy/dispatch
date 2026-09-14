#!/usr/bin/env bash
# Shared helpers for the PreToolUse/Bash git-commit hooks.
# Source, don't execute: `source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"`.

# Files this commit will actually contain. A `git add` in the same command
# leaves the index empty at PreToolUse time (the hook runs first), so fall
# back to the working tree -- `git add -A && git commit` otherwise fires no hook.
files_for_commit() {
  local cmd="$1" staged
  staged="$(git diff --cached --name-only 2>/dev/null || true)"
  if printf '%s' "$cmd" | grep -qE '\bgit\s+add\b'; then
    printf '%s\n%s\n' "$staged" "$(git status --porcelain 2>/dev/null | sed 's/^...//')"
  else
    printf '%s\n' "$staged"
  fi | sed '/^$/d' | sort -u
}
