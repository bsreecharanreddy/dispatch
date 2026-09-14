# dispatch — Status

Authoritative record of where implementation stands against
`docs/design/` (not yet written). Updated **in the same commit as the work
it describes**, never as a follow-up.

## Current position

**Repo scaffolded, 2026-09-14. Nothing built yet.** This commit carries only
conventions and tooling: `pyproject.toml` (uv, ruff, mypy --strict, pytest,
dev deps at live-checked current versions), the `Makefile` gate targets, CI
skeleton, `.claude/` hooks (STATUS.md same-commit check, story-bank
reminder), dual MIT/Apache-2.0 licensing, and this doc structure — all
carried over from
[almanac](https://github.com/bsreecharanreddy/almanac)'s conventions where
applicable, adapted where the stack differs (no Spark/dbt/Databricks here;
GPU rental via lightweight scripts instead of Terraform).

Deliberately **not** carried over: almanac's project-specific
`.claude/skills/` (leakage-review, paid-window, design-decision). Those
exist because a specific incident already happened on almanac — dispatch
has had none yet, so its skills folder starts empty. See CLAUDE.md's
`.claude/` tooling section.

**Next step**: finish brainstorming the system architecture (model choice,
kernel scope, phase breakdown) and write it up as
`docs/design/YYYY-MM-DD-dispatch-system-design.md`, then a phase 0
implementation plan in `docs/plans/`, before any code lands.
