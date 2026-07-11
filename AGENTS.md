# Repository Guidelines

## Scope

This file governs Git workflow and commit message format across this repository.

## Dependency Management

`uv` is the canonical dependency and environment manager for the root RoboDojo
project. Declare root dependencies in `pyproject.toml`, keep `uv.lock` current,
and run native commands with `uv run --locked`.

Policy-owned environments under `XPolicyLab` remain independent and may use
their own dependency tooling.

## Git Flow And Worktrees

`main` is the protected working trunk. Do not direct-push to `main`.

`Makefile` is the tracked entry point for common development, evaluation, and
storage workflows. Keep its help output and `.env.example` synchronized when
adding or changing configurable targets.

Before starting implementation work, fetch `origin`, confirm local `main` has
no tracked-file changes, and confirm it is current with `origin/main` when that
remote branch exists. If local `main` has diverged from `origin/main`, stop and
coordinate instead of rewriting shared trunk state.

Every implementation should start in an ephemeral Git worktree branched from
the current `main`. Create worktrees under the repository root:

```text
.worktrees/<slug>
```

Use branch names that identify the type of work:

```text
feat/<slug>
fix/<slug>
docs/<slug>
chore/<slug>
```

Completion means the work has landed on `main` and the disposable worktree has
been closed. For normal implementation work, do not send a final "complete"
response while the implementation branch or its ephemeral worktree is still
open, unless the user explicitly asks to keep it open.

Complete implementation work in this order:

1. Verify the implementation and review the intended changes.
2. Commit the worktree changes.
3. Fetch `origin` and confirm `main` is still current with `origin/main` when
   that remote branch exists.
4. Rebase the work branch if needed.
5. Integrate the work branch into `main`.
6. Push `main`.
7. Remove the `.worktrees/<slug>` worktree.
8. Delete the implementation branch unless the user asked to keep it.

If merge, push, worktree removal, or branch deletion cannot be completed, stop
at the blocking point and report exactly what remains, why it remains, and
which branch or worktree still needs attention. Coordinate instead of forcing
history when there is non-fast-forward remote divergence or a non-trivial
conflict.

Prefer integration into `main` in this order: fast-forward; rebase and
fast-forward; merge commit only as a fallback.

## Commit Messages

Commit messages must follow:

```text
<gitmoji> (scope): <subject>
```

Use these categories:

```text
✨ feat       Introduce new features
🐛 fix        Fix a bug
📝 docs       Documentation only changes
💄 style      Code style changes
♻️ refactor   Refactoring without behavior change
⚡️ perf       Improve performance
✅ test       Add or update tests
🔧 build      Build system or dependency changes
👷 ci         CI configuration and scripts
🔒 chore      Other changes not affecting src or tests
⏪️ revert     Revert a previous commit
```

Examples:

```text
📝 docs(plan): add staged reimplementation roadmap
🔧 build(project): initialize uv scaffold
✨ feat(data): add versioned dataset registry
```

Commit bodies are optional. When present, explain what and why, not how, and
wrap body text at 72 characters.
