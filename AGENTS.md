# Repository Guidelines

## Scope

This is the canonical contributor guide for the repository. It governs
architecture and ownership boundaries, dependency management, validation, Git
workflow, and commit message format. Do not create a second agent- or
tool-specific contributor guide that duplicates or overrides these rules.

Human-facing usage documentation lives at
<https://robodojo-benchmark.com/doc/>.

## Upstream Compatibility

Keep shared contract boundaries as close as practical to the official
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) repositories. Those
upstreams are the reference for repository layout, configuration keys and
semantics, launcher arguments, process flow, transport behavior, and
observation/action wire formats.

Before changing a shared boundary, inspect the corresponding implementation in
both upstream repositories. Prefer additive, backward-compatible extensions
over replacements or local reinterpretations. In particular:

- Preserve upstream names, paths, argument order, configuration meanings,
  WebSocket behavior, and observation/action shapes when practical.
- Keep fork-specific behavior in narrow RoboDojo adapters such as the typed CLI
  orchestration or `scripts/eval_policy.sh`; do not push local requirements
  into XPolicyLab policy adapters or RoboDojo task/environment interfaces.
- If divergence is unavoidable, document the upstream-to-local mapping at the
  adapter and cover both the upstream-compatible path and the extension with
  tests.
- Treat the ability to rebase, cherry-pick, or port future upstream changes
  without broad rewrites as an architectural acceptance criterion.

Run `make upstream-check` to compare the reviewed revisions in
`upstream_sync.yml` with the official repositories. Adopting or intentionally
declining a reported change requires updating its mapping or disposition and
the reviewed commit. Unmapped shared-boundary changes must be classified before
the reviewed revision advances.

The configured `XPolicyLab` submodule may point to a working fork. The official
XPolicyLab repository remains the contract reference. Update the gitlink pin
intentionally, and do not edit submodule contents as part of RoboDojo-owned
changes unless the task explicitly includes coordinated XPolicyLab work.

## Repository Layout And Ownership

```text
src/robodojo/core/           settings, paths, models, storage, processes
src/robodojo/policy/         policy adapter validation and launching
src/robodojo/sim/            simulator managers, tasks, evaluation, scene export
src/robodojo/orchestration/  coordinated policy/simulator process lifecycle
src/robodojo/workflows/      install, download, storage, result, Docker workflows
configs/                     robot, scene, simulator, and camera YAML
task/RoboDojo/config/        task scene/object YAML
scripts/eval_policy.sh       private XPolicyLab compatibility shim
docker/                      container evaluation support
XPolicyLab/                  policy servers and adapters (submodule)
third_party/                 IsaacLab and curobo (submodules)
Assets/                      downloaded assets; not tracked by Git
eval_result/                 runtime evaluation output
```

RoboDojo owns `src/robodojo/`, root configuration, task definitions, the Typer
CLI, and root install/assets/storage workflows. XPolicyLab owns policy code,
policy-specific dependencies and training, checkpoints, `deploy.yml`, policy
servers, and `XPolicyLab/policy/<POLICY>/setup_eval_*` scripts.

New code imports through the `robodojo.*` namespace. Core and policy code must
not import `robodojo.sim` or simulator dependencies. Orchestration may import
the lightweight simulator launcher but must not initialize Isaac or Torch. The
simulator uses `XPolicyLab/client_server/ws/model_client.py` for WebSocket
transport.

## Dependency Management

`uv` is the canonical dependency and environment manager for the root RoboDojo
project. Declare root dependencies in `pyproject.toml`, keep `uv.lock` current,
and run native commands with `uv run --locked`.

Policy-owned environments under `XPolicyLab` remain independent and may use
their own dependency tooling.

## Evaluation Boundary And Flow

Run native commands as `uv run --extra sim --locked robodojo <command>` when
simulator dependencies are required. The main commands are `doctor`, `eval`,
`server`, `client`, `smoke`, `benchmark`, `summarize`, and `tasks`.
`upstream check` is the lightweight maintenance command for reviewing official
repository changes.

Single-machine evaluation follows this boundary:

```text
robodojo eval
  -> robodojo.orchestration
    -> robodojo.policy -> (cwd: policy directory) setup_eval_policy_server.sh
    -> robodojo.sim.evaluation.main
```

- The upstream XPolicyLab adapter surface includes `eval.sh`,
  `setup_eval_policy_server.sh`, and `deploy.yml`. RoboDojo's managed path
  consumes the setup script and deployment configuration directly rather than
  invoking policy `eval.sh`.
- Run `setup_eval_*` scripts with the policy directory as their working
  directory so upstream-relative paths continue to work.
- WebSocket is the default transport (`protocol: ws` in `deploy.yml`).
- Split evaluation preserves the same boundary: `robodojo server` starts the
  XPolicyLab adapter and `robodojo client` starts the simulator client.
- Full evaluation-infrastructure acceptance is sequential. A successful smoke
  or benchmark run must exit zero and write `_result.json` with
  `eval_time >= 1`; an exit code alone is insufficient.

## Naming And Task Conventions

- Use lowercase `snake_case` for configs, scripts, Python packages, and Python
  module files. Use PascalCase for Python classes.
- Keep task Python filename, YAML name, exported environment class, and
  layout/result paths aligned. Existing uppercase asset-name exceptions such as
  `play_Xylophone`, `swap_T`, and `push_T` retain their upstream spelling.
- Keep upstream submodule and asset directory casing, including `XPolicyLab/`
  and `Assets/`.
- Task registration imports `robodojo.sim.tasks.<task_name>` and expects the
  exported environment class name to match the module basename.
- Task success checks must call `reward_manager.check(...)` or an equivalent
  meaningful check; never leave success trivially true.
- Reset all task-owned state in both `reset()` and `soft_reset()` when it must
  not leak between episodes.
- Do not use `print` in task logic; use the shared logger.

## Validation And Review

Use the smallest validation loop that proves a change. Common fast checks are:

```bash
bash -n scripts/eval_policy.sh
uv run --locked robodojo tasks --format json --check
uv run --extra sim --locked robodojo doctor --skip-policy
uv run --locked ruff check .
git diff --check
```

For evaluation infrastructure, also run dry-run `robodojo eval` and
`robodojo smoke` commands against a representative XPolicyLab policy before
requesting review. Runtime changes require a one-episode evaluation when the
simulator and policy are available.

During review, verify the applicable items:

- Task Python/YAML names and labels match, success checks are meaningful, and
  episode limits are suitable.
- Framework changes are backward compatible and add no unnecessary mandatory
  constructor arguments.
- Shared config changes have safe defaults and do not silently alter unrelated
  tasks.
- Evaluation-boundary changes preserve the official upstream contracts or
  clearly isolate and test any documented adapter-level divergence.
- No debug prints, breakpoints, or commented-out debug code remain.

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

## Common Pitfalls

- A trivial success check produces invalid evaluation metrics.
- Mismatched YAML and Python labels can fail silently.
- Missing `soft_reset()` state handling leaks state between episodes.
- Step limits that are too short truncate long-horizon tasks.
- Camera or shared-config changes can unintentionally affect every task.
- A zero policy-launcher exit code does not prove an evaluation completed;
  verify the result artifact.
- Policy logic belongs in XPolicyLab, not in RoboDojo orchestration.
- Running `setup_eval_*` outside the policy directory breaks upstream-relative
  paths.
- Submodule content is not owned by this repository unless a coordinated
  submodule change is explicitly in scope.
