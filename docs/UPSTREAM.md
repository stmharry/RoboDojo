# Adopting official upstream changes

`upstream_sync.yml` records the official RoboDojo and XPolicyLab revisions that
have been reviewed against this fork. It also describes how upstream paths map
into the local package layout and which differences are intentional.

Run the audit from the repository root:

```bash
make upstream-check
make upstream-check ARGS="--format json"
make upstream-check ARGS="--project robodojo"
```

The command fetches official history into a temporary bare checkout. It does
not modify the working tree, remotes, submodules, or manifest. Exit status `0`
means the official heads and mapped local contracts match the reviewed
baseline. Status `1` means upstream review or local adoption is pending. Status
`2` means the manifest, mapping, history, fetch, or ancestry check is invalid.

## Review workflow

1. Run `make upstream-check` and inspect every reported path and parity failure.
2. Port changes through the mapped local boundary:
   - New task modules go to `src/robodojo/sim/tasks/`; rewrite only first-party
     import prefixes and keep task behavior aligned.
   - New task YAML goes to `configs/task/` with the upstream filename and
     semantics unchanged.
   - New scenes go to `configs/scene/`; select them through the existing scene
     resolver rather than changing the policy adapter contract.
   - Framework changes go to the mapped simulator or evaluation module while
     retaining upstream public parameters and wire formats. Local optional
     extensions remain backward compatible.
   - XPolicyLab changes are first integrated into the policy-owned fork. Update
     the root gitlink only after the new official revision is an ancestor of
     the selected fork commit.
3. Add or update tests for any intentional divergence. New upstream paths must
   receive a mapping or an explicit `upstream-only` classification with a
   rationale; they must never be silently ignored.
4. Run the validation required by `AGENTS.md`, including task inventory and a
   representative evaluation dry run when runtime boundaries changed.
5. Advance `reviewed_commit` (and `local_fork.reviewed_commit` when applicable)
   in `upstream_sync.yml` only after the mapped code and tests are complete.
   Re-run the audit and require a clean result in the same commit.

For local fixtures or an already-fetched upstream checkout, select one project
and avoid network access with `--source`:

```bash
uv run --locked robodojo upstream check \
  --project robodojo \
  --source /path/to/official/RoboDojo \
  --ref main
```

The checker compares shared YAML semantically, task Python bodies without
fork-specific import prefixes, and framework public call signatures with
backward-compatible optional local extensions. Adapter replacements remain
manual review boundaries and are recorded as structured intentional
divergences in the manifest.
