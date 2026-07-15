# Experiment setup and preflight

RoboDojo separates policy mutation from launch-time validation. Configure one
experiment in `.env`, prepare it explicitly, validate it, and then launch it:

```bash
cp .env.example .env
make policy-setup
make preflight
make preflight DEEP=true
make eval PUBLISH=false
```

`make policy-setup` is the only new setup mutation interface. It invokes the
selected policy's optional `prepare_eval_policy.sh` from the policy directory
with this argument prefix:

```text
<dataset> <task> <ckpt> <env> <action> <seed> <gpu> <policy-env>
```

The policy owns dependency installation, runtime resolution, checkpoint
download, and integrity rules. RoboDojo does not reproduce those rules.
Adapters without the optional setup hook report a warning and retain their
legacy README-driven setup.

Fast preflight is read-only. It validates:

- the root `.venv`, `uv.lock`, and installed simulator distributions;
- task code/YAML, environment components, scene profile, and selected layout;
- generated robot manifests and declared output checksums;
- policy and simulator GPU indices;
- S3 URI and AWS CLI presence when publication is requested;
- required adapter files, the resolved uv project or Conda environment, and
  XPolicyLab imports in that environment;
- explicit checkpoint paths, plus policy-owned embodiment, action, checkpoint,
  source, and pinned-integrity checks from `check_eval_policy.sh`.

Legacy adapters without `check_eval_policy.sh` receive generic checks and a
warning for unsupported policy-specific validation. Opaque checkpoint aliases
also warn unless a policy hook owns their resolution. Warnings do not block a
launch; failures do.

`make preflight DEEP=true` first requires fast preflight to pass. It then starts
the normal `setup_eval_policy_server.sh` on a temporary loopback port, waits for
readiness, and always terminates the process group on success, early policy
exit, or timeout. It never starts Isaac Sim and never publishes.

Human reports use `PASS`, `WARN`, and `FAIL` with exact remediation. JSON is
available from the CLI:

```bash
uv run --extra sim --locked --no-sync robodojo preflight \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task general_pickup \
  --ckpt pi05_yam_molmoact2 \
  --policy-env uv \
  --env-cfg bimanual_yam \
  --scene molmo_yam \
  --action-type joint \
  --policy-gpu 0 \
  --env-gpu 1 \
  --format json
```

The JSON object has a stable top-level `status` and a `checks` array whose
records contain `name`, `status`, `detail`, and optional `remediation`.

Real `eval`, `server`, `smoke`, and `benchmark` commands run fast preflight
before selecting a free port or starting policy/simulator processes. A sweep
runs the shared gate once, not once per child task. Dry runs intentionally skip
preflight and remain command-rendering-only. Make launch targets use uv's
`--no-sync` mode, so a missing or stale root environment fails with `make sync`
instead of changing the environment during launch.
