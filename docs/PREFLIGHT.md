# Experiment setup and preflight

RoboDojo separates repository-local setup from launch-time validation. Select a
tracked recipe and run the complete local workflow with one Make invocation:

```bash
make recipes
make eval RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
```

The Make target displays a grouped terminal table. Use the direct
`robodojo recipes --format plain|json` formats for automation.

The recipe resolves four typed components: policy, environment, scene, and task
protocol. A recipe cannot be combined with component overrides. Direct CLI
manual mode requires all four named components, which prevents a partial
selection from silently changing the policy, embodiment, scene, or benchmark
behavior. The Makefile defaults to seed 0, automatic policy and simulator GPU
selection, the protocol's native episode count, `INFO` verbosity, no scene
export, and no publication. Override workflow controls with assignments such as
`make eval RECIPE=<name> EVAL_NUM=25`, `EXPORT_SCENE=true`, `PUBLISH=true`, or
`VERBOSITY=DEBUG`. Make also loads an optional ignored `.env` from the repository
root. Entries use Make syntax and should use `?=` so Make arguments and exported
process variables retain precedence. Direct `robodojo` commands continue to
read the process environment only. `PRESET` is rejected with a migration error.

A normal `make eval` runs idempotent setup first and then calls the managed
evaluation. The managed evaluation performs fast preflight exactly once before
starting policy and simulator processes. `DRY_RUN=true` skips both setup and
preflight so it remains mutation-free. Standalone setup and deep preflight are
available when diagnosing readiness:

```bash
make setup RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
make preflight RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
make preflight RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup DEEP=true
```

GPU selectors accept a nonnegative physical device index or lowercase `auto`.
When both roles are automatic, Python ranks `nvidia-smi` devices by free memory,
then by lowest index, assigns the simulator first, and assigns the policy a
distinct second device. An explicit peer is validated and excluded when the
other role is automatic. Single-role `setup`, `client`, and scene-only export
use the most-free device and therefore support one-GPU hosts. CLI flags override
exported `POLICY_GPU` and `ENV_GPU`, which override the `auto` default:

```bash
make eval RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup POLICY_GPU=0 ENV_GPU=1
export POLICY_GPU=0 ENV_GPU=1
make eval RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
```

On a direct CLI call, `--policy-gpu 2 --env-gpu 3` takes precedence over those
exported values.

`make setup` is the consolidated mutation interface. It validates host tools,
initializes pinned submodules without overwriting dirty work, synchronizes the
locked Python 3.11 simulator environment, downloads the base asset bundle,
builds robot/scene/task assets inferred from the configured profiles, and then
invokes the selected policy's optional `prepare_eval_policy.sh`. The hook runs
from the policy directory with this argument prefix:

```text
<dataset> <task> <ckpt> <env> <action> <seed> <gpu> <policy-env>
```

The policy owns dependency installation, runtime resolution, checkpoint
download, and integrity rules. RoboDojo does not reproduce those rules.
Adapters without the optional setup hook report a warning and retain their
legacy README-driven setup.

Fast preflight is read-only. It validates:

- the root `.venv`, `uv.lock`, and installed simulator distributions;
- the policy/environment, scene/environment, and protocol/scene compatibility
  edges before launch;
- canonical task code/YAML, protocol horizon and evaluation count, environment
  components, scene profile, and every protocol-selected layout;
- task label/support-plane contracts and environment robot roots in the resolved
  scene workspace frame;
- scene-owned task asset builds and identities, resolved against the protocol's
  base task;
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
  --recipe pi05-bimanual_yam-molmo_yam-general_pickup \
  --policy-gpu auto \
  --env-gpu auto \
  --format json
```

Setup provides the same human and JSON reporting style. Direct CLI callers can
run every stage or repeat `--only` to select stages while preserving the same
complete recipe:

```bash
uv run --locked robodojo setup \
  --recipe pi05-bimanual_yam-molmo_yam-general_pickup --only root
uv run --locked robodojo setup \
  --recipe pi05-bimanual_yam-molmo_yam-general_pickup --only assets
uv run --locked robodojo setup \
  --recipe pi05-bimanual_yam-molmo_yam-general_pickup --only policy
```

The preflight JSON object has a stable top-level `status` and a `checks` array whose
records contain `name`, `status`, `detail`, and optional `remediation`.

Real `eval`, `server`, `snapshots`, `smoke`, and `benchmark` commands run fast preflight
before selecting a free port or starting policy/simulator processes. Sweeps
accept explicit repeated recipes; they never synthesize arbitrary policy/task
combinations. Dry runs intentionally skip
preflight; automatic selectors still inspect `nvidia-smi` so the rendered
commands contain concrete devices, while numeric dry runs remain GPU-query-free.
Scene-only export runs simulator-side preflight and does not require a policy
GPU, adapter, checkpoint, or publication configuration. After the setup phase,
the launch phase of `make eval` uses uv's `--no-sync` mode so managed evaluation
cannot modify the environment while processes are starting.
