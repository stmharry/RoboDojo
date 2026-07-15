# Experiment setup and preflight

RoboDojo separates repository-local setup from launch-time validation.
Select one experiment with Make arguments or exported process variables,
prepare it explicitly, validate it, and then launch it:

```bash
export TASK=general_pickup
export ENV_CFG=bimanual_yam
export SCENE=molmo_yam
export POLICY_DIR=XPolicyLab/policy/Pi_05
export POLICY_ENV=uv
export CKPT=pi05_yam_molmoact2
make setup
make preflight
make preflight DEEP=true
make eval PUBLISH=false
```

The Makefile defaults to the RoboDojo dataset, joint actions, seed 0, automatic
policy and simulator GPU selection, one episode, scene export, and publication.
Override any default with a Make assignment such as `make eval EVAL_NUM=25` or
disable the export with `make eval EXPORT_SCENE=false`; repository `.env` files
are not loaded.

GPU selectors accept a nonnegative physical device index or lowercase `auto`.
When both roles are automatic, Python ranks `nvidia-smi` devices by free memory,
then by lowest index, assigns the simulator first, and assigns the policy a
distinct second device. An explicit peer is validated and excluded when the
other role is automatic. Single-role `setup`, `client`, and scene-only export
use the most-free device and therefore support one-GPU hosts. CLI flags override
exported `POLICY_GPU` and `ENV_GPU`, which override the `auto` default:

```bash
make eval POLICY_GPU=0 ENV_GPU=1
export POLICY_GPU=0 ENV_GPU=1
make eval
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
  --policy-gpu auto \
  --env-gpu auto \
  --format json
```

Setup provides the same human and JSON reporting style. Direct CLI callers can
run every stage or repeat `--only` to select stages:

```bash
uv run --locked robodojo setup --only root
uv run --locked robodojo setup --only assets \
  --task general_pickup --env-cfg bimanual_yam --scene molmo_yam
uv run --locked robodojo setup --only policy \
  --policy-dir XPolicyLab/policy/Pi_05 --task general_pickup \
  --ckpt pi05_yam_molmoact2 --policy-env uv --env-cfg bimanual_yam \
  --action-type joint
```

The preflight JSON object has a stable top-level `status` and a `checks` array whose
records contain `name`, `status`, `detail`, and optional `remediation`.

Real `eval`, `server`, `smoke`, and `benchmark` commands run fast preflight
before selecting a free port or starting policy/simulator processes. A sweep
runs the shared gate once, not once per child task. Dry runs intentionally skip
preflight; automatic selectors still inspect `nvidia-smi` so the rendered
commands contain concrete devices, while numeric dry runs remain GPU-query-free.
Scene-only export runs simulator-side preflight and does not require a policy
GPU, adapter, checkpoint, or publication configuration. Make launch targets use
uv's `--no-sync` mode, so a missing or stale root environment fails with
`make setup` instead of changing the environment during launch.
