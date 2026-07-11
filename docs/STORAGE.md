# S3-backed storage

RoboDojo supports a read-only Mountpoint for S3 as its durable data view. It
does not write through Mountpoint: downloads, evaluations, and training first
write to a local POSIX filesystem and publish completed payloads with AWS CLI.

## Configuration

On `mv-53`, configure the service environment outside the repository:

```bash
export ROBODOJO_STORAGE_ROOT=/home/harry/s3/moonlake-harry-data/robodojo
export ROBODOJO_S3_URI=s3://moonlake-harry-data/robodojo
export ROBODOJO_LOCAL_SCRATCH_ROOT=/path/to/local/nvme/robodojo
```

The canonical durable layout is:

```text
assets/
datasets/
model_weights/<policy>/<checkpoint>/
runs/eval_result/RoboDojo/
runs/smoke/
```

`ROBODOJO_ASSETS_ROOT`, `ROBODOJO_DATA_ROOT`,
`ROBODOJO_MODEL_ROOT`, `ROBODOJO_CHECKPOINT_ROOT`,
`ROBODOJO_EVAL_ROOT`, and `ROBODOJO_RUN_ROOT` may override individual read
roots. With no storage variables, the existing repository-local defaults are
unchanged. `ROBO_DOJO_DATA_ROOT` remains a legacy data-root fallback.

Summaries always read the durable evaluation root, but storage mode writes the
generated Markdown to local scratch at `runs/reports/_summary.md`. Override the
destination with `ROBODOJO_SUMMARY_PATH` or, at highest precedence:

```bash
bash scripts/robodojo.sh summarize --output /local/path/summary.md
```

Do not put AWS credentials in this repository or `.env`. The storage helper
uses the standard AWS CLI credential provider chain. The mount must remain
read-only even when the publishing IAM user has prefix-scoped write access.
On mv-53, use `AWS_PROFILE=robodojo-runtime`. The Docker smoke helper passes the
profile name and mounts `~/.aws` read-only. A protected Docker-compatible
`ROBODOJO_AWS_ENV_FILE` remains supported for hosts that cannot use a named
profile, but it is not required in the project `.env`.

## Publication

Inspect the configured read mount, local scratch, and AWS CLI availability:

```bash
bash scripts/robodojo_storage.sh doctor
# Equivalent read-only status check:
bash scripts/robodojo_storage.sh status
```

The wrapper is managed by `uv` and executes against the locked project
environment with `--frozen --no-sync`; it never creates a separate tool venv.

Publish materialized payloads (never Git/LFS metadata or tool cache layouts):

```bash
bash scripts/robodojo_storage.sh publish-assets /local/stage/Assets
bash scripts/robodojo_storage.sh publish-data RoboDojo_demo /local/stage/RoboDojo_demo
bash scripts/robodojo_storage.sh publish-checkpoint SmolVLA run-10000 /local/stage/run-10000
bash scripts/robodojo_storage.sh publish-reference-cache openarm REVISION /local/stage/reference
# Generic form for an approved canonical subpath:
bash scripts/robodojo_storage.sh publish /local/stage/payload datasets/example
```

The helper uploads `_MANIFEST.json` and then `_COMPLETE.json` last. Completed
destinations are immutable unless `--replace` is explicitly requested. A
failed publication leaves its local source intact. Publication excludes nested
`.cache` and `.git` trees, lock files, and partial/incomplete temporary files;
materialized weights, model shards, tokenizers, and configuration files remain.

Evaluations write videos, resume manifests, and `_result.json` below local
scratch. A successful `eval_policy.sh` invocation publishes the completed run
to `runs/eval_result/RoboDojo/`; `_result.json` and `_COMPLETE.json` are the
last objects uploaded. Absolute `--ckpt` paths remain unchanged for the policy
server and use a safe basename in the result directory. Use `--ckpt-label` to
choose a stable label.

Active checkpoints and optimizer state must remain local. Publish a checkpoint
only after its writer has closed it. Policies that need POSIX behavior or lower
latency can materialize a completed checkpoint locally:

```bash
bash scripts/robodojo_storage.sh materialize-checkpoint SmolVLA run-10000 /local/runtime/run-10000
# Generic alias for a completed manifested payload:
bash scripts/robodojo_storage.sh hydrate /storage/robodojo/model_weights/SmolVLA/run-10000 /local/runtime/run-10000
```

Optional compatibility links are local-only and must name their destination
explicitly. The helper never replaces a real directory or a mismatched link:

```bash
bash scripts/robodojo_storage.sh link assets "$PWD/Assets"
bash scripts/robodojo_storage.sh link datasets "$PWD/data"
bash scripts/robodojo_storage.sh link checkpoint /local/policy/checkpoints/run-10000 \
  --policy SmolVLA --checkpoint run-10000
```

Keep Git/LFS checkouts, Hugging Face lock/symlink caches, `.venv`, uv/pip,
Omniverse, Warp, CUDA, Triton, TorchInductor, JAX, temporary video streams, and
distributed-training state on local storage. Only materialized immutable
payloads belong in the durable prefix.
