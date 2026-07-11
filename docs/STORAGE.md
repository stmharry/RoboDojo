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
robodojo summarize --output /local/path/summary.md
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
robodojo storage doctor
# Equivalent read-only status check:
robodojo storage status
```

Run storage commands through `uv run --locked robodojo storage ...`; they use
the same lightweight locked project environment as the rest of the CLI.

Publish materialized payloads (never Git/LFS metadata or tool cache layouts):

```bash
robodojo storage publish-assets /local/stage/Assets
robodojo storage publish-data RoboDojo_demo /local/stage/RoboDojo_demo
robodojo storage publish-checkpoint SmolVLA run-10000 /local/stage/run-10000
robodojo storage publish-reference-cache openarm REVISION /local/stage/reference
# Generic form for an approved canonical subpath:
robodojo storage publish /local/stage/payload datasets/example
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
robodojo storage materialize-checkpoint SmolVLA run-10000 /local/runtime/run-10000
# Generic alias for a completed manifested payload:
robodojo storage hydrate /storage/robodojo/model_weights/SmolVLA/run-10000 /local/runtime/run-10000
```

Optional compatibility links are local-only and must name their destination
explicitly. The helper never replaces a real directory or a mismatched link:

```bash
robodojo storage link assets "$PWD/Assets"
robodojo storage link datasets "$PWD/data"
robodojo storage link checkpoint /local/policy/checkpoints/run-10000 \
  --policy SmolVLA --checkpoint run-10000
```

Keep Git/LFS checkouts, Hugging Face lock/symlink caches, `.venv`, uv/pip,
Omniverse, Warp, CUDA, Triton, TorchInductor, JAX, temporary video streams, and
distributed-training state on local storage. Only materialized immutable
payloads belong in the durable prefix.
