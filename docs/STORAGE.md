# Local storage and S3 publication

RoboDojo keeps all runtime data in one writable local root. By default this is
`.robodojo/` below the repository root, regardless of the directory from which
the CLI is invoked. Override it when needed:

```bash
export ROBODOJO_STORAGE_ROOT=/local/nvme/robodojo
```

The canonical layout is:

```text
assets/
datasets/
model_weights/<policy>/<checkpoint>/
runs/eval_result/RoboDojo/
runs/smoke/
runs/reports/
.cache/
.staging/
```

There are no checkout-local `Assets`, `data`, or `eval_result` compatibility
paths. Active evaluations and downloads write directly below the local storage
root. Keep Git/LFS metadata and temporary downloads below `.cache`; `.staging`
is reserved for transactional storage operations.

## Optional S3 remote

Configure a dedicated remote prefix and standard AWS CLI credentials:

```bash
export ROBODOJO_S3_URI=s3://your-bucket/robodojo
export AWS_PROFILE=robodojo-runtime
```

S3 is not mounted, and setting `ROBODOJO_S3_URI` alone never triggers a bucket
sync. Direct CLI evaluations stay local unless publication is explicitly
requested:

```bash
robodojo eval --publish <other-eval-options>
```

The Make workflow opts in by default with `PUBLISH=true`. Disable publication
for one run with `make eval PUBLISH=false`; only the `eval` target receives this
default, not client or sweep commands.

Inspect the writable local root and optional remote access with:

```bash
uv run --locked robodojo storage doctor
```

Publish a completed local payload explicitly:

```bash
robodojo storage publish-assets .robodojo/assets
robodojo storage publish-data demo .robodojo/datasets/demo
robodojo storage publish-checkpoint SmolVLA run-10000 /local/checkpoints/run-10000
robodojo storage publish .robodojo/datasets/example datasets/example
```

With `--publish`, a successful evaluation publishes its completed timestamped
run. RoboDojo validates the S3 prefix and AWS CLI before starting the expensive
evaluation; an upload failure returns nonzero but leaves the local result in
place. Payload files are uploaded first, followed by `_MANIFEST.json`,
`_result.json` when present, and `_COMPLETE.json` last. Completed remote
destinations are immutable unless `--replace` is explicit.

Restore exactly one manifested payload into its canonical local location:

```bash
robodojo storage pull assets
robodojo storage pull datasets/demo
robodojo storage pull model_weights/SmolVLA/run-10000
```

Pulls stage downloads below `.robodojo/.staging`, verify the completion marker,
manifest hash, file sizes, and SHA-256 digests, then install the payload. An
existing local destination is preserved unless `--replace` is supplied.

AWS credentials must not be committed or stored in the project `.env`. The
Docker smoke workflow mounts the same local storage root and passes the named
AWS profile or protected `ROBODOJO_AWS_ENV_FILE` when configured.
