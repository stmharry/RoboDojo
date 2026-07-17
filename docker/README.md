# Run RoboDojo in Docker

The image contains the RoboDojo simulator/client package and its `sim` extra.
Policy models and policy-specific environments remain outside the container;
the client connects to an XPolicyLab server over WebSocket.

## Prerequisites

- Linux with a CUDA 12.8-compatible NVIDIA driver.
- `uv` installed on the host.
- RoboDojo assets downloaded with `uv run --locked robodojo workspace assets download`.
- A reachable policy server bound to a non-loopback interface for remote clients.

Install Docker and the NVIDIA runtime when needed:

```bash
uv run --locked robodojo workspace docker install
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

## Build

```bash
uv run --locked robodojo workspace docker build --image robodojo:cuda12.8
```

The Dockerfile installs the committed lockfile with `--extra sim`, copies the
package and configuration tree, and uses `robodojo` as its entrypoint.

## Smoke evaluation

Start a policy server on the host or another machine, then run:

```bash
uv run --locked robodojo workspace docker smoke \
  --image robodojo:cuda12.8 \
  --policy-port 9999 \
  --policy pi05_arx_x5 \
  --task-protocol stack_bowls \
  --environment arx_x5
```

The default Docker workflow uses host networking and mounts the single local
`.robodojo/` storage root read-write. Override the policy and environment
options for your external server.

When `AWS_PROFILE` is set, `~/.aws` is mounted read-only and the profile is
forwarded. `ROBODOJO_AWS_ENV_FILE` may name an explicit protected Docker env
file; the workflow rejects missing files.

## Manual invocation

Because `robodojo` is the image entrypoint, commands follow the normal CLI:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v "$PWD/.robodojo:/workspace/RoboDojo/.robodojo" \
  robodojo:cuda12.8 \
  eval client --policy-profile pi05_arx_x5 --environment arx_x5 \
  --scene default --task-protocol stack_bowls \
  --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1
```

Use `robodojo workspace docker monitor` for an existing smoke log and
`robodojo workspace docker clean` to remove labeled leftover smoke containers.
