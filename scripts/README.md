# Compatibility scripts

RoboDojo workflow logic lives in the installable `robodojo` package and is
exposed through the Typer CLI:

```bash
uv run --extra sim --locked robodojo --help
```

`eval_policy.sh` is the sole root-owned shell script. It is a private,
logic-free compatibility adapter for unchanged XPolicyLab launchers that still
call this repository path. New integrations should invoke `robodojo eval client`
or `robodojo eval run` directly.

Policy-specific scripts remain owned by the `XPolicyLab` submodule.
