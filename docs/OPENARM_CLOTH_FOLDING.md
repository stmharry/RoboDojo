# OpenArm cloth folding

RoboDojo provides one OpenArm cloth-folding environment:
`openarm_cloth_folding`. It uses the checkpoint-compatible 16-dimensional
right-first action contract and the available DYNA camera hardware: one
Waveshare base camera and two Arducam wrist cameras.

## Build the assets

The builder pins the OpenArm Isaac Lab source and the LeRobot hardware
modifications, then generates the extended-arm functional twin, enlarged jaws,
camera holders, and their named optical frames.

```bash
uv run --locked bash scripts/assets/build_openarm_cloth_folding.sh
```

Generated assets are written under `Assets/Robots/openarm/` and remain
untracked.

## Install the policy

```bash
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
PYTHONPATH=. conda run -n lerobot-pi05 python \
  XPolicyLab/policy/LeRobot_Pi05_OpenArm/smoke_test.py
```

Verify the complete installation with:

```bash
uv run --locked bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --env-cfg openarm_cloth_folding \
  --ckpt folding_final --policy-env lerobot-pi05
```

## Evaluate or smoke-test

Run one recorded episode:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --locked bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 --eval-num 1
```

### Export the composed pre-rollout scene

The normal evaluator can export the first fully reset scene immediately before
policy motion. Use `--export-scene` to continue the rollout after exporting, or
`--export-scene-only` to skip policy-server startup, checkpoint loading,
inference, and actions:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --locked bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding --action-type joint --seed 0 \
  --layout-id 0 --env-gpu 0 --policy-env lerobot-pi05 \
  --export-scene-only
```

The default output is `scene_snapshot/` inside the evaluation run directory;
`--export-scene-dir PATH` selects an explicit ignored artifact directory. The
bundle contains readable `scene_referenced.usda`, portable
`scene_flattened.usdc`, `scene_manifest.json`, and any reachable non-USD files
under `dependencies/`. The manifest is authoritative for the effective
postprocessed fisheye projection and records unresolved dependencies.

USD preserves the authored PhysX, articulation, collision, particle, and cloth
schemas plus observable reset state. PhysX contact caches, GPU buffers, tensor
handles, and solver warm-start state are runtime-only and cannot be serialized.
Generic USD viewers may also render MDL materials differently from Isaac RTX.

For a transport and camera smoke test without loading checkpoint weights, add
`ROBODOJO_OPENARM_ZERO_ACTION=1` and optionally limit the run with
`ROBODOJO_OPENARM_SMOKE_STEPS`.

The adapter preserves the checkpoint prompt, three-camera ordering, 30 Hz
observations, 30-step chunks, and the right-first 16-D state/action layout.

Authoritative build inputs are pinned in
`scripts/assets/openarm_sources.json`.
