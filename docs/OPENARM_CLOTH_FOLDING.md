# OpenArm cloth folding

RoboDojo provides one OpenArm cloth-folding environment:
`openarm_cloth_folding`. It uses the checkpoint-compatible 16-dimensional
right-first action contract and the available DYNA camera hardware: one
Waveshare base camera and two Arducam wrist cameras.

## Build the assets

The builder pins the OpenArm Isaac Lab source and the LeRobot hardware
modifications, then generates the extended-arm functional twin, enlarged jaws,
camera holders, and their named optical frames.

The upstream OpenArm fingers include a dark gripping surface, and the
[physical assembly guide](https://github.com/enactic/openarm/blob/main/website/versioned_docs/version-1.0/hardware/assembly-guide/gripper-sub-assembly.mdx)
recommends applying 3M tape there. The upstream Isaac asset renders that region
as part of each rigid finger mesh; it is not a separate soft or deformable body.
The [cloth-folding hardware package](https://huggingface.co/datasets/lerobot/openarms-hardware-modifications)
provides enlarged `jaw_normal.stl` and AnySkin-compatible `jaw_anyskin.stl`
variants, but no standalone soft-pad or tape CAD. RoboDojo continues to use the
enlarged normal jaws.

The generated wrist-camera holder assets remain available, but this camera
profile sets their hardware mounts to `enabled: false` because their geometry
occludes the wrist-camera views. Scene construction therefore attaches both
wrist cameras directly to their link-7 targets at the configured optical poses;
it does not instantiate the wrist holder visuals or collisions. The head-camera
holder remains enabled.

```bash
uv run --extra sim --locked robodojo assets build-openarm
```

Generated assets are written under `.robodojo/assets/Robots/openarm/` by
default and remain untracked.

## Install the policy

```bash
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
PYTHONPATH=. conda run -n lerobot-pi05 python \
  XPolicyLab/policy/LeRobot_Pi05_OpenArm/smoke_test.py
```

Verify the complete installation with:

```bash
uv run --extra sim --locked robodojo doctor \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --env-cfg openarm_cloth_folding \
  --ckpt folding_final --policy-env lerobot-pi05
```

## Evaluate or smoke-test

Run one recorded episode:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
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
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
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
`configs/tooling/openarm/sources.json`.
