# OpenArm

RoboDojo provides the generic `openarm` environment profile. It reuses the
standard RoboDojo scene and ARX X5 evaluation layouts while selecting the
OpenArm embodiment, its 30 Hz camera rig, and its 30 Hz simulator timing.

The LeRobot Pi0.5 folding checkpoint uses the profile's 16-dimensional,
right-first action contract and three cameras: one Waveshare base camera and
two Arducam wrist cameras. Cloth folding remains an ordinary RoboDojo task and
does not have a separate scene or environment profile.

## Build the assets

The builder pins the OpenArm Isaac Lab source and LeRobot hardware
modifications, then generates the extended-arm functional twin, enlarged jaws,
camera holders, and named optical frames.

The generated wrist-camera holder assets remain available, but the camera
profile does not instantiate their geometry because it occludes the wrist
views. Both wrist cameras attach directly to link 7 at `[0.02, 0.0, 0.12]`
with intrinsic XYZ orientation `[180.0, 0.0, -90.0]`. The head-camera holder
remains enabled.

```bash
uv run --extra sim --locked robodojo assets build-openarm
```

Generated assets are written under `.robodojo/assets/Robots/openarm/` by
default and remain untracked. Authoritative source revisions, checksums, build
parameters, and the generated robot configuration live in
`configs/tooling/openarm.yml`.

## Install the folding policy

```bash
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
PYTHONPATH=. conda run -n lerobot-pi05 python \
  XPolicyLab/policy/LeRobot_Pi05_OpenArm/smoke_test.py
```

Verify the installation with:

```bash
uv run --extra sim --locked robodojo doctor \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --env-cfg openarm \
  --ckpt folding_final --policy-env lerobot-pi05
```

## Evaluate

Run one recorded episode:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 --eval-num 1
```

Export the composed pre-rollout scene without starting the policy:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm --action-type joint --seed 0 \
  --layout-id 0 --env-gpu 0 --policy-env lerobot-pi05 \
  --export-scene-only
```

The default export is `scene_snapshot/` inside the evaluation run directory.
Use `--export-scene-dir PATH` to select another ignored artifact directory.

For a transport and camera smoke test without checkpoint weights, set
`ROBODOJO_OPENARM_ZERO_ACTION=1` and optionally limit the rollout with
`ROBODOJO_OPENARM_SMOKE_STEPS`.
