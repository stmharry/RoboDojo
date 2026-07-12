# OpenArm

`openarm_lerobot` is the runnable reference profile for reproducing LeRobot's
published cloth-folding setup in simulation. It uses the upstream-modified
OpenArm embodiment, published camera models, the standard RoboDojo scene and
ARX X5 layouts, the right-first 16-D policy contract, and 30 Hz timing. The
ambiguous `openarm` profile remains removed.

The office-specific `openarm_wowrobo_v1_1` and `openarm_anvil_v2` profiles are
reserved for the later transfer phase. Each remains release-blocked until its manifest under
`configs/calibration/` contains measured robot, camera, and controller data.
Vendor specifications and another robot's calibration are not substitutes for
serial-numbered office measurements. Cloth folding remains an ordinary task
and does not have a separate scene or environment profile.

Promoting a manifest to `measured` requires checksummed raw captures plus the
robot geometry and gripper conversion, three-camera ChArUco calibration and
camera settings, and controller step-response fit listed in that manifest.
Populate the corresponding robot, camera, and simulator configuration only
from those results.

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
  --task fold_clothes --env-cfg openarm_lerobot \
  --ckpt folding_final --policy-env lerobot-pi05
```

## Evaluate

Run one recorded episode:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_lerobot --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 --eval-num 1
```

Export the composed pre-rollout scene without starting the policy:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_lerobot --action-type joint --seed 0 \
  --layout-id 0 --env-gpu 0 --policy-env lerobot-pi05 \
  --export-scene-only
```

The default export is `scene_snapshot/` inside the evaluation run directory.
Use `--export-scene-dir PATH` to select another ignored artifact directory.

## Diagnose action drift

Set `ROBODOJO_OPENARM_TRACE=1` to write `openarm_trace.jsonl` beside
`_result.json`. The trace contains model chunks, queue timing and indices,
measured states, interpolated targets, safety clamps, and initial camera image
statistics. Compare the adapter paths without changing the checkpoint:

```bash
# Legacy absolute-action re-anchoring behavior.
ROBODOJO_OPENARM_TRACE=1 ROBODOJO_OPENARM_RTC_MODE=current <evaluation command>

# Pinned LeRobot semantics: retain original model-space RTC leftovers.
ROBODOJO_OPENARM_RTC_MODE=official <evaluation command>

# Wait for each queue to drain and infer without an overlapping RTC prefix.
ROBODOJO_OPENARM_RTC_MODE=synchronous <evaluation command>
```

`official` is the default. `official` and `synchronous` enable tracing automatically. Set
`ROBODOJO_OPENARM_TRACE_PATH` to choose an explicit trace destination. Scene
exports record effective and published camera FOVs, their diagonal error, and
whether a zero-distortion fisheye postprocess is active.

The launcher stops office-profile runs with
`hardware calibration is not release-ready` while the selected office
manifest remains `pending_measurement`. These profiles must not substitute
office camera or controller values into the `openarm_lerobot` baseline.
