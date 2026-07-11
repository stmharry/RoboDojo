# Layered OpenARM cloth-folding camera rigs

RoboDojo exposes two sensor profiles over one embodiment, scene, policy tensor
contract, and upstream camera stand. `openarm_cloth_folding` is the checkpoint's
policy-original rig. `openarm_cloth_folding_dyna` substitutes only the
availability-driven Waveshare base sensor and projection and is the evaluated
counterpart. Wrist cameras and every embodiment parameter are identical.
Both profiles use the generated CAD holder hierarchy; only DYNA is run and
published by the workflow below.

Camera configuration is normalized into four layers: sensor identity and stream,
mount target and optical roll, projection, and capture key/annotators. Scene and
robot managers publish fixture and link mounts; the camera manager resolves them.
Existing flat camera YAML remains supported by the legacy normalizer.

## Setup

```bash
uv run --locked bash scripts/assets/build_openarm_cloth_folding.sh
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
PYTHONPATH=. conda run -n lerobot-pi05 python \
  XPolicyLab/policy/LeRobot_Pi05_OpenArm/smoke_test.py
uv run --locked bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --env-cfg openarm_cloth_folding_dyna \
  --ckpt folding_final --policy-env lerobot-pi05
```

## One recorded episode

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --locked bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding_dyna --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 \
  --eval-num 1
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
  --env-cfg openarm_cloth_folding_dyna --action-type joint --seed 0 \
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

The adapter follows LeRobot's pinned real-robot evaluation loop: one
`predict_action_chunk` call per inference, asynchronous Real-Time Chunking with
a 30-action queue, execution horizon 20, guidance 5.0, LINEAR prefix attention,
relative-prefix re-anchoring, and 30→90 Hz action interpolation. The simulator
holds those targets for a repeating 3/3/2 physics-tick pattern at 240 Hz while
observations, rewards, and videos remain at 30 Hz.

## First-frame and visual validation

Run a hold-position episode without loading checkpoint weights, then validate
its camera streams against pinned reference frames:

```bash
ROBODOJO_OPENARM_ZERO_ACTION=1 ROBODOJO_OPENARM_SMOKE_STEPS=30 \
  ROBODOJO_OPENARM_VALIDATION_MASKS=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  uv run --locked bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding_dyna --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 \
  --eval-num 1

uv run --locked python scripts/validate_openarm_visuals.py \
  /path/to/generated/run --profile-id openarm_dyna --allow-partial \
  --mask-run-dir /path/to/generated/run
```

Generate the CAD-anchor diagrams, pinned article contact sheets, dataset-state
extracts, and the machine-readable calibration report with:

```bash
PYTHONPATH=. conda run -n lerobot-pi05 python \
  scripts/calibrate_openarm_cameras.py /path/to/generated/run \
  --output-dir /path/to/generated/run/camera_calibration
```

For a pose-matched zero-action capture, export the compact JSON object written
to `camera_calibration/matched_state_environment.json` as
`ROBODOJO_OPENARM_CALIBRATION_STATES`. The evaluator writes the pinned 16-D
states immediately before observations 0, 10, and 30; this affects only the
calibration harness and never a policy rollout.

The validator streams frames 0, 10, and 30 directly from the pinned episode
video URLs into `.cache/`; it does not download or process the training dataset.
It produces base/left/right comparison sheets and `visual_validation.json`.
Wrist acceptance uses holder/gripper instance masks, including the requirement
that both silhouettes enter from the bottom of their frames; cloth pixels never
participate in that gate.
The official data is a validation oracle only: it is not used to optimize
camera transforms, robot geometry, cloth state, or scoring.

The adapter always uses the checkpoint's prompt `Fold the T-shirt properly.`,
right-first 16-D state/action ordering, saved normalization processors, and
30-step chunks. Generated assets, checkpoints, and results remain ignored.

The tracked source manifest pins OpenARM Isaac Lab at
`bad82e23716e6941c2de78ccb978f57c78b37734` and the supplied hardware changes
at `ffe34b93c070343042eb9412fbfeffce16139947`. The builder checksums and
instantiates `head camera holder v4.stl` and `arducam_holder.step/.stl`, records
named `MountFrame` and `OpticalFrame` prims, and authors collision-enabled
holder USDs. A camera mount specifies the final named optical-frame target;
CameraManager derives the holder attachment as `target × inverse(optical)`.
The wrist targets retain the original link-7 centers `[0.05, 0, 0.12]` and
`[0.035, 0, 0.12]`, their original viewing axis, and zero runtime roll. This is
a pure left `+90°` / right `−90°` correction from the previous formulation.
Validation uses instance masks for holders and grippers so black cloth cannot
be classified as robot silhouette. Isaac Sim 5.1's tiled
Replicator path renders its native OpenCV-fisheye schema black, so the manager
uses a calibrated pinhole backing projection and applies the same explicit
equidistant model deterministically to captured RGB frames.

Authoritative references:

- [Pinned official folding setup](https://huggingface.co/spaces/lerobot/robot-folding/tree/170e1d479579e0b4be1afe0c99ebf868b24803db)
- [Pinned camera-holder CAD](https://huggingface.co/datasets/lerobot/openarms-hardware-modifications/tree/ffe34b93c070343042eb9412fbfeffce16139947)
- [Pinned LeRobot evaluation script](https://github.com/huggingface/lerobot/blob/1396b9fab7aecddd10006c33c47a487ffdcb54b4/examples/rtc/eval_with_real_robot.py)
- [Pinned checkpoint model card](https://huggingface.co/lerobot-data-collection/folding_final/blob/695abe40dbf3aac04efda59c1501d748681fa0fb/README.md)
- [DYNA camera note](https://moonlakeai.slack.com/archives/C0BCJPA3T9R/p1782508159343489)
- [Waveshare OV2710 Camera (A), SKU 14121](https://www.waveshare.com/wiki/OV2710_2MP_USB_Camera_%28A%29)

Validation-only references:

- [Pinned training dataset](https://huggingface.co/datasets/lerobot-data-collection/level2_final_quality3/tree/2e1b2e913cd367d74dc4481736954eed4a051ddc)
- [Interactive episode viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=lerobot-data-collection/level2_final_quality3)
