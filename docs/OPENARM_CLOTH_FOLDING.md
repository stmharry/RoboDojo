# Protocol-corrected OpenARM cloth folding

This profile mirrors the documented LeRobot cloth-folding setup: dual OpenARM with
the 5 cm upper-arm extension and larger jaws, a white folding table, one base
camera, two wrist cameras, and a native LeRobot π0.5 checkpoint.

## Setup

```bash
bash scripts/assets/build_openarm_cloth_folding.sh
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
PYTHONPATH=. conda run -n lerobot-pi05 python \
  XPolicyLab/policy/LeRobot_Pi05_OpenArm/smoke_test.py
bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --env-cfg openarm_cloth_folding \
  --ckpt folding_final --policy-env lerobot-pi05
```

## One recorded episode

```bash
OMNI_KIT_ACCEPT_EULA=YES bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 \
  --eval-env RoboDojo --eval-num 1
```

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
  OMNI_KIT_ACCEPT_EULA=YES \
  bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/LeRobot_Pi05_OpenArm \
  --task fold_clothes --ckpt folding_final \
  --env-cfg openarm_cloth_folding --action-type joint --seed 0 \
  --policy-gpu 0 --env-gpu 1 --policy-env lerobot-pi05 \
  --eval-env RoboDojo --eval-num 1

conda run -n RoboDojo python scripts/validate_openarm_visuals.py \
  /path/to/generated/run --allow-partial
```

The validator streams frames 0, 10, and 30 directly from the pinned episode
video URLs into `.cache/`; it does not download or process the training dataset.
It produces base/left/right comparison sheets and `visual_validation.json`.
The official data is a validation oracle only: it is not used to optimize
camera transforms, robot geometry, cloth state, or scoring.

The adapter always uses the checkpoint's prompt `Fold the T-shirt properly.`,
right-first 16-D state/action ordering, saved normalization processors, and
30-step chunks. Generated assets, checkpoints, and results remain ignored.

The tracked source manifest pins OpenARM Isaac Lab at
`bad82e23716e6941c2de78ccb978f57c78b37734` and the supplied hardware changes
at `ffe34b93c070343042eb9412fbfeffce16139947`. Camera placement, fisheye
intrinsics, and orientation derive from the documented physical mounting roles,
published camera/lens specifications, and OpenARM CAD. They remain explicit in
the camera and generated robot configuration YAML. Isaac Sim 5.1's tiled
Replicator path renders its native OpenCV-fisheye schema black, so the manager
uses a calibrated pinhole backing projection and applies the same explicit
equidistant model deterministically to captured RGB frames.

Authoritative references:

- [Official folding setup](https://huggingface.co/spaces/lerobot/robot-folding)
- [Pinned LeRobot evaluation script](https://github.com/huggingface/lerobot/blob/1396b9fab7aecddd10006c33c47a487ffdcb54b4/examples/rtc/eval_with_real_robot.py)
- [Pinned checkpoint model card](https://huggingface.co/lerobot-data-collection/folding_final/blob/695abe40dbf3aac04efda59c1501d748681fa0fb/README.md)

Validation-only references:

- [Pinned training dataset](https://huggingface.co/datasets/lerobot-data-collection/level2_final_quality3/tree/2e1b2e913cd367d74dc4481736954eed4a051ddc)
- [Interactive episode viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=lerobot-data-collection/level2_final_quality3)
