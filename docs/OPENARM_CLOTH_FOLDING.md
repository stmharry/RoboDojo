# Slack-mirrored OpenARM cloth folding

This profile mirrors the real LeRobot cloth-folding setup: dual OpenARM with
the 5 cm upper-arm extension and larger jaws, a white folding table, one base
camera, two wrist cameras, and a native LeRobot π0.5 checkpoint.

## Setup

```bash
bash scripts/assets/build_openarm_cloth_folding.sh
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/install.sh lerobot-pi05
bash XPolicyLab/policy/LeRobot_Pi05_OpenArm/download_checkpoint.sh lerobot-pi05
conda run -n lerobot-pi05 python \
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

The adapter always uses the checkpoint's prompt `Fold the T-shirt properly.`,
right-first 16-D state/action ordering, saved normalization processors, and
30-step chunks. Generated assets, checkpoints, and results remain ignored.

The tracked source manifest pins OpenARM Isaac Lab at
`bad82e23716e6941c2de78ccb978f57c78b37734` and the supplied hardware changes
at `ffe34b93c070343042eb9412fbfeffce16139947`. Camera placement, fisheye
intrinsics, orientation, and black/white scene appearance were calibrated to
real dataset revision `2e1b2e913cd367d74dc4481736954eed4a051ddc` and remain
explicit in the camera and generated robot configuration YAML.
