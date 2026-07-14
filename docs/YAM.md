# Bimanual YAM cloth-folding baseline

The `bimanual_yam` environment profile is the public-upstream simulation
baseline for the MolmoAct2 YAM policy. It keeps RoboDojo's existing
`fold_clothes` task and ARX X5 layout data while replacing only the simulator,
robot, and camera contracts.

Build the licensed I2RT-derived runtime asset before launching the profile:

```bash
make assets-yam
```

The build checks out I2RT at the revision pinned in
`configs/tooling/yam.yml`, snapshots the original inputs and license, produces
a normalized URDF with convex collisions, converts it to `YAM.usd`, and writes
a checksummed manifest below `.robodojo/assets/Robots/yam/`. Generated assets
remain untracked.

The policy-facing contract is 30 Hz RGB plus 14 absolute joint values ordered
as left six joints, left normalized gripper, right six joints, right normalized
gripper. Camera order is `cam_head`, `cam_left_wrist`, then
`cam_right_wrist`. The full pinned source and observation contract is recorded
in `configs/reference/bimanual_yam.yml`.

## Post-reset visual audit

Scene-only exports can opt into a two-second, no-policy visual audit:

```bash
ROBODOJO_SCENE_VISUAL_AUDIT=1 uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/MolmoACT2 \
  --task fold_clothes \
  --ckpt molmoact2_bimanual_yam \
  --policy-env uv \
  --env-cfg bimanual_yam \
  --seed 0 \
  --layout-id 0 \
  --export-scene-only
```

The post-reset USD remains in `scene_snapshot/`. The sibling
`visual_audit/` directory contains reset and held RGB PNGs for every camera,
two contact sheets, exact run metadata, and `metrics.json`. The hold advances
exactly two simulated seconds with the reset actuator targets unchanged and
does not call a policy or the policy observation path. Cloth support is
reported as a geometric table-proximity measurement; direct cloth contact
counts are explicitly null because the current runtime does not expose a safe
contact-count API for this diagnostic.

`ROBODOJO_SCENE_VISUAL_AUDIT=1` is rejected unless
`--export-scene-only` is also present.
