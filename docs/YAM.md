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
