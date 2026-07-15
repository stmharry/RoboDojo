# Bimanual YAM embodiment

The `bimanual_yam` environment profile owns the YAM robot, camera rig,
controllers, reset state, and state/action dimensions. It reuses the
`fold_clothes` task and compatible saved layouts without selecting a policy or
a policy-specific workspace.

Build the licensed I2RT-derived runtime asset before launching the profile:

```bash
make assets-yam
```

The build checks out I2RT at the revision pinned in
`configs/tooling/yam.yml`, snapshots the original inputs and license, produces
a normalized URDF with convex collisions, converts it to `YAM.usd`, and writes
a checksummed manifest below `.robodojo/assets/Robots/yam/`. Generated assets
remain untracked.

Rebuild with `make assets-yam` after updating RoboDojo. Existing generated YAM
assets are not migrated in place. In particular, the canonical build now
publishes the generated `gripper/wrist_camera_mount` frame; an asset that still
contains the historical `molmo_link6` prim is stale. The build retains the
historical reference revision as provenance for the frame transform.

The embodiment contract is 30 Hz RGB plus 14 absolute values ordered as left
six joints, left normalized gripper, right six joints, and right normalized
gripper. Camera keys are `cam_head`, `cam_left_wrist`, and
`cam_right_wrist`. The source, geometry, controller, reset, camera, and final
hardware-calibration contracts are recorded in
`configs/reference/bimanual_yam.yml`. Checkpoint revisions, inference horizons,
policy camera mappings, and policy-specific joint-sign bridges belong to the
policy adapter and are intentionally absent from that reference.

## Independent workspace selection

Scene selection comes from task metadata or the explicit `--scene` option; it
is not stored in an environment profile. The typed `molmo_yam` scene profile
selects its rendering component, an explicitly bundled layout source, and typed
asset recipes required by those layouts. Bundled layouts are never silently
shadowed by a downloaded layout with the same name; `default` and `conveyor`
instead select their downloaded asset layouts explicitly. A scene can be
composed with any compatible policy and embodiment. Its geometry, fixture
transforms, physics materials, camera stand, HDR, and intensity match `default`;
only visible room and tabletop appearance differs.

The referenced room receives an off-white PreviewSurface. Replayed layouts
retain their geometry and physics but replace the legacy white tabletop
material with the packaged material-0122 Mahogany MDL. Scene selection never
changes the robot, camera, task, or policy contract.

For `general_pickup`, `molmo_yam` selects a bundled, training-aligned
single-ball layout. The task's upstream instruction, lift reward, labels, and
episode limit remain unchanged.

The scene's typed `fold_clothes` recipe derives a topology-preserving
short-sleeve shirt from the downloaded `Top_Long/00009` garment before scene
construction. The generated catalog entry includes `object.usd`, inherited
metadata, and `derivation.json`. The derivation manifest pins source and output
hashes plus the transform and builder versions; generated USD references stay
catalog-relative so the artifact is portable across storage roots. Valid
outputs are reused, while changed inputs rebuild under a lock and publish
atomically. The recipe retains the source mesh topology and reward-point vertex
indices, places the garment at the table center in its native frame, and uses
the cloth spring parameters from
Isaac Sim 5.1's particle-cloth example. The canonical task instruction, reward,
labels, and episode limit remain unchanged. Newton VBD was also investigated,
but is not part of this profile because the pinned Isaac Sim 5.1 runtime does
not provide that backend.

Evaluation results, resume manifests, and scene exports record separate hashes
for the scene profile, the exact ordered layout set, and prepared scene assets.
A resumed run is rejected if any of those inputs changed.

For example, a YAM evaluation may opt into that workspace explicitly:

```bash
uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/MolmoACT2 \
  --task general_pickup \
  --ckpt molmoact2_bimanual_yam \
  --policy-env molmoact2 \
  --env-cfg bimanual_yam \
  --action-type joint \
  --scene molmo_yam
```

The same scene remains independently composable with other policies:

```bash
uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/MolmoACT2 \
  --task fold_clothes \
  --ckpt molmoact2_bimanual_yam \
  --policy-env molmoact2 \
  --env-cfg bimanual_yam \
  --action-type joint \
  --scene molmo_yam
```

The released PI0.5 fine-tune uses the same embodiment and dataset-frame
contract. Prepare its pinned checkpoint once, then evaluate it with the scene
selected independently:

```bash
bash XPolicyLab/policy/Pi_05/prepare_checkpoint.sh pi05_yam_molmoact2
uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task general_pickup \
  --ckpt pi05_yam_molmoact2 \
  --policy-env uv \
  --env-cfg bimanual_yam \
  --action-type joint \
  --scene molmo_yam
```

`pi05_yam_molmoact2` requires `bimanual_yam`, but it does not require the
`molmo_yam` scene. The latter is the recommended training-matched appearance.

## Canonical appearance and hardware calibration

The dark/light YAM skin and generated `D405_proxy.usd` housings are embodiment
assets. Their historical public image and geometry references remain pinned as
provenance in `configs/tooling/yam.yml`, but the generated material and prim
names are hardware-oriented rather than policy-oriented.

Both wrist mounts render a D405 housing with collision and physics disabled.
The `D405` default prim publishes an identity `OpticalFrame` child so a
referenced holder retains the named frame and configured camera pose exactly.
No housing is attached to the top camera.

The accepted left/right wrist corrections and visual-only jaw transforms are
retained under the `yam_hardware_calibration_v1` source tag. They are final
embodiment values, not a runtime fitting workflow. The jaw adapter targets only
the two render prims, rejects collision and rigid-body prims, and leaves
physics, contacts, and the proprioceptive state/action contract unchanged. The
camera corrections and jaw transforms intentionally affect rendered RGB.

The one-off released-frame fitter, frame manifest, landmark annotations, and
YAM matched-replay diagnostic used to establish those values have been
removed. OpenArm's independent calibration and replay workflow is unchanged.

## Post-reset visual audit

Scene-only exports can opt into a two-second, no-policy visual audit:

```bash
ROBODOJO_SCENE_VISUAL_AUDIT=1 uv run --extra sim --locked robodojo eval \
  --policy-dir XPolicyLab/policy/MolmoACT2 \
  --task fold_clothes \
  --ckpt molmoact2_bimanual_yam \
  --policy-env uv \
  --env-cfg bimanual_yam \
  --action-type joint \
  --scene molmo_yam \
  --seed 0 \
  --layout-id 0 \
  --export-scene-only
```

The post-reset USD remains in `scene_snapshot/`. The sibling `visual_audit/`
directory contains reset and held RGB PNGs for every camera, two contact
sheets, exact run metadata, and `metrics.json`. The hold advances exactly two
simulated seconds with reset actuator targets unchanged and does not call a
policy or the policy observation path.

`ROBODOJO_SCENE_VISUAL_AUDIT=1` is rejected unless `--export-scene-only` is
also present.
