# Bimanual YAM setup contracts

`bimanual_yam` is the internal shared policy contract for YAM and is not
directly selectable. Use `bimanual_yam_molmoact2` for the classic MolmoAct2
roots, world-mounted top camera, and light-gray D405 housings. Use
`bimanual_yam_moonlake_office` for the Moonlake roots, fixture-mounted top
camera, and charcoal D405 housings. Both setups share the same 30 Hz
observation, 14-value action, controller, max-open reset, and finger-collision
contract.

Selectable environment profiles are concrete embodiment realizations, not
policy interfaces. A measured tuning candidate receives a versioned name,
extends one of these setups, and replaces only the changed sim, robot, or camera
component references. It may change geometry, collision, dynamics, controller,
reset, calibration, camera, or appearance while retaining
`embodiment: bimanual_yam`; changing the state/action schema requires a
new base policy contract. `variant.derived_for` records which policy, scene, or
protocol motivated a candidate but does not make the environment exclusive to
that policy. The scene must still list every intended environment profile
explicitly, and every evaluated combination remains an explicit recipe or
complete manual selection.

The normal setup workflow reads the selected profile's explicit `asset_builds`
and builds the licensed I2RT-derived runtime asset before launch:

```bash
make setup RECIPE=molmoact2-bimanual_yam-molmo_yam-general_pickup
```

For a granular support operation, use
`uv run --extra sim --locked robodojo assets build-yam`.

The build checks out I2RT at the revision pinned in
`configs/tooling/yam.yml`, snapshots the original inputs and license, produces
a normalized URDF with convex collisions, converts it to `YAM.usd`, and writes
a checksummed manifest below `.robodojo/assets/Robots/yam/`. Generated assets
remain untracked.

Re-run `make setup` after updating RoboDojo. Existing generated YAM
assets are not migrated in place. In particular, the canonical build now
publishes the generated `gripper/wrist_camera_mount` frame; an asset that still
contains the historical `molmo_link6` prim is stale. The build retains the
historical reference revision as provenance for the frame transform.

The shared policy contract is 30 Hz RGB plus 14 absolute values ordered as left
six joints, left normalized gripper, right six joints, and right normalized
gripper. Camera keys are `cam_head`, `cam_left_wrist`, and
`cam_right_wrist`. The source, geometry, controller, reset, camera, and final
hardware-calibration contracts are recorded in
`configs/reference/bimanual_yam.yml`. Checkpoint revisions, inference horizons,
executed chunk prefixes, policy camera mappings, and policy-specific joint-sign
bridges belong to the policy adapter and are intentionally absent from that
reference. These values are declared in each adapter's `eval_contracts.yml` and
implemented by its policy code; the simulator applies the returned commands at
the environment's control rate, while the protocol alone owns episode length.

## Independent workspace selection

Scene selection comes only from the explicit recipe or the complete four-part
manual selection; task metadata never selects it. The typed `molmo_yam` scene profile
selects its rendering component, an explicitly bundled layout source, and typed
asset recipes required by those layouts. Bundled layouts are never silently
shadowed by a downloaded layout with the same name; `default` and `conveyor`
instead select their downloaded asset layouts explicitly. A scene can be
composed with any compatible policy and embodiment. Its geometry, fixture
transforms, physics materials, camera stand, HDR, and intensity match `default`;
the room appearance also remains unchanged.

Replayed layouts retain their geometry and physics but replace the legacy white
tabletop material with the packaged material-0122 Mahogany MDL. This is the
only scene-level appearance override: keeping the referenced room materials is
required for the classic MolmoAct2 wrist-camera domain. Scene profiles do not
override the robot, camera, task, or policy contract. Camera resolution/aspect
ratio, intrinsics, focal length, and mounts at acquisition are owned by the
camera component selected by the environment profile: Moonlake uses 640x480
streams, while the classic MolmoAct2 setup uses 640x360. Scene exports and
evaluation videos retain that source geometry. Checkpoint-facing projection is
owned separately by the policy adapter. The pinned public MolmoAct2 YAM
checkpoint presents Moonlake RGB through an exact vertical center crop from
rows 60 through 419, preserving focal scale while mapping the source principal
point from `cy=240` to the checkpoint's `cy=180`. The generic PI0.5 YAM
adapter applies the same 640x360 source-geometry contract before OpenPI's
checkpoint resize; the simulator still records the native Moonlake stream.
Moonlake robot roots also belong to its environment profile, not its scene
profile.
The base YAM environment also declares robot-root offsets in the scene-owned
`Table` frame. Preflight and simulator startup validate those offsets against
the selected saved layout; they never rewrite robot or object poses.

For `general_pickup`, every policy and scene receives RoboDojo's canonical
`Pick up the <target> by 10 cm.` instruction, 200-step limit, `target` label,
and 10 cm lift reward. The classic layout retains its single green ball, while
the Moonlake protocol covers 20 tracked ball layouts across four table-depth
rows, five lateral positions, and four colors. The task YAML owns the label and
support plane, while each selected saved layout owns the exact replay position.
No environment or scene profile overrides the instruction.

`moonlake_office_general_pickup` is an explicit 20-episode benchmark protocol
over that unchanged task. It selects the `general_pickup` layout family,
applies a 400-step horizon, and is compatible only with the Moonlake scene.
Result paths and publication identities use the protocol name; artifacts
record both the protocol and its `general_pickup` base task. Runtime evaluation
fails instead of silently accepting a partial result if the requested episode
count exceeds the available layouts.

Container placement is a separate behavior under `pack_item_into_container`.
That task owns the `item` and `container` roles, box-and-lid instruction,
containment/closure reward, 1300-step horizon, six packing layouts, and the
`moonlake_packing` task asset build. None of those dependencies are inherited
by `general_pickup`.

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

Evaluation results and resume manifests record the policy descriptor and
execution summary, policy and environment profile names, domain-shift label,
embodiment and scene asset manifests, and both repository revisions. Scene
exports record separate hashes for the scene profile, the exact ordered layout
set, and prepared scene assets.
A resumed run is rejected if any of those inputs changed.

For example, a YAM evaluation may opt into that workspace explicitly:

```bash
uv run --extra sim --locked robodojo eval \
  --recipe molmoact2-bimanual_yam-molmo_yam-general_pickup
```

The same scene remains independently composable with other policies:

```bash
uv run --extra sim --locked robodojo eval \
  --recipe molmoact2-bimanual_yam-molmo_yam-fold_clothes
```

The released PI0.5 fine-tune uses the same embodiment and dataset-frame
contract. Prepare its pinned checkpoint once, then evaluate it with the scene
selected independently:

```bash
bash XPolicyLab/policy/Pi_05/prepare_checkpoint.sh pi05_yam_molmoact2
uv run --extra sim --locked robodojo eval \
  --recipe pi05-bimanual_yam-molmo_yam-general_pickup
```

`pi05_yam_molmoact2` consumes the shared `bimanual_yam` policy contract, but it
does not require the `molmo_yam` scene. The latter is the recommended
training-matched appearance. Selecting Moonlake instead is allowed and recorded
as domain shift. The pickup-trained `pi05_yam_abc_pickplace` profile declares
Moonlake as its reference setup and retains its grasp-triggered 8/50-action
execution and joint-height correction inside XPolicyLab. That correction
depends on the predicted action chunk and is therefore policy execution logic,
not a static embodiment calibration.

## Canonical appearance and hardware calibration

The generated arm aliases and D405 housings are setup-owned render assets.
`YAM_molmoact2.usd` and `YAM_moonlake_office.usd` have the same physics hash;
their D405 proxies provide the explicit light-gray and charcoal visual
domains. Provenance remains pinned in `configs/tooling/yam.yml`.

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
  --recipe molmoact2-bimanual_yam-molmo_yam-fold_clothes \
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
