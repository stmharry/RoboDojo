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

Rebuild with `make assets-yam` after updating RoboDojo. Existing generated YAM
assets are not migrated in place. The canonical build removes an empty
`/yam/root/visuals` prim left by the Isaac URDF importer for the geometry-free
root link; it does not replace that prim with the visible base or duplicate any
robot geometry.

The policy-facing contract is 30 Hz RGB plus 14 absolute joint values ordered
as left six joints, left normalized gripper, right six joints, right normalized
gripper. Camera order is `cam_head`, `cam_left_wrist`, then
`cam_right_wrist`. The full pinned source and observation contract is recorded
in `configs/reference/bimanual_yam.yml`.

The profile uses the isolated `molmo_yam` scene. Its geometry, fixture
transforms, physics materials, camera stand, HDR, and intensity are identical
to `default`; only visible appearance differs. The referenced room receives a
stronger off-white PreviewSurface and the material-0122 Mahogany table
receives a deterministic solid-brown fallback because its MDL currently
renders near-white on the generated cube. The fallback matches color, not
wood-grain detail. Runtime visual audit must verify the table is no longer
near-white and the room's cabinet materials do not show through.

Both wrist mounts render the generated `D405_proxy.usd` housing with collision
and physics disabled. The `D405` default prim publishes an identity
`OpticalFrame` child so a referenced holder retains the named frame and the
configured camera pose exactly. No housing is attached to the top camera.

## Matched-frame wrist calibration

`configs/reference/bimanual_yam_matched_frames.yml` pins 24 released frames,
eight each from `allenai/18122025-foldclo-01`,
`allenai/18122025-foldclo-13`, and `allenai/24122025-foldclo-05`. The first two
datasets are training evidence and the third is held out. Source images,
simulator renders, and reports remain untracked. The checked-in contract stores
immutable provenance and selected sample IDs; the checksummed companion
`bimanual_yam_landmark_annotations.json` stores the exact released states,
camera/video checksums, uncertainty, mesh derivation, and 192 reviewed 2-D/3-D
landmarks without redistributing any source image or video.

The released wrist views support distal clamp/fingertip landmarks only. The
arm is outside both wrist fields of view and the reviewed images contain only
the gripper edges; no additional arm feature remains visible and stable across
the selected states. The fit therefore does not invent arm landmarks or infer
them from simulator renders.

Inspect the fail-closed status without downloading or changing data:

```bash
uv run --locked python scripts/fit_yam_matched_frames.py \
  --manifest configs/reference/bimanual_yam_matched_frames.yml
```

Reproduce the accepted bounded fit with:

```bash
uv run --locked python scripts/fit_yam_matched_frames.py --fit \
  --manifest configs/reference/bimanual_yam_matched_frames.yml \
  --camera-config configs/camera/bimanual_yam.yml \
  --report .robodojo/calibration/molmo_yam/fit_report.json
```

The fitter projects mesh-derived `molmo_link6` points through the configured
pinhole poses and derives image Jacobians at runtime. It solves only the first
two datasets, mirrors the right correction, and uses the third only for
acceptance. Shared correction is limited to 5 mm/2 degrees; per-arm residuals
to 2 mm/0.5 degree; and an optional visual clamp to 3 mm/1 degree. Held-out
median error must be at most 8 px and improve by at least 30%. A `complete`
manifest persists the accepted metrics and final left/right camera poses.

The accepted held-out median is 4.657 px, down from 29.618 px (84.3%); the
left/right corrected medians are 4.392 px and 6.025 px, respectively. Camera
corrections are per-arm residuals around one mirrored shared fit and are
composed after SAPIEN-to-USD conversion and optical roll, before named D405
frame alignment. Two separate 3 mm/1 degree visual-only corrections move the
left and right jaw render prims symmetrically in each arm's `molmo_link6`
frame. Their runtime adapter rejects any collision or rigid-body prim; physics,
contacts, joint state, and policy observations remain unchanged.

The persisted fit also records a reset-geometry refinement for the left wrist:
the fitted rotation-vector parameters receive `[+0.8, -0.25, 0]` degrees after
the two-session least-squares fit. This moves seed-0/layout-0 cloth visibility
from 23.77% to 25.27% at 640x360 while the residual remains 0.4353 degrees,
inside the 0.5-degree per-arm bound. The visual clamp is then re-fitted. On the
held-out collection session the resulting median landmark error is 4.64 px for
the left wrist and 6.02 px for the right (4.79 px aggregate), still an 83.8%
reduction from baseline. The refinement, source revision, and projection gate
are recorded in the matched-frame reference contract.

Released near-home frames use grippers around `g=0.98–1.0`; the intentionally
preserved simulator reset at `q=-0.02 m` is only `g≈0.421`. Reset contact sheets
therefore cannot assess wrist aperture alignment. Matched replay commands each
released 14-D state, including `q=-0.0475g`, before capture. This workflow does
not change head pose, intrinsics, roots, reset, actions, or collisions.

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
