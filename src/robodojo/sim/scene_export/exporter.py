"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from pxr import Usd, UsdGeom, UsdUtils

from robodojo.sim.environment.global_configs import ROOT_DIR
from robodojo.sim.scene_export.contracts import (
    SCENE_EXPORT_FORMAT_VERSION,
    ExportIdentity,
    completed_export_matches,
)
from robodojo.sim.scene_export.preview import PREVIEW_NAME, create_blender_preview
from robodojo.sim.scene_export.services.bundling import _bundle_flattened_assets, _make_referenced_paths_portable
from robodojo.sim.scene_export.services.manifest import (
    _config_hashes,
    _git_revision,
    _runtime_versions,
    _sha256_file,
    _source_revisions,
)
from robodojo.sim.scene_export.services.state_capture import _apply_snapshot_state, _live_guard, _stage_layer_stack
from robodojo.sim.scene_export.services.validation import _validate_flattened, _validate_reopened_stage

logger = logging.getLogger(__name__)

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MANIFEST_NAME = "scene_manifest.json"
REFERENCED_NAME = "scene_referenced.usda"
FLATTENED_NAME = "scene_flattened.usdc"


def export_scene_snapshot(env, output_dir: str | os.PathLike[str], layout_id: int) -> Path:
    """Export one fully reset environment and return the completed directory."""
    repo_root = Path(ROOT_DIR).resolve()
    output = Path(output_dir).expanduser().resolve()
    revision, dirty = _git_revision(repo_root)
    identity = ExportIdentity(
        task=str(env.task_name),
        task_protocol=str(env.task_protocol),
        episode_horizon=int(env.step_lim),
        evaluation_episodes=int(env.eval_cfg["evaluation_episodes"]),
        recipe=env.recipe,
        experiment_hash=env.experiment_hash,
        environment=str(env.environment),
        scene=str(env.scene),
        seed=int(env.eval_seed),
        layout_id=int(layout_id),
        repository_revision=revision,
        environment_profile_hash=str(env.environment_profile_hash),
        embodiment=str(env.embodiment),
        scene_profile_hash=str(env.scene_profile_hash),
        layout_set_hash=str(env.layout_set_hash),
        scene_asset_hash=str(env.scene_asset_hash),
    )
    if output.exists():
        if completed_export_matches(output, identity):
            logger.info("[scene-export] reusing completed snapshot: %s", output)
            return output
        raise FileExistsError(f"scene export directory already exists and does not match this run: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        stage = env.stage
        before = _live_guard(env, stage)
        state = before.pop("state")
        referenced_layer = UsdUtils.FlattenLayerStack(stage)
        referenced_stage = Usd.Stage.Open(referenced_layer)
        _apply_snapshot_state(referenced_stage, state)
        del referenced_stage
        referenced_dependencies = _make_referenced_paths_portable(referenced_layer, temporary, repo_root)
        referenced_layer.Export(str(temporary / REFERENCED_NAME))
        reopened_referenced = Usd.Stage.Open(str(temporary / REFERENCED_NAME), load=Usd.Stage.LoadAll)
        _validate_reopened_stage(reopened_referenced, state, "referenced")

        flattened_layer = stage.Flatten()
        flattened_stage = Usd.Stage.Open(flattened_layer)
        _apply_snapshot_state(flattened_stage, state)
        bundled, unresolved = _bundle_flattened_assets(flattened_stage.GetRootLayer(), temporary, repo_root)
        flattened_stage.GetRootLayer().Export(str(temporary / FLATTENED_NAME))

        reopened_flattened = Usd.Stage.Open(str(temporary / FLATTENED_NAME), load=Usd.Stage.LoadAll)
        _validate_reopened_stage(reopened_flattened, state, "flattened")
        external_arcs, internal_arcs = _validate_flattened(reopened_flattened)
        if external_arcs:
            raise RuntimeError(f"flattened export retained external USD composition arcs: {external_arcs}")

        preview = create_blender_preview(
            reopened_flattened,
            temporary,
            [camera["sensor_path"] for camera in state["cameras"]],
        )

        layout_path = Path(env.seed_manager.seed_info[int(layout_id)]["scene_layout"]).resolve()
        root_layer = stage.GetRootLayer()
        manifest = {
            "format_version": SCENE_EXPORT_FORMAT_VERSION,
            "complete": True,
            "identity": identity.to_dict(),
            "created_at": datetime.now(UTC).isoformat(),
            "snapshot_boundary": "post_reset_pre_rollout",
            "task": env.task_name,
            "task_protocol": env.task_protocol,
            "episode_horizon": int(env.step_lim),
            "evaluation_episodes": int(env.eval_cfg["evaluation_episodes"]),
            "recipe": env.recipe,
            "experiment_hash": env.experiment_hash,
            "environment": {
                "name": env.environment,
                "camera_profile_id": env.camera_rig.profile_id,
                "embodiment": env.embodiment,
                "sha256": env.environment_profile_hash,
            },
            "scene": env.scene,
            "scene_profile": {
                "component": env.scene_component,
                "layout_set": env.layout_set,
                "layout_source": env.layout_source,
                "sha256": env.scene_profile_hash,
                "layout_set_sha256": env.layout_set_hash,
                "scene_asset_sha256": env.scene_asset_hash,
            },
            "seed": int(env.eval_seed),
            "layout": {
                "id": int(layout_id),
                "path": str(layout_path),
                "sha256": _sha256_file(layout_path),
            },
            "repository": {"revision": revision, "dirty": dirty},
            "source_revisions": _source_revisions(repo_root),
            "config_sha256": _config_hashes(repo_root, env),
            "runtime_versions": _runtime_versions(),
            "stage": {
                "up_axis": UsdGeom.GetStageUpAxis(stage),
                "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                "time_codes_per_second": stage.GetTimeCodesPerSecond(),
                "source_root_layer": root_layer.identifier,
                "source_session_layer": stage.GetSessionLayer().identifier,
                "source_layer_stack": [layer.identifier for layer in _stage_layer_stack(stage)],
                "exported_start_time_code": 0,
                "exported_end_time_code": 0,
            },
            "simulation": {
                "dt": float(env.dt),
                "device": str(env.device),
                "use_fabric": bool(env.use_fabric),
                "timeline_time_at_capture": before["timeline_time"],
                "simulation_time_at_capture": before["simulation_time"],
            },
            "cameras": state["cameras"],
            "articulations": state["robots"],
            "cloth": state["cloth"],
            "prim_inventory": state["inventory"],
            "artifacts": {
                "referenced_usda": {
                    "path": REFERENCED_NAME,
                    "sha256": _sha256_file(temporary / REFERENCED_NAME),
                },
                "flattened_usdc": {
                    "path": FLATTENED_NAME,
                    "sha256": _sha256_file(temporary / FLATTENED_NAME),
                },
                "preview_usdz": {
                    "path": PREVIEW_NAME,
                    "sha256": _sha256_file(temporary / PREVIEW_NAME),
                },
            },
            "preview": preview,
            "dependencies": {
                "referenced_external": referenced_dependencies,
                "flattened_bundled": bundled,
                "unresolved": unresolved,
                "flattened_internal_arcs": internal_arcs,
            },
            "limitations": [
                "PhysX contact caches, GPU buffers, tensor handles, and solver warm-start state are not "
                "serializable to USD.",
                "The manifest is authoritative for postprocessed fisheye projection; UsdGeom.Camera stores "
                "the backing camera.",
                "Generic USD viewers may not reproduce Isaac RTX/MDL appearance exactly.",
                "scene_preview.usdz uses portable approximations and is not an MDL-faithful simulation artifact.",
            ],
        }
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        after = _live_guard(env, stage)
        after.pop("state")
        if after != before:
            raise RuntimeError("scene export mutated the live simulator stage or runtime state")
        os.replace(temporary, output)
        logger.info(
            "[scene-export] wrote referenced USDA, flattened USDC, Blender preview USDZ, and manifest to %s",
            output,
        )
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
