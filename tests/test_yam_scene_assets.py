import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_scene_profile
from robodojo.sim import scene_assets
from robodojo.sim.scene_assets import (
    inspect_scene_assets,
    prepare_scene_assets,
    reshape_long_sleeves_for_yam_scene,
    update_garment_metadata,
    validate_scene_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def test_simulator_entrypoint_only_inspects_prepared_scene_assets():
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")
    assert "inspect_scene_assets" in source
    assert "prepare_scene_assets" not in source


@pytest.fixture(scope="module")
def isaac_app():
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    yield app
    app.close()


def test_yam_scene_shirt_derivation_preserves_torso_and_topology():
    points = np.array(
        [
            [0.04, -0.15, 0.0],
            [-0.04, 0.04, 0.01],
            [0.19, 0.14, 0.0],
            [-0.19, 0.14, 0.0],
        ],
        dtype=np.float32,
    )
    result = reshape_long_sleeves_for_yam_scene(points)
    np.testing.assert_array_equal(result[:2], points[:2])
    assert result.shape == points.shape
    assert np.all(np.abs(result[2:, 0]) < np.abs(points[2:, 0]))
    assert np.all(np.abs(result[2:, 1] - 0.109) < np.abs(points[2:, 1] - 0.109))


def test_inherited_garment_metadata_keeps_source_landmarks_and_updates_geometry():
    source = {
        "geometry": {"vertices": 4, "faces": 2},
        "passive": {"functional": {"left_hem": {"id": [3]}}},
    }
    points = np.array([[-0.2, -0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.01], [0.2, 0.1, 0.01]])
    metadata = update_garment_metadata(source, points, face_count=3)

    assert metadata["geometry"]["vertices"] == 4
    assert metadata["geometry"]["faces"] == 3
    assert metadata["geometry"]["aligned_bbox"]["extents"] == pytest.approx([0.4, 0.2, 0.01])
    assert metadata["passive"] == source["passive"]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_garment(root: Path) -> tuple[Path, Path]:
    from pxr import Sdf, Usd, UsdGeom, Vt

    source = root / "Object/RoboDojo/Garment/Top_Long/00009"
    source.mkdir(parents=True)
    object_path = source / "object.usd"
    stage = Usd.Stage.CreateNew(str(object_path))
    mesh = UsdGeom.Mesh.Define(stage, "/Garment")
    stage.SetDefaultPrim(mesh.GetPrim())
    points = np.asarray(
        [
            [-0.20, -0.10, 0.0],
            [0.20, -0.10, 0.0],
            [-0.20, 0.14, 0.01],
            [0.20, 0.14, 0.01],
        ],
        dtype=np.float32,
    )
    usd_points = Vt.Vec3fArray.FromNumpy(points)
    mesh.CreatePointsAttr(usd_points)
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(usd_points))
    mesh.GetPrim().CreateAttribute("fixture:asset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    stage.GetRootLayer().Save()
    (source / "texture.png").write_bytes(b"fixture texture")
    metadata_path = source / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "geometry": {"vertices": 4, "faces": 2},
                "passive": {"functional": {"left_hem": {"id": [2]}, "right_hem": {"id": [3]}}},
            }
        ),
        encoding="utf-8",
    )
    return object_path, metadata_path


def test_typed_scene_asset_preparation_is_task_scoped(tmp_path):
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")
    assert validate_scene_assets(scene, "general_pickup", root=tmp_path) == ()
    prepared = prepare_scene_assets(scene, "general_pickup", root=tmp_path)
    assert prepared.artifacts == ()
    assert len(prepared.identity_hash) == 64


def test_scene_asset_inspection_does_not_prepare_missing_outputs(tmp_path):
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")
    source = tmp_path / "Object/RoboDojo/Garment/Top_Long/00009"
    source.mkdir(parents=True)
    (source / "object.usd").write_bytes(b"read-only inspection fixture")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "Object/RoboDojo/Garment/Top_Long/00012"

    with pytest.raises(FileNotFoundError, match="run make setup"):
        inspect_scene_assets(scene, "fold_clothes", root=tmp_path)

    assert not destination.exists()


def test_scene_garment_derivation_is_manifested_idempotent_and_source_sensitive(tmp_path, isaac_app):
    from pxr import Sdf, Usd, UsdGeom

    source_object, source_metadata = _write_source_garment(tmp_path)
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")

    first = prepare_scene_assets(scene, "fold_clothes", root=tmp_path)
    assert inspect_scene_assets(scene, "fold_clothes", root=tmp_path) == first
    destination = tmp_path / "Object/RoboDojo/Garment/Top_Long/00012"
    object_path = destination / "object.usd"
    metadata_path = destination / "metadata.json"
    manifest_path = destination / "derivation.json"
    assert object_path.is_file() and metadata_path.is_file() and manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["builder_version"] == 1
    assert manifest["recipe"]["transform"] == "yam_short_sleeve_v1"
    assert manifest["inputs"]["object"]["sha256"] == _sha256(source_object)
    assert manifest["outputs"]["object"]["sha256"] == _sha256(object_path)
    assert manifest["topology"] == {"vertices": 4, "faces": 2}
    derived_metadata = json.loads(metadata_path.read_text())
    assert derived_metadata["passive"] == json.loads(source_metadata.read_text())["passive"]

    stage = Usd.Stage.Open(str(object_path))
    [mesh] = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    derived_points = np.asarray(mesh.GetPointsAttr().Get())
    assert len(derived_points) == 4
    assert abs(derived_points[0, 0]) < 0.20
    asset_path = mesh.GetPrim().GetAttribute("fixture:asset").Get()
    assert isinstance(asset_path, Sdf.AssetPath)
    assert asset_path.path == "../00009/texture.png"

    output_mtime = object_path.stat().st_mtime_ns
    second = prepare_scene_assets(scene, "fold_clothes", root=tmp_path)
    assert second == first
    assert object_path.stat().st_mtime_ns == output_mtime

    metadata_path.write_text("{}", encoding="utf-8")
    repaired = prepare_scene_assets(scene, "fold_clothes", root=tmp_path)
    assert repaired == first
    assert json.loads(metadata_path.read_text())["passive"] == json.loads(source_metadata.read_text())["passive"]

    source_payload = json.loads(source_metadata.read_text())
    source_payload["source_revision"] = 2
    source_metadata.write_text(json.dumps(source_payload), encoding="utf-8")
    rebuilt = prepare_scene_assets(scene, "fold_clothes", root=tmp_path)
    assert rebuilt.identity_hash != first.identity_hash
    assert json.loads(manifest_path.read_text())["inputs"]["metadata"]["sha256"] == _sha256(source_metadata)


def test_failed_scene_asset_rebuild_leaves_previous_publication_intact(monkeypatch, tmp_path, isaac_app):
    _, source_metadata = _write_source_garment(tmp_path)
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")
    prepare_scene_assets(scene, "fold_clothes", root=tmp_path)
    destination = tmp_path / "Object/RoboDojo/Garment/Top_Long/00012"
    before = {name: (destination / name).read_bytes() for name in ("object.usd", "metadata.json", "derivation.json")}

    source_metadata.write_text('{"geometry":{"vertices":4,"faces":2},"changed":true}', encoding="utf-8")

    def fail_derivation(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scene_assets, "_derive_garment", fail_derivation)
    with pytest.raises(RuntimeError, match="boom"):
        prepare_scene_assets(scene, "fold_clothes", root=tmp_path)

    assert {name: (destination / name).read_bytes() for name in before} == before
