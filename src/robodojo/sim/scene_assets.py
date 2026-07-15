"""Typed, deterministic preparation of assets owned by scene profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

from robodojo.core.models import SceneCatalogAsset, SceneGarmentVariantRecipe
from robodojo.core.profiles import SceneProfile
from robodojo.core.storage import assets_root as runtime_assets_root

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DERIVATION_MANIFEST = "derivation.json"
DERIVATION_SCHEMA_VERSION = 1
DERIVATION_BUILDER_VERSION = 1


@dataclass(frozen=True)
class ResolvedSceneAssetRecipe:
    recipe: SceneGarmentVariantRecipe
    source_root: Path
    source_object: Path
    source_metadata: Path
    destination_root: Path


@dataclass(frozen=True)
class PreparedSceneAsset:
    destination_root: Path
    derivation_hash: str
    manifest_hash: str


@dataclass(frozen=True)
class PreparedSceneAssets:
    artifacts: tuple[PreparedSceneAsset, ...]
    identity_hash: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_root(root: Path, asset: SceneCatalogAsset) -> Path:
    return root / "Object" / "RoboDojo" / asset.object_type / asset.category / f"{asset.index:05d}"


def _source_object(root: Path) -> Path | None:
    for filename in ("object.usdz", "object.usd"):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def validate_scene_assets(
    profile: SceneProfile,
    task_name: str,
    *,
    root: Path | None = None,
) -> tuple[ResolvedSceneAssetRecipe, ...]:
    """Resolve task recipes and fail before stage construction when inputs are absent."""

    root = (root or runtime_assets_root()).resolve()
    resolved = []
    for recipe in profile.document.task_assets.get(task_name, ()):
        source_root = _catalog_root(root, recipe.source)
        destination_root = _catalog_root(root, recipe.destination)
        source_object = _source_object(source_root)
        source_metadata = source_root / "metadata.json"
        if source_object is None or not source_metadata.is_file():
            raise FileNotFoundError(
                f"scene asset source is incomplete under {source_root}; download RoboDojo assets first"
            )
        resolved.append(
            ResolvedSceneAssetRecipe(
                recipe=recipe,
                source_root=source_root,
                source_object=source_object,
                source_metadata=source_metadata,
                destination_root=destination_root,
            )
        )
    return tuple(resolved)


def reshape_long_sleeves_for_yam_scene(points: np.ndarray) -> np.ndarray:
    """Return the version-one topology-preserving YAM short-sleeve transform."""

    import numpy as np

    result = np.asarray(points, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"garment points must have shape (N, 3), got {result.shape}")
    distance = np.abs(result[:, 0])
    sleeve_weight = np.clip((distance - 0.10) / 0.06, 0.0, 1.0)
    sleeve_sign = np.where(result[:, 0] < 0.0, -1.0, 1.0)
    target_x = sleeve_sign * (0.10 + (distance - 0.10) * 0.43)
    target_y = 0.109 + (result[:, 1] - 0.109) * 0.5
    result[:, 0] += sleeve_weight * (target_x - result[:, 0])
    result[:, 1] += sleeve_weight * (target_y - result[:, 1])
    return result


def update_garment_metadata(metadata: dict, points: np.ndarray, face_count: int) -> dict:
    """Update geometry facts while retaining source functional landmarks."""

    import numpy as np

    if face_count <= 0:
        raise ValueError("garment mesh must contain at least one face")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"garment points must have shape (N, 3), got {points.shape}")

    result = json.loads(json.dumps(metadata))
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    extents = upper - lower
    bounds = [
        [x, y, z]
        for x in (float(lower[0]), float(upper[0]))
        for y in (float(lower[1]), float(upper[1]))
        for z in (float(lower[2]), float(upper[2]))
    ]
    geometry = result.setdefault("geometry", {})
    geometry.update(
        {
            "faces": int(face_count),
            "vertices": int(len(points)),
            "aligned_bbox": {"vertices": bounds, "extents": extents.astype(float).tolist()},
            "oriented_bbox": {"vertices": bounds, "extents": extents.astype(float).tolist()},
            "radius": float(extents.max() * 0.5),
        }
    )
    return result


def _derivation_contract(resolved: ResolvedSceneAssetRecipe, root: Path) -> dict[str, Any]:
    catalog = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(resolved.source_root.rglob("*"))
        if path.is_file()
    ]
    inputs = {
        "object": {
            "path": resolved.source_object.relative_to(root).as_posix(),
            "sha256": _sha256(resolved.source_object),
        },
        "metadata": {
            "path": resolved.source_metadata.relative_to(root).as_posix(),
            "sha256": _sha256(resolved.source_metadata),
        },
        "catalog": catalog,
    }
    payload = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "builder_version": DERIVATION_BUILDER_VERSION,
        "recipe": resolved.recipe.model_dump(mode="json"),
        "inputs": inputs,
    }
    return {**payload, "identity_hash": _canonical_hash(payload)}


def _valid_manifest(destination: Path, expected_identity: str) -> tuple[dict[str, Any], str] | None:
    manifest_path = destination / DERIVATION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        manifest.get("schema_version") != DERIVATION_SCHEMA_VERSION
        or manifest.get("builder_version") != DERIVATION_BUILDER_VERSION
        or manifest.get("identity_hash") != expected_identity
    ):
        return None
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"object", "metadata"}:
        return None
    for output in outputs.values():
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            return None
        path = destination / output["path"]
        if not path.is_file() or _sha256(path) != output.get("sha256"):
            return None
    return manifest, _canonical_hash(manifest)


def _derive_garment(
    resolved: ResolvedSceneAssetRecipe,
    output_object: Path,
    output_metadata: Path,
) -> dict[str, int]:
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, Vt

    source_stage = Usd.Stage.Open(str(resolved.source_object))
    if source_stage is None:
        raise RuntimeError(f"could not open scene garment source {resolved.source_object}")
    if not source_stage.Flatten().Export(str(output_object)):
        raise RuntimeError(f"could not export flattened scene garment to {output_object}")

    stage = Usd.Stage.Open(str(output_object))
    if stage is None:
        raise RuntimeError(f"could not open flattened scene garment {output_object}")
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one garment mesh in {resolved.source_object}, found {len(meshes)}")
    mesh = meshes[0]
    source_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    if resolved.recipe.transform == "yam_short_sleeve_v1":
        points = reshape_long_sleeves_for_yam_scene(source_points)
    else:  # Pydantic rejects unknown transforms before simulator startup.
        raise ValueError(f"unsupported garment transform: {resolved.recipe.transform}")
    usd_points = Vt.Vec3fArray.FromNumpy(points)
    mesh.GetPointsAttr().Set(usd_points)
    mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(usd_points))
    face_count = len(mesh.GetFaceVertexCountsAttr().Get())

    source_root = resolved.source_root.resolve()
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            container, separator, package_member = value.path.partition("[")
            container_path = Path(container)
            if not container_path.is_absolute():
                continue
            resolved_container = container_path.resolve()
            try:
                resolved_container.relative_to(source_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"derived garment contains an absolute asset outside its source catalog: {value.path}"
                ) from exc
            portable_container = Path(os.path.relpath(resolved_container, output_object.parent)).as_posix()
            portable_path = portable_container + (separator + package_member if separator else "")
            attribute.Set(Sdf.AssetPath(portable_path))
    stage.GetRootLayer().Save()

    with resolved.source_metadata.open(encoding="utf-8") as stream:
        source_metadata = json.load(stream)
    derived_metadata = update_garment_metadata(source_metadata, points, face_count)
    with output_metadata.open("w", encoding="utf-8") as stream:
        json.dump(derived_metadata, stream, indent=2)
        stream.write("\n")
    return {"vertices": int(len(points)), "faces": int(face_count)}


def _prepare_recipe(resolved: ResolvedSceneAssetRecipe, root: Path) -> PreparedSceneAsset:
    from filelock import FileLock

    destination = resolved.destination_root
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.lock")
    with FileLock(str(lock_path), timeout=120):
        contract = _derivation_contract(resolved, root)
        if cached := _valid_manifest(destination, contract["identity_hash"]):
            _, manifest_hash = cached
            return PreparedSceneAsset(destination, contract["identity_hash"], manifest_hash)

        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        try:
            output_object = staging / "object.usd"
            output_metadata = staging / "metadata.json"
            topology = _derive_garment(resolved, output_object, output_metadata)
            if _derivation_contract(resolved, root)["identity_hash"] != contract["identity_hash"]:
                raise RuntimeError(f"scene asset source changed during preparation: {resolved.source_root}")
            manifest = {
                **contract,
                "outputs": {
                    "object": {"path": output_object.name, "sha256": _sha256(output_object)},
                    "metadata": {"path": output_metadata.name, "sha256": _sha256(output_metadata)},
                },
                "topology": topology,
            }
            staged_manifest = staging / DERIVATION_MANIFEST
            staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            destination.mkdir(parents=True, exist_ok=True)
            os.replace(output_object, destination / output_object.name)
            os.replace(output_metadata, destination / output_metadata.name)
            os.replace(staged_manifest, destination / DERIVATION_MANIFEST)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        verified = _valid_manifest(destination, contract["identity_hash"])
        if verified is None:
            raise RuntimeError(f"scene asset publication failed verification: {destination}")
        _, manifest_hash = verified
        logger.info(
            "prepared scene asset %s from %s (sha256=%s)",
            destination,
            resolved.source_root,
            manifest_hash,
        )
        return PreparedSceneAsset(destination, contract["identity_hash"], manifest_hash)


def prepare_scene_assets(
    profile: SceneProfile,
    task_name: str,
    *,
    root: Path | None = None,
) -> PreparedSceneAssets:
    """Prepare every typed recipe for a task and return its runtime identity."""

    root = (root or runtime_assets_root()).resolve()
    recipes = validate_scene_assets(profile, task_name, root=root)
    artifacts = tuple(_prepare_recipe(recipe, root) for recipe in recipes)
    identity = _canonical_hash(
        [
            {
                "destination": artifact.destination_root.relative_to(root).as_posix(),
                "derivation_hash": artifact.derivation_hash,
                "manifest_hash": artifact.manifest_hash,
            }
            for artifact in artifacts
        ]
    )
    return PreparedSceneAssets(artifacts=artifacts, identity_hash=identity)


def inspect_scene_assets(
    profile: SceneProfile,
    task_name: str,
    *,
    root: Path | None = None,
) -> PreparedSceneAssets:
    """Return prepared scene assets without creating or repairing any files."""

    root = (root or runtime_assets_root()).resolve()
    recipes = validate_scene_assets(profile, task_name, root=root)
    artifacts: list[PreparedSceneAsset] = []
    for recipe in recipes:
        contract = _derivation_contract(recipe, root)
        verified = _valid_manifest(recipe.destination_root, contract["identity_hash"])
        if verified is None:
            raise FileNotFoundError(
                f"prepared scene asset is missing or stale: {recipe.destination_root}; run make setup"
            )
        _, manifest_hash = verified
        artifacts.append(PreparedSceneAsset(recipe.destination_root, contract["identity_hash"], manifest_hash))
    identity = _canonical_hash(
        [
            {
                "destination": artifact.destination_root.relative_to(root).as_posix(),
                "derivation_hash": artifact.derivation_hash,
                "manifest_hash": artifact.manifest_hash,
            }
            for artifact in artifacts
        ]
    )
    return PreparedSceneAssets(artifacts=tuple(artifacts), identity_hash=identity)
