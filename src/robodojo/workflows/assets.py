"""Source-pinned robot asset build preparation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
from urllib.request import urlopen

import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import EnvironmentProfile, SceneProfile
from robodojo.core.storage import assets_root, storage_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_robot_builds(profile: EnvironmentProfile) -> tuple[str, ...]:
    """Return generated robot assets required by an environment profile."""

    return tuple(profile.document.asset_builds)


def generated_robot_error(name: str) -> str | None:
    """Return a read-only integrity error for one generated robot asset."""

    root = assets_root() / "Robots" / name
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"generated {name} manifest is missing or invalid: {manifest_path}: {exc}"
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        for relative, expected in outputs.items():
            output = root / relative
            if not output.is_file():
                return f"generated {name} asset is missing: {output}"
            if _sha256(output) != expected:
                return f"generated {name} asset checksum mismatch: {output}"
    else:
        output_name = manifest.get("output")
        if output_name and not (root / str(output_name)).is_file():
            return f"generated {name} asset is missing: {root / str(output_name)}"
    if name == "yam" and "provenance" in manifest:
        repository = RepositoryPaths.resolve(Path(__file__).resolve().parents[3])
        expected_manifest_hash = _sha256(repository.yam_manifest)
        actual_manifest_hash = manifest.get("provenance", {}).get("build_manifest_sha256")
        if actual_manifest_hash != expected_manifest_hash:
            return f"generated yam asset is stale: {manifest_path}"
    return None


def ensure_generated_robot(paths: RepositoryPaths, name: str) -> bool:
    """Build a required generated robot only when its manifest is invalid."""

    if generated_robot_error(name) is None:
        return False
    builders = {"yam": build_yam, "openarm": build_openarm}
    code = builders[name](paths)
    if code != 0:
        raise RuntimeError(f"{name} asset builder exited {code}")
    if error := generated_robot_error(name):
        raise RuntimeError(error)
    return True


def _fixture_paths(paths: RepositoryPaths, name: str) -> tuple[Path, Path, Path]:
    if name != "moonlake_office":
        raise ValueError(f"unsupported generated scene asset: {name}")
    tooling = paths.moonlake_office_manifest
    specification = yaml.safe_load(tooling.read_text(encoding="utf-8")) or {}
    root = assets_root() / "Object" / "RoboDojo" / "Geometry" / specification["fixture"]["category"]
    return tooling, root, root / "manifest.json"


def _packing_asset_error(paths: RepositoryPaths) -> str | None:
    tooling = paths.moonlake_packing_manifest
    specification = yaml.safe_load(tooling.read_text(encoding="utf-8")) or {}
    root = assets_root()
    manifest_path = root / "manifests" / "moonlake_packing.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"generated moonlake_packing manifest is missing or invalid: {manifest_path}: {exc}"
    if manifest.get("tooling_manifest_sha256") != _sha256(tooling):
        return f"generated moonlake_packing manifest is stale: {manifest_path}"
    records = manifest.get("assets")
    expected = specification.get("assets")
    if not isinstance(records, dict) or not isinstance(expected, dict) or set(records) != set(expected):
        return f"generated moonlake_packing manifest has an invalid asset inventory: {manifest_path}"
    filenames = {"object.usd", "metadata.json", "description.json", "provenance.json"}
    for key, asset in expected.items():
        relative = (
            Path("Object/RoboDojo") / str(asset["object_type"]) / str(asset["category"]) / f"{int(asset['index']):05d}"
        )
        record = records[key]
        if not isinstance(record, dict) or record.get("asset") != relative.as_posix():
            return f"generated moonlake_packing asset path mismatch for {key}: {manifest_path}"
        files = record.get("files")
        if not isinstance(files, dict) or set(files) != filenames:
            return f"generated moonlake_packing file inventory is invalid for {key}: {manifest_path}"
        for filename in sorted(filenames):
            output = root / relative / filename
            if not output.is_file():
                return f"generated moonlake_packing asset is missing: {output}"
            if files[filename] != _sha256(output):
                return f"generated moonlake_packing asset checksum mismatch: {output}"
    return None


def generated_fixture_error(paths: RepositoryPaths, name: str) -> str | None:
    """Return a read-only integrity error for a scene-declared fixture build."""

    if name == "moonlake_packing":
        return _packing_asset_error(paths)
    tooling, root, manifest_path = _fixture_paths(paths, name)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"generated {name} fixture manifest is missing or invalid: {manifest_path}: {exc}"
    if manifest.get("build_manifest_sha256") != _sha256(tooling):
        return f"generated {name} fixture manifest is stale: {manifest_path}"
    output_name = manifest.get("output")
    output = root / str(output_name or "")
    if not output_name or not output.is_file():
        return f"generated {name} fixture output is missing: {output}"
    if manifest.get("output_sha256") != _sha256(output):
        return f"generated {name} fixture checksum mismatch: {output}"
    return None


def ensure_generated_fixture(paths: RepositoryPaths, name: str) -> bool:
    """Build a scene-declared fixture only when its pinned manifest is invalid."""

    if generated_fixture_error(paths, name) is None:
        return False
    builders = {
        "moonlake_office": build_moonlake_office,
        "moonlake_packing": build_moonlake_packing,
    }
    if name not in builders:
        raise ValueError(f"unsupported generated scene asset: {name}")
    code = builders[name](paths)
    if code != 0:
        raise RuntimeError(f"{name} fixture builder exited {code}")
    if error := generated_fixture_error(paths, name):
        raise RuntimeError(error)
    return True


def required_fixture_builds(scene: SceneProfile, task_name: str | None = None) -> tuple[str, ...]:
    builds = list(scene.document.asset_builds)
    if task_name is not None:
        builds.extend(scene.document.task_asset_builds.get(task_name, ()))
    return tuple(dict.fromkeys(builds))


def build_openarm(paths: RepositoryPaths) -> int:
    manifest = yaml.safe_load(paths.openarm_manifest.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    cache = storage_root() / ".cache" / "openarm"
    source = cache / "openarm_isaac_lab"
    hardware = cache / "hardware"
    output = assets_root() / "Robots" / "openarm"
    hardware.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    source_spec = sources["openarm_isaac_lab"]
    if not (source / ".git").is_dir():
        subprocess.run(["git", "clone", source_spec["repository"], str(source)], check=True)
    revision = source_spec["revision"]
    subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)

    hardware_spec = sources["hardware_modifications"]
    for name, expected in hardware_spec["sha256"].items():
        destination = hardware / name
        url = f"{hardware_spec['repository'].rstrip('/')}/resolve/{hardware_spec['revision']}/{quote(name)}"
        if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            with urlopen(url) as response, destination.open("wb") as stream:  # noqa: S310 - pinned URL and checksum
                stream.write(response.read())
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {actual} != {expected}")

    (output / "manifest.json").unlink(missing_ok=True)
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    command = [
        sys.executable,
        "-m",
        "robodojo.workflows.assets_openarm",
        "--source-root",
        str(source),
        "--hardware-root",
        str(hardware),
        "--output-root",
        str(output),
        "--manifest",
        str(paths.openarm_manifest),
    ]
    code = subprocess.run(command, cwd=paths.root, env=env).returncode
    if code == 0 and not (output / "manifest.json").is_file():
        raise RuntimeError("OpenARM build completed without manifest.json")
    return code


def build_yam(paths: RepositoryPaths) -> int:
    manifest = yaml.safe_load(paths.yam_manifest.read_text(encoding="utf-8"))
    source_spec = manifest["sources"]["i2rt"]
    cache = storage_root() / ".cache" / "yam"
    source = cache / "i2rt"
    output = assets_root() / "Robots" / "yam"
    output.mkdir(parents=True, exist_ok=True)

    if not (source / ".git").is_dir():
        subprocess.run(["git", "clone", source_spec["repository"], str(source)], check=True)
    revision = source_spec["revision"]
    subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)

    (output / "manifest.json").unlink(missing_ok=True)
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    command = [
        sys.executable,
        "-m",
        "robodojo.workflows.assets_yam",
        "--source-root",
        str(source),
        "--output-root",
        str(output),
        "--manifest",
        str(paths.yam_manifest),
    ]
    code = subprocess.run(command, cwd=paths.root, env=env).returncode
    if code == 0 and not (output / "manifest.json").is_file():
        raise RuntimeError("YAM build completed without manifest.json")
    return code


def build_moonlake_office(paths: RepositoryPaths) -> int:
    """Build the pinned internal Moonlake office fixture without executing upstream code."""
    manifest = yaml.safe_load(paths.moonlake_office_manifest.read_text(encoding="utf-8"))
    source_spec = manifest["sources"]["spatio_monorepo"]
    cache = storage_root() / ".cache" / "moonlake_office"
    source = cache / "spatio_monorepo"
    output = assets_root() / "Object" / "RoboDojo" / "Geometry" / manifest["fixture"]["category"]
    output.mkdir(parents=True, exist_ok=True)

    if not (source / ".git").is_dir():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", source_spec["repository"], str(source)],
            check=True,
        )
    source_files = [entry["path"] for entry in source_spec["files"].values()]
    subprocess.run(["git", "-C", str(source), "sparse-checkout", "init", "--no-cone"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "sparse-checkout", "set", "--no-cone", *source_files],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", source_spec["revision"]],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "--force", "FETCH_HEAD"], check=True)

    (output / "manifest.json").unlink(missing_ok=True)
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
    }
    command = [
        sys.executable,
        "-m",
        "robodojo.workflows.assets_moonlake_office",
        "--source-root",
        str(source),
        "--output-root",
        str(output),
        "--manifest",
        str(paths.moonlake_office_manifest),
    ]
    code = subprocess.run(command, cwd=paths.root, env=env).returncode
    if code == 0 and not (output / "manifest.json").is_file():
        raise RuntimeError("Moonlake office build completed without manifest.json")
    return code


def build_moonlake_packing(paths: RepositoryPaths) -> int:
    """Build the internal procedural assets for the Moonlake packing task."""
    output = assets_root()
    shared_manifest = output / "manifests" / "moonlake_packing.json"
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
    }
    command = [
        sys.executable,
        "-m",
        "robodojo.workflows.assets_moonlake_packing",
        "--output-root",
        str(output),
        "--manifest",
        str(paths.moonlake_packing_manifest),
    ]
    code = subprocess.run(command, cwd=paths.root, env=env).returncode
    if code == 0 and not shared_manifest.is_file():
        raise RuntimeError("Moonlake packing build completed without its shared manifest")
    return code
