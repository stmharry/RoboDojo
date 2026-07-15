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

    robot_config = yaml.safe_load(profile.component_paths["robot"].read_text(encoding="utf-8")) or {}
    names = {str(item.get("robot_name", "")) for item in robot_config.get("robots", [])}
    return tuple(sorted(names.intersection({"yam", "openarm"})))


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


def generated_fixture_error(paths: RepositoryPaths, name: str) -> str | None:
    """Return a read-only integrity error for a scene-declared fixture build."""

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
    if name != "moonlake_office":
        raise ValueError(f"unsupported generated scene asset: {name}")
    code = build_moonlake_office(paths)
    if code != 0:
        raise RuntimeError(f"{name} fixture builder exited {code}")
    if error := generated_fixture_error(paths, name):
        raise RuntimeError(error)
    return True


def required_fixture_builds(scene: SceneProfile) -> tuple[str, ...]:
    return tuple(scene.document.asset_builds)


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
