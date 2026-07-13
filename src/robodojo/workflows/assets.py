"""Source-pinned robot asset build preparation."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from urllib.parse import quote
from urllib.request import urlopen

import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import assets_root, storage_root


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
