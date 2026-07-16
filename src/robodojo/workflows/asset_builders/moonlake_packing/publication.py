"""Atomic publication for Moonlake packing assets."""

import os
from pathlib import Path
import shutil
import tempfile

import yaml

from robodojo.workflows.asset_builders.moonlake_packing import geometry, validation as validator


def _publish_paths(staged: list[tuple[Path, Path]]) -> None:
    def remove(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    published = []
    try:
        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = destination.with_name(f".{destination.name}.moonlake-packing-backup")
            remove(backup)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(source, destination)
            except Exception:
                if backup.exists():
                    os.replace(backup, destination)
                raise
            published.append((destination, backup))
    except Exception:
        for destination, backup in reversed(published):
            remove(destination)
            if backup.exists():
                os.replace(backup, destination)
        raise
    for _, backup in published:
        remove(backup)


def build(output_root: Path, manifest_path: Path) -> dict:
    geometry._load_pxr()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".moonlake-packing-", dir=output_root))
    staged_publications = []
    records = {}
    try:
        for key, spec in manifest["assets"].items():
            relative = Path("Object/RoboDojo") / spec["object_type"] / spec["category"] / f"{int(spec['index']):05d}"
            instance = staging_root / relative
            metadata, validation = validator._author_asset(key, spec, instance)
            provenance = {
                "format_version": 1,
                "distribution": manifest["distribution"],
                "reference": manifest["references"][key if key != "container" else "gift_box"],
                "specification": spec,
                "tooling_manifest_sha256": geometry.sha256(manifest_path),
                "outputs": {
                    name: {"geometry.sha256": geometry.sha256(instance / name)}
                    for name in ("object.usd", "metadata.json", "description.json")
                },
            }
            geometry._write_json(instance / "provenance.json", provenance)
            records[key] = {
                "asset": relative.as_posix(),
                "metadata": metadata,
                "validation": validation,
                "files": {
                    name: geometry.sha256(instance / name)
                    for name in ("object.usd", "metadata.json", "description.json", "provenance.json")
                },
            }
            staged_publications.append((instance, output_root / relative))

        shared = {
            "format_version": 1,
            "distribution": manifest["distribution"],
            "tooling_manifest_sha256": geometry.sha256(manifest_path),
            "assets": records,
        }
        staged_manifest = staging_root / "moonlake_packing.json"
        geometry._write_json(staged_manifest, shared)
        staged_publications.append((staged_manifest, output_root / "manifests" / "moonlake_packing.json"))
        _publish_paths(staged_publications)
        return shared
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        build(args.output_root, args.manifest)
    finally:
        simulation_app.close()
