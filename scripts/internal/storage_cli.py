#!/usr/bin/env python3
"""Publish immutable RoboDojo payloads with AWS CLI.

The durable Mountpoint view is never written. Payloads are assembled on local
storage and published directly to S3, with completion metadata uploaded last.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.storage import (  # noqa: E402
    assets_root,
    checkpoint_root,
    data_root,
    eval_work_root,
    local_scratch_root,
    model_root,
    s3_uri,
    storage_root,
)

INTERNAL_FILES = {"_MANIFEST.json", "_COMPLETE.json"}
EXCLUDED_DIRS = {".cache", ".git"}
EXCLUDED_SUFFIXES = (".lock", ".partial", ".part", ".tmp", ".incomplete")
CANONICAL_TOP_LEVEL = {"assets", "datasets", "model_weights", "runs"}


def _aws(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _base_uri() -> str:
    base = s3_uri()
    if not base or not base.startswith("s3://"):
        raise SystemExit("ROBODOJO_S3_URI must be set to the dedicated s3://.../robodojo prefix")
    return base.rstrip("/")


def _destination(relative: str) -> str:
    clean = relative.strip("/")
    if not clean or ".." in Path(clean).parts:
        raise SystemExit(f"invalid storage destination: {relative!r}")
    if Path(clean).parts[0] not in CANONICAL_TOP_LEVEL:
        raise SystemExit(f"destination must use a canonical storage root: {relative!r}")
    destination = f"{_base_uri()}/{clean}"
    if not destination.startswith(_base_uri() + "/"):
        raise SystemExit("refusing destination outside ROBODOJO_S3_URI")
    return destination


def _s3_location(uri: str) -> tuple[str, str]:
    bucket_and_key = uri.removeprefix("s3://")
    if "/" not in bucket_and_key:
        return bucket_and_key, ""
    bucket, key = bucket_and_key.split("/", 1)
    return bucket, key


def _payload_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        excluded_part = next(
            (
                part
                for part in relative.parts
                if part in EXCLUDED_DIRS
                or part.endswith(EXCLUDED_SUFFIXES)
                or ".partial." in part
                or ".part-" in part
            ),
            None,
        )
        if excluded_part is not None or path.name in INTERNAL_FILES:
            continue
        if path.is_symlink():
            raise SystemExit(f"payload contains unsupported symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(source: Path, relative: str) -> tuple[dict, dict]:
    files = _payload_files(source)
    entries = [
        {
            "path": str(path.relative_to(source)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    created = datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- system Python 3.10 compatibility
    manifest = {
        "schema_version": 1,
        "destination": relative.strip("/"),
        "created_at": created,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }
    complete = {
        "schema_version": 1,
        "created_at": created,
    }
    return manifest, complete


def publish(source: Path, relative: str, *, replace: bool = False, dry_run: bool = False) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise SystemExit(f"publish source is not a directory: {source}")
    destination = _destination(relative)
    if dry_run:
        print(f"publish {source} -> {destination}")
        return

    bucket, key = _s3_location(destination)
    completed = (
        _aws(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            f"{key}/_COMPLETE.json",
            check=False,
        ).returncode
        == 0
    )
    if completed and not replace:
        raise SystemExit(f"destination is already complete: {destination}")
    if completed and replace:
        _aws("s3", "rm", destination, "--recursive", "--only-show-errors")

    manifest, complete = _metadata(source, relative)
    local_scratch_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_scratch_root()) as temporary:
        temporary_path = Path(temporary)
        manifest_path = temporary_path / "_MANIFEST.json"
        complete_path = temporary_path / "_COMPLETE.json"
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        complete["manifest_sha256"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        manifest_path.write_text(manifest_text, encoding="utf-8")
        complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")

        sync_args = [
            "s3",
            "sync",
            str(source),
            destination,
            "--only-show-errors",
            "--exclude",
            ".git/*",
            "--exclude",
            "*/.git/*",
            "--exclude",
            ".cache/*",
            "--exclude",
            "*/.cache/*",
            "--exclude",
            "*.lock",
            "--exclude",
            "*.lock/*",
            "--exclude",
            "*.partial",
            "--exclude",
            "*.partial/*",
            "--exclude",
            "*.partial.*",
            "--exclude",
            "*.partial.*/*",
            "--exclude",
            "*.part",
            "--exclude",
            "*.part/*",
            "--exclude",
            "*.part-*",
            "--exclude",
            "*.part-*/*",
            "--exclude",
            "*.tmp",
            "--exclude",
            "*.tmp/*",
            "--exclude",
            "*.incomplete",
            "--exclude",
            "*.incomplete/*",
            "--exclude",
            "_MANIFEST.json",
            "--exclude",
            "_COMPLETE.json",
        ]
        result_path = source / "_result.json"
        if result_path.is_file():
            sync_args.extend(["--exclude", "_result.json"])
        _aws(*sync_args)
        _aws("s3", "cp", str(manifest_path), f"{destination}/_MANIFEST.json", "--only-show-errors")
        if result_path.is_file():
            _aws("s3", "cp", str(result_path), f"{destination}/_result.json", "--only-show-errors")
        _aws("s3", "cp", str(complete_path), f"{destination}/_COMPLETE.json", "--only-show-errors")
    print(f"published {source} -> {destination}")


def _find_eval_run(run_id: str) -> Path:
    matches = [path.parent for path in eval_work_root().rglob("_result.json") if path.parent.name == run_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one completed local eval run named {run_id!r}, found {len(matches)}")
    return matches[0]


def _verify_materialized(path: Path) -> None:
    manifest_path = path / "_MANIFEST.json"
    complete_path = path / "_COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise SystemExit(f"durable payload is incomplete: {path}")
    manifest_bytes = manifest_path.read_bytes()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != complete.get("manifest_sha256"):
        raise SystemExit(f"durable payload manifest hash mismatch: {path}")
    manifest = json.loads(manifest_bytes)
    for entry in manifest.get("files", []):
        candidate = path / entry["path"]
        if not candidate.is_file() or candidate.stat().st_size != entry["size"]:
            raise SystemExit(f"materialized payload failed size verification: {candidate}")
        if _sha256(candidate) != entry["sha256"]:
            raise SystemExit(f"materialized payload failed hash verification: {candidate}")


def materialize(source: Path, destination: Path) -> None:
    if destination.exists():
        raise SystemExit(f"materialize destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)
    try:
        _verify_materialized(destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print(f"materialized {source} -> {destination}")


def link_payload(source: Path, destination: Path) -> None:
    """Create an explicit local compatibility link to a durable read path."""
    source = source.resolve()
    destination = destination.expanduser().absolute()
    if not source.is_dir():
        raise SystemExit(f"link source is not a completed directory: {source}")
    if destination.is_symlink():
        if destination.resolve() == source:
            print(f"link already configured: {destination} -> {source}")
            return
        raise SystemExit(f"refusing to replace existing symlink: {destination}")
    if destination.exists():
        raise SystemExit(f"refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)
    print(f"linked {destination} -> {source}")


def doctor() -> None:
    root = storage_root()
    if root is None:
        raise SystemExit("ROBODOJO_STORAGE_ROOT is not configured")
    if not root.is_dir():
        raise SystemExit(f"durable read mount is unavailable: {root}")
    if not os.access(root, os.R_OK | os.X_OK):
        raise SystemExit(f"durable read mount is not readable: {root}")
    if shutil.which("findmnt") is not None:
        options = subprocess.run(
            ["findmnt", "-no", "OPTIONS", "--target", str(root)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip().split(",")
        if "ro" not in options:
            raise SystemExit(f"durable Mountpoint must be mounted read-only: {root}")
    scratch = local_scratch_root()
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / ".storage-doctor"
    probe.write_text("probe\n", encoding="utf-8")
    replacement = scratch / ".storage-doctor.replaced"
    os.replace(probe, replacement)
    replacement.unlink()
    if shutil.which("aws") is None:
        raise SystemExit("aws CLI is not installed")
    _base_uri()
    print(f"storage mount readable: {root}")
    print(f"local scratch supports replace/delete: {scratch}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("status")

    def add_publish(name: str, positional: tuple[str, ...]) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        for argument in positional:
            child.add_argument(argument)
        child.add_argument("--replace", action="store_true")
        child.add_argument("--dry-run", action="store_true")
        return child

    add_publish("publish-assets", ("source",))
    add_publish("publish-data", ("dataset", "source"))
    add_publish("publish-model", ("policy", "model", "source"))
    add_publish("publish-checkpoint", ("policy", "checkpoint", "source"))
    add_publish("publish-reference-cache", ("name", "revision", "source"))
    eval_parser = add_publish("publish-eval", ("source",))
    eval_parser.add_argument("--run-id")
    add_publish("publish-run", ("kind", "run_id", "source"))
    add_publish("publish", ("source", "relative"))

    for name in ("materialize-model", "materialize-checkpoint"):
        child = subparsers.add_parser(name)
        child.add_argument("policy")
        child.add_argument("name")
        child.add_argument("destination")
    hydrate_parser = subparsers.add_parser("hydrate")
    hydrate_parser.add_argument("source")
    hydrate_parser.add_argument("destination")
    link_parser = subparsers.add_parser("link")
    link_parser.add_argument("kind", choices=("assets", "datasets", "checkpoint"))
    link_parser.add_argument("destination")
    link_parser.add_argument("--policy")
    link_parser.add_argument("--checkpoint")

    args = parser.parse_args(argv)
    if args.command in {"doctor", "status"}:
        doctor()
        return 0
    if args.command == "hydrate":
        materialize(Path(args.source), Path(args.destination))
        return 0
    if args.command == "link":
        if args.kind == "assets":
            source = assets_root()
        elif args.kind == "datasets":
            source = data_root()
        else:
            if not args.policy or not args.checkpoint:
                parser.error("link checkpoint requires --policy and --checkpoint")
            source = checkpoint_root() / args.policy / args.checkpoint
        link_payload(source, Path(args.destination))
        return 0
    if args.command == "materialize-model":
        materialize(model_root() / args.policy / args.name, Path(args.destination))
        return 0
    if args.command == "materialize-checkpoint":
        materialize(checkpoint_root() / args.policy / args.name, Path(args.destination))
        return 0

    source = Path(args.source)
    if args.command == "publish":
        relative = args.relative
    elif args.command == "publish-assets":
        relative = "assets"
    elif args.command == "publish-data":
        relative = f"datasets/{args.dataset}"
    elif args.command == "publish-model":
        relative = f"model_weights/{args.policy}/{args.model}"
    elif args.command == "publish-checkpoint":
        relative = f"model_weights/{args.policy}/{args.checkpoint}"
    elif args.command == "publish-reference-cache":
        relative = f"datasets/reference_cache/{args.name}/{args.revision}"
    elif args.command == "publish-eval":
        if args.run_id:
            source = _find_eval_run(args.run_id)
        relative = str(Path("runs/eval_result/RoboDojo") / source.resolve().relative_to(eval_work_root().resolve()))
    else:
        relative = f"runs/{args.kind}/{args.run_id}"
    publish(source, relative, replace=args.replace, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
