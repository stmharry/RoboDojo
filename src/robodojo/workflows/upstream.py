"""Detect upstream changes and route them through the local project shape."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import yaml

from robodojo.core.models import UpstreamProject
from robodojo.core.paths import RepositoryPaths

VALID_DISPOSITIONS = frozenset({"mapped", "replaced", "submodule-owned"})


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _load_manifest(paths: RepositoryPaths) -> dict[str, Any]:
    manifest_path = paths.root / "upstream_sync.yml"
    payload: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError(f"upstream sync schema_version must be 1: {manifest_path}")
    projects = payload.get("projects")
    if not isinstance(projects, dict) or not projects:
        raise ValueError(f"upstream sync manifest has no projects: {manifest_path}")
    for name, config in projects.items():
        for field in ("repository", "ref", "reviewed_commit", "mappings"):
            if not config.get(field):
                raise ValueError(f"upstream project {name} is missing {field}")
        for rule in config["mappings"]:
            if rule.get("kind") not in {"exact", "prefix"}:
                raise ValueError(f"upstream project {name} has invalid mapping kind")
            if rule.get("disposition") not in VALID_DISPOSITIONS:
                raise ValueError(f"upstream project {name} has invalid mapping disposition")
            if not isinstance(rule.get("upstream"), str) or not isinstance(rule.get("local"), str):
                raise ValueError(f"upstream project {name} has invalid mapping paths")
    return payload


def _map_path(path: str, mappings: list[dict[str, str]]) -> dict[str, str]:
    for rule in mappings:
        upstream = rule["upstream"]
        if rule["kind"] == "exact" and path == upstream:
            return {"upstream_path": path, "local_path": rule["local"], "disposition": rule["disposition"]}
        if rule["kind"] == "prefix" and path.startswith(upstream):
            suffix = path[len(upstream) :]
            return {
                "upstream_path": path,
                "local_path": rule["local"] + suffix,
                "disposition": rule["disposition"],
            }
    return {"upstream_path": path, "local_path": "", "disposition": "unmapped"}


def _inspect_checkout(name: str, config: dict[str, Any], checkout: Path, ref: str) -> dict[str, Any]:
    reviewed = str(config["reviewed_commit"])
    current = _git(checkout, "rev-parse", ref).stdout.strip()
    if _git(checkout, "cat-file", "-e", f"{reviewed}^{{commit}}", check=False).returncode != 0:
        raise ValueError(f"reviewed commit {reviewed} is unavailable in {checkout}")
    if _git(checkout, "merge-base", "--is-ancestor", reviewed, current, check=False).returncode != 0:
        raise ValueError(f"reviewed commit {reviewed} is not an ancestor of {current}")
    changed = _git(checkout, "diff", "--name-only", f"{reviewed}..{current}").stdout.splitlines()
    mappings = [_map_path(path, config["mappings"]) for path in changed]
    status = "clean" if not mappings else "pending"
    if any(item["disposition"] == "unmapped" for item in mappings):
        status = "error"
    return {
        "project": name,
        "repository": config["repository"],
        "ref": ref,
        "reviewed_commit": reviewed,
        "current_commit": current,
        "status": status,
        "changes": mappings,
    }


def _inspect_remote(name: str, config: dict[str, Any], ref: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"robodojo-upstream-{name}-") as directory:
        checkout = Path(directory)
        subprocess.run(["git", "init", "--bare", "--quiet", str(checkout)], check=True)
        _git(
            checkout,
            "fetch",
            "--quiet",
            "--filter=blob:none",
            config["repository"],
            f"+refs/heads/{ref}:refs/upstream/current",
        )
        result = _inspect_checkout(name, config, checkout, "refs/upstream/current")
        result["ref"] = ref
        return result


def check_upstreams(
    paths: RepositoryPaths,
    *,
    project: UpstreamProject = UpstreamProject.ALL,
    ref: str | None = None,
    source: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Return a structured upstream report and its stable CLI exit code."""
    try:
        manifest = _load_manifest(paths)
    except Exception as exc:
        return {"status": "error", "projects": [], "error": str(exc)}, 2

    selected = list(manifest["projects"])
    if project != UpstreamProject.ALL:
        selected = [project.value]
    if source is not None and len(selected) != 1:
        return {"status": "error", "projects": [], "error": "--source requires one project"}, 2

    results = []
    for name in selected:
        config = manifest["projects"].get(name)
        if config is None:
            results.append({"project": name, "status": "error", "changes": [], "error": "unknown project"})
            continue
        selected_ref = ref or str(config["ref"])
        try:
            if source is None:
                result = _inspect_remote(name, config, selected_ref)
            else:
                result = _inspect_checkout(name, config, source.expanduser().resolve(), selected_ref)
        except Exception as exc:
            result = {"project": name, "status": "error", "changes": [], "error": str(exc)}
        results.append(result)

    if any(result["status"] == "error" for result in results):
        status, code = "error", 2
    elif any(result["status"] == "pending" for result in results):
        status, code = "pending", 1
    else:
        status, code = "clean", 0
    return {"status": status, "projects": results}, code


def format_upstream_report(report: dict[str, Any]) -> str:
    if not report.get("projects"):
        return f"[ERROR] upstream: {report.get('error', 'unknown error')}"
    lines = []
    for project in report["projects"]:
        status = project["status"].upper()
        if project["status"] == "error" and project.get("error"):
            lines.append(f"[{status}] {project['project']}: {project['error']}")
            continue
        lines.append(
            f"[{status}] {project['project']}: {project['reviewed_commit'][:12]} -> {project['current_commit'][:12]}"
        )
        for change in project["changes"]:
            target = change["local_path"] or "<unmapped>"
            lines.append(f"  {change['disposition']}: {change['upstream_path']} -> {target}")
    return "\n".join(lines)


def json_upstream_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
