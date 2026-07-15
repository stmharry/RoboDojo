"""Detect official upstream changes and validate the local compatibility surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import yaml

from robodojo.core.models import UpstreamProject
from robodojo.core.paths import RepositoryPaths

VALID_MAPPING_KINDS = frozenset({"exact", "prefix"})
VALID_DISPOSITIONS = frozenset({"mirrored", "adapted", "replaced", "submodule-owned", "upstream-only"})
VALID_COMPARISONS = frozenset({"yaml-semantic", "yaml-subset", "python-body", "python-api", "inventory", "manual"})
REQUIRED = "<required>"


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
        if not isinstance(config, dict):
            raise ValueError(f"upstream project {name} must be a mapping")
        for field in ("repository", "ref", "reviewed_commit", "mappings"):
            if not config.get(field):
                raise ValueError(f"upstream project {name} is missing {field}")
        if not isinstance(config["mappings"], list):
            raise ValueError(f"upstream project {name} mappings must be a list")
        for rule in config["mappings"]:
            if not isinstance(rule, dict):
                raise ValueError(f"upstream project {name} has a non-mapping rule")
            if rule.get("kind") not in VALID_MAPPING_KINDS:
                raise ValueError(f"upstream project {name} has invalid mapping kind")
            if rule.get("disposition") not in VALID_DISPOSITIONS:
                raise ValueError(f"upstream project {name} has invalid mapping disposition")
            if rule.get("comparison") not in VALID_COMPARISONS:
                raise ValueError(f"upstream project {name} has invalid comparison mode")
            if not isinstance(rule.get("upstream"), str) or not rule["upstream"]:
                raise ValueError(f"upstream project {name} has an invalid upstream path")
            if not isinstance(rule.get("local", ""), str):
                raise ValueError(f"upstream project {name} has an invalid local path")
            if rule["comparison"] != "manual" and not rule.get("local"):
                raise ValueError(f"upstream project {name} comparison {rule['comparison']} requires a local path")

        local_fork = config.get("local_fork")
        if local_fork is not None:
            if not isinstance(local_fork, dict) or not local_fork.get("path") or not local_fork.get("reviewed_commit"):
                raise ValueError(f"upstream project {name} has an invalid local_fork")

    divergences = payload.get("intentional_divergences")
    if not isinstance(divergences, list) or not divergences:
        raise ValueError("upstream sync manifest must document intentional_divergences")
    for divergence in divergences:
        if not isinstance(divergence, dict):
            raise ValueError("intentional divergences must be mappings")
        for field in ("id", "upstream", "local", "rationale", "tests"):
            if not divergence.get(field):
                raise ValueError(f"intentional divergence is missing {field}")
        for test_path in divergence["tests"]:
            if not (paths.root / test_path).is_file():
                raise ValueError(f"intentional divergence {divergence['id']} references missing test {test_path}")
    return payload


def _local_target(rule: dict[str, Any], upstream_path: str) -> str:
    local = str(rule.get("local", ""))
    if not local:
        return ""
    if rule["kind"] == "prefix" and rule.get("append_suffix", True):
        return local + upstream_path[len(rule["upstream"]) :]
    return local


def _mapping_for(path: str, mappings: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for rule in mappings:
        upstream = rule["upstream"]
        if rule["kind"] == "exact" and path == upstream:
            candidates.append((1, len(upstream), rule))
        elif rule["kind"] == "prefix" and path.startswith(upstream):
            candidates.append((0, len(upstream), rule))
    if not candidates:
        return None
    _, _, rule = max(candidates, key=lambda candidate: candidate[:2])
    return {
        "upstream_path": path,
        "local_path": _local_target(rule, path),
        "disposition": rule["disposition"],
        "comparison": rule["comparison"],
    }


def _tree_paths(repository: Path, commit: str) -> list[str]:
    output = _git(repository, "ls-tree", "-r", "-z", "--name-only", commit).stdout
    return [path for path in output.split("\0") if path]


def _blob(repository: Path, commit: str, path: str) -> str:
    return _git(repository, "show", f"{commit}:{path}").stdout


def _changed_paths(repository: Path, old: str, new: str) -> list[dict[str, str]]:
    fields = _git(repository, "diff", "--name-status", "-z", "--find-renames", f"{old}..{new}").stdout.split("\0")
    fields = [field for field in fields if field]
    changes: list[dict[str, str]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            old_path, path = fields[index : index + 2]
            index += 2
            changes.append({"status": status, "old_path": old_path, "path": path})
        else:
            path = fields[index]
            index += 1
            changes.append({"status": status, "path": path})
    return changes


class _DropImports(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None


def _python_body(source: str) -> str:
    tree = _DropImports().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def _semantic_subset(upstream: Any, local: Any) -> bool:
    if isinstance(upstream, dict):
        return isinstance(local, dict) and all(
            key in local and _semantic_subset(value, local[key]) for key, value in upstream.items()
        )
    if isinstance(upstream, list):
        return (
            isinstance(local, list)
            and len(upstream) == len(local)
            and all(
                _semantic_subset(upstream_value, local_value)
                for upstream_value, local_value in zip(upstream, local, strict=True)
            )
        )
    return upstream == local


def _default_value(node: ast.expr | None) -> str:
    return REQUIRED if node is None else ast.dump(node, include_attributes=False)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    arguments = node.args
    positional_nodes = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (len(positional_nodes) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    return {
        "kind": "callable",
        "posonly_count": len(arguments.posonlyargs),
        "positional": [argument.arg for argument in positional_nodes],
        "positional_defaults": [_default_value(default) for default in defaults],
        "vararg": arguments.vararg is not None,
        "kwonly": [argument.arg for argument in arguments.kwonlyargs],
        "kwonly_defaults": [_default_value(default) for default in arguments.kw_defaults],
        "kwarg": arguments.kwarg is not None,
    }


def _api_surface(source: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(source)
    surface: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            surface[node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            surface[f"class:{node.name}"] = {"kind": "class"}
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    method.name == "__init__" or not method.name.startswith("_")
                ):
                    surface[f"{node.name}.{method.name}"] = _signature(method)
    return surface


def _signature_is_compatible(upstream: dict[str, Any], local: dict[str, Any]) -> bool:
    if upstream["kind"] != local.get("kind"):
        return False
    if upstream["kind"] == "class":
        return True
    if upstream["posonly_count"] != local["posonly_count"]:
        return False
    upstream_positional = upstream["positional"]
    local_positional = local["positional"]
    if local_positional[: len(upstream_positional)] != upstream_positional:
        return False
    if local["positional_defaults"][: len(upstream_positional)] != upstream["positional_defaults"]:
        return False
    if any(default == REQUIRED for default in local["positional_defaults"][len(upstream_positional) :]):
        return False
    if upstream["vararg"] and not local["vararg"]:
        return False
    local_kwonly = dict(zip(local["kwonly"], local["kwonly_defaults"], strict=True))
    for name, default in zip(upstream["kwonly"], upstream["kwonly_defaults"], strict=True):
        if local_kwonly.get(name) != default:
            return False
    upstream_kwonly = set(upstream["kwonly"])
    if any(default == REQUIRED for name, default in local_kwonly.items() if name not in upstream_kwonly):
        return False
    return not upstream["kwarg"] or local["kwarg"]


def _comparison_failure(
    comparison: str,
    upstream_source: str,
    local_path: Path,
) -> str | None:
    if comparison == "inventory":
        return None if local_path.exists() or local_path.is_symlink() else "mapped local path is missing"
    if not local_path.is_file():
        return "mapped local file is missing"
    try:
        local_source = local_path.read_text(encoding="utf-8")
        if comparison == "yaml-semantic":
            return None if yaml.safe_load(upstream_source) == yaml.safe_load(local_source) else "YAML semantics differ"
        if comparison == "yaml-subset":
            return (
                None
                if _semantic_subset(yaml.safe_load(upstream_source), yaml.safe_load(local_source))
                else "upstream YAML semantics are not preserved"
            )
        if comparison == "python-body":
            return None if _python_body(upstream_source) == _python_body(local_source) else "Python bodies differ"
        if comparison == "python-api":
            upstream_api = _api_surface(upstream_source)
            local_api = _api_surface(local_source)
            missing = sorted(set(upstream_api) - set(local_api))
            changed = sorted(
                name
                for name in set(upstream_api) & set(local_api)
                if not _signature_is_compatible(upstream_api[name], local_api[name])
            )
            details = []
            if missing:
                details.append(f"missing public API: {', '.join(missing)}")
            if changed:
                details.append(f"incompatible signatures: {', '.join(changed)}")
            return "; ".join(details) or None
    except (OSError, SyntaxError, ValueError, yaml.YAMLError) as exc:
        return f"comparison failed: {exc}"
    raise ValueError(f"unsupported comparison: {comparison}")


def _check_alignment(
    paths: RepositoryPaths,
    repository: Path,
    reviewed: str,
    mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = []
    checked = 0
    for upstream_path in _tree_paths(repository, reviewed):
        mapped = _mapping_for(upstream_path, mappings)
        if mapped is None or mapped["comparison"] == "manual":
            continue
        local_path = paths.root / mapped["local_path"]
        upstream_source = "" if mapped["comparison"] == "inventory" else _blob(repository, reviewed, upstream_path)
        failure = _comparison_failure(mapped["comparison"], upstream_source, local_path)
        checked += 1
        if failure:
            failures.append({**mapped, "message": failure})
    return {
        "status": "pending" if failures else "clean",
        "checked": checked,
        "failures": failures,
    }


def _inspect_local_fork(
    paths: RepositoryPaths,
    config: dict[str, Any],
    official_reviewed: str,
) -> tuple[dict[str, Any], str]:
    local_config = config["local_fork"]
    submodule = paths.root / local_config["path"]
    expected = str(local_config["reviewed_commit"])
    result: dict[str, Any] = {
        "path": local_config["path"],
        "reviewed_commit": expected,
        "gitlink_commit": "",
        "checkout_commit": "",
        "official_reviewed_is_ancestor": False,
    }
    if not submodule.is_dir():
        result["error"] = f"submodule checkout is missing: {submodule}"
        return result, "error"
    gitlink = _git(paths.root, "ls-files", "--stage", "--", local_config["path"], check=False)
    checkout = _git(submodule, "rev-parse", "HEAD", check=False)
    gitlink_fields = gitlink.stdout.strip().split()
    if gitlink.returncode != 0 or len(gitlink_fields) < 2 or checkout.returncode != 0:
        result["error"] = "could not resolve the XPolicyLab gitlink and checkout"
        return result, "error"
    result["gitlink_commit"] = gitlink_fields[1]
    result["checkout_commit"] = checkout.stdout.strip()
    if _git(submodule, "cat-file", "-e", f"{official_reviewed}^{{commit}}", check=False).returncode != 0:
        result["error"] = f"official reviewed commit {official_reviewed} is unavailable in the local fork"
        return result, "error"
    is_ancestor = (
        _git(
            submodule,
            "merge-base",
            "--is-ancestor",
            official_reviewed,
            expected,
            check=False,
        ).returncode
        == 0
    )
    result["official_reviewed_is_ancestor"] = is_ancestor
    if not is_ancestor:
        result["error"] = f"official reviewed commit {official_reviewed} is not an ancestor of {expected}"
        return result, "error"
    if result["gitlink_commit"] != expected or result["checkout_commit"] != result["gitlink_commit"]:
        result["message"] = "XPolicyLab gitlink or checkout differs from the reviewed local fork commit"
        return result, "pending"
    return result, "clean"


def _inspect_checkout(
    name: str,
    config: dict[str, Any],
    checkout: Path,
    ref: str,
    paths: RepositoryPaths,
) -> dict[str, Any]:
    reviewed = str(config["reviewed_commit"])
    current_result = _git(checkout, "rev-parse", ref, check=False)
    if current_result.returncode != 0:
        raise ValueError(f"could not resolve ref {ref} in {checkout}")
    current = current_result.stdout.strip()
    if _git(checkout, "cat-file", "-e", f"{reviewed}^{{commit}}", check=False).returncode != 0:
        raise ValueError(f"reviewed commit {reviewed} is unavailable in {checkout}")
    if _git(checkout, "merge-base", "--is-ancestor", reviewed, current, check=False).returncode != 0:
        raise ValueError(f"reviewed commit {reviewed} is not an ancestor of {current}")

    mappings = config["mappings"]
    reviewed_paths = set(_tree_paths(checkout, reviewed))
    current_paths = set(_tree_paths(checkout, current))
    unmapped_paths = sorted(path for path in reviewed_paths | current_paths if _mapping_for(path, mappings) is None)
    changes = []
    for change in _changed_paths(checkout, reviewed, current):
        mapped = _mapping_for(change["path"], mappings)
        if mapped is None:
            mapped = {
                "upstream_path": change["path"],
                "local_path": "",
                "disposition": "unmapped",
                "comparison": "manual",
            }
        record = {"status": change["status"], **mapped}
        if "old_path" in change:
            record["old_path"] = change["old_path"]
            old_mapping = _mapping_for(change["old_path"], mappings)
            record["old_local_path"] = old_mapping["local_path"] if old_mapping else ""
        changes.append(record)

    alignment = _check_alignment(paths, checkout, reviewed, mappings)
    status = "pending" if changes or alignment["status"] == "pending" else "clean"
    result: dict[str, Any] = {
        "project": name,
        "repository": config["repository"],
        "ref": ref,
        "reviewed_commit": reviewed,
        "current_commit": current,
        "status": status,
        "changes": changes,
        "alignment": alignment,
        "unmapped_paths": unmapped_paths,
    }
    if unmapped_paths:
        result["status"] = "error"
        result["error"] = f"upstream contains {len(unmapped_paths)} unmapped path(s)"

    if config.get("local_fork"):
        local_fork, fork_status = _inspect_local_fork(paths, config, reviewed)
        result["local_fork"] = local_fork
        if fork_status == "error":
            result["status"] = "error"
            result["error"] = local_fork["error"]
        elif fork_status == "pending" and result["status"] == "clean":
            result["status"] = "pending"
    return result


def _inspect_remote(
    name: str,
    config: dict[str, Any],
    ref: str,
    paths: RepositoryPaths,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"robodojo-upstream-{name}-") as directory:
        checkout = Path(directory)
        subprocess.run(["git", "init", "--bare", "--quiet", str(checkout)], check=True)
        fetch = _git(checkout, "fetch", "--quiet", config["repository"], ref, check=False)
        if fetch.returncode != 0:
            detail = fetch.stderr.strip() or fetch.stdout.strip() or "unknown fetch error"
            raise RuntimeError(f"could not fetch {config['repository']} {ref}: {detail}")
        result = _inspect_checkout(name, config, checkout, "FETCH_HEAD", paths)
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
                result = _inspect_remote(name, config, selected_ref, paths)
            else:
                result = _inspect_checkout(name, config, source.expanduser().resolve(), selected_ref, paths)
        except Exception as exc:
            result = {"project": name, "status": "error", "changes": [], "error": str(exc)}
        results.append(result)

    if any(result["status"] == "error" for result in results):
        status, code = "error", 2
    elif any(result["status"] == "pending" for result in results):
        status, code = "pending", 1
    else:
        status, code = "clean", 0
    return {"schema_version": 1, "status": status, "projects": results}, code


def format_upstream_report(report: dict[str, Any]) -> str:
    if not report.get("projects"):
        return f"[ERROR] upstream: {report.get('error', 'unknown error')}"
    lines = []
    for project in report["projects"]:
        status = project["status"].upper()
        if project["status"] == "error" and not project.get("current_commit"):
            lines.append(f"[{status}] {project['project']}: {project.get('error', 'unknown error')}")
            continue
        lines.append(
            f"[{status}] {project['project']}: {project['reviewed_commit'][:12]} -> {project['current_commit'][:12]}"
        )
        for change in project.get("changes", []):
            target = change["local_path"] or "<documented upstream-only>"
            old = f" (from {change['old_path']})" if change.get("old_path") else ""
            lines.append(
                f"  {change['status']} {change['upstream_path']}{old} -> {target} "
                f"[{change['disposition']}/{change['comparison']}]"
            )
        alignment = project.get("alignment", {})
        lines.append(f"  alignment: {alignment.get('checked', 0)} mapped file(s) checked")
        for failure in alignment.get("failures", []):
            lines.append(f"  parity: {failure['upstream_path']} -> {failure['local_path']}: {failure['message']}")
        for path in project.get("unmapped_paths", []):
            lines.append(f"  unmapped: {path}")
        if project.get("local_fork"):
            fork = project["local_fork"]
            lines.append(
                f"  local fork: {fork['reviewed_commit'][:12]} "
                f"(gitlink={fork['gitlink_commit'][:12] or 'missing'}, "
                f"checkout={fork['checkout_commit'][:12] or 'missing'})"
            )
            if fork.get("message"):
                lines.append(f"  parity: {fork['message']}")
        if project.get("error"):
            lines.append(f"  error: {project['error']}")
    return "\n".join(lines)


def json_upstream_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
