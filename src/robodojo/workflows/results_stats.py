"""Count per-score episode frequencies for selected policies under eval_result.

Example output (conceptually):
    stack_bowls / Pi_05: 5 x 0.5, 10 x 0.0, ...

Directory layout (same as summarize_result.py):
    <protocol>/<policy>/<environment>/<seed>_ckpt_name=...,action_type=.../<timestamp>/_result.json

Rules:
  * Only the latest timestamp folder is read for each
    (task, policy, embodiment, scene, seed).
  * Standalone tasks use the first 50 episodes.
  * Paired tasks (task + task_random) use the first 25 from each half.
  * ``*_random`` tasks are not reported on their own.
  * Scores are aggregated across all seeds by default; use ``--per-seed`` for a breakdown.

Usage:
    robodojo results stats --task stack_bowls --json-out score_stats.json
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import sys

from robodojo.core.scene_identity import ArtifactSchemaError, require_current_result_artifact
from robodojo.core.storage import eval_root
from robodojo.workflows.errors import ResultsError

DEFAULT_POLICIES = ("pi05_bimanual_yam", "pi05_bimanual_yam_pickup", "molmoact2_bimanual_yam")
POLICY_ALIASES = {
    "xiaomi_robotics_0": "Xiaomi_Robotics_0",
}

SEED_RE = re.compile(r"^(\d+)_")
STANDALONE_EPISODES = 50
PAIRED_HALF_EPISODES = 25


def normalize_policy_name(name: str) -> str:
    return POLICY_ALIASES.get(name, name)


def list_subdirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))


def load_completed_result(path: str) -> dict | None:
    try:
        with open(path) as stream:
            data = json.load(stream)
        require_current_result_artifact(data, context=f"evaluation result {path}")
        return data
    except ArtifactSchemaError:
        raise
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def result_scene_config(data: dict) -> str:
    return str(data["scene_config"])


def load_scores(data: dict) -> list[float]:
    details = data.get("details", {})
    items: list[tuple[int, float]] = []
    for key, entry in details.items():
        try:
            layout = int(key)
        except (TypeError, ValueError):
            continue
        score = float(entry.get("score", 0.0) or 0.0)
        items.append((layout, score))
    items.sort(key=lambda x: x[0])
    return [score for _, score in items]


def scan_task(
    root: str,
    task: str,
    env_config: str | None = None,
    scene_config: str | None = None,
) -> dict[tuple[str, str, str, int], list[float]]:
    """Return latest scores keyed by policy, embodiment, scene, and seed."""
    task_path = os.path.join(root, task)
    out: dict[tuple[str, str, str, int], list[float]] = {}
    latest_ts: dict[tuple[str, str, str, int], str] = {}

    for policy in list_subdirs(task_path):
        policy_path = os.path.join(task_path, policy)
        for embodiment in list_subdirs(policy_path):
            if env_config is not None and embodiment != env_config:
                continue
            emb_path = os.path.join(policy_path, embodiment)
            for run in list_subdirs(emb_path):
                match = SEED_RE.match(run)
                if not match:
                    continue
                seed = int(match.group(1))
                run_dir = os.path.join(emb_path, run)
                for ts_name in list_subdirs(run_dir):
                    result_file = os.path.join(run_dir, ts_name, "_result.json")
                    data = load_completed_result(result_file)
                    if data is None:
                        continue
                    selected_scene = result_scene_config(data)
                    if scene_config is not None and selected_scene != scene_config:
                        continue
                    key = (policy, embodiment, selected_scene, seed)
                    if key not in out or ts_name > latest_ts[key]:
                        out[key] = load_scores(data)
                        latest_ts[key] = ts_name
    return out


def ensure_unambiguous(
    task: str,
    *scans: dict[tuple[str, str, str, int], list[float]],
    policies: set[str] | None = None,
) -> None:
    """Fail before distinct embodiment/scene results collapse into one report."""
    combinations: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for scan in scans:
        for policy, embodiment, scene_config, seed in scan:
            if policies is not None and policy not in policies:
                continue
            combinations.setdefault((policy, seed), set()).add((embodiment, scene_config))
    for (policy, seed), values in sorted(combinations.items()):
        if len(values) <= 1:
            continue
        rendered = ", ".join(f"{embodiment}/{scene}" for embodiment, scene in sorted(values))
        raise ResultsError(
            f"Ambiguous results for task={task!r}, policy={policy!r}, seed={seed}: {rendered}. "
            "Pass --environment and/or --scene to select one environment/scene combination."
        )


def discover_tasks(root: str) -> list[str]:
    return sorted(name for name in list_subdirs(root) if not name.endswith("_random") and not name.startswith("_"))


def discover_random_tasks(root: str, tasks: list[str]) -> dict[str, str]:
    all_task_dirs = set(list_subdirs(root))
    random_of: dict[str, str] = {}
    for task in tasks:
        random_name = task + "_random"
        if random_name in all_task_dirs:
            random_of[task] = random_name
    return random_of


def scores_for_run(
    base_scan: dict[tuple[str, str, str, int], list[float]],
    rand_scan: dict[tuple[str, str, str, int], list[float]] | None,
    key: tuple[str, str, str, int],
    paired: bool,
) -> list[float] | None:
    base_scores = base_scan.get(key)
    if base_scores is None:
        return None

    if paired:
        if rand_scan is None:
            return None
        rand_scores = rand_scan.get(key)
        if rand_scores is None:
            return None
        if len(base_scores) < PAIRED_HALF_EPISODES or len(rand_scores) < PAIRED_HALF_EPISODES:
            return None
        return base_scores[:PAIRED_HALF_EPISODES] + rand_scores[:PAIRED_HALF_EPISODES]

    if len(base_scores) < STANDALONE_EPISODES:
        return None
    return base_scores[:STANDALONE_EPISODES]


def format_score(score: float) -> str:
    rounded = round(score, 4)
    if rounded == int(rounded):
        return str(int(rounded))
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return text


def counter_to_dict(counter: Counter[float]) -> dict[str, int]:
    return {format_score(score): count for score, count in sorted(counter.items())}


def format_distribution(counter: Counter[float]) -> str:
    if not counter:
        return "(no data)"
    parts = [f"{count} x {format_score(score)}" for score, count in sorted(counter.items())]
    return ", ".join(parts)


def collect_distributions(
    root: str,
    policies: list[str],
    tasks: list[str] | None = None,
    per_seed: bool = False,
    env_config: str | None = None,
    scene_config: str | None = None,
) -> dict:
    all_tasks = discover_tasks(root)
    if tasks:
        unknown = sorted(set(tasks) - set(all_tasks))
        if unknown:
            raise ResultsError(f"Unknown task(s): {', '.join(unknown)}")
        selected_tasks = sorted(tasks)
    else:
        selected_tasks = all_tasks

    random_of = discover_random_tasks(root, selected_tasks)
    policy_set = set(policies)

    # aggregated[task][policy] -> Counter
    aggregated: dict[str, dict[str, Counter[float]]] = defaultdict(lambda: defaultdict(Counter))
    # per_seed[task][policy][seed] -> Counter
    by_seed: dict[str, dict[str, dict[int, Counter[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )

    for task in selected_tasks:
        base_scan = scan_task(root, task, env_config, scene_config)
        rand_scan = scan_task(root, random_of[task], env_config, scene_config) if task in random_of else None
        paired = task in random_of
        scans = (base_scan,) if rand_scan is None else (base_scan, rand_scan)
        ensure_unambiguous(task, *scans, policies=policy_set)

        keys = set(base_scan)
        if rand_scan is not None:
            keys &= set(rand_scan)

        for key in sorted(keys):
            policy, _embodiment, _scene, seed = key
            if policy not in policy_set:
                continue
            scores = scores_for_run(base_scan, rand_scan, key, paired)
            if scores is None:
                continue
            aggregated[task][policy].update(scores)
            if per_seed:
                by_seed[task][policy][seed].update(scores)

    result = {
        "root": root,
        "policies": policies,
        "tasks": selected_tasks,
        "aggregated": {
            task: {policy: counter_to_dict(counter) for policy, counter in sorted(task_data.items())}
            for task, task_data in sorted(aggregated.items())
        },
    }
    if per_seed:
        result["per_seed"] = {
            task: {
                policy: {str(seed): counter_to_dict(counter) for seed, counter in sorted(seed_data.items())}
                for policy, seed_data in sorted(task_data.items())
            }
            for task, task_data in sorted(by_seed.items())
        }
    return result


def print_report(result: dict, per_seed: bool) -> None:
    policies = result["policies"]
    aggregated = result["aggregated"]

    sys.stdout.write(f"Eval root: {result['root']}\n")
    sys.stdout.write(f"Policies: {', '.join(policies)}\n")
    sys.stdout.write("\n")

    for task in result["tasks"]:
        task_data = aggregated.get(task, {})
        if not any(task_data.get(policy) for policy in policies):
            continue

        sys.stdout.write(f"## {task}\n")
        for policy in policies:
            dist = task_data.get(policy, {})
            if not dist:
                sys.stdout.write(f"  {policy}: (no completed eval)\n")
                continue
            total = sum(dist.values())
            detail = ", ".join(f"{count} x {score}" for score, count in dist.items())
            sys.stdout.write(f"  {policy} ({total} episodes): {detail}\n")

            if per_seed:
                seed_data = result.get("per_seed", {}).get(task, {}).get(policy, {})
                for seed, seed_dist in sorted(seed_data.items(), key=lambda x: int(x[0])):
                    seed_total = sum(seed_dist.values())
                    seed_detail = ", ".join(f"{count} x {score}" for score, count in seed_dist.items())
                    sys.stdout.write(f"    seed {seed} ({seed_total}): {seed_detail}\n")
        sys.stdout.write("\n")


def generate_score_report(
    *,
    results_root: Path | None = None,
    policies: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    environment: str | None = None,
    scene: str | None = None,
    per_seed: bool = False,
    json_out: Path | None = None,
) -> dict:
    """Generate, print, and optionally persist one score-distribution report."""

    root = (results_root or eval_root()).expanduser().resolve()
    if not root.is_dir():
        raise ResultsError(f"Eval result directory not found: {root}")
    selected_policies = [normalize_policy_name(name) for name in (policies or DEFAULT_POLICIES)]
    result = collect_distributions(
        root=str(root),
        policies=selected_policies,
        tasks=list(tasks) if tasks is not None else None,
        per_seed=per_seed,
        env_config=environment,
        scene_config=scene,
    )
    print_report(result, per_seed=per_seed)
    if json_out is not None:
        destination = json_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sys.stdout.write(f"Wrote JSON to {destination}\n")
    return result
