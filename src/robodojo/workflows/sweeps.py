"""Sequential smoke and benchmark sweeps."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time

from robodojo.core.experiments.catalogs import load_recipe_catalog
from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models.reports import (
    SmokeRecord,
    SmokeSummary,
)
from robodojo.core.models.requests import (
    EvaluationRequest,
    SweepRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import run_work_root
from robodojo.orchestration.evaluation import run_evaluation

logger = logging.getLogger(__name__)


def _selected_recipes(paths: RepositoryPaths, request: SweepRequest) -> list[str]:
    available = load_recipe_catalog(paths).recipes
    unknown = sorted(set(request.recipes) - set(available))
    if unknown:
        raise ValueError(f"unknown recipe(s): {', '.join(unknown)}")
    selected = list(request.recipes)
    return selected[: request.limit] if request.limit else selected


def _write_summary(summary: SmokeSummary, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts = {
        status: sum(row.status == status for row in summary.results) for status in ("PASS", "FAIL", "SKIP", "DRY_RUN")
    }
    lines = [
        f"# RoboDojo Sweep `{summary.run_id}`",
        "",
        f"- pass: `{counts['PASS']}`",
        f"- fail: `{counts['FAIL']}`",
        f"- skip: `{counts['SKIP']}`",
        f"- dry run: `{counts['DRY_RUN']}`",
        "",
        "| Status | Recipe | Exit | Seconds | Result | Log | Message |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.status} | `{row.recipe}` | {row.exit_code} | {row.elapsed_sec:.2f} | "
        f"`{row.result_path}` | `{row.log_path}` | {row.message} |"
        for row in summary.results
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(paths: RepositoryPaths, request: SweepRequest) -> int:
    try:
        assignment = resolve_gpus(policy_gpu=request.policy_gpu, env_gpu=request.environment_gpu)
    except GpuSelectionError as exc:
        logger.error("GPU selection failed: %s", exc)
        return 2
    request = request.model_copy(update={"policy_gpu": assignment.policy_gpu, "environment_gpu": assignment.env_gpu})
    recipes = _selected_recipes(paths, request)
    run_id = request.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_sweep")
    run_dir = run_work_root() / "smoke" / run_id
    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "summary.md"
    results: list[SmokeRecord] = []
    if request.resume and summary_path.is_file():
        prior = SmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        results.extend(prior.results)
    passed = {row.recipe for row in results if row.status == "PASS"}

    for recipe_name in recipes:
        if recipe_name in passed:
            continue
        experiment = resolve_recipe(paths, recipe_name)
        started = time.monotonic()
        evaluation = EvaluationRequest(
            experiment=experiment.spec(paths),
            seed=request.seed,
            policy_gpu=request.policy_gpu,
            environment_gpu=request.environment_gpu,
            eval_num=request.eval_num,
            dry_run=request.dry_run,
        )
        code = run_evaluation(paths, evaluation, preflight=True)
        status = "DRY_RUN" if request.dry_run else ("PASS" if code == 0 else "FAIL")
        record = SmokeRecord(
            status=status,
            recipe=recipe_name,
            scene=experiment.scene.name,
            exit_code=code,
            elapsed_sec=time.monotonic() - started,
            message="" if code == 0 else f"evaluation exited {code}",
        )
        results = [row for row in results if row.recipe != recipe_name]
        results.append(record)
        summary = SmokeSummary(run_id=run_id, eval_num=request.eval_num or 1, results=results)
        _write_summary(summary, summary_path, markdown_path)
        if code != 0 and request.fail_fast:
            return code
    return 1 if any(row.status == "FAIL" for row in results) else 0
