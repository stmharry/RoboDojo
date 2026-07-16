"""Resume-manifest loading and cleanup."""

from __future__ import annotations

import json
import logging
import os

from robodojo.core.artifacts.results import normalize_artifact, require_matching_scene_identity
from robodojo.core.storage import eval_work_root

logger = logging.getLogger(__name__)


def resume_manifest_path(eval_cfg, run_id):
    return os.path.join(
        str(eval_work_root()),
        eval_cfg["task_protocol"],
        eval_cfg["policy_profile"],
        eval_cfg["environment"],
        f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        f"_resume_{run_id}.json",
    )


def load_resume_manifest(eval_cfg, run_id):
    path = resume_manifest_path(eval_cfg, run_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception as exc:
        raise ValueError(f"invalid resume manifest at {path}: {exc}") from exc
    context = f"resume manifest at {path}"
    require_matching_scene_identity(eval_cfg, data, context=context)
    data = normalize_artifact(data, context=context)
    logger.info(
        "[main] resuming from manifest %s (success=%s fail=%s completed=%s abandoned=%s restart_count=%s)",
        path,
        data.get("success_nums"),
        data.get("fail_nums"),
        len(data.get("completed_layout_ids") or []),
        len(data.get("abandoned_layout_ids") or []),
        data.get("restart_count", 0),
    )
    return data


def delete_resume_manifest(env):
    try:
        path = env.resume_manifest_path()
    except Exception:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info("[main] removed resume manifest %s (eval completed)", path)
    except Exception as exc:
        logger.warning("[main] failed to unlink resume manifest %s: %s", path, exc)
