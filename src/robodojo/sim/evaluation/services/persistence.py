from __future__ import annotations

import json
import logging
import os

from robodojo.core.artifacts.results import scene_identity
from robodojo.core.storage import eval_work_root

logger = logging.getLogger(__name__)


class PersistenceService:
    def resume_manifest_path(self) -> str:
        """Stable-but-run-id-tagged path for the resume manifest.

        Lives one directory above the timestamped save_dir so that
        independent eval invocations (each with their own ROBODOJO_RUN_ID)
        never overwrite each other's manifest while still being easy to
        locate by humans.
        """
        return os.path.join(
            str(eval_work_root()),
            self.task_protocol,
            self.policy_profile,
            self.environment,
            str(self.eval_seed) + "_" + self.additional_info,
            f"_resume_{self.run_id}.json",
        )

    def persist_resume_manifest(self, restart_count: int = 0) -> str:
        """Atomically persist enough state to resume after process death.

        Writes ``_resume_<run_id>.json`` next to the timestamped save_dir.
        Always called at the end of run_eval() (best-effort) and again
        from main.py's PhysXFatalError handler (authoritative). Atomic
        via tmp + rename so a partial write cannot corrupt resume.
        """
        manifest_path = self.resume_manifest_path()
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        completed_layout_ids = sorted(
            {
                int(v["layout_id"])
                for v in self.eval_result.get("details", {}).values()
                if isinstance(v, dict) and "layout_id" in v
            }
        )
        payload = {
            "run_id": self.run_id,
            "save_dir": self.save_dir,
            "task": self.task_name,
            "task_protocol": self.task_protocol,
            "policy_name": self.policy_name,
            "policy_profile": self.policy_profile,
            "environment": self.environment,
            **scene_identity(self.eval_cfg),
            "eval_seed": self.eval_seed,
            "additional_info": self.additional_info,
            "success_nums": int(self.success_nums),
            "fail_nums": int(self.fail_nums),
            "unstable_nums": int(self.unstable_nums),
            "total_score": float(self.total_score),
            "completed_layout_ids": completed_layout_ids,
            "abandoned_layout_ids": sorted(int(s) for s in self.abandoned_seeds),
            "details": self.eval_result.get("details", {}),
            "restart_count": int(restart_count),
        }
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, default=str)
        os.replace(tmp_path, manifest_path)
        return manifest_path
