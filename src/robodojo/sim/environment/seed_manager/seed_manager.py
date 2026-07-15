from collections.abc import Iterable, Mapping
from copy import deepcopy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from robodojo.core.layouts import resolve_layout_set
from robodojo.sim.environment.global_configs import ASSETS_PATH, BENCHMARK, ENV_CONFIG_PATH
from robodojo.sim.utils.load_file import load_json

logger = logging.getLogger(__name__)


class SeedManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.num_envs: int = int(self.config["num_envs"])

        # config fields used for directory layout
        self.task_name: str = str(self.config["task_name"])
        self.layout_name: str = str(self.config["layout_name"])
        self.config_name: str = str(self.config["config_name"])
        self.layout_config_name: str = str(self.config.get("layout_config_name", self.config_name))
        self.layout_source: str = str(self.config.get("layout_source", "assets"))

        self.st_idx: int
        self.ed_idx: int
        self.type: str

        self._current_batch_seeds: List[int] | None = None

    def init_eval(
        self,
        completed_layout_ids: Iterable[int] | None = None,
        abandoned_layout_ids: Iterable[int] | None = None,
    ):
        self.eval_seed = self.config.get("seed", 0)
        resolved = resolve_layout_set(
            config_root=Path(ENV_CONFIG_PATH),
            assets_root=Path(ASSETS_PATH),
            benchmark=BENCHMARK,
            layout_set=self.layout_config_name,
            layout_source=self.layout_source,
            task=self.layout_name,
            seed=int(self.eval_seed),
        )
        self.layout_set_hash = resolved.identity_hash
        expected_hash = self.config.get("layout_set_hash")
        if expected_hash is not None and expected_hash != self.layout_set_hash:
            raise ValueError(
                f"resolved layout hash mismatch: expected {expected_hash}, found {self.layout_set_hash} "
                f"in {resolved.directory}"
            )
        logger.info(
            "[SeedManager] using %s evaluation layouts from %s (sha256=%s)",
            self.layout_source,
            resolved.directory,
            self.layout_set_hash,
        )
        self.seed_info = {layout.layout_id: {"scene_layout": str(layout.path)} for layout in resolved.layouts}

        all_layout_ids = sorted(self.seed_info)
        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(int(s) for s in (abandoned_layout_ids or []))
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            logger.warning(
                "[SeedManager] init_eval resume filter: excluded=%s remaining=%s/%s",
                len(excluded),
                len(self.seed_list),
                len(all_layout_ids),
            )
        else:
            self.seed_list = all_layout_ids
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Returns None when enough episodes have been successfully collected.
        """

        if self.idx >= self.ed_idx:
            return None
        if max_count is not None:
            batch_size = min(self.num_envs, max(0, int(max_count)))
            if batch_size == 0:
                return None
            batch = self.seed_list[self.idx : min(self.idx + batch_size, self.ed_idx)]
            self.idx += len(batch)
            self._current_batch_seeds = batch
            return batch
        if self.idx + self.num_envs > self.ed_idx:
            batch = self.seed_list[self.idx : self.ed_idx]
            result = deepcopy(batch)
            for _ in range(self.num_envs - len(result)):
                batch.append(self.seed_list[self.ed_idx - 1])  # pad with last seed if not enough remaining
        else:
            batch = self.seed_list[self.idx : self.idx + self.num_envs]
        self.idx += self.num_envs
        self._current_batch_seeds = batch
        return batch

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        seed_info = self.seed_info.get(seed)
        if seed_info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Scene layout file not found for seed {seed} at expected path {file_path}.")
        data = load_json(file_path)
        return data

    def eval_step(self):
        self._current_batch_seeds = None
