"""Pure saved-pose restoration sequencing for simulator scene batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def restore_saved_poses(
    env_idx_list: Sequence[int],
    tables: Sequence[Any | None],
    object_groups: Sequence[Sequence[dict[str, Any]]],
    sim: Any,
    *,
    settle_steps: int = 20,
) -> None:
    """Restore tables, settle, restore objects, and settle the completed batch."""
    for env_idx in env_idx_list:
        table = tables[env_idx]
        if table is not None:
            table.apply_saved_pose()

    for _ in range(settle_steps):
        sim.sim_step(render=False)

    for env_idx in env_idx_list:
        for group in object_groups[env_idx]:
            for obj in group.values():
                obj.apply_saved_pose()

    for _ in range(settle_steps):
        sim.sim_step(render=False)
