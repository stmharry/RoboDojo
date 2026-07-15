"""Small tensor-pose adapters shared by simulator object wrappers."""

from __future__ import annotations

import numpy as np
import torch


def articulation_link_pose_wxyz(articulation_view, link_name: str, prim_path: str) -> np.ndarray:
    """Read one live PhysX link transform and return ``[xyz, qwxyz]``."""
    body_index = articulation_view.get_body_index(link_name)
    physics_view = getattr(articulation_view, "_physics_view", None)
    if body_index is None or physics_view is None:
        raise RuntimeError(f"Physics link '{link_name}' is unavailable on {prim_path}")
    pose_xyzw = physics_view.get_link_transforms()[0, body_index]
    if isinstance(pose_xyzw, torch.Tensor):
        pose_xyzw = pose_xyzw.detach().cpu().numpy()
    elif hasattr(pose_xyzw, "numpy"):
        pose_xyzw = pose_xyzw.numpy()
    pose_xyzw = np.asarray(pose_xyzw, dtype=np.float32).reshape(-1)
    if pose_xyzw.size != 7 or not np.isfinite(pose_xyzw).all():
        raise RuntimeError(f"Physics returned an invalid pose for link '{link_name}' on {prim_path}")
    return np.concatenate((pose_xyzw[:3], pose_xyzw[[6, 3, 4, 5]]))
