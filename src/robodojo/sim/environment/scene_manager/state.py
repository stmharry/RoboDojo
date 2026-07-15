"""Shared tensor-state normalization for scene objects."""

from __future__ import annotations

from typing import Any

import torch


def make_root_pose_relative(root_pose: torch.Tensor, env_origin: Any) -> torch.Tensor:
    """Subtract an environment origin without changing pose dtype or device."""

    if root_pose.ndim != 1 or root_pose.numel() < 3:
        raise ValueError(f"root pose must be a flat tensor with at least three values, got {tuple(root_pose.shape)}")
    origin = torch.as_tensor(env_origin, dtype=root_pose.dtype, device=root_pose.device).flatten()
    if origin.numel() < 3:
        origin = torch.cat(
            [
                origin,
                torch.zeros(3 - origin.numel(), dtype=root_pose.dtype, device=root_pose.device),
            ]
        )
    result = root_pose.clone()
    result[:3] -= origin[:3]
    return result
