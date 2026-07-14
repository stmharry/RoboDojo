from __future__ import annotations

from typing import Any, Dict

from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
import numpy as np
import omni.kit.commands
from pxr import UsdShade


class Room(SingleGeometryPrim):
    """Geometry wrapper for the room USD asset."""

    def __init__(
        self,
        prim_path: str | None = None,
        usd_path: str | None = None,
        instance_config: Dict[str, Any] | None = None,
    ):
        """
        Initialize Room object wrapping the room geometry prim.

        Args:
            prim_path: Prim path in Isaac Sim (optional)
            usd_path: USD asset file path (optional, if None will wrap existing prim)
            instance_config: Instance configuration dictionary (optional)
        """
        if prim_path is not None:
            # Load USD asset if usd_path is provided
            if usd_path is not None:
                room_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
                if not room_prim or not room_prim.IsValid():
                    raise RuntimeError(f"Failed to load USD from {usd_path} to {prim_path}")

            collision = instance_config.get("collision", True) if instance_config else True
            SingleGeometryPrim.__init__(
                self,
                prim_path=prim_path,
                name=prim_path.split("/")[-1],
                collision=collision,
            )

            if instance_config:
                path_parts = prim_path.split("/")
                self.instance_name = path_parts[-1]
                self.category_name = self.instance_name.rsplit("_", 1)[0]
                pos = instance_config.get("default_pos", [0.0, 0.0, 0])
                rot = instance_config.get("default_ori", [1.0, 0.0, 0.0, 0.0])
                self.set_local_pose(translation=pos, orientation=rot)
                scale = instance_config.get("scale", [1.0, 1.0, 1.0])
                self.set_local_scale(scale)
                visual_color = instance_config.get("visual_color")
                if visual_color is not None:
                    self.apply_visual_color(visual_color)

    def apply_visual_color(self, color):
        """Apply an opt-in stronger material without changing room geometry."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,) or not np.all(np.isfinite(color)) or np.any((color < 0.0) | (color > 1.0)):
            raise ValueError("room visual_color must contain three finite values in [0, 1]")
        material_path = find_unique_string_name(
            self.prim_path + "/VisualColor",
            lambda path: not is_prim_path_valid(path),
        )
        self.visual_material = PreviewSurface(prim_path=material_path, color=color)
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.prim_path,
            material_path=material_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        self.visual_material_path = material_path
