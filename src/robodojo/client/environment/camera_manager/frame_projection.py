"""Reusable camera-frame projection stages, independent of capture transport."""

from __future__ import annotations

import cv2
import numpy as np


class FrameProjectionPipeline:
    def __init__(self, camera_config, camera_names, intrinsic_provider):
        self.camera_config = camera_config
        self.camera_names = camera_names
        self.intrinsic_provider = intrinsic_provider
        self._fisheye_maps = {}

    def _fisheye_map(self, cam_id: int, width: int, height: int):
        key = (cam_id, width, height)
        if key in self._fisheye_maps:
            return self._fisheye_maps[key]
        name = self.camera_names[0][cam_id]
        camera = self.camera_config[name].camera
        if camera.get("projection_backend") != "pinhole_postprocess":
            self._fisheye_maps[key] = None
            return None

        cx, cy = float(camera.cx), float(camera.cy)
        fx, fy = float(camera.fx), float(camera.fy)
        intrinsics = self.intrinsic_provider(cam_id)
        backing_fx, backing_fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = (xx - cx) / fx
        dy = (yy - cy) / fy
        theta_d = np.sqrt(dx * dx + dy * dy)
        theta = theta_d.copy()
        coefficients = np.asarray(camera.get("distortion_coefficients", [0.0] * 4), dtype=np.float32)
        for _ in range(5):
            t2 = theta * theta
            scale = (
                1.0 + coefficients[0] * t2 + coefficients[1] * t2**2 + coefficients[2] * t2**3 + coefficients[3] * t2**4
            )
            theta = np.divide(theta_d, scale, out=theta.copy(), where=np.abs(scale) > 1e-6)
        radius = np.tan(theta)
        unit_x = np.divide(dx, theta_d, out=np.zeros_like(dx), where=theta_d > 1e-8)
        unit_y = np.divide(dy, theta_d, out=np.zeros_like(dy), where=theta_d > 1e-8)
        map_x = (width / 2.0 + backing_fx * radius * unit_x).astype(np.float32)
        map_y = (height / 2.0 + backing_fy * radius * unit_y).astype(np.float32)
        self._fisheye_maps[key] = (map_x, map_y)
        return map_x, map_y

    def apply(self, cam_id: int, annotator_name: str, frames: np.ndarray) -> np.ndarray:
        projected_annotators = {"rgb", "instance_id_segmentation_fast"}
        if annotator_name not in projected_annotators:
            return frames
        height, width = frames.shape[1:3]
        remap = self._fisheye_map(cam_id, width, height)
        if remap is None:
            return frames
        map_x, map_y = remap
        interpolation = cv2.INTER_LINEAR if annotator_name == "rgb" else cv2.INTER_NEAREST
        projected = []
        for frame in frames:
            source = frame.astype(np.float32) if annotator_name != "rgb" else frame
            remapped = cv2.remap(source, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)
            if annotator_name != "rgb":
                remapped = np.rint(remapped).astype(frame.dtype)
            projected.append(remapped)
        return np.stack(projected)
