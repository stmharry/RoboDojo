from dataclasses import dataclass

import numpy as np

VALID_JAW_MODES = ("stock", "enlarged")

# The LeRobot jaw STL uses (length, width, thickness) as (X, Y, Z), while
# OpenArm finger links use (width, thickness, length) as (X, Y, Z).
JAW_CAD_TO_FINGER = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class JawRegistration:
    vertices: np.ndarray
    transform: np.ndarray
    stock_bounds: np.ndarray
    enlarged_bounds: np.ndarray
    base_plane_error_m: float
    cross_section_center_error_m: float
    width_error_m: float
    thickness_error_m: float

    def as_dict(self) -> dict:
        return {
            "transform_cad_to_finger": self.transform.tolist(),
            "stock_bounds_m": self.stock_bounds.tolist(),
            "enlarged_bounds_m": self.enlarged_bounds.tolist(),
            "rotation_determinant": float(np.linalg.det(self.transform[:3, :3])),
            "base_plane_error_m": self.base_plane_error_m,
            "cross_section_center_error_m": self.cross_section_center_error_m,
            "width_error_m": self.width_error_m,
            "thickness_error_m": self.thickness_error_m,
            "forward_extension_m": float(self.enlarged_bounds[1, 2] - self.stock_bounds[1, 2]),
        }


def validate_jaw_mode(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_JAW_MODES:
        choices = "|".join(VALID_JAW_MODES)
        raise ValueError(f"asset.jaw must be one of {choices}; got {value!r}")
    return value


def register_enlarged_jaw(
    cad_vertices_m: np.ndarray,
    stock_vertices_finger_m: np.ndarray,
) -> JawRegistration:
    """Register the published enlarged jaw to an OpenArm finger-link frame."""
    cad_vertices = _vertices(cad_vertices_m, "enlarged jaw")
    stock_vertices = _vertices(stock_vertices_finger_m, "stock jaw")
    rotated = cad_vertices @ JAW_CAD_TO_FINGER.T
    stock_bounds = np.vstack((stock_vertices.min(axis=0), stock_vertices.max(axis=0)))
    rotated_bounds = np.vstack((rotated.min(axis=0), rotated.max(axis=0)))

    stock_center = stock_bounds.mean(axis=0)
    rotated_center = rotated_bounds.mean(axis=0)
    translation = np.asarray(
        (
            stock_center[0] - rotated_center[0],
            stock_center[1] - rotated_center[1],
            stock_bounds[0, 2] - rotated_bounds[0, 2],
        )
    )
    vertices = rotated + translation
    enlarged_bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    transform = np.eye(4)
    transform[:3, :3] = JAW_CAD_TO_FINGER
    transform[:3, 3] = translation

    base_error = abs(float(enlarged_bounds[0, 2] - stock_bounds[0, 2]))
    center_error = float(np.linalg.norm(enlarged_bounds.mean(axis=0)[:2] - stock_center[:2]))
    stock_extents = stock_bounds[1] - stock_bounds[0]
    enlarged_extents = enlarged_bounds[1] - enlarged_bounds[0]
    width_error = abs(float(enlarged_extents[0] - stock_extents[0]))
    thickness_error = abs(float(enlarged_extents[1] - stock_extents[1]))

    if not np.isclose(np.linalg.det(JAW_CAD_TO_FINGER), 1.0, atol=1e-12):
        raise ValueError("enlarged jaw registration is not a proper rotation")
    if base_error > 0.0005 or center_error > 0.0005:
        raise ValueError(
            "enlarged jaw base does not align with stock jaw: "
            f"base={base_error:.6f}m center={center_error:.6f}m"
        )
    if width_error > 0.001 or thickness_error > 0.001:
        raise ValueError(
            "enlarged jaw cross-section does not match stock jaw: "
            f"width={width_error:.6f}m thickness={thickness_error:.6f}m"
        )
    return JawRegistration(
        vertices=vertices,
        transform=transform,
        stock_bounds=stock_bounds,
        enlarged_bounds=enlarged_bounds,
        base_plane_error_m=base_error,
        cross_section_center_error_m=center_error,
        width_error_m=width_error,
        thickness_error_m=thickness_error,
    )


def _vertices(value: np.ndarray, label: str) -> np.ndarray:
    vertices = np.asarray(value, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{label} vertices must have shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label} vertices must be finite")
    return vertices
