"""MuJoCo pinhole projection metadata with explicit pixel-center resizing."""

import json
import math
from functools import reduce
from pathlib import Path

import numpy as np

_DESCRIPTION = Path(__file__).resolve().parent / "yor-description"


def stereo_render_size(model, camera_names, width, height):
    """Render the full calibrated aspect, then resize to the requested output.

    MuJoCo's OpenGL renderer uses square pixels. The rectified ZED profiles have
    equal fx/fy. Rendering their native aspect first preserves the full image
    even when a client requests a different aspect ratio.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    cameras = [model.camera(name).id for name in camera_names]
    if not all(np.all(model.cam_sensorsize[c] > 0) for c in cameras):
        return width, height
    resolution = model.cam_resolution[cameras[0]]
    for c in cameras:
        focal = (
            model.cam_intrinsic[c, :2]
            / model.cam_sensorsize[c]
            * model.cam_resolution[c]
        )
        if not np.allclose(focal[0], focal[1], rtol=1e-5) or not np.array_equal(
            resolution, model.cam_resolution[c]
        ):
            raise ValueError(
                "Stereo publisher requires equal-focal, equal-resolution rectified eyes"
            )
    divisor = reduce(math.gcd, map(int, resolution))
    aspect_width, aspect_height = resolution // divisor
    scale = max(1, math.ceil(height / aspect_height))
    return int(aspect_width * scale), int(aspect_height * scale)


def camera_intrinsics(
    model, name, render_width, render_height, output_width=None, output_height=None
):
    """Return the actual OpenGL frustum K, then apply cv2.resize's pixel mapping."""
    camera = model.camera(name).id
    size = model.cam_sensorsize[camera]
    if np.all(size > 0):
        fx, fy, px, py = model.cam_intrinsic[camera].astype(float)
        focal = render_height * fy / size[1]
        cx = (render_width - 1) / 2 - px / fx * focal
        cy = (render_height - 1) / 2 - py / size[1] * render_height
    else:
        focal = render_height / 2 / math.tan(math.radians(model.cam_fovy[camera]) / 2)
        cx, cy = (render_width - 1) / 2, (render_height - 1) / 2
    width, height = output_width or render_width, output_height or render_height
    sx, sy = width / render_width, height / render_height
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(focal * sx),
        "fy": float(focal * sy),
        "cx": float((cx + 0.5) * sx - 0.5),
        "cy": float((cy + 0.5) * sy - 0.5),
    }


class SophiaCameraRenderer:
    """Four calibrated RGB/depth views; caller owns the physics/render lock.

    Wrist extrinsics are accepted photo estimates. Raw wrist output uses measured
    OpenCV distortion. Head output is already rectified. A wider internal frustum
    is remapped to each requested lens, including unequal fx/fy and pixel centers.
    The temporary render frustum is restored before returning.
    """

    names = ("head_zed_left", "head_zed_right", "left_wrist", "right_wrist")

    def __init__(self, model, width=640, height=360):
        import mujoco

        if width <= 0 or height <= 0:
            raise ValueError("Positive output dimensions required")
        self.model, self.width, self.height = model, width, height
        self.profiles = json.loads(
            (_DESCRIPTION / "calibration/camera_intrinsics_2026-10-30.json").read_text()
        )
        self.config = json.loads((_DESCRIPTION / "sophia.json").read_text())
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.maps = {}

    def profile(self, name, raw):
        if name not in self.names:
            raise ValueError(f"Unknown Sophia camera: {name}")
        if name.startswith("head_zed_"):
            profile = self.profiles["zed"]["rectified"][name.rsplit("_", 1)[-1]]
        else:
            profile = self.profiles[name.replace("wrist", "gripper")]
        sx, sy = self.width / profile["width"], self.height / profile["height"]
        k = np.array(
            [
                [profile["fx"] * sx, 0, (profile["cx"] + 0.5) * sx - 0.5],
                [0, profile["fy"] * sy, (profile["cy"] + 0.5) * sy - 0.5],
                [0, 0, 1.0],
            ]
        )
        distortion = np.array(profile["distortion"] if raw else [0.0] * 5)
        return k, distortion

    def render(self, data, name, *, raw=True):
        import cv2

        k, distortion = self.profile(name, raw)
        cid = self.model.camera(name).id
        original = self.model.cam_intrinsic[cid].copy()
        try:
            # Broad frustum covers rays near the edges of the distorted wrist
            # images. Exact desired K is recovered by the ray mapping below.
            self.model.cam_intrinsic[cid, :2] = original[:2] * 0.45
            self.model.cam_intrinsic[cid, 2:] = 0
            actual = camera_intrinsics(self.model, name, self.width, self.height)
            key = (name, raw)
            if key not in self.maps:
                u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
                pixels = np.stack((u, v), axis=-1).astype(np.float64).reshape(-1, 1, 2)
                rays = cv2.undistortPointsIter(
                    pixels,
                    k,
                    distortion,
                    None,
                    None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
                ).reshape(self.height, self.width, 2)
                mx = (rays[..., 0] * actual["fx"] + actual["cx"]).astype(np.float32)
                my = (rays[..., 1] * actual["fy"] + actual["cy"]).astype(np.float32)
                # Flag noninvertible/extreme lens regions instead of fabricating
                # pixels. Forward projection must round-trip to the output ray.
                points = np.dstack((rays, np.ones((self.height, self.width))))
                projected, _ = cv2.projectPoints(
                    points.reshape(-1, 3), np.zeros(3), np.zeros(3), k, distortion
                )
                roundtrip = np.linalg.norm(
                    projected.reshape(pixels.shape) - pixels, axis=-1
                )
                valid = roundtrip.reshape(self.height, self.width) < 0.1
                valid &= (
                    (mx >= 0)
                    & (mx <= self.width - 1)
                    & (my >= 0)
                    & (my <= self.height - 1)
                )
                self.maps[key] = mx, my, valid
            mx, my, valid = self.maps[key]
            self.renderer.update_scene(data, camera=name, scene_option=self.option)
        finally:
            self.model.cam_intrinsic[cid] = original
        self.renderer.disable_depth_rendering()
        rgb = cv2.remap(self.renderer.render(), mx, my, cv2.INTER_LINEAR)
        self.renderer.enable_depth_rendering()
        depth = cv2.remap(self.renderer.render(), mx, my, cv2.INTER_NEAREST)
        self.renderer.disable_depth_rendering()
        rgb[~valid] = 0
        depth[~valid] = np.nan
        optical = np.eye(4)
        optical[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        optical[:3, 3] = data.cam_xpos[cid]
        parent = self.model.cam_bodyid[cid]
        parent_transform = np.eye(4)
        parent_transform[:3, :3] = data.xmat[parent].reshape(3, 3)
        parent_transform[:3, 3] = data.xpos[parent]
        metadata = {
            "camera": name,
            "timestamp_s": float(data.time),
            "width": self.width,
            "height": self.height,
            "K": k.tolist(),
            "distortion": distortion.tolist(),
            "image_model": "opencv_distorted"
            if np.any(distortion)
            else "pinhole_rectified",
            "optical_axes": "x right, y down, z forward",
            "depth": "optical z in meters",
            "parent_body": self.model.body(parent).name,
            "T_world_optical": optical.tolist(),
            "T_parent_optical": (np.linalg.inv(parent_transform) @ optical).tolist(),
            "mount_status": "accepted_photo_estimate"
            if name.endswith("wrist")
            else "accepted_CAD_transform",
            "hardware_extrinsics_calibrated": False,
            "valid_pixel_fraction": float(valid.mean()),
        }
        return rgb, depth, valid.copy(), metadata

    def close(self):
        self.renderer.close()
