"""Recover articulated neck frames and reviewed part blocks from the supplied STL.

STL carries no joint mates. Recover axis lines from concentric cylindrical CAD
surfaces; preserve dimensions and record fit residuals and source triangle ranges.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DESC = ROOT / "robot/yor-description"
SOURCE = ROOT / "CAD Parts/SPINAL DESIGN.stl"
SOURCE_SHA = "abce9493faff32b848605c3744614f8044bdbf7c3f8314841db210e57dccf48d"
ROTATION = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
ORIGIN = np.array([-8.996849060058594, -26.647842407226562, -109.06756591796875])


def circle(points, initial):
    fit = least_squares(
        lambda x: np.linalg.norm(points - x[:2], axis=1) - x[2],
        initial,
        loss="soft_l1",
        f_scale=0.01,
    )
    errors = abs(np.linalg.norm(points - fit.x[:2], axis=1) - fit.x[2])
    inliers = points[errors < 0.02]
    fit = least_squares(lambda x: np.linalg.norm(inliers - x[:2], axis=1) - x[2], fit.x)
    rms = float(np.std(np.linalg.norm(inliers - fit.x[:2], axis=1)))
    if len(inliers) < 40 or rms > 0.005:
        raise ValueError("CAD cylinder-axis recovery failed")
    return fit.x, dict(inliers=len(inliers), radial_rms_mm=rms, radius_mm=float(fit.x[2]))


def write_stl(path, triangles):
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("a", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    records["normal"], records["vertices"] = normals, triangles
    path.write_bytes(
        b"Sophia spinal CAD: rigid frame conversion, SI units".ljust(80, b"\0")
        + struct.pack("<I", len(records))
        + records.tobytes()
    )


def main():
    payload = SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError("Spinal source changed; review part selections")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("a", "<u2")])
    records = np.frombuffer(payload, offset=84, dtype=dtype)
    triangles = records["vertices"].astype(float)
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    lens_centers = np.array(
        [
            [-70.48448181152344, -75.18571853637695, 144.29367065429688],
            [49.489200592041016, -75.18571853637695, 146.8067626953125],
        ]
    )
    tilt_axis = lens_centers[1] - lens_centers[0]
    tilt_axis /= np.linalg.norm(tilt_axis)
    transverse = np.stack([[0, 1, 0], np.cross(tilt_axis, [0, 1, 0])])
    radius = np.linalg.norm(vertices[:, :2] - [-9.09628, -37.29672], axis=1)
    use = (abs(radius - 3) < 0.7) & (vertices[:, 2] > 54) & (vertices[:, 2] < 61)
    pan_circle, pan_quality = circle(vertices[use, :2], [-9.09628, -37.29672, 3])
    pan = np.array([*pan_circle[:2], 54.723683])
    projected = vertices @ transverse.T
    radius = np.linalg.norm(projected - [-67.69672, 105.3], axis=1)
    axial = vertices @ tilt_axis
    use = (abs(radius - 10) < 0.7) & (axial > -30) & (axial < -20)
    tilt_circle, tilt_quality = circle(projected[use], [-67.69672, 105.3, 10])
    # Pick the point on the recovered tilt line at the pan axis's lateral datum.
    z = (tilt_circle[1] + tilt_axis[2] * pan[0]) / tilt_axis[0]
    tilt = np.array([pan[0], tilt_circle[0], z])
    lens_midpoint = np.array([-10.48698, -79.56121826171875, 145.66199493408203])
    frames = {"head_neck": ORIGIN, "head_pan": pan, "head_tilt": tilt}
    # These are contiguous reviewed part blocks, not bounding-box clipping.
    parts = [
        ("fixed_adapter", "head_neck", [(310517, 313575)]),
        ("pan_servo", "head_neck", [(372759, 383542)]),
        ("pan_servo_support", "head_neck", [(301587, 310517)]),
        ("pan_bracket", "head_pan", [(176723, 187351)]),
        ("tilt_servo", "head_pan", [(132277, 139214)]),
        ("tilt_bearing", "head_tilt", [(139214, 141039)]),
        ("tilt_bracket", "head_tilt", [(224379, 227431)]),
        ("zed_camera", "head_tilt", [(227431, 301481)]),
    ]
    assets = []
    for name, body, ranges in parts:
        source = np.concatenate([triangles[a:b] for a, b in ranges])
        transformed = (source - frames[body]) @ ROTATION.T * 0.001
        path = DESC / f"assets_nero/sophia/spinal_{name}.stl"
        write_stl(path, transformed)
        assets.append(
            dict(
                name=name,
                body=body,
                asset=str(path.relative_to(DESC / "assets_nero")),
                triangles=len(source),
                source_ranges=ranges,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                local_bounds_m=[transformed.min((0, 1)).tolist(), transformed.max((0, 1)).tolist()],
            )
        )
    right = -tilt_axis
    up = np.cross(tilt_axis, [0, 1, 0])
    camera_rotation = ROTATION @ np.column_stack([right, up, [0, 1, 0]])
    report = dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=SOURCE_SHA,
        units_scale=0.001,
        source_to_neck_rotation=ROTATION.tolist(),
        mount_origin_source_mm=ORIGIN.tolist(),
        pan_axis_source_mm=pan.tolist(),
        tilt_axis_point_source_mm=tilt.tolist(),
        tilt_axis_direction_source=tilt_axis.tolist(),
        axis_recovery={"pan": pan_quality, "tilt": tilt_quality},
        pan_position_neck_m=(ROTATION @ (pan - ORIGIN) * 0.001).tolist(),
        tilt_position_pan_m=(ROTATION @ (tilt - pan) * 0.001).tolist(),
        tilt_axis_direction_local=(ROTATION @ tilt_axis).tolist(),
        camera_midpoint_tilt_m=(ROTATION @ (lens_midpoint - tilt) * 0.001).tolist(),
        camera_rotation_local=camera_rotation.tolist(),
        parts=assets,
        notes=[
            "Mount origin follows the symmetric lower lattice attachment bolt footprint.",
            "Pan point height is an arbitrary datum along the recovered vertical axis.",
            "STL has no mates or encoder reference; fitted servo reference remains separate.",
            "Glass-front centre is a nominal optical origin, not a measured entrance pupil.",
            "Camera XY is subsequently aligned to the user-confirmed neutral overhang/centring.",
            "Small fasteners omitted; source masses and inertias remain approximations.",
            "Separate physical part hulls avoid one convex hull around the complete neck.",
        ],
    )
    path = DESC / "calibration/spinal_assembly_2026-11-04.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["pan_position_neck_m", "tilt_position_pan_m", "axis_recovery"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
