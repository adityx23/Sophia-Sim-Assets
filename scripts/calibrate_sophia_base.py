"""Reconstruct the yellow frame from real stereo, holding head/arm calibration fixed."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_camera import camera_intrinsics  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import measured_pose  # noqa: E402

OUT = ROOT / "output/asset-review/base-calibration"
DATA = ROOT / "output/asset-review/head-calibration-pipeline/data"
FRAMES = [
    (0, 90, "train"),
    (0, 930, "train"),
    (3, 1080, "train"),
    (0, 810, "validation"),
    (0, 960, "validation"),
    (1, 855, "validation"),
    (1, 1350, "validation"),
    (1, 1425, "validation"),
    (3, 795, "validation"),
    (3, 861, "validation"),
    (2, 84, "validation"),
    (2, 1590, "validation"),
]


def robust_line(x, y):
    mid = np.median(x)
    xx = x - mid
    result = least_squares(
        lambda v: v[0] * xx + v[1] - y, np.polyfit(xx, y, 1), loss="soft_l1", f_scale=0.7
    )
    a, b = result.x
    return a, b - a * mid, float(np.sqrt(np.mean(np.minimum(abs(a * xx + b - y), 5) ** 2)))


def yellow_corners(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([19, 140, 100]), np.array([40, 255, 255]))
    # This ROI was reviewed in the selected recordings; it excludes gripper cables.
    mask[:800] = 0
    mask[:, :750] = 0
    mask[:, 1300:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No yellow frame")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1500:
        raise ValueError("Too little visible frame")
    connected = [c for c in contours if cv2.contourArea(c) > 0.08 * cv2.contourArea(contour)]
    x, y, w, h = cv2.boundingRect(np.vstack(connected))
    if w < 250:
        raise ValueError("Frame top rail not sufficiently visible")
    xx = []
    yy = []
    for column in range(x + 40, x + w - 40):
        pixels = np.flatnonzero(mask[:, column])
        if len(pixels):
            xx.append(column)
            yy.append(pixels[0])
    a, b, top_rms = robust_line(np.array(xx), np.array(yy))
    ys = []
    left = []
    right = []
    for row in range(int(a * (x + w / 2) + b + 35), 1070):
        pixels = np.flatnonzero(mask[row])
        if len(pixels):
            ys.append(row)
            left.append(pixels[0])
            right.append(pixels[-1])
    if len(ys) < 10:
        raise ValueError("Insufficient side rail extent")
    points = []
    for edge in (left, right):
        aa, bb, rms = robust_line(np.array(ys), np.array(edge))
        u = (aa * b + bb) / (1 - aa * a)
        v = a * u + b
        patch = cv2.getRectSubPix(mask, (15, 15), (float(u), float(v)))
        if cv2.countNonZero(patch) < 12:
            raise ValueError("Extrapolated corner is occluded")
        points.append([u, v])
    if top_rms > 3:
        raise ValueError("Nonlinear/occluded top rail")
    return np.array(points), dict(top_line_rms_px=top_rms, mask_pixels=cv2.countNonZero(mask))


def match_one(left, right, uv):
    """Subpixel NCC with bidirectional search; uses no simulated landmark positions."""

    def match(a, b, point, reverse=False):
        low, high = 50, 180
        template = cv2.getRectSubPix(a, (31, 31), tuple(map(float, point)))
        centre = (float(point[0] + (1 if reverse else -1) * (low + high) / 2), float(point[1]))
        search = cv2.getRectSubPix(b, (high - low + 31, 35), centre)
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, where = cv2.minMaxLoc(scores)
        ix, iy = where

        def refine(values, index):
            if index <= 0 or index >= len(values) - 1:
                return 0.0
            left, c, right = values[index - 1 : index + 2]
            den = left - 2 * c + right
            return (
                float(np.clip(0.5 * (left - right) / den, -0.5, 0.5)) if abs(den) > 1e-10 else 0.0
            )

        dx = refine(scores[iy], ix)
        dy = refine(scores[:, ix], iy)
        x = point[0] + (low if reverse else -high) + ix + dx
        y = point[1] - 2 + iy + dy
        return np.array([x, y]), float(score)

    gl, gr = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (left, right)]
    right_uv, score = match(gl, gr, uv)
    recovered, reverse_score = match(gr, gl, right_uv, True)
    error = float(np.linalg.norm(recovered - uv))
    valid = bool(
        score >= 0.88 and reverse_score >= 0.88 and error <= 1.5 and abs(right_uv[1] - uv[1]) <= 1.5
    )
    return right_uv, dict(ncc=score, reverse_ncc=reverse_score, reverse_error_px=error, valid=valid)


def main():
    OUT.mkdir(exist_ok=True)
    sim = YORMujoco(
        OUT / "baseline/scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    m, d = sim.model, sim.data
    ids = [m.camera("head_zed_" + eye).id for eye in ("left", "right")]
    k = camera_intrinsics(m, "head_zed_left", 1920, 1080)
    baseline = float(np.linalg.norm(m.cam_pos[ids[0]] - m.cam_pos[ids[1]]))
    records = []
    excluded = []
    try:
        for ep, frame, split in FRAMES:
            images = []
            for eye in ("left", "right"):
                if ep == 2:
                    image = cv2.imread(str(OUT / f"data/ep2-frame{frame}-{eye}.png"))
                    ok = image is not None
                else:
                    cap = cv2.VideoCapture(str(DATA / f"episode-{ep}-zed-{eye}.mp4"))
                    cap.set(1, frame)
                    ok, image = cap.read()
                    cap.release()
                if not ok:
                    raise ValueError("Source image missing")
                images.append(image)
            try:
                corners, quality = yellow_corners(images[0])
            except ValueError as error:
                excluded.append(dict(episode=ep, frame=frame, reason=str(error)))
                continue
            native = dict(
                np.load(
                    ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
                )
            )
            for side in ("left", "right"):
                native[side + "_gripper"] = native[side + "_gripper"].reshape(-1)
            alignment = json.loads(
                ((OUT / "data" if ep == 2 else DATA) / f"episode-{ep}-alignment.json").read_text()
            )[frame]
            if not alignment["valid"]:
                excluded.append(
                    dict(episode=ep, frame=frame, reason="Invalid capture-time alignment")
                )
                continue
            index = alignment["index"]
            measured_pose(sim, d, native, index)
            lo = max(0, index - 3)
            hi = min(len(native["time_s"]) - 1, index + 3)
            speed = np.max(
                abs(
                    (native["neck_deg"][hi] - native["neck_deg"][lo])
                    / (native["time_s"][hi] - native["time_s"][lo])
                )
            )
            if speed > 5:
                excluded.append(
                    dict(
                        episode=ep,
                        frame=frame,
                        reason="Moving neck: timing sensitivity",
                        neck_speed_dps=float(speed),
                    )
                )
                continue
            for corner, uv in zip(("front_left", "front_right"), corners, strict=True):
                right, match = match_one(*images, uv)
                if not match["valid"]:
                    excluded.append(
                        dict(
                            episode=ep,
                            frame=frame,
                            corner=corner,
                            reason="Stereo rejected",
                            quality=match,
                        )
                    )
                    continue
                disparity = uv[0] - right[0]
                z = k["fx"] * baseline / disparity
                p = np.array(
                    [(uv[0] - k["cx"]) * z / k["fx"], -(uv[1] - k["cy"]) * z / k["fy"], -z]
                )
                world = d.cam_xpos[ids[0]] + d.cam_xmat[ids[0]].reshape(3, 3) @ p
                chassis = m.body("sophia_chassis").id
                local = (world - d.xpos[chassis]) @ d.xmat[chassis].reshape(3, 3)
                records.append(
                    dict(
                        episode=ep,
                        frame=frame,
                        split=split,
                        corner=corner,
                        left_px=uv.tolist(),
                        right_px=right.tolist(),
                        world_m=world.tolist(),
                        chassis_m=local.tolist(),
                        raw_lift_m=float(native["lift_height"][index, 0]),
                        qpos=d.qpos.tolist(),
                        camera_world=d.cam_xpos[ids[0]].tolist(),
                        camera_rotation=d.cam_xmat[ids[0]].reshape(3, 3).tolist(),
                        quality=match,
                    )
                )
            for eye, image in enumerate(images):
                for record in [r for r in records if r["episode"] == ep and r["frame"] == frame]:
                    uv = record["left_px" if eye == 0 else "right_px"]
                    cv2.circle(image, tuple(np.rint(uv).astype(int)), 7, (0, 0, 255), 2)
                cv2.imwrite(str(OUT / f"review/stereo-{ep}-{frame}-{eye}.jpg"), image)
    finally:
        sim.stop()
    report = dict(
        intrinsics=k,
        baseline_m=baseline,
        records=records,
        excluded=excluded,
        method=(
            "Yellow-rail line intersections and real-image subpixel stereo, "
            "with accepted camera/arm transforms held fixed"
        ),
    )
    (OUT / "stereo-base-points.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in records:
        print(r["episode"], r["frame"], r["corner"], np.round(r["world_m"], 4), r["quality"]["ncc"])
    print("excluded", excluded)


if __name__ == "__main__":
    main()
