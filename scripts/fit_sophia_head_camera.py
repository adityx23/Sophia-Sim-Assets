"""Fit a fixed head-camera transform from manually reviewed fingertip landmarks.

All observations are native-video pixels. Kinematics use measured joints at
camera capture times, never commanded or physics-replayed joints. The fitted
transform is relative to the moving head_tilt body, so neck motion is retained.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_camera import camera_intrinsics  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import measured_pose, writer  # noqa: E402

REPLAY = ROOT / "output/real2sim-videos/pi-atlas-episode-0"
NATIVE = ROOT / "output/asset-review/hf-sanity-check/native-replay/episode_000000.npz"
OUT = ROOT / "output/asset-review/head-camera-fit"
TIMES = [3, 5, 31, 32, 42, 43, 44, 45, 46, 48, 50, 52, 54]
TIP = np.array([0.0, 0.0, 0.0035])  # centre of vendor mesh's distal end face


def project(points, camera_pos, camera_rot, k):
    v = (points - camera_pos) @ camera_rot
    return np.stack(
        [k["fx"] * v[:, 0] / -v[:, 2] + k["cx"], k["cy"] - k["fy"] * v[:, 1] / -v[:, 2]], axis=-1
    )


def prepare(episode=0, times=None):
    OUT.mkdir(parents=True, exist_ok=True)
    sim = YORMujoco(
        REPLAY / "scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    m, d = sim.model, sim.data
    native_path = NATIVE.with_name(f"episode_{episode:06d}.npz")
    native = dict(np.load(native_path))
    for side in ("left", "right"):
        native[side + "_gripper"] = native[side + "_gripper"].reshape(-1)
    alignment = (
        REPLAY / "zed-frame-alignment.json"
        if episode == 0
        else OUT / f"episode-{episode}-alignment.json"
    )
    rows = json.loads(alignment.read_text())
    k = camera_intrinsics(m, "head_zed_left", 1920, 1080)
    cid = m.camera("head_zed_left").id
    b = m.body("head_tilt").id
    r0 = np.empty(9)
    mujoco.mju_quat2Mat(r0, m.cam_quat[cid])
    r0 = r0.reshape(3, 3)
    t0 = m.cam_pos[cid].copy()
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    video = (
        REPLAY / "real-zed-left-source.mp4"
        if episode == 0
        else OUT / f"real-zed-episode-{episode}.mp4"
    )
    prefix = "" if episode == 0 else f"episode-{episode}-"
    cap = cv2.VideoCapture(str(video))
    records = []
    names = [f"{s}_{a}" for s in ("left", "right") for a in ("a", "b")]
    with mujoco.Renderer(m, height=540, width=960) as renderer:
        for t in times or TIMES:
            frame = t * 30
            row = rows[frame]
            assert row["annotation.camera_valid.zed_left"]
            seconds = row["source.camera_capture_ns.zed_left"] / 1e9 - native["robot_timestamp"][0]
            i = int(
                np.clip(
                    np.searchsorted(native["time_s"], seconds, side="right") - 1,
                    0,
                    len(native["time_s"]) - 1,
                )
            )
            measured_pose(sim, d, native, i)
            rb, tb = d.xmat[b].reshape(3, 3), d.xpos[b].copy()
            neck_parent = m.body_parentid[m.body("head_neck").id]
            mount_to_tilt = rb.T @ d.xmat[neck_parent].reshape(3, 3)
            world = []
            for s in ("left", "right"):
                for a in ("a", "b"):
                    body = m.body(f"{s}_gripper_finger_{a}").id
                    world.append(d.xpos[body] + d.xmat[body].reshape(3, 3) @ TIP)
            points = (np.array(world) - tb) @ rb
            records.append(
                dict(
                    second=t,
                    video_frame=frame,
                    source_time_s=float(seconds),
                    measured_index=i,
                    neck_deg=native["neck_deg"][i].tolist(),
                    mount_to_tilt_rotation=mount_to_tilt.tolist(),
                    point_names=names,
                    points_head_tilt=points.tolist(),
                )
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, real = cap.read()
            assert ok
            cv2.imwrite(str(OUT / f"{prefix}real-{t:02d}s.jpg"), real)
            delta = Rotation.from_rotvec(np.deg2rad([-10.0, 0, 0])).as_matrix()
            rr = r0 @ delta
            q = Rotation.from_matrix(rr).as_quat(scalar_first=True)
            m.cam_quat[cid] = q
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera="head_zed_left", scene_option=option)
            rendered = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            uv = project(points, t0, rr, k) / 2
            for name, pt in zip(names, uv, strict=True):
                if np.isfinite(pt).all() and np.max(np.abs(pt)) < 10000:
                    xy = tuple(np.round(pt).astype(int))
                    cv2.circle(rendered, xy, 5, (0, 255, 255), 1)
                    cv2.putText(rendered, name, xy, 0, 0.4, (0, 255, 255), 1)
            panel = np.vstack([cv2.resize(real, (960, 540)), rendered])
            cv2.imwrite(str(OUT / f"{prefix}landmarks-{t:02d}s.jpg"), panel)
    cap.release()
    (OUT / f"{prefix}kinematics.json").write_text(
        json.dumps(
            dict(
                intrinsics=k,
                camera_pos=t0.tolist(),
                camera_rotation=r0.tolist(),
                records=records,
                landmark_local_m=TIP.tolist(),
                native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                source_model_sha256=hashlib.sha256(
                    (REPLAY / "robot.mjcf").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


def observations(source, labels):
    points, observed, held, metadata = [], [], [], []
    for row in source["records"]:
        annotation = labels["frames"].get(str(row["second"]))
        if annotation is None:
            continue
        for side in ("left", "right"):
            names = [side + "_a", side + "_b"]
            if not all(name in annotation["pixels"] for name in names):
                continue
            points.append(
                [row["points_head_tilt"][row["point_names"].index(name)] for name in names]
            )
            observed.append(np.mean([annotation["pixels"][name] for name in names], axis=0))
            held.append(annotation["split"] == "validation")
            metadata.append(dict(second=row["second"], landmark=side + "_tip_pair_midpoint"))
    return np.array(points), np.array(observed), np.array(held), metadata


def stereo_matches():
    labels = json.loads((OUT / "annotations.json").read_text())
    cap = cv2.VideoCapture(str(OUT / "real-zed-right-episode-0.mp4"))
    records = []
    for t, annotation in labels["frames"].items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t) * 30)
        ok, right = cap.read()
        assert ok
        left = cv2.imread(str(OUT / f"real-{int(t):02d}s.jpg"))
        lg, rg = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (left, right)]
        for name, uv in annotation["pixels"].items():
            x, y = uv
            r = 10
            patch = lg[y - r : y + r + 1, x - r : x + r + 1]
            low, high = max(r, x - 220), x - 130
            scores = cv2.matchTemplate(
                rg[y - r - 2 : y + r + 3, low - r : high + r + 1], patch, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, where = cv2.minMaxLoc(scores)
            xr, yr = where[0] + low, where[1] + y - 2
            records.append(
                dict(second=int(t), landmark=name, left_px=uv, right_px=[xr, yr], ncc=float(score))
            )
            cv2.circle(right, (xr, yr), 7, (0, 255, 255), 1)
            cv2.putText(right, name, (xr, yr - 10), 0, 0.6, (0, 255, 255), 1)
        cv2.imwrite(str(OUT / f"stereo-matches-{int(t):02d}s.jpg"), right)
    cap.release()
    (OUT / "stereo-observations.json").write_text(json.dumps(records, indent=2) + "\n")


def fit(
    stereo=False,
    pitch_only=False,
    translation_frame="head_tilt",
    rigid_tilt=False,
    fixed_neck_x_m=None,
    output_report=None,
):
    source = json.loads((OUT / "kinematics.json").read_text())
    labels = json.loads((OUT / "annotations.json").read_text())
    r0, t0 = np.array(source["camera_rotation"]), np.array(source["camera_pos"])
    points, observed, held, metadata = observations(source, labels)
    record_map = {v["second"]: v for v in source["records"]}
    mount_rotations = np.array(
        [record_map[v["second"]]["mount_to_tilt_rotation"] for v in metadata]
    )
    right_observed = np.full_like(observed, np.nan)
    if stereo:
        matches = json.loads((OUT / "stereo-observations.json").read_text())
        for i, meta in enumerate(metadata):
            side = meta["landmark"].split("_")[0]
            pair = [
                v
                for v in matches
                if v["second"] == meta["second"] and v["landmark"].startswith(side + "_")
            ]
            # Conservative NCC threshold and left-finger disparity consistency.
            disparity = [v["left_px"][0] - v["right_px"][0] for v in pair]
            if (
                len(pair) == 2
                and min(v["ncc"] for v in pair) >= 0.7
                and (side != "left" or abs(disparity[0] - disparity[1]) <= 5)
            ):
                right_observed[i] = np.mean([v["right_px"] for v in pair], axis=0)
    kr = camera_intrinsics(
        mujoco.MjModel.from_xml_path(str(REPLAY / "scene.mjcf")), "head_zed_right", 1920, 1080
    )
    baseline_m = 0.1199918518066

    def camera_positions(x):
        nominal = t0
        if rigid_tilt:
            rd = r0 @ Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix() @ r0.T
            nominal = rd @ t0
        if translation_frame == "neck_mount":
            return nominal + mount_rotations @ (x[3:] * 0.01)
        return np.tile(nominal + x[3:] * 0.01, (len(points), 1))

    def predict(x):
        return (
            project(
                points.reshape(-1, 3),
                np.repeat(camera_positions(x), 2, axis=0),
                r0 @ Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix(),
                source["intrinsics"],
            )
            .reshape(-1, 2, 2)
            .mean(axis=1)
        )

    def residual(x, mask):
        left = (predict(x)[mask] - observed[mask]).ravel()
        if not stereo:
            return left
        rot = r0 @ Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix()
        right_pos = camera_positions(x) + rot @ [baseline_m, 0, 0]
        pred = (
            project(points.reshape(-1, 3), np.repeat(right_pos, 2, axis=0), rot, kr)
            .reshape(-1, 2, 2)
            .mean(axis=1)
        )
        valid = mask & np.isfinite(right_observed).all(axis=1)
        return np.r_[left, (pred[valid] - right_observed[valid]).ravel()]

    active = np.array([0, 3, 4, 5] if pitch_only else list(range(6)))
    if fixed_neck_x_m is not None:
        if translation_frame != "neck_mount":
            raise ValueError("Fixed neck X requires neck_mount translation frame")
        active = active[active != 3]

    def expand(v):
        x = np.zeros(6)
        if fixed_neck_x_m is not None:
            x[3] = fixed_neck_x_m * 100
        x[active] = v
        return x

    lower = np.array([-30, -15, -15, -15, -15, -15])[active]
    upper = np.array([30, 15, 15, 15, 15, 15])[active]

    def solve(initial, mask, noise=None):
        def fun(v):
            value = residual(expand(v), mask)
            return value if noise is None else value - noise

        return least_squares(
            fun, initial[active], loss="soft_l1", f_scale=4, bounds=(lower, upper), max_nfev=2000
        )

    candidates = []
    for pitch in [-20.0, -10.0, 0.0, 10.0]:
        initial = np.array([pitch, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = solve(initial, ~held)
        candidates.append(result)
    result = min(candidates, key=lambda r: r.cost)
    x = expand(result.x)
    fitted = predict(x)
    baseline = predict(np.zeros(6))

    def stats(pred, mask):
        errors = np.linalg.norm(pred[mask] - observed[mask], axis=1)
        return dict(
            n=int(mask.sum()),
            median_px=float(np.median(errors)),
            rmse_px=float(np.sqrt(np.mean(errors**2))),
            max_px=float(errors.max()),
        )

    singular = np.linalg.svd(result.jac, compute_uv=False)
    rng = np.random.default_rng(20261103)
    noise_fits = []
    for _ in range(100):
        noise = rng.normal(0, 6, residual(x, ~held).shape)
        trial = solve(x, ~held, noise)
        noise_fits.append(expand(trial.x))
    rotation_only = least_squares(
        lambda v: residual(np.r_[v, [0.0, 0.0, 0.0]], ~held),
        [-10.0, 0.0, 0.0],
        loss="soft_l1",
        f_scale=4,
        bounds=([-30, -15, -15], [30, 15, 15]),
    )
    rotation_prediction = predict(np.r_[rotation_only.x, [0.0, 0.0, 0.0]])
    report = dict(
        method=(
            "Robust fixed camera/neck alignment; fingertip-pair image midpoints; "
            "measured joints; fixed supplied intrinsics"
        ),
        stereo_constraints=bool(stereo),
        rotation_model="pitch_only_yaw_and_roll_retained" if pitch_only else "full_3_axis_rotation",
        translation_frame=translation_frame,
        rigid_tilt_assembly=bool(rigid_tilt),
        fixed_neck_parent_x_m=fixed_neck_x_m,
        stereo_valid_pair_observations=int(np.isfinite(right_observed).all(axis=1).sum()),
        parameter_units=(
            "Rotation vector degrees in original camera axes; translation cm in "
            + ("head_neck parent axes" if translation_frame == "neck_mount" else "head_tilt axes")
        ),
        delta_rotation_vector_deg=x[:3].tolist(),
        delta_translation_m=(x[3:] * 0.01).tolist(),
        fitted_camera_pos=(
            camera_positions(x)[0] - mount_rotations[0] @ (x[3:] * 0.01)
            if translation_frame == "neck_mount"
            else camera_positions(x)[0]
        ).tolist(),
        fitted_camera_rotation=(r0 @ Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix()).tolist(),
        baseline={s: stats(baseline, mask) for s, mask in [("train", ~held), ("validation", held)]},
        fitted={s: stats(fitted, mask) for s, mask in [("train", ~held), ("validation", held)]},
        jacobian_singular_values=singular.tolist(),
        annotation_noise_sensitivity=dict(
            method=(
                "100 refits with independent Gaussian 6px noise; sensitivity only, "
                "not a physical accuracy confidence interval"
            ),
            parameter_p05_p50_p95=np.percentile(noise_fits, [5, 50, 95], axis=0).tolist(),
        ),
        rotation_only=dict(
            delta_rotation_vector_deg=rotation_only.x.tolist(),
            train=stats(rotation_prediction, ~held),
            validation=stats(rotation_prediction, held),
        ),
        solutions=[dict(parameters=expand(r.x).tolist(), cost=float(r.cost)) for r in candidates],
        observations=[
            dict(
                **meta,
                split="validation" if h else "train",
                observed_px=o.tolist(),
                baseline_px=b.tolist(),
                fitted_px=f.tolist(),
            )
            for meta, h, o, b, f in zip(metadata, held, observed, baseline, fitted, strict=True)
        ],
        caveats=[
            "Manual fingertip annotations; distal-face centre can be partly occluded.",
            "Conditional on current arm/gripper kinematics and supplied camera intrinsics.",
            "Fixed optical transform cannot uniquely isolate neck zero versus mount alignment.",
            "Single episode; validation frames withheld from fitting, not an independent session.",
        ],
    )
    name = (
        ("rigid-" if rigid_tilt else "")
        + ("mount-" if translation_frame == "neck_mount" else "")
        + ("pitch-" if pitch_only else "")
        + ("stereo-fit-report.json" if stereo else "fit-report.json")
    )
    (OUT / (output_report or name)).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["delta_rotation_vector_deg", "delta_translation_m", "baseline", "fitted"]
            },
            indent=2,
        )
    )
    return report


def validate(report_name):
    report = json.loads((OUT / report_name).read_text())
    source = json.loads((OUT / "episode-1-kinematics.json").read_text())
    labels = json.loads((OUT / "episode-1-annotations.json").read_text())
    points, observed, _, metadata = observations(source, labels)
    results = {}
    for label, pos, rot in [
        ("baseline", source["camera_pos"], source["camera_rotation"]),
        ("fitted", report["fitted_camera_pos"], report["fitted_camera_rotation"]),
    ]:
        if label == "fitted" and report["translation_frame"] == "neck_mount":
            rowmap = {v["second"]: v for v in source["records"]}
            rs = np.array([rowmap[v["second"]]["mount_to_tilt_rotation"] for v in metadata])
            pos = np.repeat(np.array(pos) + rs @ report["delta_translation_m"], 2, axis=0)
        uv = project(points.reshape(-1, 3), pos, np.array(rot), source["intrinsics"])
        uv = uv.reshape(-1, 2, 2).mean(axis=1)
        errors = np.linalg.norm(uv - observed, axis=1)
        results[label] = dict(
            n=len(errors),
            rmse_px=float(np.sqrt(np.mean(errors**2))),
            median_px=float(np.median(errors)),
            max_px=float(errors.max()),
            observations=[
                dict(**meta, error_px=float(e)) for meta, e in zip(metadata, errors, strict=True)
            ],
        )
    report["independent_episode_validation"] = dict(
        dataset_episode=1,
        excluded_from_fitting=True,
        source_video_sha256=source["video_sha256"],
        results=results,
    )
    (OUT / report_name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def review(report_name, video=False):
    report = json.loads((OUT / report_name).read_text())
    sim = YORMujoco(
        REPLAY / "scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    m, d = sim.model, sim.data
    cid = m.camera("head_zed_left").id
    old_pos, old_quat = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
    neck, tilt = m.body("head_neck").id, m.body("head_tilt").id
    neck_pos, tilt_quat = m.body_pos[neck].copy(), m.body_quat[tilt].copy()
    r0 = np.empty(9)
    mujoco.mju_quat2Mat(r0, old_quat)
    rd = np.array(report["fitted_camera_rotation"]) @ r0.reshape(3, 3).T
    tilt_delta_quat = Rotation.from_matrix(rd).as_quat(scalar_first=True)
    fitted_tilt_quat = np.empty(4)
    mujoco.mju_mulQuat(fitted_tilt_quat, tilt_quat, tilt_delta_quat)
    new_quat = Rotation.from_matrix(report["fitted_camera_rotation"]).as_quat(scalar_first=True)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    panels = []
    process = writer(OUT / "episode-0-camera-fit-comparison.mp4", 1920, 420, 30) if video else None
    selections = (
        [(0, i) for i in range(42 * 30, 55 * 30)]
        if video
        else [(0, 5 * 30), (0, 48 * 30), (1, 2 * 30), (1, 48 * 30), (1, 56 * 30)]
    )
    current_episode = None
    cap = None
    try:
        with mujoco.Renderer(m, height=360, width=640) as renderer:
            for episode, frame in selections:
                if episode != current_episode:
                    if cap is not None:
                        cap.release()
                    a = dict(np.load(NATIVE.with_name(f"episode_{episode:06d}.npz")))
                    for side in ("left", "right"):
                        a[side + "_gripper"] = a[side + "_gripper"].reshape(-1)
                    alignment = (
                        REPLAY / "zed-frame-alignment.json"
                        if episode == 0
                        else OUT / f"episode-{episode}-alignment.json"
                    )
                    rows = json.loads(alignment.read_text())
                    path = (
                        REPLAY / "real-zed-left-source.mp4"
                        if episode == 0
                        else OUT / f"real-zed-episode-{episode}.mp4"
                    )
                    cap = cv2.VideoCapture(str(path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    current_episode = episode
                if not video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, real = cap.read()
                assert ok
                capture = rows[frame]["source.camera_capture_ns.zed_left"]
                assert capture and rows[frame]["annotation.camera_valid.zed_left"]
                seconds = capture / 1e9 - a["robot_timestamp"][0]
                i = int(
                    np.clip(
                        np.searchsorted(a["time_s"], seconds, side="right") - 1,
                        0,
                        len(a["time_s"]) - 1,
                    )
                )
                measured_pose(sim, d, a, i)
                views = [cv2.cvtColor(cv2.resize(real, (640, 360)), cv2.COLOR_BGR2RGB)]
                for fitted in (False, True):
                    m.body_pos[neck], m.body_quat[tilt] = neck_pos, tilt_quat
                    m.cam_pos[cid], m.cam_quat[cid] = old_pos, old_quat
                    if fitted:
                        if report["translation_frame"] == "neck_mount":
                            m.body_pos[neck] = neck_pos + report["delta_translation_m"]
                        if report["rigid_tilt_assembly"]:
                            m.body_quat[tilt] = fitted_tilt_quat
                        else:
                            m.cam_pos[cid], m.cam_quat[cid] = report["fitted_camera_pos"], new_quat
                    mujoco.mj_forward(m, d)
                    renderer.update_scene(d, camera="head_zed_left", scene_option=option)
                    views.append(renderer.render().copy())
                canvas = np.full((420, 1920, 3), 22, np.uint8)
                for j, (im, label) in enumerate(
                    zip(
                        views,
                        ["REAL ZED", "SIM - original camera", "SIM - dataset-fitted camera"],
                        strict=True,
                    )
                ):
                    canvas[30:390, j * 640 : (j + 1) * 640] = im
                    cv2.putText(canvas, label, (j * 640 + 12, 22), 0, 0.6, (240, 240, 240), 1)
                cv2.putText(
                    canvas,
                    f"Episode {episode} | {frame / 30:.2f}s | measured arm/neck poses "
                    "| scene not reconstructed",
                    (12, 411),
                    0,
                    0.56,
                    (240, 240, 240),
                    1,
                )
                if video:
                    process.stdin.write(canvas.tobytes())
                    if frame % 150 == 0:
                        print(f"Camera review video: {frame / 30:.1f}s", flush=True)
                else:
                    cv2.imwrite(
                        str(OUT / f"comparison-episode-{episode}-{frame // 30:02d}s.jpg"),
                        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                    )
                    panels.append(canvas)
        if video:
            process.stdin.close()
            assert process.wait() == 0
        else:
            cv2.imwrite(
                str(OUT / "camera-fit-before-after.jpg"),
                cv2.cvtColor(np.vstack(panels), cv2.COLOR_RGB2BGR),
            )
    finally:
        if cap is not None:
            cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def candidate_model(output, report_name=None):
    """Write an explicitly opt-in candidate without replacing production assets."""
    desc = ROOT / "robot/yor-description"
    sys.path.insert(0, str(desc))
    from sophia_sources import apply_dataset_head_alignment

    config = json.loads((desc / "sophia.json").read_text())
    if config["head_alignment"]["enabled"]:
        raise ValueError("Generate the baseline model with head_alignment disabled first")
    config["head_alignment"]["enabled"] = True
    model = ET.parse(desc / "sophia.mjcf")
    root = model.getroot()
    if report_name is None:
        apply_dataset_head_alignment(root, config)
    else:
        report = json.loads((OUT / report_name).read_text())
        if config["measured"].get("neck_mount_coaxial_with_pillar", False) and not np.allclose(
            np.array(report["delta_translation_m"])[:2], 0, atol=1e-9
        ):
            raise ValueError("Head fit violates user-confirmed concentric neck/pillar mounting")
        if (
            report["translation_frame"] != "neck_mount"
            or not report["rigid_tilt_assembly"]
            or report["rotation_model"] != "pitch_only_yaw_and_roll_retained"
        ):
            raise ValueError("Candidate assembly requires rigid mount translation and fixed tilt")
        baseline = mujoco.MjModel.from_xml_path(str(desc / "scene-sophia.mjcf"))
        r0 = np.empty(9)
        mujoco.mju_quat2Mat(r0, baseline.cam_quat[baseline.camera("head_zed_left").id])
        rd = np.array(report["fitted_camera_rotation"]) @ r0.reshape(3, 3).T
        neck = root.find(".//body[@name='head_neck']")
        position = np.fromstring(neck.get("pos"), sep=" ") + report["delta_translation_m"]
        neck.set("pos", " ".join(f"{v:.12g}" for v in position))
        root.find(".//body[@name='head_tilt']").set(
            "quat",
            " ".join(f"{v:.12g}" for v in Rotation.from_matrix(rd).as_quat(scalar_first=True)),
        )
        ET.SubElement(
            root.find("custom"),
            "text",
            name="dataset_head_fit_report_sha256",
            data=hashlib.sha256((OUT / report_name).read_bytes()).hexdigest(),
        )
    root.find("compiler").set("meshdir", str(desc / "assets_nero"))
    root.find("custom/text[@name='ik_model']").set("data", str(desc / "sophia-ik.mjcf"))
    output.mkdir(parents=True, exist_ok=True)
    ET.indent(model, space="  ")
    model.write(output / "robot-dataset-fit.mjcf", encoding="unicode")
    scene = ET.parse(desc / "scene-sophia.mjcf")
    scene.getroot().find("include").set("file", "robot-dataset-fit.mjcf")
    scene.write(output / "scene-dataset-fit.mjcf", encoding="unicode")
    return output / "scene-dataset-fit.mjcf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["prepare", "fit", "stereo-matches", "validate", "review", "video", "candidate"],
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--times", type=int, nargs="+")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--pitch-only", action="store_true")
    parser.add_argument(
        "--translation-frame", choices=["head_tilt", "neck_mount"], default="head_tilt"
    )
    parser.add_argument("--rigid-tilt", action="store_true")
    parser.add_argument("--fixed-neck-x-m", type=float)
    parser.add_argument("--fit-output-report")
    parser.add_argument("--candidate-report")
    parser.add_argument("--candidate-output", type=Path, default=OUT)
    parser.add_argument("--report", default="rigid-mount-pitch-stereo-fit-report.json")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.episode, args.times)
    elif args.action == "stereo-matches":
        stereo_matches()
    elif args.action == "validate":
        validate(args.report)
    elif args.action in ("review", "video"):
        review(args.report, args.action == "video")
    elif args.action == "candidate":
        print(candidate_model(args.candidate_output, args.candidate_report))
    else:
        fit(
            args.stereo,
            args.pitch_only,
            args.translation_frame,
            args.rigid_tilt,
            args.fixed_neck_x_m,
            args.fit_output_report,
        )
