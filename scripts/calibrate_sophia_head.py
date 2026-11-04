"""Constrained stereo calibration and acceptance report; never controls hardware.

Run prepare_sophia_head_calibration.py with pyarrow first. Reviewed image labels
are separate inputs. Only episode-0 training labels enter parameter estimation.
All other frames/episodes remain validation; no runtime promotion is automatic.
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
from scripts.render_continuous_teleop import measured_pose, scene  # noqa: E402

OUT = ROOT / "output/asset-review/head-calibration-pipeline"
CACHE = ROOT / "output/asset-review/head-camera-fit"
PARAMETERS = [
    "joint_tilt_delta_deg",
    "joint_pan_delta_deg",
    "camera_pitch_down_deg",
    "camera_yaw_deg",
    "camera_height_delta_m",
]
LOWER = np.array([-20.0, -5.0, -20.0, -5.0, -0.03])
UPPER = -LOWER


def stereo_match(left, right, pixels):
    """Measured-image matching only; independent of simulated camera parameters."""
    lg, rg = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (left, right)]
    results = []
    for uv in pixels:
        x, y = np.rint(uv).astype(int)
        radius = 10
        if min(x, y) < radius + 4 or y >= lg.shape[0] - radius - 4:
            return None
        template = lg[y - radius : y + radius + 1, x - radius : x + radius + 1]
        low, high = max(radius, x - 350), x - 15
        if high <= low:
            return None
        scores = cv2.matchTemplate(
            rg[y - radius - 3 : y + radius + 4, low - radius : high + radius + 1],
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, where = cv2.minMaxLoc(scores)
        xr, yr = low + where[0], y - 3 + where[1]
        other = scores.copy()
        other[:, max(0, where[0] - 8) : where[0] + 9] = -1
        uniqueness = score - float(other.max())
        patch = rg[yr - radius : yr + radius + 1, xr - radius : xr + radius + 1]
        reverse_low, reverse_high = xr + 15, min(lg.shape[1] - radius - 1, xr + 350)
        reverse = cv2.matchTemplate(
            lg[yr - radius - 3 : yr + radius + 4, reverse_low - radius : reverse_high + radius + 1],
            patch,
            cv2.TM_CCOEFF_NORMED,
        )
        _, reverse_score, _, loc = cv2.minMaxLoc(reverse)
        reverse_error = float(np.hypot(reverse_low + loc[0] - x, loc[1] + yr - 3 - y))
        results.append(
            dict(
                left_px=list(map(float, uv)),
                right_px=[int(xr), int(yr)],
                ncc=score,
                uniqueness=uniqueness,
                reverse_error_px=reverse_error,
                valid=score >= 0.8
                and uniqueness >= 0.02
                and reverse_score >= 0.8
                and reverse_error <= 3,
            )
        )
    return results


def rotation(x):
    return Rotation.from_rotvec(np.deg2rad([-x[2], x[3], 0])).as_matrix()


def predicted(samples, x, local_positions, local_rotation, midpoint, tilt_axis=None):
    """Exact articulated camera FK; rotates/translates the complete stereo rig."""
    mount = local_rotation @ rotation(x) @ local_rotation.T
    camera_local = midpoint + (local_positions - midpoint) @ mount.T
    camera_local[:, 2] += x[4]
    local_r = mount @ local_rotation
    result, depths, camera_points = [], [], []
    for s in samples:
        # The delta pan rotates about the actual pan pivot, not the neck root.
        pan_delta = s["pan_r"] @ Rotation.from_rotvec([0, 0, np.deg2rad(x[1])]).as_matrix()
        pan_delta = pan_delta @ s["pan_r"].T
        pivot = s["pan_t"] + pan_delta @ (s["tilt_t"] - s["pan_t"])
        axis = np.array([0, -1, 0] if tilt_axis is None else tilt_axis)
        r = pan_delta @ s["tilt_r"] @ Rotation.from_rotvec(axis * np.deg2rad(x[0])).as_matrix()
        cam_r = r @ local_r
        cameras = pivot + camera_local @ r.T
        eyes, zs = [], []
        for eye, camera in enumerate(cameras):
            v = (s["points"] - camera) @ cam_r
            k = s["intrinsics"][eye]
            eyes.append(
                np.stack(
                    [
                        k["cx"] + k["fx"] * v[:, 0] / -v[:, 2],
                        k["cy"] - k["fy"] * v[:, 1] / -v[:, 2],
                    ],
                    axis=-1,
                ).mean(0)
            )
            zs.append(float(-v[:, 2].mean()))
        result.append(eyes)
        depths.append(zs)
        camera_points.append((s["points"].mean(0) - cameras[0]) @ cam_r)
    return np.array(result), np.array(depths), np.array(camera_points)


def candidate(base_dir, x, home, zero, local_rotation, midpoint):
    """Create a reviewable model; saved Home/ranges retain hardware meaning."""
    dest = OUT / "candidate"
    dest.mkdir(exist_ok=True)
    tree = ET.parse(base_dir / "robot.mjcf")
    root = tree.getroot()
    newzero = zero - np.array([-x[1], x[0]])
    root.find("custom/numeric[@name='neck_zero_deg']").set("data", " ".join(map(str, newzero)))
    tilt = root.find(".//body[@name='head_tilt']")
    mount = local_rotation @ rotation(x) @ local_rotation.T
    qmount = Rotation.from_matrix(mount).as_quat(scalar_first=True)
    for node in tilt:
        if node.get("name", "").startswith(("head_zed", "zed_")):
            pos = np.fromstring(node.get("pos", "0 0 0"), sep=" ")
            pos = midpoint + mount @ (pos - midpoint) + [0, 0, x[4]]
            node.set("pos", " ".join(map(str, pos)))
            quat = np.fromstring(node.get("quat", "1 0 0 0"), sep=" ")
            if node.tag == "camera":
                quat = Rotation.from_matrix(local_rotation).as_quat(scalar_first=True)
            result = np.empty(4)
            mujoco.mju_mulQuat(result, qmount, quat)
            node.set("quat", " ".join(map(str, result)))
            node.attrib.pop("xyaxes", None)
            node.attrib.pop("euler", None)
    # Mapping changes apply to joint limits and every saved pose/controller goal.
    model = mujoco.MjModel.from_xml_path(str(base_dir / "scene.mjcf"))
    for name, value in (("pan", x[1]), ("tilt", x[0])):
        delta = np.deg2rad(value)
        joint = root.find(f".//joint[@name='head_{name}_joint']")
        joint.set("range", " ".join(map(str, np.fromstring(joint.get("range"), sep=" ") + delta)))
        for key in root.findall("keyframe/key"):
            q, ctrl = [np.fromstring(key.get(field), sep=" ") for field in ("qpos", "ctrl")]
            q[model.joint(f"head_{name}_joint").qposadr[0]] += delta
            ctrl[model.actuator(f"head_{name}_pos").id] += delta
            key.set("qpos", " ".join(map(str, q)))
            key.set("ctrl", " ".join(map(str, ctrl)))
    tree.write(dest / "robot.mjcf")
    scene_tree = ET.parse(base_dir / "scene.mjcf")
    scene_tree.write(dest / "scene.mjcf")
    return dest / "scene.mjcf", newzero


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--scene", type=Path, default=ROOT / "robot/yor-description/scene-sophia.mjcf"
    )
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(exist_ok=True)
    base = OUT / "baseline"
    base.mkdir(exist_ok=True)
    scene(base)
    if args.scene.resolve() != (ROOT / "robot/yor-description/scene-sophia.mjcf").resolve():
        include = ET.parse(args.scene).getroot().find("include").get("file")
        source = ET.parse(args.scene.resolve().parent / include)
        source.getroot().find("compiler").set(
            "meshdir", str(ROOT / "robot/yor-description/assets_nero")
        )
        source.write(base / "robot.mjcf")
    sim = YORMujoco(
        base / "scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    m, d = sim.model, sim.data
    cids = [m.camera("head_zed_" + eye).id for eye in ("left", "right")]
    local_positions = m.cam_pos[cids].copy()
    local_rotation = Rotation.from_quat(m.cam_quat[cids[0]], scalar_first=True).as_matrix()
    midpoint = local_positions.mean(0)
    tilt_axis = m.jnt_axis[m.joint("head_tilt_joint").id].copy()
    intrinsics = [camera_intrinsics(m, "head_zed_" + eye, 1920, 1080) for eye in ("left", "right")]
    samples, matching = [], []
    try:
        label_source = ROOT / "robot/yor-description/calibration/head_camera_labels_2026-11-04.json"
        all_labels = json.loads(label_source.read_text())["episodes"]
        for ep in (0, 1, 3, 8, 17):
            labels = all_labels[str(ep)]
            native = dict(
                np.load(
                    ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
                )
            )
            for side in ("left", "right"):
                native[side + "_gripper"] = native[side + "_gripper"].reshape(-1)
            alignment = json.loads((OUT / f"data/episode-{ep}-alignment.json").read_text())
            videos = [
                cv2.VideoCapture(str(OUT / f"data/episode-{ep}-zed-{eye}.mp4"))
                for eye in ("left", "right")
            ]
            for second, annotation in labels.items():
                frame = round(float(second) * 30)
                row = alignment[frame]
                if not row["valid"]:
                    continue
                images = []
                for cap in videos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, im = cap.read()
                    if not ok:
                        raise ValueError(f"Missing source frame {ep}:{frame}")
                    images.append(im)
                measured_pose(sim, d, native, row["index"])
                for side in ("left", "right"):
                    names = [side + "_" + letter for letter in "ab"]
                    if not all(name in annotation["pixels"] for name in names):
                        continue
                    pixels = [annotation["pixels"][name] for name in names]
                    matches = stereo_match(*images, pixels)
                    stereo_valid = matches is not None and all(v["valid"] for v in matches)
                    if stereo_valid:
                        disparities = [v["left_px"][0] - v["right_px"][0] for v in matches]
                        separation = np.linalg.norm(
                            np.subtract(matches[0]["right_px"], matches[1]["right_px"])
                        )
                        stereo_valid = bool(
                            abs(disparities[0] - disparities[1]) <= (8 if side == "left" else 35)
                            and separation > 6
                        )
                    reviewed = annotation.get("reviewed_right_pixels", {})
                    if all(name in reviewed for name in names):
                        matches = [
                            dict(
                                left_px=pixel,
                                right_px=reviewed[name],
                                method="visually_reviewed_real_stereo_correspondence",
                                valid=True,
                            )
                            for name, pixel in zip(names, pixels, strict=True)
                        ]
                        stereo_valid = True
                    right_uv = (
                        np.mean([v["right_px"] for v in matches], 0)
                        if stereo_valid
                        else [np.nan] * 2
                    )
                    matching.append(
                        dict(
                            episode=ep,
                            second=second,
                            side=side,
                            matches=matches,
                            stereo_valid=stereo_valid,
                        )
                    )
                    points = []
                    for letter in "ab":
                        b = m.body(f"{side}_gripper_finger_{letter}").id
                        points.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ [0, 0, 0.0035])
                    pan, tilt = m.body("head_pan").id, m.body("head_tilt").id
                    split = (
                        "train" if ep == 0 and annotation["split"] == "train" else f"episode_{ep}"
                    )
                    samples.append(
                        dict(
                            episode=ep,
                            second=float(second),
                            side=side,
                            split=split,
                            qpos=d.qpos.copy(),
                            index=row["index"],
                            points=np.array(points),
                            observed=np.array([np.mean(pixels, 0), right_uv]),
                            stereo_valid=stereo_valid,
                            intrinsics=intrinsics,
                            pan_r=d.xmat[pan].reshape(3, 3).copy(),
                            pan_t=d.xpos[pan].copy(),
                            tilt_r=d.xmat[tilt].reshape(3, 3).copy(),
                            tilt_t=d.xpos[tilt].copy(),
                            neck_deg=native["neck_deg"][row["index"]].tolist(),
                        )
                    )
            for cap in videos:
                cap.release()
        observed = np.array([s["observed"] for s in samples])
        valid = np.isfinite(observed).all(2)
        training = np.array([s["split"] == "train" for s in samples])
        groups = {
            name: np.array([s["split"] == name for s in samples])
            for name in sorted({s["split"] for s in samples})
        }

        def predict(x):
            return predicted(samples, x, local_positions, local_rotation, midpoint, tilt_axis)[0]

        def residual(x, mask):
            return (predict(x) - observed)[valid & mask[:, None]].ravel()

        def scores(x):
            uv, depths, points = predicted(samples, x, local_positions, local_rotation, midpoint)
            results = {}
            for group, mask in groups.items():
                results[group] = {}
                for eye, name in enumerate(("left", "right")):
                    use = mask & valid[:, eye]
                    error = uv[use, eye] - observed[use, eye]
                    results[group][name] = dict(
                        n=int(use.sum()),
                        rmse_px=float(np.sqrt(np.mean(np.sum(error**2, 1)))) if use.any() else None,
                    )
            return results

        fits = {}
        for name, active in {
            "baseline": [],
            "joint_tilt": [0],
            "joint_pan_tilt": [0, 1],
            "optical_pitch": [2],
            "joint_and_optical_pitch": [0, 2],
            "joint_tilt_height": [0, 4],
            "optical_pitch_height": [2, 4],
            "joint_pitch_height": [0, 2, 4],
            "joint_pan_tilt_pitch_height": [0, 1, 2, 4],
        }.items():

            def expand(v, active=active):
                x = np.zeros(5)
                x[active] = v
                return x

            if active:
                solutions = []
                for start in (-8.0, 0.0, 8.0):
                    initial = np.zeros(len(active))
                    initial[0] = start
                    solutions.append(
                        least_squares(
                            lambda v, expand=expand: residual(expand(v), training),
                            initial,
                            bounds=(LOWER[active], UPPER[active]),
                            loss="soft_l1",
                            f_scale=6,
                            x_scale="jac",
                            max_nfev=500,
                        )
                    )
                solution = min(solutions, key=lambda s: s.cost)
                x = expand(solution.x)
                singular = np.linalg.svd(solution.jac, compute_uv=False).tolist()
                cost = float(solution.cost)
            else:
                x, singular = np.zeros(5), []
                cost = float(np.sum(residual(x, training) ** 2) / 2)
            fits[name] = dict(
                parameters=dict(zip(PARAMETERS, x.tolist(), strict=True)),
                scores=scores(x),
                train_cost=cost,
                singular_values=singular,
                at_bound=bool(
                    active and np.any(np.isclose(abs(x[active]), UPPER[active], atol=1e-4))
                ),
            )
        # Model choice uses training only, never the held-out episode scores.
        eligible = [
            (v["train_cost"] + 12 * sum(abs(np.array(list(v["parameters"].values()))) > 1e-8), n)
            for n, v in fits.items()
            if n != "baseline" and not v["at_bound"]
        ]
        selected = min(eligible)[1]
        x = np.array(list(fits[selected]["parameters"].values()))
        candidate_scene, newzero = candidate(
            base, x, sim._neck_home_deg, sim._neck_zero_deg, local_rotation, midpoint
        )
        uv, depths, camera_points = predicted(
            samples, x, local_positions, local_rotation, midpoint, tilt_axis
        )
        compiled = mujoco.MjModel.from_xml_path(str(candidate_scene))
        compiled_data = mujoco.MjData(compiled)
        fk_errors = []
        for i, sample in enumerate(samples):
            compiled_data.qpos[:] = sample["qpos"]
            compiled_data.qpos[sim._pan_adr] += np.deg2rad(x[1])
            compiled_data.qpos[sim._tilt_adr] += np.deg2rad(x[0])
            mujoco.mj_forward(compiled, compiled_data)
            for eye, cid in enumerate(cids):
                v = (sample["points"] - compiled_data.cam_xpos[cid]) @ compiled_data.cam_xmat[
                    cid
                ].reshape(3, 3)
                k = intrinsics[eye]
                exact = np.stack(
                    [
                        k["cx"] + k["fx"] * v[:, 0] / -v[:, 2],
                        k["cy"] - k["fy"] * v[:, 1] / -v[:, 2],
                    ],
                    axis=-1,
                ).mean(0)
                fk_errors.append(float(abs(exact - uv[i, eye]).max()))
            sample["predicted_pixels"] = uv[i].tolist()
            if sample["stereo_valid"]:
                left, right = sample["observed"]
                disparity = left[0] - right[0]
                z = (
                    intrinsics[0]["fx"]
                    * np.linalg.norm(local_positions[1] - local_positions[0])
                    / disparity
                )
                point = np.array(
                    [
                        (left[0] - intrinsics[0]["cx"]) * z / intrinsics[0]["fx"],
                        -(left[1] - intrinsics[0]["cy"]) * z / intrinsics[0]["fy"],
                        -z,
                    ]
                )
                sample["stereo_point_error_m"] = float(np.linalg.norm(point - camera_points[i]))
        if max(fk_errors) > 1e-5:
            raise AssertionError(f"Candidate XML and fitting FK disagree: {max(fk_errors)} px")
        stereo_scores = {}
        for name, mask in groups.items():
            errors = [
                s["stereo_point_error_m"]
                for s, use in zip(samples, mask, strict=True)
                if use and "stereo_point_error_m" in s
            ]
            stereo_scores[name] = dict(
                n=len(errors), rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None
            )
        serializable = [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in s.items()
                if k not in ("pan_r", "pan_t", "tilt_r", "tilt_t", "points")
            }
            for s in samples
        ]
        report = dict(
            training="episode 0 train labels only",
            selected=selected,
            fits=fits,
            geometric_zero_hardware_deg=newzero.tolist(),
            hardware_home_deg=sim._neck_home_deg.tolist(),
            samples=serializable,
            candidate_scene=str(candidate_scene),
            candidate_robot_sha256=hashlib.sha256(
                candidate_scene.with_name("robot.mjcf").read_bytes()
            ).hexdigest(),
            labels_sha256=hashlib.sha256(label_source.read_bytes()).hexdigest(),
            constraints=(
                "fixed neck/pillar XY, lattice dimensions, neutral lens XY, "
                "stereo baseline and intrinsics"
            ),
            changes_promoted=False,
            compiled_fk_max_error_px=max(fk_errors),
            stereo_point_scores=stereo_scores,
        )
        (OUT / "stereo-matches.json").write_text(json.dumps(matching, indent=2) + "\n")
        (OUT / "fit-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"selected": selected, "fit": fits[selected]}, indent=2))
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
