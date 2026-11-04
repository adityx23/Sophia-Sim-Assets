"""Separate camera-reference error from a fixed arm-mount reference error.

Fits training pixels only; keeps flange centres, link lengths, servo readings and
camera CAD geometry fixed. An effective mount rotation is not a physical survey.
"""

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_camera import camera_intrinsics  # noqa: E402
from scripts.calibrate_sophia_head import predicted  # noqa: E402

from scripts import calibrate_sophia_head as calibration  # noqa: E402

OUT = ROOT / "output/asset-review/full-neck-cad-calibration"


def main():
    source = json.loads((OUT / "fit-report.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(OUT / "baseline/scene.mjcf"))
    d = mujoco.MjData(m)
    cids = [m.camera("head_zed_" + eye).id for eye in ("left", "right")]
    local_positions = m.cam_pos[cids].copy()
    local_rotation = Rotation.from_quat(m.cam_quat[cids[0]], scalar_first=True).as_matrix()
    midpoint = local_positions.mean(0)
    axis = m.jnt_axis[m.joint("head_tilt_joint").id].copy()
    samples = []
    for original in source["samples"]:
        d.qpos[:] = original["qpos"]
        mujoco.mj_forward(m, d)
        sample = dict(original)
        points = []
        for letter in "ab":
            b = m.body(f"{sample['side']}_gripper_finger_{letter}").id
            points.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ [0, 0, 0.0035])
        sample["points"] = np.array(points)
        for name in ("pan", "tilt"):
            b = m.body("head_" + name).id
            sample[name + "_r"], sample[name + "_t"] = (
                d.xmat[b].reshape(3, 3).copy(),
                d.xpos[b].copy(),
            )
        parent = m.body("Lift_Top").id
        sample["mount_r"] = d.xmat[parent].reshape(3, 3).copy()
        sample["mount_t"] = d.xpos[m.body("right_arm_base_link").id].copy()
        sample["intrinsics"] = [
            camera_intrinsics(m, "head_zed_" + eye, 1920, 1080) for eye in ("left", "right")
        ]
        samples.append(sample)
    observed = np.array([s["observed"] for s in samples])
    valid = np.isfinite(observed).all(2)
    training = np.array([s["split"] == "train" for s in samples])

    def prediction(x):
        changed = []
        rotation = Rotation.from_rotvec(np.deg2rad(x[2:])).as_matrix()
        for sample in samples:
            s = dict(sample)
            if s["side"] == "right":
                r = s["mount_r"] @ rotation @ s["mount_r"].T
                s["points"] = (s["points"] - s["mount_t"]) @ r.T + s["mount_t"]
            changed.append(s)
        return predicted(
            changed, [x[0], x[1], 0, 0, 0], local_positions, local_rotation, midpoint, axis
        )[0]

    initial = np.array(
        [
            source["fits"]["joint_pan_tilt"]["parameters"]["joint_tilt_delta_deg"],
            source["fits"]["joint_pan_tilt"]["parameters"]["joint_pan_delta_deg"],
            0,
            0,
            0,
        ]
    )
    solutions = []
    for start in (initial, [13, 0, 0, 5, 0], [13, 0, 0, -5, 0]):
        result = least_squares(
            lambda x: (prediction(x) - observed)[valid & training[:, None]].ravel(),
            start,
            bounds=([-20, -5, -15, -15, -15], [20, 5, 15, 15, 15]),
            loss="soft_l1",
            f_scale=6,
            max_nfev=1000,
        )
        solutions.append(result)
    result = min(solutions, key=lambda r: r.cost)
    uv = prediction(result.x)
    scores = {}
    for ep in (0, 1, 3, 8, 17):
        for side in ("left", "right"):
            mask = np.array(
                [
                    s["episode"] == ep and s["side"] == side and s["split"] != "train"
                    for s in samples
                ]
            )
            if not mask.any():
                continue
            error = uv[mask, 0] - observed[mask, 0]
            scores[f"episode_{ep}_{side}"] = dict(
                n=int(mask.sum()),
                rmse_px=float(np.sqrt(np.mean(np.sum(error**2, 1)))),
                mean_error_px=error.mean(0).tolist(),
            )
    report = dict(
        parameters=result.x.tolist(),
        parameter_names=[
            "neck_tilt_delta_deg",
            "neck_pan_delta_deg",
            "right_mount_rx_deg",
            "right_mount_ry_deg",
            "right_mount_rz_deg",
        ],
        scores=scores,
        training_cost=result.cost,
        jacobian_singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist(),
        method=(
            "Training-only effective right-arm reference rotation about fixed flange centre; "
            "CAD camera rigid geometry fixed."
        ),
        notes=[
            "Mount-axis rotation and first-joint zero are partly indistinguishable.",
            "Do not interpret this image-derived reference as a measured mechanical mount angle.",
        ],
    )
    (OUT / "arm-frame-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    # Materialize the full rigid-reference correction, then score its actual FK.
    calibration.OUT = OUT
    correction = np.array([result.x[0], result.x[1], 0.0, 0.0, 0.0])
    path, zero = calibration.candidate(
        OUT / "baseline",
        correction,
        np.array(source["hardware_home_deg"]),
        np.array(source["current_zero"])
        if "current_zero" in source
        else np.array(m.numeric("neck_zero_deg").data),
        local_rotation,
        midpoint,
    )
    tree = ET.parse(path.with_name("robot.mjcf"))
    root = tree.getroot()
    body = root.find(".//body[@name='right_arm_base_link']")
    base_rotation = Rotation.from_quat(
        m.body_quat[m.body("right_arm_base_link").id], scalar_first=True
    ).as_matrix()
    new_rotation = Rotation.from_rotvec(np.deg2rad(result.x[2:])).as_matrix() @ base_rotation
    body.set(
        "quat", " ".join(map(str, Rotation.from_matrix(new_rotation).as_quat(scalar_first=True)))
    )
    body.attrib.pop("euler", None)
    tree.write(path.with_name("robot.mjcf"))
    final_model = mujoco.MjModel.from_xml_path(str(path))
    final_data = mujoco.MjData(final_model)
    # Recompute the sim-only hanging reference under the calibrated mount.
    # Keep its intended EE pose, rather than retaining now-unsafe old joint angles.
    mujoco.mj_resetDataKeyframe(m, d, m.key("reference").id)
    mujoco.mj_forward(m, d)
    site = m.site("right_arm_ee").id
    target_position = d.site_xpos[site].copy()
    target_rotation = d.site_xmat[site].reshape(3, 3).copy()
    final_data.qpos[:] = final_model.key("reference").qpos
    joint_ids = [final_model.joint(f"right_arm_joint{i}").id for i in range(1, 8)]
    addresses = final_model.jnt_qposadr[joint_ids]
    reference_initial = final_data.qpos[addresses].copy()

    def reference_error(q):
        final_data.qpos[addresses] = q
        mujoco.mj_forward(final_model, final_data)
        dp = final_data.site_xpos[site] - target_position
        dr = Rotation.from_matrix(
            target_rotation.T @ final_data.site_xmat[site].reshape(3, 3)
        ).as_rotvec()
        return np.r_[dp * 1000, dr * 100, (q - reference_initial) * 0.002]

    reference = least_squares(
        reference_error,
        reference_initial,
        bounds=(final_model.jnt_range[joint_ids, 0], final_model.jnt_range[joint_ids, 1]),
        max_nfev=3000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    reference_residual = reference_error(reference.x)
    if np.linalg.norm(reference_residual[:3]) > 0.01:
        raise ValueError("Calibrated reference pose IK failed")
    key = root.find("keyframe/key[@name='reference']")
    q, ctrl = [np.fromstring(key.get(k), sep=" ") for k in ("qpos", "ctrl")]
    q[addresses] = reference.x
    for i, angle in enumerate(reference.x, 1):
        ctrl[final_model.actuator(f"right_arm_joint{i}_pos").id] = angle
    key.set("qpos", " ".join(map(str, q)))
    key.set("ctrl", " ".join(map(str, ctrl)))
    tree.write(path.with_name("robot.mjcf"))
    final_model = mujoco.MjModel.from_xml_path(str(path))
    final_data = mujoco.MjData(final_model)
    stereo_scores, group_scores = {}, {}
    fk_error = 0.0
    output_samples = []
    xyz_errors = {}
    for i, sample in enumerate(samples):
        final_data.qpos[:] = sample["qpos"]
        final_data.qpos[m.joint("head_pan_joint").qposadr[0]] += np.deg2rad(result.x[1])
        final_data.qpos[m.joint("head_tilt_joint").qposadr[0]] += np.deg2rad(result.x[0])
        mujoco.mj_forward(final_model, final_data)
        p = []
        for letter in "ab":
            b = final_model.body(f"{sample['side']}_gripper_finger_{letter}").id
            p.append(final_data.xpos[b] + final_data.xmat[b].reshape(3, 3) @ [0, 0, 0.0035])
        p = np.array(p)
        exact = []
        for eye, cid in enumerate(cids):
            v = (p - final_data.cam_xpos[cid]) @ final_data.cam_xmat[cid].reshape(3, 3)
            k = sample["intrinsics"][eye]
            exact.append(
                np.stack(
                    [
                        k["cx"] + k["fx"] * v[:, 0] / -v[:, 2],
                        k["cy"] - k["fy"] * v[:, 1] / -v[:, 2],
                    ],
                    axis=-1,
                ).mean(0)
            )
        fk_error = max(fk_error, float(np.max(abs(np.array(exact) - uv[i]))))
        entry = dict(source["samples"][i])
        entry["predicted_pixels"] = np.array(exact).tolist()
        if sample["stereo_valid"]:
            left, right = observed[i]
            k = sample["intrinsics"][0]
            z = (
                k["fx"]
                * np.linalg.norm(local_positions[1] - local_positions[0])
                / (left[0] - right[0])
            )
            point = np.array(
                [(left[0] - k["cx"]) * z / k["fx"], -(left[1] - k["cy"]) * z / k["fy"], -z]
            )
            cid = cids[0]
            actual = (p.mean(0) - final_data.cam_xpos[cid]) @ final_data.cam_xmat[cid].reshape(3, 3)
            error = float(np.linalg.norm(actual - point))
            entry["stereo_point_error_m"] = error
            xyz_errors.setdefault(sample["split"], []).append(error)
        output_samples.append(entry)
    if fk_error > 1e-5:
        raise ValueError("Arm-frame fit and compiled FK disagree")
    for group in sorted({s["split"] for s in samples}):
        mask = np.array([s["split"] == group for s in samples])
        group_scores[group] = {}
        for eye, name in enumerate(["left", "right"]):
            use = mask & valid[:, eye]
            e = uv[use, eye] - observed[use, eye]
            group_scores[group][name] = dict(
                n=int(use.sum()),
                rmse_px=float(np.sqrt(np.mean(np.sum(e**2, 1)))) if use.any() else None,
            )
        errors = xyz_errors.get(group, [])
        stereo_scores[group] = dict(
            n=len(errors), rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        )
    source["fits"]["full_cad_and_arm_reference"] = dict(
        parameters=dict(zip(calibration.PARAMETERS, correction.tolist(), strict=True)),
        scores=group_scores,
        at_bound=False,
        train_cost=result.cost,
    )
    source.update(
        selected="full_cad_and_arm_reference",
        samples=output_samples,
        stereo_point_scores=stereo_scores,
        candidate_scene=str(path),
        candidate_robot_sha256=hashlib.sha256(
            path.with_name("robot.mjcf").read_bytes()
        ).hexdigest(),
        compiled_fk_max_error_px=fk_error,
        geometric_zero_hardware_deg=zero.tolist(),
        right_mount_rotvec_parent_deg=result.x[2:].tolist(),
        changes_promoted=False,
    )
    source["right_reference_q"] = reference.x.tolist()
    source["reference_ee_error_m"] = float(np.linalg.norm(reference_residual[:3]) / 1000)
    feature_source = json.loads(
        (ROOT / "robot/yor-description/calibration/head_camera_labels_2026-11-04.json").read_text()
    )["independent_wrist_features"]
    feature_rows = []
    for second, features in feature_source["frames"].items():
        sample = next(s for s in samples if s["episode"] == 0 and s["second"] == float(second))
        for model, data, label in [
            (m, d, "before_arm_reference"),
            (final_model, final_data, "after_arm_reference"),
        ]:
            data.qpos[:] = sample["qpos"]
            data.qpos[m.joint("head_pan_joint").qposadr[0]] += np.deg2rad(result.x[1])
            data.qpos[m.joint("head_tilt_joint").qposadr[0]] += np.deg2rad(result.x[0])
            mujoco.mj_forward(model, data)
            cid = cids[0]
            R = data.cam_xmat[cid].reshape(3, 3)
            k = sample["intrinsics"][0]
            for body, pixel in features.items():
                v = (data.xpos[model.body(body).id] - data.cam_xpos[cid]) @ R
                uv_feature = np.array(
                    [k["cx"] + k["fx"] * v[0] / -v[2], k["cy"] - k["fy"] * v[1] / -v[2]]
                )
                feature_rows.append(
                    dict(
                        second=second,
                        body=body,
                        model=label,
                        observed=pixel,
                        predicted=uv_feature.tolist(),
                        error_px=float(np.linalg.norm(uv_feature - pixel)),
                    )
                )
    source["independent_wrist_features"] = dict(
        method=feature_source["method"],
        uncertainty_px=20,
        observations=feature_rows,
        rmse_px={
            label: float(
                np.sqrt(np.mean([v["error_px"] ** 2 for v in feature_rows if v["model"] == label]))
            )
            for label in ["before_arm_reference", "after_arm_reference"]
        },
    )
    (OUT / "combined-fit-report.json").write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
