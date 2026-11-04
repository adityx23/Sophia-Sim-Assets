"""Validate a head candidate and render measured states; never controls hardware.

Writes explicit pass/fail gates. Sampled collision sweeps are not continuous proofs.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_collision import SelfCollisionGuard  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import measured_pose, writer  # noqa: E402

OUT = ROOT / "output/asset-review/head-calibration-pipeline"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--fit-report", default="fit-report.json")
    args = parser.parse_args()
    OUT = args.output.resolve()
    fit = json.loads((OUT / args.fit_report).read_text())
    candidate_path = Path(fit["candidate_scene"]).with_name("robot.mjcf")
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != fit["candidate_robot_sha256"]:
        raise ValueError("Candidate does not match its fit report; rerun calibration")
    selection = fit["fits"][fit["selected"]]
    x = selection["parameters"]
    delta = np.deg2rad([x["joint_pan_delta_deg"], x["joint_tilt_delta_deg"]])
    baseline = mujoco.MjModel.from_xml_path(str(OUT / "baseline/scene.mjcf"))
    candidate = mujoco.MjModel.from_xml_path(fit["candidate_scene"])
    data = [mujoco.MjData(model) for model in (baseline, candidate)]
    models = [baseline, candidate]
    neck = [baseline.joint(f"head_{name}_joint").qposadr[0] for name in ("pan", "tilt")]
    home = np.array(fit["hardware_home_deg"])
    zeros = [np.array(model.numeric("neck_zero_deg").data) for model in models]
    invariants = []
    for model, d, zero in zip(models, data, zeros, strict=True):
        mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
        feedback = zero + np.rad2deg(d.qpos[neck]) * [-1, 1]
        invariants.append(bool(np.allclose(feedback, home, atol=1e-6)))
        d.qpos[neck] = 0
        mujoco.mj_forward(model, d)
        torso = model.body("Lift_Top").id
        ids = [model.camera("head_zed_" + eye).id for eye in ("left", "right")]
        local = (d.cam_xpos[ids].mean(0) - d.xpos[torso]) @ d.xmat[torso].reshape(3, 3)
        invariants.append(bool(np.allclose(local[:2], [-0.0508, 0], atol=1e-9)))
        invariants.append(
            abs(np.linalg.norm(d.cam_xpos[ids[1]] - d.cam_xpos[ids[0]]) - 0.1199918518066) < 1e-9
        )
    for name in ("head_neck", "head_pan", "head_tilt"):
        invariants.append(
            bool(
                np.allclose(
                    baseline.body_pos[baseline.body(name).id],
                    candidate.body_pos[candidate.body(name).id],
                )
            )
        )
    guards = [SelfCollisionGuard(model, 0.00001) for model in models]
    collision_cases, failures, preexisting = 0, [], 0
    baseline_examples = []

    def contacts(qbase, context, calibrated_pose=None):
        nonlocal collision_cases, preexisting
        qcandidate = qbase.copy()
        qcandidate[neck] += delta
        if calibrated_pose is not None:
            qcandidate = calibrated_pose
        hits = [guard._contacts(q) for guard, q in zip(guards, [qbase, qcandidate], strict=True)]
        collision_cases += 1
        preexisting += any(value < -0.001 for value in hits[0].values())
        if len(baseline_examples) < 20:
            for pair, depth in hits[0].items():
                if depth < -0.001:
                    baseline_examples.append(
                        dict(
                            context=context,
                            geoms=[baseline.geom(g).name for g in pair],
                            distance_m=depth,
                        )
                    )
        for pair, depth in hits[1].items():
            if depth < -0.001 and depth < hits[0].get(pair, 0) - 0.001:
                failures.append(
                    dict(
                        context=context,
                        geoms=[candidate.geom(g).name for g in pair],
                        candidate_distance_m=depth,
                        baseline_distance_m=hits[0].get(pair),
                    )
                )

    calib = json.loads((ROOT / "robot/neck/neck_calibration.json").read_text())
    limits = []
    for name in ("pan", "tilt"):
        motor = calib["motors"][name]
        half = (motor["range_max"] - motor["range_min"]) * 180 / 4095
        limits.append([-half, half])
    for pose in ("home", "reference", "lowered", "raised"):
        for height in (0.0, 0.5588):
            for pan in np.linspace(*limits[0], 9):
                for tilt in np.linspace(*limits[1], 13):
                    q = baseline.key(pose).qpos.copy()
                    q[neck] = np.deg2rad(np.array([pan, tilt]) - zeros[0]) * [-1, 1]
                    q[baseline.joint("lift_joint").qposadr[0]] = height
                    q[baseline.joint("sophia_middle_slide").qposadr[0]] = height / 2
                    qc = candidate.key(pose).qpos.copy()
                    qc[neck] = np.deg2rad(np.array([pan, tilt]) - zeros[1]) * [-1, 1]
                    qc[candidate.joint("lift_joint").qposadr[0]] = height
                    qc[candidate.joint("sophia_middle_slide").qposadr[0]] = height / 2
                    contacts(
                        q,
                        dict(
                            pose=pose,
                            lift_m=height,
                            pan_deg=pan,
                            tilt_deg=tilt,
                            calibrated_named_pose=True,
                        ),
                        qc,
                    )
    sim = YORMujoco(
        OUT / "baseline/scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    trajectory_count = 0
    try:
        for ep in (0, 1, 3, 8, 17):
            a = dict(
                np.load(
                    ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
                )
            )
            for side in ("left", "right"):
                a[side + "_gripper"] = a[side + "_gripper"].reshape(-1)
            for i in range(0, len(a["time_s"]), 3):
                measured_pose(sim, sim.data, a, i)
                if np.isfinite(sim.data.qpos).all():
                    contacts(sim.data.qpos.copy(), dict(episode=ep, row=i))
                    trajectory_count += 1
    finally:
        sim.stop()
    blocked = []
    for sample in fit["samples"]:
        d = data[1]
        d.qpos[:] = sample["qpos"]
        d.qpos[neck] += delta
        mujoco.mj_forward(candidate, d)
        for eye in ("left", "right"):
            origin = d.cam_xpos[candidate.camera("head_zed_" + eye).id]
            for letter in "ab":
                b = candidate.body(f"{sample['side']}_gripper_finger_{letter}").id
                point = d.xpos[b] + d.xmat[b].reshape(3, 3) @ [0, 0, 0.0035]
                direction = point - origin
                distance = np.linalg.norm(direction)
                direction /= distance
                geom = np.array([-1], dtype=np.int32)
                ray = mujoco.mj_ray(
                    candidate,
                    d,
                    origin + 0.003 * direction,
                    direction,
                    np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8),
                    1,
                    -1,
                    geom,
                )
                if ray >= 0 and ray < distance - 0.015:
                    name = candidate.geom(geom[0]).name or "unnamed"
                    if any(
                        token in name
                        for token in ("neck", "tower", "plate", "head_", "zed_", "spinal_")
                    ):
                        blocked.append(
                            dict(
                                episode=sample["episode"],
                                second=sample["second"],
                                side=sample["side"],
                                eye=eye,
                                obstruction=name,
                            )
                        )
    gates = dict(
        enough_training_stereo_pairs=fit["stereo_point_scores"]["train"]["n"] >= 6,
        home_and_mount_constraints=bool(all(invariants)),
        analytical_vs_compiled_fk=fit["compiled_fk_max_error_px"] < 1e-5,
        left_validation_rmse_under_25px=all(
            v["left"]["rmse_px"] is not None and v["left"]["rmse_px"] <= 25
            for k, v in selection["scores"].items()
            if k != "train"
        ),
        stereo_validation_rmse_under_35px=all(
            v["right"]["n"] >= 2 and v["right"]["rmse_px"] <= 35
            for k, v in selection["scores"].items()
            if k != "train"
        ),
        stereo_point_validation_under_30mm=all(
            v["n"] >= 2 and v["rmse_m"] <= 0.03
            for k, v in fit["stereo_point_scores"].items()
            if k != "train"
        ),
        improvement_at_least_50_percent_each_episode=all(
            v["left"]["rmse_px"] <= 0.5 * fit["fits"]["baseline"]["scores"][k]["left"]["rmse_px"]
            for k, v in selection["scores"].items()
            if k != "train"
        ),
        no_new_or_deeper_penetration_over_1mm=not failures,
        reviewed_visible_fingers_unobstructed_by_head_mount=not blocked,
        fit_not_at_parameter_bounds=not selection["at_bound"],
    )
    # Segmentation supplements the line-of-sight checks: quantify visible mount
    # area without assuming the real plate is never visible at a downward tilt.
    visibility = []
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    for model, d, label in zip(models, data, ("baseline", "candidate"), strict=True):
        mount_ids = [
            g
            for g in range(model.ngeom)
            if any(
                token in (model.geom(g).name or "")
                for token in (
                    "spinal_",
                    "tower_",
                    "neck_mount",
                    "neck_fixed",
                    "head_top",
                    "shoulder_bridge",
                    "pan_bracket",
                )
            )
        ]
        with mujoco.Renderer(model, height=180, width=320) as renderer:
            renderer.enable_segmentation_rendering()
            seen = set()
            for sample in fit["samples"]:
                key = (sample["episode"], sample["second"])
                if key in seen:
                    continue
                seen.add(key)
                d.qpos[:] = sample["qpos"]
                if label == "candidate":
                    d.qpos[neck] += delta
                mujoco.mj_forward(model, d)
                for eye in ("left", "right"):
                    renderer.update_scene(d, camera="head_zed_" + eye, scene_option=option)
                    segmentation = renderer.render()
                    mask = np.isin(segmentation[:, :, 0], mount_ids)
                    mask &= segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                    visibility.append(
                        dict(
                            model=label,
                            episode=key[0],
                            second=key[1],
                            eye=eye,
                            mount_fraction=float(mask.mean()),
                            central_fraction=float(mask[:162].mean()),
                        )
                    )
    neck_angles = np.array([s["neck_deg"] for s in fit["samples"]])
    report = dict(
        gates=gates,
        accepted=bool(all(gates.values())),
        promoted=False,
        named_reference_recomputed_with_ik=bool(fit.get("right_reference_q")),
        reference_ee_error_m=fit.get("reference_ee_error_m"),
        independent_wrist_features=fit.get("independent_wrist_features"),
        thresholds_set_before_fitting=dict(
            left_rmse_px=25, right_rmse_px=35, stereo_point_rmse_m=0.03, new_penetration_m=0.001
        ),
        collision_poses=collision_cases,
        trajectory_poses=trajectory_count,
        baseline_poses_with_penetrations=preexisting,
        baseline_penetration_examples=baseline_examples,
        new_or_deeper_collision_count=len(failures),
        collision_examples=failures[:40],
        blocked_mount_rays=blocked,
        mount_segmentation_visibility=visibility,
        visual_neck_angle_range_deg=[neck_angles.min(0).tolist(), neck_angles.max(0).tolist()],
        coverage=(
            "Both visible arms; five episodes; full hardware pan/tilt grid at both lift "
            "endpoints and four arm poses; native trajectories sampled at about 10Hz."
        ),
        limitations=[
            "Sparse landmarks and previously inspected development episodes "
            "are not a final untouched test set.",
            "Sampled collision checks cannot prove clearance for all intermediate robot poses.",
            "A camera mount change retains approximate tilt-body inertials; "
            "dynamic identification remains separate.",
        ],
    )
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_video:
        fit["validation_accepted"] = report["accepted"]
        render_video(fit, candidate, data[1], neck, delta)
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "gates": gates,
                "collision_poses": collision_cases,
                "collision_failures": len(failures),
            },
            indent=2,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


def render_video(fit, model, d, neck, delta):
    a = dict(np.load(ROOT / "output/asset-review/hf-sanity-check/native-replay/episode_000000.npz"))
    alignment = json.loads((OUT / "data/episode-0-alignment.json").read_text())
    sim = YORMujoco(
        OUT / "baseline/scene.mjcf", ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True
    )
    for side in ("left", "right"):
        a[side + "_gripper"] = a[side + "_gripper"].reshape(-1)
    path = OUT / "episode-0-real-vs-sim-zed-calibration-candidate.mp4"
    encoder = writer(path, 1920, 640, 30)
    cap = cv2.VideoCapture(str(OUT / "data/episode-0-zed-left.mp4"))
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    try:
        with mujoco.Renderer(model, height=540, width=960) as renderer:
            for i, row in enumerate(alignment):
                ok, real = cap.read()
                if not ok:
                    raise ValueError(f"Real video ended at {i}")
                measured_pose(sim, sim.data, a, row["index"])
                d.qpos[:] = sim.data.qpos
                d.qpos[neck] += delta
                mujoco.mj_forward(model, d)
                renderer.update_scene(d, camera="head_zed_left", scene_option=option)
                image = np.full((640, 1920, 3), 22, np.uint8)
                image[60:600, :960] = cv2.cvtColor(cv2.resize(real, (960, 540)), cv2.COLOR_BGR2RGB)
                image[60:600, 960:] = renderer.render()
                for text, position in [
                    ("REAL - recorded ZED", (16, 25)),
                    ("SIM - constrained stereo candidate; measured joints", (976, 25)),
                    (
                        f"Episode 0 | {i / 30:.2f}s | "
                        + (
                            "Validated calibration"
                            if fit.get("validation_accepted")
                            else "Unaccepted candidate"
                        ),
                        (16, 52),
                    ),
                ]:
                    cv2.putText(image, text, position, 0, 0.65, (245, 245, 245), 1)
                if not row["valid"]:
                    cv2.putText(
                        image,
                        "Invalid or stale synchronized source sample",
                        (16, 625),
                        0,
                        0.6,
                        (255, 180, 0),
                        1,
                    )
                encoder.stdin.write(image.tobytes())
                if i in (150, 960, 1440):
                    cv2.imwrite(
                        str(OUT / f"candidate-preview-{i}.jpg"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    )
    finally:
        cap.release()
        encoder.stdin.close()
        status = encoder.wait()
        sim.stop()
    if status:
        raise RuntimeError(f"ffmpeg exited {status}")
    (OUT / "video.json").write_text(
        json.dumps(
            dict(
                path=str(path),
                frames=len(alignment),
                fps=30,
                measured_state_replay=True,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
