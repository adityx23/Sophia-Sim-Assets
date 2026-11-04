"""Validate, optionally promote, and render the offline Sophia base correction.

No network/hardware access. Replay is measured-pose visualization, not a dynamics
or sim-to-real transfer test. Use --promote --video after the stereo fit/build.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "robot/yor-description"))
from build_sophia import build  # noqa: E402
from robot.sim_camera import camera_intrinsics  # noqa: E402
from robot.sim_collision import SelfCollisionGuard  # noqa: E402
from robot.sophia_recording_alignment import recording_lift_offset  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import measured_pose, writer  # noqa: E402

OUT = ROOT / "output/asset-review/base-calibration"
DATA = ROOT / "output/asset-review/head-calibration-pipeline/data"
DESC = ROOT / "robot/yor-description"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(ep):
    path = ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
    a = dict(np.load(path))
    for side in ("left", "right"):
        a[side + "_gripper"] = a[side + "_gripper"].reshape(-1)
    return path, a


def contacts(guard, model, q):
    return {
        tuple(sorted(model.geom(g).name for g in pair)): value
        for pair, value in guard._contacts(q).items()
        if value < -0.001
    }


def render(sim, cfg, full=False):
    m, d = sim.model, sim.data
    path, a = native(0)
    offset = recording_lift_offset(path, 0, cfg)
    alignment = json.loads((DATA / "episode-0-alignment.json").read_text())
    video = OUT / "episode-0-real-vs-sim-zed-base-corrected.mp4"
    encoder = writer(video, 1920, 640, 30) if full else None
    cap = cv2.VideoCapture(str(DATA / "episode-0-zed-left.mp4"))
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    selected = [60, 90, 930, 960, 1425]
    try:
        with mujoco.Renderer(m, height=540, width=960) as renderer:
            for i, row in enumerate(alignment):
                if not full and i not in selected:
                    continue
                if not full:
                    cap.set(1, i)
                ok, real = cap.read()
                if not ok:
                    raise ValueError(f"Real video ended at {i}")
                measured_pose(sim, d, a, row["index"], lift_extension_offset_m=offset)
                renderer.update_scene(d, camera="head_zed_left", scene_option=option)
                image = np.full((640, 1920, 3), 22, np.uint8)
                image[60:600, :960] = cv2.cvtColor(cv2.resize(real, (960, 540)), cv2.COLOR_BGR2RGB)
                image[60:600, 960:] = renderer.render()
                for text, xy in [
                    ("REAL - recorded ZED", (16, 25)),
                    ("SIM - corrected base; measured joints", (976, 25)),
                    (f"Episode 0 | {i / 30:.2f}s | locked head / arm calibration", (16, 52)),
                    (
                        f"Historical recording height alignment: {offset * 1000:.1f} mm; "
                        "live lift map unchanged",
                        (16, 625),
                    ),
                ]:
                    cv2.putText(image, text, xy, 0, 0.62, (245, 245, 245), 1)
                if not row["valid"]:
                    cv2.putText(image, "STALE source pose", (1450, 52), 0, 0.65, (255, 170, 0), 1)
                if encoder:
                    encoder.stdin.write(image.tobytes())
                if i in selected:
                    cv2.imwrite(
                        str(OUT / f"preview-{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
    finally:
        cap.release()
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed")
    if full:
        (OUT / "video.json").write_text(
            json.dumps(
                dict(
                    path=str(video),
                    sha256=digest(video),
                    frames=len(alignment),
                    fps=30,
                    width=1920,
                    height=640,
                    duration_s=len(alignment) / 30,
                    measured_pose_replay=True,
                    historical_lift_offset_m=offset,
                    limitations=(
                        "No real environment reconstruction; "
                        "not a physical replay or dynamics validation."
                    ),
                ),
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((OUT / "candidate-config.json").read_text())
    profile = json.loads((DESC / cfg["base_alignment"]["source"]).read_text())
    if digest(OUT / "candidate-profile.json") != cfg["base_alignment"]["sha256"]:
        raise ValueError("Fit and staged configuration differ; rebuild the candidate")
    if digest(OUT / "stereo-base-points.json") != profile["evidence_sha256"]:
        raise ValueError("Stereo evidence changed; rerun the base fit")
    current = json.loads((DESC / "sophia.json").read_text())
    if {k: v for k, v in cfg.items() if k != "base_alignment"} != {
        k: v for k, v in current.items() if k != "base_alignment"
    }:
        raise ValueError("Other model settings changed; rebuild and review the base candidate")
    # Use the same review floor/lighting as the previous accepted video.
    review_scene = ET.parse(OUT / "baseline/scene.mjcf")
    review_scene.getroot().find("include").set("file", str(OUT / "candidate/sophia.mjcf"))
    review_scene.write(OUT / "candidate/review-scene.mjcf")
    sims = [
        YORMujoco(scene, ik, headless=True)
        for scene, ik in [
            (OUT / "baseline/scene.mjcf", DESC / "sophia-ik.mjcf"),
            (OUT / "candidate/review-scene.mjcf", OUT / "candidate/sophia-ik.mjcf"),
        ]
    ]
    try:
        models = [s.model for s in sims]
        base, candidate = models
        locked = {}
        for attr in (
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "cam_pos",
            "cam_quat",
            "cam_intrinsic",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "actuator_gainprm",
        ):
            locked[attr] = bool(np.array_equal(getattr(base, attr), getattr(candidate, attr)))
        locked["neck_zero"] = bool(
            np.array_equal(
                base.numeric("neck_zero_deg").data, candidate.numeric("neck_zero_deg").data
            )
        )
        locked["wheel_geoms"] = all(
            np.array_equal(
                base.geom_pos[g], candidate.geom_pos[candidate.geom(base.geom(g).name).id]
            )
            for g in range(base.ngeom)
            if base.geom(g).name and base.geom_bodyid[g] not in (base.body("sophia_chassis").id, 0)
        )
        # The previous assertion also covers arm/head geometry positions, not only wheels.
        guards = [SelfCollisionGuard(m, 0.00001) for m in models]
        named_failures, recorded_failures, baseline_hits = [], [], []
        case_count = 0

        def check(qs, context, destination):
            nonlocal case_count
            hits = [contacts(g, m, q) for g, m, q in zip(guards, models, qs, strict=True)]
            case_count += 1
            if hits[0]:
                baseline_hits.append(context)
            for pair, depth in hits[1].items():
                if depth < hits[0].get(pair, 0) - 0.001:
                    destination.append(
                        dict(
                            context=context, pair=pair, depth_m=depth, previous_m=hits[0].get(pair)
                        )
                    )

        limits = [
            candidate.jnt_range[candidate.joint("head_" + s + "_joint").id] for s in ("pan", "tilt")
        ]
        for pose in ("home", "reference", "lowered", "raised"):
            for height in (0.0, 0.3, 0.5588):
                for pan in np.linspace(*limits[0], 9):
                    for tilt in np.linspace(*limits[1], 13):
                        qs = []
                        for m in models:
                            q = m.key(pose).qpos.copy()
                            q[m.joint("lift_joint").qposadr[0]] = height
                            q[m.joint("sophia_middle_slide").qposadr[0]] = height / 2
                            q[m.joint("head_pan_joint").qposadr[0]] = pan
                            q[m.joint("head_tilt_joint").qposadr[0]] = tilt
                            qs.append(q)
                        check(qs, dict(pose=pose, height_m=height), named_failures)
        named_count = case_count
        projection_errors = []
        for ep in (0, 1, 2, 3, 8, 17):
            path, a = native(ep)
            off = recording_lift_offset(path, ep, cfg)
            for i in range(0, len(a["time_s"]), 3):
                for s, offset in zip(sims, (0.0, off), strict=True):
                    measured_pose(s, s.data, a, i, lift_extension_offset_m=offset)
                check(
                    [s.data.qpos.copy() for s in sims], dict(episode=ep, row=i), recorded_failures
                )
                for side in ("left", "right"):
                    values = []
                    for s in sims:
                        cid = s.model.camera("head_zed_left").id
                        p = (
                            s.data.site_xpos[s.model.site(side + "_arm_ee").id]
                            - s.data.cam_xpos[cid]
                        )
                        values.append(s.data.cam_xmat[cid].reshape(3, 3).T @ p)
                    projection_errors.append(float(np.linalg.norm(values[1] - values[0])))
        # Confirm the four fixed tape measurements on compiled geometry.
        s = sims[1]
        m = s.model
        d = s.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        ids = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(("upper_rail_", "lower_rail_", "corner_post_"))
            and not m.geom(g).name.endswith("_collision")
        ]
        low = np.min([d.geom_xpos[g] - m.geom_size[g] for g in ids], axis=0)
        high = np.max([d.geom_xpos[g] + m.geom_size[g] for g in ids], axis=0)
        expected = cfg["measured"]
        measurements_ok = bool(
            np.allclose(
                [high[0] - low[0], high[1] - low[1], high[2] - low[2], low[2]],
                [
                    expected["base_frame_depth_m"],
                    expected["base_frame_width_m"],
                    expected["base_frame_height_m"],
                    expected["base_frame_floor_clearance_m"],
                ],
                atol=1e-8,
            )
        )
        # Score the compiled geometry/cameras, not merely the analytical fit.
        # The upright grounded chassis is fixed in these native recordings.
        compiled_scores = {
            split: {eye: [] for eye in ("left", "right")} for split in ("train", "validation")
        }
        evidence = json.loads((OUT / "stereo-base-points.json").read_text())
        for record in evidence["records"]:
            ep = record["episode"]
            path, _ = native(ep)
            offset = recording_lift_offset(path, ep, cfg)
            d.qpos[:] = record["qpos"]
            d.qpos[m.joint("lift_joint").qposadr[0]] += offset
            d.qpos[m.joint("sophia_middle_slide").qposadr[0]] += offset / 2
            mujoco.mj_forward(m, d)
            point = np.array(
                [high[0], high[1] if record["corner"] == "front_left" else low[1], high[2]]
            )
            for eye in ("left", "right"):
                cid = m.camera("head_zed_" + eye).id
                k = camera_intrinsics(m, "head_zed_" + eye, 1920, 1080)
                p = d.cam_xmat[cid].reshape(3, 3).T @ (point - d.cam_xpos[cid])
                uv = [k["cx"] + k["fx"] * p[0] / -p[2], k["cy"] - k["fy"] * p[1] / -p[2]]
                compiled_scores[record["split"]][eye].append(
                    float(np.linalg.norm(np.array(uv) - record[eye + "_px"]))
                )
        compiled_scores = {
            split: {
                eye: dict(n=len(values), rmse_px=float(np.sqrt(np.mean(np.square(values)))))
                for eye, values in eyes.items()
            }
            for split, eyes in compiled_scores.items()
        }
        # New named-pose collisions are never silently exempted. The normal guard
        # must reject them; the geometry remains honest even at unsafe low lift.
        guard_catches = []
        for context in {json.dumps(x["context"], sort_keys=True) for x in named_failures}:
            c = json.loads(context)
            q = m.key(c["pose"]).qpos.copy()
            q[m.joint("lift_joint").qposadr[0]] = c["height_m"]
            q[m.joint("sophia_middle_slide").qposadr[0]] = c["height_m"] / 2
            # Base/arm contacts do not depend on neck grid position.
            guard_catches.append(guards[1].check(m.key("home").qpos, q) is not None)
        gates = dict(
            locked_frames_and_dynamics=all(locked.values()),
            confirmed_dimensions=measurements_ok,
            compiled_landmarks_match_fit=all(
                eye["rmse_px"] < 15 for eye in compiled_scores["validation"].values()
            ),
            enough_heldout_landmarks=profile["metrics"]["validation"]["n"] >= 8,
            heldout_stereo_rmse_under_20mm=profile["metrics"]["validation"]["aligned"]["rmse_m"]
            < 0.02,
            geometry_only_projection_improves_over_80_percent=profile["metrics"]["validation"][
                "geometry_only"
            ]["rmse_px"]
            < 0.2 * profile["metrics"]["validation"]["before"]["rmse_px"],
            arm_camera_relative_pose_unchanged=max(projection_errors) < 1e-9,
            no_new_recorded_penetrations=not recorded_failures,
            unsafe_named_poses_guarded=all(guard_catches),
        )
        report = dict(
            accepted=all(gates.values()),
            promoted=False,
            gates=gates,
            locked_arrays=locked,
            measured_frame_min_world_m=low.tolist(),
            measured_frame_max_world_m=high.tolist(),
            collision_cases=case_count,
            named_cases=named_count,
            recorded_cases=case_count - named_count,
            new_named_contacts=len(named_failures),
            new_named_examples=named_failures[:20],
            new_recorded_contacts=len(recorded_failures),
            new_recorded_examples=recorded_failures[:30],
            baseline_contact_poses=len(baseline_hits),
            arm_camera_max_relative_error_m=max(projection_errors),
            metrics=profile["metrics"],
            compiled_landmark_scores=compiled_scores,
            hardware_modified=False,
            limitations=profile["limitations"]
            + ["Discrete collision samples are not a continuous clearance proof."],
        )
        (OUT / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        render(sims[1], cfg, False)
        if not report["accepted"]:
            raise SystemExit("Base candidate failed acceptance; defaults unchanged.")
        if args.promote:
            cfg["base_alignment"]["status"] = (
                "validated_stereo_base_geometry_with_recording_scoped_height_alignment"
            )
            (DESC / "sophia.json").write_text(json.dumps(cfg, indent=2) + "\n")
            build(cfg)
            report.update(promoted=True, canonical_model_sha256=digest(DESC / "sophia.mjcf"))
            (OUT / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.video:
            render(sims[1], cfg, True)
    finally:
        for s in sims:
            s.stop()


if __name__ == "__main__":
    main()
