"""Extended Sophia evaluations; no hardware, network or realtime waits.

MUJOCO_GL=egl python robot/yor-description/evaluate_sophia.py

Exit 1 means at least one tested condition failed. Workspace probes deliberately
include colliding configurations: those must be rejected by the motion guard.
Raw collision counts and closeups remain in the report for geometry review.
They do not certify the real robot's dynamics or collision-free workspace.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from validate_sophia import HERE, qadr, reset, step
from sophia_sources import head_calibration


def penetration(model, data):
    """Worst robot-to-robot penetration; exclude the expected floor contacts."""
    worst = {"depth_m": 0.0, "geoms": [], "bodies": []}
    for contact in data.contact:
        ids = (int(contact.geom1), int(contact.geom2))
        if any(model.geom_bodyid[g] == 0 for g in ids):
            continue
        if -contact.dist > worst["depth_m"]:
            worst = {
                "depth_m": float(-contact.dist),
                "geoms": [model.geom(g).name or f"geom_{g}" for g in ids],
                "bodies": [model.body(model.geom_bodyid[g]).name for g in ids],
            }
    return worst


def tilt_deg(model, data):
    # The source chassis's local -X points up at the upright home pose.
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return float(np.rad2deg(np.arccos(np.clip(-rotation[2, 0], -1, 1))))


def render_failure(model, qpos, contact, output, name):
    """Save a close view of the worst pose, with visuals and collision shapes."""
    from PIL import Image

    data = reset(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    ids = [int(g[5:]) if g.startswith("geom_") else model.geom(g).id for g in contact["geoms"]]
    camera.lookat[:] = data.geom_xpos[ids].mean(axis=0)
    camera.distance = .42 if name == "neck-interference" else .8
    camera.azimuth, camera.elevation = 140, -15
    option = mujoco.MjvOption()
    with mujoco.Renderer(model, height=640, width=800) as renderer:
        for label, visible in (("visual", 2), ("collision", 3)):
            option.geomgroup[2] = visible == 2
            option.geomgroup[3] = visible == 3
            renderer.update_scene(data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"{name}-{label}.png")


def evaluate(model, output):
    checks = []
    sys.path.insert(0, str(HERE.parents[1]))
    from robot.sim_collision import SelfCollisionGuard

    guard = SelfCollisionGuard(model, model.numeric("self_collision_margin_m").data[0])

    def check(name, ok, **details):
        checks.append({"name": name, "passed": bool(ok), **details})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {details}", flush=True)

    for pose in ("lowered", "home", "reference", "raised"):
        data = reset(model, pose)
        start = data.qpos[:2].copy()
        max_tilt = max_depth = 0.0
        finite = True
        for _ in range(600):
            step(model, data, .1)
            max_tilt = max(max_tilt, tilt_deg(model, data))
            max_depth = max(max_depth, penetration(model, data)["depth_m"])
            finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        drift = float(np.linalg.norm(data.qpos[:2] - start))
        check(f"{pose}: 60 s gravity hold", finite and drift < .02 and max_tilt < 2
              and max_depth < .001 and not data.warning.number.any(),
              drift_m=drift, max_tilt_deg=max_tilt, max_self_penetration_m=max_depth,
              warnings=data.warning.number.tolist())

    # A grid exposes configuration-dependent interference hidden by home checks.
    data = reset(model)
    neck_start = data.qpos.copy()
    worst = {"depth_m": 0.0}
    worst_pose = None
    colliding = 0
    rejected_collisions = 0
    pan_range, tilt_range = (model.joint(f"head_{axis}_joint").range for axis in ("pan", "tilt"))
    zero = model.numeric("neck_zero_deg").data
    for pan in np.linspace(*pan_range, 25):
        for tilt in np.linspace(*tilt_range, 19):
            data.qpos[qadr(model, "head_pan_joint")] = pan
            data.qpos[qadr(model, "head_tilt_joint")] = tilt
            mujoco.mj_forward(model, data)
            contact = penetration(model, data)
            colliding += contact["depth_m"] > .0001
            if contact["depth_m"] > .0001:
                rejected_collisions += guard.check(neck_start, data.qpos) is not None
            if contact["depth_m"] > worst["depth_m"]:
                worst = {**contact, "api_pan_deg": float(zero[0] - np.rad2deg(pan)),
                         "api_tilt_deg": float(zero[1] + np.rad2deg(tilt))}
                worst_pose = data.qpos.copy()
    check("neck: guard rejects colliding combinations in 475-pose sweep", rejected_collisions == colliding,
          colliding_configurations=int(colliding), worst=worst,
          rejected_colliding_configurations=int(rejected_collisions),
          penetration_tolerance_m=.0001)
    if colliding:
        render_failure(model, worst_pose, worst, output, "neck-interference")

    arm_names = [f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)]
    arm_q = [qadr(model, name) for name in arm_names]
    arm_act = [model.actuator(f"{name}_pos").id for name in arm_names]
    a, b = model.key("home").qpos[arm_q], model.key("reference").qpos[arm_q]
    for lift in (0.0, .1, .3, float(model.joint("lift_joint").range[1])):
        data = reset(model)
        data.qpos[qadr(model, "lift_joint")] = lift
        data.qpos[qadr(model, "sophia_middle_slide")] = lift / 2
        path_start = data.qpos.copy()
        worst = {"depth_m": 0.0}
        worst_pose = None
        colliding = 0
        for fraction in np.linspace(0, 1, 101):
            data.qpos[arm_q] = (1-fraction)*a + fraction*b
            mujoco.mj_forward(model, data)
            contact = penetration(model, data)
            colliding += contact["depth_m"] > .0001
            if contact["depth_m"] > worst["depth_m"]:
                worst = {**contact, "path_fraction": float(fraction)}
                worst_pose = data.qpos.copy()
        rejection = guard.check(path_start, data.qpos)
        check(f"home-to-hanging at lift {lift:.4f} m: guard {'rejects interference' if colliding else 'allows clear path'}",
              (rejection is not None) if colliding else (rejection is None),
              samples=101, colliding_configurations=int(colliding), worst=worst,
              guard_rejection=rejection,
              penetration_tolerance_m=.0001)
        if colliding:
            render_failure(model, worst_pose, worst, output, f"arm-interference-lift-{lift:.4f}")

    # Simultaneous, smooth motion close to home. Grippers traverse their full
    # physical opening and the lift traverses its full measured stroke twice.
    data = reset(model, "lowered")
    step(model, data, 2)
    lift_act = model.actuator("Lift").id
    pan_act, tilt_act = (model.actuator(f"head_{axis}_pos").id for axis in ("pan", "tilt"))
    grip_act = [model.actuator(f"{side}_gripper_finger_{finger}_pos").id
                for side in ("left", "right") for finger in "ab"]
    travel = float(model.joint("lift_joint").range[1])
    max_arm_error = max_lift_error = max_depth = max_coupling = 0.0
    for n in range(round(30 / model.opt.timestep)):
        phase = 2*np.pi*n*model.opt.timestep/15
        target = a + .08*np.sin(phase)*np.cos(np.arange(14))
        data.ctrl[arm_act] = target
        data.ctrl[lift_act] = travel*(1-np.cos(phase))/2
        data.ctrl[pan_act], data.ctrl[tilt_act] = .7*np.sin(phase), .4*np.sin(phase)
        data.ctrl[grip_act] = .0508*(1+np.cos(phase))/2
        mujoco.mj_step(model, data)
        max_arm_error = max(max_arm_error, float(np.max(abs(data.qpos[arm_q]-target))))
        max_lift_error = max(max_lift_error, float(abs(data.qpos[qadr(model, "lift_joint")]-data.ctrl[lift_act])))
        max_coupling = max(max_coupling, float(abs(data.qpos[qadr(model, "sophia_middle_slide")]
                                                 - data.qpos[qadr(model, "lift_joint")]/2)))
        max_depth = max(max_depth, penetration(model, data)["depth_m"])
    check("30 s simultaneous arms, lift, neck and gripper motion",
          max_arm_error < .15 and max_lift_error < .01 and max_coupling < .002
          and max_depth < .001 and not data.warning.number.any()
          and np.isfinite(data.qpos).all(), max_arm_tracking_error_rad=max_arm_error,
          max_lift_tracking_error_m=max_lift_error, max_stage_coupling_error_m=max_coupling,
          max_self_penetration_m=max_depth, warnings=data.warning.number.tolist())

    # Exercise the actual swerve calculation without launching its realtime
    # thread, so every run takes exactly the same physics steps.
    sys.path.insert(0, str(HERE.parents[1]))
    from robot.yor_mujoco import YORMujoco

    def base_pose(sim):
        transform = sim._T_world_robot()
        center = transform[:2, 3] + transform[:2, :2] @ sim._chassis_center
        return {"x": float(center[0]), "y": float(center[1]),
                "theta": float(np.arctan2(transform[1, 0], transform[0, 0]))}

    for pose in ("lowered", "raised"):
        sim = YORMujoco(model_path=HERE / "scene-sophia.mjcf", headless=True)
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, m.key(pose).id)
        step(m, d, 2)
        max_tilt = max_depth = 0.0
        displacements = []
        for velocity in ([.2, 0, 0], [0, .2, 0], [0, 0, .5]):
            before = base_pose(sim)
            for target, duration in ((velocity, 2), ([0, 0, 0], 1)):
                sim.base_target[:] = target
                for _ in range(round(duration/m.opt.timestep)):
                    sim._tick_base(m.opt.timestep)
                    mujoco.mj_step(m, d)
                    max_tilt = max(max_tilt, tilt_deg(m, d))
                    max_depth = max(max_depth, penetration(m, d)["depth_m"])
            after = base_pose(sim)
            c, s = np.cos(before["theta"]), np.sin(before["theta"])
            dx, dy = after["x"]-before["x"], after["y"]-before["y"]
            yaw = (after["theta"]-before["theta"]+np.pi) % (2*np.pi)-np.pi
            displacements.append([float(c*dx+s*dy), float(-s*dx+c*dy), float(yaw)])
        diagonal = np.diag(displacements)
        off_axis = np.asarray(displacements)-np.diag(diagonal)
        check(f"{pose}: forward, strafe and yaw with stops",
              np.all(diagonal > [.2, .2, .5]) and np.max(abs(off_axis)) < .1
              and max_tilt < 5 and max_depth < .001 and not d.warning.number.any(),
              displacements_xy_m_yaw_rad=displacements, max_tilt_deg=max_tilt,
              max_self_penetration_m=max_depth, warnings=d.warning.number.tolist())
        sim.stop()

    from PIL import Image

    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for pose in ("home", "raised"):
            data = reset(model, pose)
            cameras = [model.camera(f"head_zed_{side}").id for side in ("left", "right")]
            baseline = float(np.linalg.norm(data.cam_xpos[cameras[0]]-data.cam_xpos[cameras[1]]))
            parallel = np.allclose(data.cam_xmat[cameras[0]], data.cam_xmat[cameras[1]], atol=1e-7)
            for side in ("left", "right"):
                renderer.update_scene(data, camera=f"head_zed_{side}", scene_option=option)
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                Image.fromarray(rgb).save(output / f"stereo-{pose}-{side}.png")
                check(f"{pose}: {side} stereo RGB/depth render",
                      parallel and abs(baseline-head_calibration()["baseline_m"]) < 1e-8 and float(rgb.std()) > 3
                      and np.isfinite(depth).all() and np.min(depth) > 0,
                      configured_baseline_m=baseline, rgb_std=float(rgb.std()),
                      depth_min_max_m=[float(depth.min()), float(depth.max())],
                      camera_intrinsics_source="Orin 2026-10-30; zed.rectified",
                      camera_to_robot_extrinsics_still_unmeasured=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "output" / "asset-review" / "evals")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
    checks = evaluate(model, args.output)
    passed = sum(c["passed"] for c in checks)
    source_files = ["scene-sophia.mjcf", "sophia.mjcf", "sophia-ik.mjcf", "sophia.json", "evaluate_sophia.py",
                    "../yor_mujoco.py", "../sim_collision.py"]
    report = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_files},
              "mujoco_version": mujoco.__version__, "timestep_s": float(model.opt.timestep),
              "checks": checks, "passed_count": passed, "total_count": len(checks),
              "passed": passed == len(checks)}
    (args.output / "extended-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{passed}/{len(checks)} passed", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
