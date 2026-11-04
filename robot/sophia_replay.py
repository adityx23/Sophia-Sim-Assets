"""Workstation-only trajectory generation and MuJoCo replay. No robot transport."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTION = ROOT / "robot/yor-description"
TRAJECTORY_SCHEMA = "sophia.replay.trajectory.v1"
RECORDING_SCHEMA = "sophia.replay.recording.v1"
POSE_FRAME = "local_world_base_center_xyyaw_m_rad"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def trajectory_hash(trajectory):
    return digest({k: v for k, v in trajectory.items() if k != "sha256"})


def _vector(value, n, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected {n} finite numbers")
    return a


def validate_trajectory(trajectory):
    from robot.sophia_policy import joint_limits

    t = trajectory
    if t.get("schema") != TRAJECTORY_SCHEMA or t.get("sha256") != trajectory_hash(t):
        raise ValueError("Trajectory schema/hash mismatch")
    if t.get("side") != "left" or t.get("base_command_frame") != "body_x_forward_y_left_yaw_ccw":
        raise ValueError("Unsupported arm or base frame")
    frames, hz = t["frames"], t["control_hz"]
    if t.get("no_objects") is not True or not np.isfinite(t["duration_s"]):
        raise ValueError("Object-free replay and finite duration required")
    if not np.isfinite(hz) or hz != 30 or len(frames) < 2:
        raise ValueError("This replay version requires a 30 Hz trajectory")
    times = np.array([f["time_s"] for f in frames])
    if not np.allclose(times, np.arange(len(frames)) / hz, atol=1e-9, rtol=0):
        raise ValueError("Trajectory timestamps must start at zero and follow the 30 Hz grid")
    if [f["seq"] for f in frames] != list(range(len(frames))):
        raise ValueError("Trajectory sequence must be contiguous from zero")
    if abs(t["duration_s"] - times[-1]) > 1e-9:
        raise ValueError("Duration does not match the command timeline")
    if any(
        not np.isfinite(f["preview_time_s"]) or abs(f["preview_time_s"] - 1 / hz) > 1e-9
        for f in frames
    ):
        raise ValueError("Replay preview_time must equal one control period")
    q = np.array([_vector(f["left_joints"], 7, "left_joints") for f in frames])
    base = np.array([_vector(f["base_velocity"], 3, "base_velocity") for f in frames])
    grip = np.array([f["left_gripper"] for f in frames])
    limits = joint_limits("left")
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("Joint target exceeds model limits")
    if not np.isfinite(grip).all() or np.any(grip < 0) or np.any(grip > 1):
        raise ValueError("Gripper opening must be in [0,1]")
    if np.any(abs(base[:, 0]) > 0.050001) or np.any(abs(base[:, 1:]) > 1e-10):
        raise ValueError("Initial replay supports forward speed <= 0.05 m/s only")
    if max(np.max(abs(base[0])), np.max(abs(base[-1]))) > 1e-10:
        raise ValueError("Base must start and finish stopped")
    velocity = np.diff(q, axis=0) * hz
    acceleration = np.diff(velocity, axis=0) * hz
    if np.max(abs(velocity)) > 0.5001 or np.max(abs(acceleration)) > 1.501:
        raise ValueError("Arm command rate/acceleration exceeds replay bounds")
    if np.max(abs(np.diff(base, axis=0) * hz)) > 0.101:
        raise ValueError("Base command acceleration exceeds replay bounds")
    initial = t["initial"]
    if not np.allclose(q[0], _vector(initial["left_joints"], 7, "initial joints"), atol=1e-10):
        raise ValueError("Initial joint targets disagree")
    if not np.allclose(q[-1], q[0], atol=1e-8) or grip[0] != 1 or grip[-1] != 1:
        raise ValueError("Arm must return to its initial pose with an open gripper")
    if (
        not np.isfinite(initial["lift_extension_m"])
        or not 0 <= initial["lift_extension_m"] <= 0.5588
    ):
        raise ValueError("Invalid initial lift extension")
    _vector(initial["right_joints"], 7, "right hold joints")
    _vector(initial["neck_joint_rad"], 2, "neck hold joints")
    return t


def load_trajectory(path):
    return validate_trajectory(json.loads(Path(path).read_text()))


def build_trajectory():
    from robot.sophia_env import solve_grasp_ik
    from robot.sophia_profile import lift_extension_to_encoder
    from robot.yor_mujoco import HOME_Q, YORMujoco

    sim = YORMujoco(DESCRIPTION / "scene-sophia.mjcf", headless=True)
    try:
        stages = [
            ("align", [0.40, 0.13045, 0.92], 1.0),
            ("reach", [0.54, 0.13045, 0.90], 1.0),
            ("lower_to_virtual_pick", [0.54, 0.13045, 0.80], 1.0),
            ("close_empty_gripper", None, 0.0),
            ("lift_empty_gripper", [0.54, 0.13045, 0.92], 0.0),
            ("move_to_virtual_place", [0.49, 0.27045, 0.92], 0.0),
            ("lower_to_virtual_place", [0.49, 0.27045, 0.80], 0.0),
            ("open_gripper", None, 1.0),
            ("retract", [0.40, 0.13045, 0.92], 1.0),
            ("return_to_start", "home", 1.0),
        ]
        hz, frames = 30, []
        q, grip = HOME_Q["left"].copy(), 1.0

        def append(stage, base, joints, opening):
            frames.append(
                {
                    "seq": len(frames),
                    "time_s": len(frames) / hz,
                    "stage": stage,
                    "base_velocity": list(map(float, base)),
                    "left_joints": joints.tolist(),
                    "left_gripper": float(opening),
                    "preview_time_s": 1 / hz,
                }
            )

        def smooth(u):
            return 10 * u**3 - 15 * u**4 + 6 * u**5

        # Six seconds, 1 s ramps, 4 s cruise: integral = 0.25 commanded meters.
        for i in range(10 * hz):
            seconds = i / hz
            if seconds < 2:
                stage, speed = "settle", 0.0
            elif seconds < 3:
                stage, speed = "drive_ramp_up", 0.05 * smooth(seconds - 2)
            elif seconds < 7:
                stage, speed = "drive_forward", 0.05
            elif seconds < 8:
                stage, speed = "drive_ramp_down", 0.05 * (1 - smooth(seconds - 7))
            else:
                stage, speed = "stop_and_settle", 0.0
            append(stage, [speed, 0, 0], q, grip)
        for stage, target, opening in stages:
            if target is None:
                next_q = q.copy()
            elif isinstance(target, str):
                next_q = HOME_Q["left"].copy()
            else:
                sim.data.qpos[sim.arms["left"].qpos_adr] = q
                next_q = solve_grasp_ik(sim, target)
            distance = float(np.max(abs(next_q - q)))
            duration = max(0.8, 1.875 * distance / 0.5, math.sqrt(5.78 * distance / 1.5))
            ticks = math.ceil(duration * hz)
            for tick in range(ticks):
                fraction = smooth(tick / ticks)
                append(
                    stage,
                    [0, 0, 0],
                    q + fraction * (next_q - q),
                    grip + fraction * (opening - grip),
                )
            q, grip = next_q, opening
            for _ in range(12):
                append(stage + "_hold", [0, 0, 0], q, grip)
        for _ in range(2 * hz + 1):
            append("finish", [0, 0, 0], q, 1.0)
        result = {
            "schema": TRAJECTORY_SCHEMA,
            "name": "forward_and_unloaded_pick_place",
            "side": "left",
            "control_hz": hz,
            "base_command_frame": "body_x_forward_y_left_yaw_ccw",
            "joint_semantics": "absolute radians, joint1..joint7, 0=closed/1=open gripper",
            "no_objects": True,
            "nominal_forward_distance_m": 0.25,
            "duration_s": frames[-1]["time_s"],
            "initial": {
                "left_joints": HOME_Q["left"].tolist(),
                "right_joints": HOME_Q["right"].tolist(),
                "left_gripper": 1.0,
                "right_gripper": 1.0,
                "lift_extension_m": 0.3,
                "lift_encoder_m_assumed": float(lift_extension_to_encoder(0.3)),
                "neck_joint_rad": sim.data.qpos[[sim._pan_adr, sim._tilt_adr]].tolist(),
                "neck_deg": sim._neck_deg(),
            },
            "initialization_note": (
                "Simulation setup only; physical start configuration "
                "must be established separately."
            ),
            "source_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in (
                    "robot/yor-description/sophia.mjcf",
                    "robot/yor-description/sophia.json",
                    "robot/sophia_replay.py",
                )
            },
            "frames": frames,
        }
        result["sha256"] = trajectory_hash(result)
        return validate_trajectory(result)
    finally:
        sim.stop()


def simulate(trajectory, output, *, video=False):
    import cv2
    import mujoco
    from robot.yor_mujoco import DRIVE_VEL_SCALE, YORMujoco

    validate_trajectory(trajectory)
    for name in (
        "robot/yor-description/sophia.mjcf",
        "robot/yor-description/sophia.json",
    ):
        if (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            != trajectory["source_sha256"][name]
        ):
            raise ValueError(f"Reference model changed: {name}; build a new trajectory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sim = YORMujoco(DESCRIPTION / "scene-sophia.mjcf", headless=True)
    sim._initialized = True  # Manual simulation clock; no RPC server or worker.
    m, d, initial = sim.model, sim.data, trajectory["initial"]
    writer = renderer = None
    samples, commands, qpos = [], [], []
    failure = None
    max_self_penetration = 0.0
    try:
        for side, arm in sim.arms.items():
            q = np.array(initial[side + "_joints"])
            d.qpos[arm.qpos_adr] = d.ctrl[arm.act_ids] = q
            arm.q_cmd, arm.q_goal = q.copy(), q.copy()
            arm.q_vel[:] = 0
            arm.gripper_cmd = arm.gripper_goal = initial[side + "_gripper"]
            for act, opened, _ in arm.gripper_acts:
                d.ctrl[act] = opened
                d.qpos[m.jnt_qposadr[m.actuator_trnid[act, 0]]] = opened
        d.qpos[sim._lift_adr] = sim.lift_setpoint = sim._lift_goal = initial["lift_extension_m"]
        d.qpos[m.joint("sophia_middle_slide").qposadr[0]] = sim.lift_setpoint / 2
        d.ctrl[sim._lift_act] = sim.lift_setpoint
        sim._neck_q_cmd = sim._neck_goal = np.array(initial["neck_joint_rad"])
        d.qpos[[sim._pan_adr, sim._tilt_adr]] = sim._neck_q_cmd
        d.ctrl[[sim._pan_act, sim._tilt_act]] = sim._neck_q_cmd
        mujoco.mj_forward(m, d)
        for _ in range(round(1 / m.opt.timestep)):
            sim._tick_controls(m.opt.timestep)
            mujoco.mj_step(m, d)
        if sim._guard_holding:
            raise RuntimeError("Starting posture was rejected by the simulation guard")
        d.time = 0.0
        initial_state = {
            "left_joints": sim.get_left_joint_positions().tolist(),
            "right_joints": sim.get_right_joint_positions().tolist(),
            "left_gripper": sim.get_left_gripper_pose(),
            "lift_extension_m": sim.get_lift_height(),
            "neck_deg": sim._neck_deg(),
            "steer_rad": sim.get_base_encoders()["steer_rad"],
        }
        if video:
            renderer = mujoco.Renderer(m, height=720, width=1280)
            writer = cv2.VideoWriter(
                str(output / "replay-preview-raw.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (1280, 720),
            )
            if not writer.isOpened():
                raise RuntimeError("Video encoder unavailable")
        for frame in trajectory["frames"]:
            while d.time < frame["time_s"] - 1e-10:
                sim._tick_controls(m.opt.timestep)
                mujoco.mj_step(m, d)
                for c in d.contact:
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    if b1 and b2 and frozenset((b1, b2)) not in sim._collision_guard.ignored:
                        max_self_penetration = max(max_self_penetration, -float(c.dist))
                if sim._guard_holding or d.warning.number.any() or not np.isfinite(d.qpos).all():
                    failure = "self_collision_or_physics_failure"
                    break
            if failure:
                break
            accepted = sim.set_left_joint_target(
                frame["left_joints"], frame["left_gripper"], frame["preview_time_s"]
            )
            if not accepted:
                failure = "command_rejected"
                break
            sim.set_base_velocity(np.array(frame["base_velocity"]))
            commands.append({**frame, "dispatch_time_s": float(d.time), "status": "accepted"})
            mujoco.mj_forward(m, d)
            pose = sim.get_pose()
            encoders = sim.get_base_encoders()
            steer = np.array(encoders["steer_rad"])
            speed = np.array(encoders["drive_vel"]) / DRIVE_VEL_SCALE
            rows = np.array(
                [
                    [1, 0, -y] if component == 0 else [0, 1, x]
                    for x, y in sim._module_xy
                    for component in (0, 1)
                ]
            )
            wheel_xy = np.column_stack([speed * np.cos(steer), speed * np.sin(steer)]).ravel()
            body_velocity = np.linalg.lstsq(rows, wheel_xy, rcond=None)[0]
            samples.append(
                {
                    "seq": len(samples),
                    "time_s": float(d.time),
                    "left_joints": sim.get_left_joint_positions().tolist(),
                    "left_gripper": sim.get_left_gripper_pose(),
                    "base_pose": [pose["x"], pose["y"], pose["theta"]],
                    "base_pose_time_s": float(d.time),
                    "base_velocity_body": body_velocity.tolist(),
                    "steer_rad": encoders["steer_rad"],
                    "drive_m_s": (np.array(encoders["drive_vel"]) / DRIVE_VEL_SCALE).tolist(),
                }
            )
            qpos.append(d.qpos.copy())
            if renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.25, -0.13, 0.8]
                camera.distance, camera.azimuth, camera.elevation = 2.65, 135, -18
                option = mujoco.MjvOption()
                option.geomgroup[3] = 0
                renderer.update_scene(d, camera, scene_option=option)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 0), (1280, 86), (25, 30, 35), -1)
                cv2.putText(
                    image,
                    "SOPHIA | unloaded trajectory replay | SIMULATION",
                    (25, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"{frame['time_s']:.1f}s  {frame['stage'].replace('_', ' ')}",
                    (25, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (210, 225, 230),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(image)
        if max_self_penetration > 0.002:
            failure = failure or "self_penetration"
        result = {
            "schema": RECORDING_SCHEMA,
            "run_id": "sim-" + trajectory["sha256"][:12],
            "trajectory_sha256": trajectory["sha256"],
            "source": "simulation",
            "clock": "seconds_from_run_start",
            "pose_frame": POSE_FRAME,
            "pose_source": "mujoco_ground_truth",
            "complete": failure is None,
            "base_velocity_source": "wheel_kinematics",
            "initial_state": initial_state,
            "end_reason": failure or "complete",
            "commands": commands,
            "samples": samples,
            "events": [],
            "hardware_validated": False,
            "max_self_penetration_m": max_self_penetration,
            "model_sha256": trajectory["source_sha256"],
        }
        write_json(output / "recording.json", result)
        np.savez_compressed(
            output / "physics-states.npz",
            qpos=np.array(qpos),
            time_s=[r["time_s"] for r in samples],
        )
        if failure:
            raise RuntimeError(f"Simulation replay failed: {failure}; see {output}")
        return result
    finally:
        sim.stop()
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
