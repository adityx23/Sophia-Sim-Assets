"""Offline recorded-command physics replay and labelled measured-pose reference."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.sophia_profile import encoder_to_lift_extension  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402


def stamp(t):
    return f"{int(t) // 60:02d}:{t % 60:05.2f}"


def scene(output):
    desc = ROOT / "robot/yor-description"
    model = ET.parse(desc / "sophia.mjcf")
    model.getroot().find("compiler").set("meshdir", str(desc / "assets_nero"))
    model.write(output / "robot.mjcf")
    tree = ET.parse(desc / "scene-sophia.mjcf")
    tree.getroot().find("include").set("file", "robot.mjcf")
    asset = tree.getroot().find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="replay_grid",
        type="2d",
        builtin="checker",
        rgb1=".28 .31 .34",
        rgb2=".34 .37 .40",
        width="256",
        height="256",
    )
    material = asset.find("material[@name='sophia_floor']")
    material.set("texture", "replay_grid")
    material.set("texrepeat", "2 2")
    material.set("texuniform", "true")
    ET.SubElement(tree.getroot().find("visual"), "quality", shadowsize="1024", offsamples="1")
    tree.write(output / "scene.mjcf")
    return output / "scene.mjcf"


def measured_pose(sim, data, a, i, *, lift_extension_offset_m=0.0):
    """Replay feedback; historical height alignment is explicit and opt-in.

    An offset is a recording-specific pose alignment, never a hardware command
    or a change to the live encoder conversion. Callers must verify provenance.
    """
    for side, arm in sim.arms.items():
        data.qpos[arm.qpos_adr] = a[side + "_joints"][i]
        for act, opened, closed in arm.gripper_acts:
            adr = sim.model.jnt_qposadr[sim.model.actuator_trnid[act, 0]]
            data.qpos[adr] = closed + a[side + "_gripper"][i] * (opened - closed)
    h = float(encoder_to_lift_extension(a["lift_height"][i, 0])) + lift_extension_offset_m
    bounds = sim.model.jnt_range[sim.model.joint("lift_joint").id]
    if not np.isfinite(h) or not bounds[0] <= h <= bounds[1]:
        raise ValueError("Recorded lift pose plus alignment is outside physical travel")
    data.qpos[sim._lift_adr] = h
    data.qpos[sim.model.joint("sophia_middle_slide").qposadr[0]] = h / 2
    data.qpos[[sim._pan_adr, sim._tilt_adr]] = np.deg2rad(a["neck_deg"][i] - sim._neck_zero_deg) * [
        -1,
        1,
    ]
    mujoco.mj_forward(sim.model, data)


def initialize(sim, a):
    d = sim.data
    mujoco.mj_resetDataKeyframe(sim.model, d, sim.model.key("home").id)
    measured_pose(sim, d, a, 0)
    for side, arm in sim.arms.items():
        arm.q_cmd = arm.q_goal = a[side + "_joints"][0].copy()
        arm.q_vel[:] = 0
        arm.command_acceleration_limit = 15.0
        arm.command_velocity = np.zeros(7)
        d.ctrl[arm.act_ids] = arm.q_cmd
        arm.gripper_cmd = arm.gripper_goal = float(a[side + "_gripper"][0])
        for act, opened, closed in arm.gripper_acts:
            d.ctrl[act] = closed + arm.gripper_cmd * (opened - closed)
    sim.lift_setpoint = float(d.qpos[sim._lift_adr])
    sim._lift_goal = sim.lift_setpoint
    d.ctrl[sim._lift_act] = sim.lift_setpoint
    sim._neck_q_cmd = d.qpos[[sim._pan_adr, sim._tilt_adr]].copy()
    sim._neck_goal = sim._neck_q_cmd.copy()
    d.ctrl[[sim._pan_act, sim._tilt_act]] = sim._neck_goal
    mujoco.mj_forward(sim.model, d)


def writer(path, w, h, fps):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def text(image, line, y, size=0.55):
    cv2.putText(
        image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, size, (235, 240, 245), 1, cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = scene(out)
    desc = ROOT / "robot/yor-description"
    sim = YORMujoco(path, ik_model_path=desc / "sophia-ik.mjcf", headless=True)
    model = sim.model
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.13, -0.21, 0.87]
    camera.distance = 2.7
    camera.azimuth = 30
    camera.elevation = -14
    reference = mujoco.MjData(model)
    report = dict(
        source=str(args.capture.resolve()),
        mode="offline; no robot connection",
        preview_s=2 / 30,
        preview_basis="Pi VR bridge source default; per-command dispatch times unavailable",
        initialization="measured initial state once per clip; physics follows changed targets",
        unknown_objects="No object/table poses recorded. No objects or grasp constraints invented.",
        command_gaps="Gap-free task windows simulated; full reference uses measured poses.",
        base_reference="Reference base fixed; task base commands zero. No measured base pose.",
        replay_limits=(
            "Getter timestamps, not exact dispatch times. Initial velocities zero; "
            "cached targets sent only when changed. No live preflight alters this "
            "offline replay; physics contacts stay enabled."
        ),
        model_sha256=hashlib.sha256((desc / "sophia.mjcf").read_bytes()).hexdigest(),
        clips=[],
    )
    process = writer(out / "pick-place-command-replay.mp4", 1280, 720, args.fps)
    try:
        with mujoco.Renderer(model, height=640, width=640) as renderer:
            for _clip, f in enumerate(
                sorted(args.capture.glob("*-task-*.npz"), key=lambda p: int(p.stem.split("-")[-1]))
            ):
                a = dict(np.load(f))
                t = a["time_s"]
                relative = t - t[0]
                initialize(sim, a)
                mujoco.mj_resetDataKeyframe(model, reference, model.key("home").id)
                assert max(np.diff(t)) < 0.1
                assert np.max(abs(a["base_cmd_vel"])) < 1e-7
                last = {s: a[s + "_joints_cmd"][0].copy() for s in sim.arms}
                index = 0
                errors = {s: [] for s in sim.arms}
                poses = []
                refposes = []
                for frame, elapsed in enumerate(np.arange(0, relative[-1], 1 / args.fps)):
                    while sim.data.time < elapsed:
                        while index + 1 < len(t) and relative[index + 1] <= sim.data.time:
                            index += 1
                        for side, arm in sim.arms.items():
                            q = a[side + "_joints_cmd"][index]
                            g = a[side + "_gripper_cmd"][index]
                            if np.isfinite(q).all() and not np.array_equal(q, last[side]):
                                arm.set_goal(q, 2 / 30)
                                last[side] = q.copy()
                            if np.isfinite(g):
                                arm.gripper_goal = float(g)
                            arm.tick(sim.data, model.opt.timestep)
                        sim._tick_base(model.opt.timestep)
                        sim._tick_lift(model.opt.timestep)
                        neck = np.deg2rad(a["neck_cmd_deg"][index] - sim._neck_zero_deg) * [-1, 1]
                        sim._neck_q_cmd += np.clip(
                            neck - sim._neck_q_cmd,
                            -np.deg2rad(175.78125) * model.opt.timestep,
                            np.deg2rad(175.78125) * model.opt.timestep,
                        )
                        sim.data.ctrl[[sim._pan_act, sim._tilt_act]] = sim._neck_q_cmd
                        mujoco.mj_step(model, sim.data)
                        if sim.data.warning.number.any() or not np.isfinite(sim.data.qpos).all():
                            raise RuntimeError("Nonfinite/warning during physics replay")
                    sample = min(np.searchsorted(relative, elapsed, side="right") - 1, len(t) - 1)
                    measured_pose(sim, reference, a, sample)
                    image = np.full((720, 1280, 3), [18, 24, 30], dtype=np.uint8)
                    renderer.update_scene(sim.data, camera, scene_option=option)
                    image[60:700, :640] = renderer.render()
                    renderer.update_scene(reference, camera, scene_option=option)
                    image[60:700, 640:] = renderer.render()
                    text(image, "SIM PHYSICS: recorded command replay", 23)
                    cv2.putText(
                        image,
                        "MEASURED JOINTS: pose reference",
                        (660, 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.53,
                        (235, 240, 245),
                        1,
                        cv2.LINE_AA,
                    )
                    label = f"{f.stem} | recording {stamp(t[0] + elapsed)} | 1x speed"
                    text(image, label, 48)
                    text(
                        image,
                        "Robot motion only: real object/table poses were not recorded.",
                        716,
                        0.47,
                    )
                    process.stdin.write(image.tobytes())
                    for side, arm in sim.arms.items():
                        errors[side].append(
                            sim.data.qpos[arm.qpos_adr] - a[side + "_joints"][sample]
                        )
                    poses.append(sim.data.qpos.copy())
                    refposes.append(reference.qpos.copy())
                    if frame == int(args.fps * 10):
                        cv2.imwrite(
                            str(out / f"{f.stem}-preview.jpg"),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        )
                    if frame % (args.fps * 10) == 0:
                        print(f"{f.stem}: rendered {elapsed:.0f}/{relative[-1]:.0f} s", flush=True)
                np.savez_compressed(
                    out / (f.stem + "-comparison.npz"),
                    simulation_qpos=poses,
                    measured_reference_qpos=refposes,
                )
                report["clips"].append(
                    dict(
                        source=f.name,
                        source_start_s=float(t[0]),
                        duration_s=float(relative[-1]),
                        joint_rmse_rad={
                            s: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).tolist()
                            for s, e in errors.items()
                        },
                    )
                )
                sim.data.time = 0
        process.stdin.close()
        assert process.wait() == 0
        npz = dict(np.load(args.capture / "native-signals.npz"))
        t = npz["time_s"]
        fps = 10
        mujoco.mj_resetDataKeyframe(model, reference, model.key("home").id)
        process = writer(out / "full-measured-motion-reference.mp4", 960, 720, fps)
        with mujoco.Renderer(model, height=640, width=960) as renderer:
            for i, elapsed in enumerate(np.arange(0, t[-1], 1 / fps)):
                j = max(0, np.searchsorted(t, elapsed, side="right") - 1)
                measured_pose(sim, reference, npz, j)
                image = np.full((720, 960, 3), [18, 24, 30], dtype=np.uint8)
                renderer.update_scene(reference, camera, scene_option=option)
                image[60:700] = renderer.render()
                text(
                    image,
                    "MEASURED MOTION REFERENCE - pose reconstruction, not physics replay",
                    24,
                    0.56,
                )
                text(image, f"Recording {stamp(elapsed)} | 1x speed | base pose not recorded", 48)
                text(
                    image,
                    "Joint, neck and gripper telemetry shown; missing intervals are flagged.",
                    716,
                    0.45,
                )
                if j + 1 < len(t) and t[j + 1] - t[j] > 0.1:
                    text(image, "DATA GAP: holding last recorded pose", 110, 0.7)
                process.stdin.write(image.tobytes())
                if i % (fps * 60) == 0:
                    print(f"Full measured reference: {elapsed:.0f}/{t[-1]:.0f} s", flush=True)
        process.stdin.close()
        assert process.wait() == 0
        (out / "replay-report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        sim.stop()
    print(out, flush=True)


if __name__ == "__main__":
    main()
