"""Record and render a scripted, contact-driven mobile manipulation demo.

MUJOCO_GL=egl .venv-sophia/bin/python scripts/render_sophia_demo.py

Robot controls drive the physics. Both parcels are free bodies throughout the
simulation: no weld, mocap attachment, applied carrying force or pose resets.
The renderer replays saved physics states after task validation succeeds.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTION = ROOT / "robot/yor-description"
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import HOME_Q, YORMujoco  # noqa: E402

SIDES = ("left", "right")
SOURCE_Y = -0.20955
DEST_Y = SOURCE_Y + 1.6
STATION_X = 1.62
PARCEL_X = 1.46
PARCEL_Z = 0.79
Y_OFFSETS = {"left": 0.34, "right": -0.34}
GRASP_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def numbers(values):
    return " ".join(f"{float(value):.10g}" for value in values)


def make_scene(output):
    root = ET.parse(DESCRIPTION / "sophia.mjcf").getroot()
    root.find("compiler").set("meshdir", str(DESCRIPTION / "assets_nero"))
    # Generated scenes live outside yor-description; keep generic VR launchers'
    # matching IK model resolution valid, just like the absolute mesh directory.
    root.find("custom/text[@name='ik_model']").set("data", str(DESCRIPTION / "sophia-ik.mjcf"))
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="demo_tiles",
        type="2d",
        builtin="checker",
        rgb1="0.24 0.29 0.34",
        rgb2="0.29 0.34 0.39",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="demo_floor",
        texture="demo_tiles",
        texrepeat="1.2 1.2",
        reflectance="0.05",
        texuniform="true",
    )
    ET.SubElement(
        asset,
        "texture",
        name="demo_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.18 0.25 0.33",
        rgb2="0.78 0.84 0.89",
        width="512",
        height="3072",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.38 0.38 0.38",
        diffuse="0.65 0.65 0.65",
        specular="0.15 0.15 0.15",
    )
    ET.SubElement(visual, "map", znear="0.02", zfar="30")
    world = root.find("worldbody")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="8 8 .05", material="demo_floor", condim="4"
    )
    ET.SubElement(
        world,
        "light",
        pos="1 -2 4",
        dir="0.2 0.5 -1",
        diffuse="0.8 0.8 0.8",
        directional="true",
        castshadow="true",
    )
    ET.SubElement(
        world,
        "light",
        pos="-2 3 3",
        dir="0 -0.3 -1",
        diffuse="0.4 0.45 0.5",
        directional="true",
        castshadow="false",
    )
    for station, y, color in (
        ("pickup", SOURCE_Y, "0.12 0.52 0.66 1"),
        ("delivery", DEST_Y, "0.12 0.64 0.45 1"),
    ):
        ET.SubElement(
            world,
            "geom",
            name=f"{station}_top",
            type="box",
            pos=numbers([STATION_X, y, 0.725]),
            size=".27 .62 .025",
            rgba="0.42 0.48 0.53 1",
            friction="0.8 .005 .0001",
        )
        ET.SubElement(
            world,
            "geom",
            type="box",
            pos=numbers([STATION_X, y, 0.686]),
            size=".275 .625 .013",
            rgba=color,
            contype="0",
            conaffinity="0",
        )
        for dx in (-0.21, 0.21):
            for dy in (-0.54, 0.54):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{station}_leg_{dx}_{dy}",
                    type="box",
                    pos=numbers([STATION_X + dx, y + dy, 0.345]),
                    size=".025 .025 .345",
                    rgba="0.15 0.20 0.25 1",
                )
        for side in SIDES:
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=numbers([PARCEL_X, y + Y_OFFSETS[side], 0.751]),
                size=".065 .065 .001",
                rgba=color,
                contype="0",
                conaffinity="0",
            )
    # Painted route markings have no contact geometry.
    for y in np.linspace(SOURCE_Y, DEST_Y, 9):
        ET.SubElement(
            world,
            "geom",
            type="box",
            pos=numbers([0.45, y, 0.0006]),
            size=".018 .045 .0005",
            rgba="0.69 0.80 0.83 1",
            contype="0",
            conaffinity="0",
        )
    initial = []
    for side, color in (("left", "0.96 0.29 0.12 1"), ("right", "0.12 0.54 0.98 1")):
        position = [PARCEL_X, SOURCE_Y + Y_OFFSETS[side], PARCEL_Z]
        parcel = ET.SubElement(world, "body", name=f"{side}_parcel", pos=numbers(position))
        ET.SubElement(parcel, "freejoint", name=f"{side}_parcel_free")
        ET.SubElement(
            parcel,
            "geom",
            name=f"{side}_parcel_geom",
            type="box",
            size=".018 .022 .04",
            mass=".025",
            rgba=color,
            friction="1 .005 .0001",
            condim="4",
            solref=".005 1",
        )
        # A small light label makes each free parcel easier to follow in the film.
        ET.SubElement(
            parcel,
            "geom",
            type="box",
            pos="-.0182 0 .012",
            size=".0002 .013 .014",
            rgba="0.95 0.96 0.94 1",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        initial.extend([*position, 1, 0, 0, 0])
    for key in root.find("keyframe"):
        key.set("qpos", key.get("qpos") + " " + numbers(initial))
    ET.indent(root)
    path = output / "demo-scene.mjcf"
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


class Demo:
    def __init__(self, path, fps):
        self.sim = YORMujoco(path, ik_model_path=DESCRIPTION / "sophia-ik.mjcf", headless=True)
        self.m, self.d = self.sim.model, self.sim.data
        self.fps, self.next_frame = fps, 0.0
        self.frames, self.times, self.labels, self.events = [], [], [], []
        self.phase = "Ready to collect"
        self.max_robot_penetration = 0.0
        self.max_table_penetration = 0.0
        self.max_carry_offset = {side: 0.0 for side in SIDES}
        self.reach_checks = []
        self.carrying = False
        self.parcels = {side: self.m.body(f"{side}_parcel").id for side in SIDES}
        self.robot_bodies = set(range(1, min(self.parcels.values())))
        self.finger_bodies = {
            side: {self.m.body(f"{side}_gripper_finger_{f}").id for f in "ab"} for side in SIDES
        }
        # SelfCollisionGuard's baseline scene has no free task objects. Ignore
        # parcels only in its private proximity model; all live physics contacts
        # remain enabled, including finger/parcel and robot/table interactions.
        guard = self.sim._collision_guard
        for g in range(self.m.ngeom):
            if int(self.m.geom_bodyid[g]) in self.parcels.values():
                guard.model.geom_contype[g] = guard.model.geom_conaffinity[g] = 0
        self.scratch = mujoco.MjData(self.m)
        for arm in self.sim.arms.values():
            arm.gripper_cmd = arm.gripper_goal = 1.0

    def base_pose(self):
        transform = self.sim._T_world_robot()
        center = transform[:2, 3] + transform[:2, :2] @ self.sim._chassis_center
        return np.r_[center, math.atan2(transform[1, 0], transform[0, 0])]

    def record_event(self, name):
        self.phase = name
        event = {
            "phase": name,
            "time_s": float(self.d.time),
            "base_pose": self.base_pose().tolist(),
            "parcel_positions": {side: self.d.xpos[b].tolist() for side, b in self.parcels.items()},
        }
        self.events.append(event)
        print(json.dumps(event), flush=True)

    def tick(self):
        self.sim._tick_controls(self.m.opt.timestep)
        if self.sim._guard_holding:
            raise RuntimeError(f"Self-contact guard held: {self.sim.get_motion_status()}")
        mujoco.mj_step(self.m, self.d)
        if self.d.warning.number.any() or not np.isfinite(self.d.qpos).all():
            raise RuntimeError("Physics warning or non-finite state")
        for contact in self.d.contact:
            a, b = [int(self.m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if a in self.robot_bodies and b in self.robot_bodies:
                if any({a, b} <= fingers for fingers in self.finger_bodies.values()):
                    continue
                self.max_robot_penetration = max(self.max_robot_penetration, -float(contact.dist))
            names = [self.m.geom(g).name or "" for g in (contact.geom1, contact.geom2)]
            if any(name.startswith(("pickup_", "delivery_")) for name in names) and (
                a in self.robot_bodies or b in self.robot_bodies
            ):
                self.max_table_penetration = max(self.max_table_penetration, -float(contact.dist))
        if self.carrying:
            for side, body in self.parcels.items():
                distance = np.linalg.norm(
                    self.d.xpos[body] - self.d.site_xpos[self.m.site(f"{side}_gripper_center").id]
                )
                self.max_carry_offset[side] = max(self.max_carry_offset[side], float(distance))
                if distance > 0.055:
                    raise RuntimeError(f"Dropped {side} parcel at {self.d.time}: offset {distance}")
        if self.d.time + 1e-8 >= self.next_frame:
            self.frames.append(self.d.qpos.copy())
            self.times.append(float(self.d.time))
            self.labels.append(self.phase)
            self.next_frame += 1 / self.fps

    def hold(self, seconds, label=None):
        if label:
            self.record_event(label)
        for _ in range(round(seconds / self.m.opt.timestep)):
            self.tick()

    def drive(self, x, y, yaw=0.0, label="Navigate", speed=0.22, timeout=20.0):
        self.record_event(label)
        start = self.d.time
        while self.d.time - start < timeout:
            current = self.base_pose()
            offset = np.array([x, y]) - current[:2]
            theta_error = (yaw - current[2] + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(offset) < 0.010 and abs(theta_error) < 0.012:
                break
            world_velocity = np.clip(1.8 * offset, -speed, speed)
            c, s = np.cos(current[2]), np.sin(current[2])
            self.sim.base_target[:] = [
                c * world_velocity[0] + s * world_velocity[1],
                -s * world_velocity[0] + c * world_velocity[1],
                np.clip(2 * theta_error, -0.45, 0.45),
            ]
            self.tick()
        else:
            raise RuntimeError(f"Base failed to arrive: {self.base_pose()} target {x, y, yaw}")
        self.sim.base_target[:] = 0
        self.hold(0.5)

    def ik(self, side, target, rotation=GRASP_ROTATION):
        m, d = self.m, self.scratch
        d.qpos[:] = self.d.qpos
        arm = self.sim.arms[side]
        joints = [m.joint(f"{side}_arm_joint{i}").id for i in range(1, 8)]
        dofs = m.jnt_dofadr[joints]
        limits = m.jnt_range[joints]
        site = m.site(f"{side}_gripper_center").id
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(500):
            mujoco.mj_forward(m, d)
            error_p = np.asarray(target) - d.site_xpos[site]
            error_r = Rotation.from_matrix(rotation @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(error_p) < 0.0003 and np.linalg.norm(error_r) < 0.003:
                return d.qpos[arm.qpos_adr].copy()
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            jac = np.vstack([jacp[:, dofs], 0.35 * jacr[:, dofs]])
            error = np.r_[error_p, 0.35 * error_r]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.00005 * np.eye(6), error)
            dq *= min(1.0, 0.08 / max(np.max(abs(dq)), 1e-9))
            d.qpos[arm.qpos_adr] = np.clip(
                d.qpos[arm.qpos_adr] + dq, limits[:, 0] + 0.001, limits[:, 1] - 0.001
            )
        raise RuntimeError(f"IK failed for {side} at {target}: pos {error_p}, rot {error_r}")

    def move_grasps(self, targets, seconds, label):
        self.record_event(label)
        commands = {side: np.array(target, dtype=float) for side, target in targets.items()}
        # IK alone does not account for the position servos' load-dependent
        # steady-state error. Close the loop on simulated endpoint feedback,
        # using ordinary guarded joint commands without changing gains/forces.
        for attempt in range(4):
            duration = seconds if attempt == 0 else 0.6
            goals = {
                side: (self.ik(side, target), None, duration) for side, target in commands.items()
            }
            if not self.sim._set_arm_targets(goals):
                raise RuntimeError(f"Arm plan rejected: {self.sim.get_motion_status()}")
            self.hold(duration + 0.5)
            errors = {
                side: target - self.d.site_xpos[self.m.site(f"{side}_gripper_center").id]
                for side, target in targets.items()
            }
            if max(np.linalg.norm(error) for error in errors.values()) <= 0.003:
                break
            for side, error in errors.items():
                commands[side] += error * min(1.0, 0.020 / max(np.linalg.norm(error), 1e-9))
        self.reach_checks.append(
            {
                "phase": label,
                "correction_steps": attempt,
                "endpoint_error_m": {side: float(np.linalg.norm(e)) for side, e in errors.items()},
            }
        )
        if any(np.linalg.norm(error) > 0.012 for error in errors.values()):
            raise RuntimeError(f"Arm tracking error after feedback: {self.reach_checks[-1]}")

    def relative_targets(self, x, z):
        cx, cy, yaw = self.base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        return {
            side: np.array([cx + c * x - s * offset, cy + s * x + c * offset, z])
            for side, offset in Y_OFFSETS.items()
        }

    def grips(self, opening, label):
        self.record_event(label)
        for arm in self.sim.arms.values():
            arm.gripper_goal = opening
        self.hold(1.2)

    def require_grasp(self):
        for side, body in self.parcels.items():
            touching = set()
            for contact in self.d.contact:
                a, b = [int(self.m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                if a == body:
                    touching.add(b)
                if b == body:
                    touching.add(a)
            if not self.finger_bodies[side] <= touching:
                raise RuntimeError(f"{side} parcel not held by both fingers: {touching}")
        self.carrying = True

    def run(self):
        self.hold(1.5, "01  /  Collect two parcels")
        self.drive(0.9, SOURCE_Y, label="02  /  Drive to the pickup station")
        self.move_grasps(self.relative_targets(0.40, 0.92), 2.2, "03  /  Align both grippers")
        goals = {
            side: np.array([PARCEL_X, SOURCE_Y + Y_OFFSETS[side], PARCEL_Z + 0.10])
            for side in SIDES
        }
        self.move_grasps(goals, 1.5, "03  /  Reach above the parcels")
        goals = {side: self.d.xpos[self.parcels[side]].copy() for side in SIDES}
        self.move_grasps(goals, 1.5, "04  /  Lower into grasp position")
        self.grips(0.0, "04  /  Close both grippers")
        self.require_grasp()
        goals = {side: target + np.array([0.0, 0.0, 0.16]) for side, target in goals.items()}
        self.move_grasps(goals, 1.8, "05  /  Lift the parcels")
        self.move_grasps(self.relative_targets(0.40, 0.92), 1.8, "05  /  Retract for transport")
        self.drive(0.43, SOURCE_Y, label="06  /  Back away from the station")
        self.drive(0.43, DEST_Y, label="07  /  Carry both parcels to delivery", speed=0.24)
        self.drive(0.9, DEST_Y, label="08  /  Dock at the delivery station")
        goals = {
            side: np.array([PARCEL_X, DEST_Y + Y_OFFSETS[side], PARCEL_Z + 0.12]) for side in SIDES
        }
        self.move_grasps(goals, 2.0, "09  /  Position parcels over the targets")
        # Preserve the measured in-grasp offset instead of assuming the object
        # stayed perfectly centered while under gravity.
        goals = {
            side: np.array([PARCEL_X, DEST_Y + Y_OFFSETS[side], PARCEL_Z + 0.003])
            + self.d.site_xpos[self.m.site(f"{side}_gripper_center").id]
            - self.d.xpos[self.parcels[side]]
            for side in SIDES
        }
        self.move_grasps(goals, 1.8, "10  /  Set both parcels down")
        self.carrying = False
        self.grips(1.0, "10  /  Release the parcels")
        goals = {side: target + np.array([0.0, 0.0, 0.14]) for side, target in goals.items()}
        self.move_grasps(goals, 1.6, "11  /  Clear the delivery station")
        self.move_grasps(self.relative_targets(0.35, 0.85), 1.6, "11  /  Retract both arms")
        self.drive(0.42, DEST_Y, label="12  /  Delivery complete")
        targets = {side: (HOME_Q[side], 1.0, 2.0) for side in SIDES}
        if not self.sim._set_arm_targets(targets):
            raise RuntimeError("Unable to return arms home")
        self.hold(2.5, "12  /  Two parcels delivered")
        delivered = {}
        for side, body in self.parcels.items():
            position = self.d.xpos[body]
            target = np.array([PARCEL_X, DEST_Y + Y_OFFSETS[side], PARCEL_Z])
            adr = self.m.joint(f"{side}_parcel_free").dofadr[0]
            speed = np.linalg.norm(self.d.qvel[adr : adr + 3])
            supported = any(
                {int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])} == {0, body}
                and any(self.m.geom(g).name == "delivery_top" for g in (c.geom1, c.geom2))
                for c in self.d.contact
            )
            passed = bool(
                np.linalg.norm(position[:2] - target[:2]) < 0.06
                and abs(position[2] - 0.79) < 0.015
                and speed < 0.02
                and supported
            )
            delivered[side] = {
                "passed": passed,
                "position_m": position.tolist(),
                "target_m": target.tolist(),
                "speed_mps": float(speed),
                "supported_by_delivery_table": supported,
            }
        success = (
            all(row["passed"] for row in delivered.values())
            and self.max_robot_penetration < 0.002
            and self.max_table_penetration < 0.001
        )
        report = {
            "success": success,
            "task": "Drive, grasp two free parcels, carry to another station, place and release.",
            "control": (
                "Scripted swerve targets and guarded joint-space paths from Cartesian IK "
                "with simulated endpoint feedback; "
                "not a learned policy."
            ),
            "object_transport": (
                "MuJoCo frictional gripper contacts only; "
                "no welds, attachments or applied object forces."
            ),
            "hardware_validated": False,
            "duration_s": float(self.d.time),
            "frames": len(self.frames),
            "fps": self.fps,
            "parcels": delivered,
            "max_carry_offset_m": self.max_carry_offset,
            "max_robot_self_penetration_m": self.max_robot_penetration,
            "max_robot_table_penetration_m": self.max_table_penetration,
            "mujoco_warnings": self.d.warning.number.tolist(),
            "reach_checks": self.reach_checks,
            "events": self.events,
        }
        if not success:
            raise RuntimeError(f"Delivery validation failed: {json.dumps(report)}")
        return report


def render_video(model, frames, labels, output, fps, width=1280, height=720):
    destination = output / "sophia_mobile_manipulation.mp4"
    log = (output / "ffmpeg.log").open("w")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    camera.lookat[:] = [0.85, 0.59, 0.75]
    camera.distance, camera.azimuth, camera.elevation = 4.45, 145.0, -24.0
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for i, (q, label) in enumerate(zip(frames, labels, strict=True)):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                chapter = int(label.split()[0])
                if chapter in (3, 4, 5, 9, 10, 11):
                    y = SOURCE_Y if chapter < 6 else DEST_Y
                    camera.lookat[:] = [1.17, y, 0.94]
                    camera.distance, camera.azimuth, camera.elevation = 2.65, 140.0, -19.0
                else:
                    camera.lookat[:] = [0.85, 0.59, 0.75]
                    camera.distance, camera.azimuth, camera.elevation = 4.45, 145.0, -24.0
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = renderer.render().copy()
                # Restrained titles make the action legible without covering it.
                cv2.rectangle(frame, (0, 0), (width, 76), (16, 24, 32), -1)
                cv2.putText(
                    frame,
                    "SOPHIA  /  MOBILE MANIPULATION",
                    (28, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.73,
                    (236, 242, 246),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "MuJoCo simulation  |  Scripted pick, carry & place",
                    (29, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (163, 186, 199),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(frame, (0, height - 58), (width, height), (16, 24, 32), -1)
                cv2.putText(
                    frame,
                    label,
                    (28, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.67,
                    (231, 240, 243),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{i / fps:04.1f} s",
                    (width - 118, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (163, 186, 199),
                    1,
                    cv2.LINE_AA,
                )
                progress = int(width * (i + 1) / len(frames))
                cv2.rectangle(frame, (0, height - 4), (progress, height - 1), (42, 198, 170), -1)
                process.stdin.write(frame.tobytes())
                if i in (0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1):
                    cv2.imwrite(
                        str(output / f"preview-{i:04d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    )
                if i % (fps * 5) == 0:
                    print(f"Rendered {i}/{len(frames)} frames", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"Video encoding failed; see {output / 'ffmpeg.log'}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/sophia-mobile-manipulation")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--scene-only", action="store_true", help="build the workcell for interactive VR; no replay"
    )
    args = parser.parse_args()
    if args.scene_only and (args.simulate_only or args.render_only):
        parser.error("--scene-only cannot be combined with replay/render modes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.scene_only:
        print(make_scene(output))
        return
    if args.render_only:
        model = mujoco.MjModel.from_xml_path(str(output / "demo-scene.mjcf"))
        trajectory = np.load(output / "trajectory.npz")
        report = json.loads((output / "task-report.json").read_text())
        if not report["success"]:
            raise RuntimeError("Refusing to render an unsuccessful task as a completed demo")
        video = render_video(model, trajectory["qpos"], trajectory["labels"], output, report["fps"])
    else:
        scene = make_scene(output)
        demo = Demo(scene, args.fps)
        try:
            report = demo.run()
            report["source_sha256"] = {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [DESCRIPTION / "sophia.mjcf", DESCRIPTION / "sophia.json", Path(__file__)]
            }
            (output / "task-report.json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(
                output / "trajectory.npz", qpos=demo.frames, times=demo.times, labels=demo.labels
            )
            print(
                f"TASK PASSED: {report['duration_s']:.2f} seconds, {report['frames']} frames",
                flush=True,
            )
            if args.simulate_only:
                return
            video = render_video(demo.m, demo.frames, demo.labels, output, args.fps)
        finally:
            demo.sim.stop()
    print(f"VIDEO: {video}", flush=True)


if __name__ == "__main__":
    main()
