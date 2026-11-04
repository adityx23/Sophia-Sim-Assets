"""
MuJoCo stand-in for robot/yor.py: serves the same RPC methods on the same port
(5557), so teleop clients (telestick, joystick, vr_teleop, yor_teleop, ...) run
against the simulation unchanged.

    python robot/yor_mujoco.py [--headless] [--port 5557] [--gripper agx|custom]

Physics model: yor-description/scene-full.mjcf (swerve base, lift, Nero arms,
grippers, pan/tilt head with ZED cameras); --gripper agx loads scene-full-agx.mjcf,
the same robot with the stock AgileX parallel-jaw grippers (make_agx_variant.py). Where a client can observe it, the
sim behaves like the real robot:
  - arms: ArmNode's SingleArmIK solver, home joint angles and joint speed limit.
    Legacy models use nero-welded-base-and-lift.mjcf; Sophia selects its generated
    companion IK model with measured mounts. EE poses use that model's world
    frame at zero lift, retaining ArmNode's lift-independent API convention.
  - grippers: 0 = closed, 1 = open (CanGripper convention); enable/disable like the
    SameHand (a disabled gripper ignores targets).
  - neck: yor.py's set_neck_position / get_neck_position / get_neck_limits /
    home_neck, in the robot's convention (degrees, + pan = head right, + tilt =
    head down). Sophia embeds calibrated angle offsets and home; legacy models
    use 0 = straight ahead and level.
  - base: [vx, vy, omega] in the robot frame, held until the next command (the
    real BaseController re-sends its last target), acceleration-limited like
    base_motor.py, driven through the 4 swerve modules of the model.
  - lift: up / down / stop at a fixed speed, like the Pico lift.
The calls vr-teleop's yor_bridge.py uses are listed in
vr-teleop/docs/YOR_RPC_INTERFACE.md. Sim-only: get_sim_pose, and get_lift_height
returns the real height.

Robot frame = the IK model's world frame: x forward, y left, z up, with its
origin 0.21 m to the right of the chassis centre (where the CAD put it).
"""
import argparse
import atexit
import functools
import math
import sys
import threading
import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from commlink import RPCServer  # noqa: E402

from robot.arm.ik_solver import SingleArmIK  # noqa: E402
from robot.sim_collision import SelfCollisionGuard  # noqa: E402
from robot.sim_camera import camera_intrinsics, stereo_render_size  # noqa: E402

YOR_PORT = 5557
SIM_MODEL = _HERE / "yor-description" / "scene-full.mjcf"
SIM_MODELS = {  # --gripper choice -> physics model
    "custom": SIM_MODEL,
    "agx": _HERE / "yor-description" / "scene-full-agx.mjcf",  # stock AgileX grippers
}
IK_MODEL = _HERE / "yor-description" / "nero-welded-base-and-lift.mjcf"  # same as yor.py

CONTROL_HZ = 100  # command updates per second; physics substeps fill the gap
VIEWER_HZ = 50

SIDES = ("left", "right")
# ArmNode.home_position (robot/arm/arm.py) -- keep in sync
HOME_Q = {
    "left": np.array([1.8317, 1.3773, -1.7357, 1.5211, -0.1130, -0.0954, 0.0973]),
    "right": np.array([-1.8391, 1.3285, 1.7432, 1.5277, -0.0006, 0.1011, 0.1097]),
}
JOINT_VEL_MAX = 3.0  # rad/s, ArmNode's config.joint_vel_max
HOME_DURATION_S = 2.0  # ArmNode.home() waits 2 s after reset_to_home()
IK_FAIL_HOLD_THRESHOLD = 10  # ArmNode: after this many failed solves in a row, hold instead of sending

# Gripper finger (joint1) angle at "closed". +q closes the finger onto the fixed
# jaw on both sides (output/grippers/build_gripper.py); 0 is open. Placeholder:
# the real servo travel is unknown, this is 95% of the collision-free range.
GRIPPER_CLOSED_FRACTION = 0.95
GRIPPER_SPEED = 2.0  # full open<->close strokes per second, placeholder

# base_motor.py: per-axis acceleration limit of the S-curve profile, and the
# raw drive-velocity unit (raw / DRIVE_VEL_SCALE = m/s)
BASE_ACCEL_MAX = np.array([1.9, 1.9, 6.5])
DRIVE_VEL_SCALE = 2.0
# Swerve modules in base_motor.py order [FL, FR, RR, RL]: (steer joint, drive joint)
MODULES = (
    ("front_left_steer", "drive_front_left"),
    ("front_right_steer", "drive_front_right"),
    ("back_right_steer", "drive_back_right"),
    ("back_left_steer", "drive_back_left"),
)

LIFT_SPEED_MPS = 0.05  # Pico lift speed is unknown -- placeholder
NECK_SPEED_RAD_S = 1.0  # simulation rate limit for guarded neck trajectories

# Simulated ZED stereo for the VR headset: rendered here, published locally, and
# streamed by vr-teleop's video server through its "zmq" camera source.
VIDEO_PUB_ADDR = "tcp://127.0.0.1:5560"
ZED_CAMERAS = ("head_zed_left", "head_zed_right")


def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            print(f"Warning: {func.__name__} called before YOR was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _pose_matrix(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _model_numeric(model, name, default):
    values = np.asarray(default, dtype=float)
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if index >= 0:
        start, size = model.numeric_adr[index], model.numeric_size[index]
        candidate = model.numeric_data[start:start + size]
        if candidate.shape != values.shape or not np.isfinite(candidate).all():
            raise ValueError(f"Invalid {name}")
        values = candidate
    return values.copy()


class _SimArm:
    """One Nero arm + gripper: ArmNode's IK, nerolib-like joint interpolation."""

    def __init__(self, side, model, ik_model_path):
        self.side = side
        joint_names = [f"{side}_arm_joint{i}" for i in range(1, 8)]
        self.ik = SingleArmIK(ik_model_path, joint_names=joint_names, ee_frame=f"{side}_arm_ee")
        self.ik_lock = threading.Lock()
        self.qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
        self.act_ids = np.array([model.actuator(n + "_pos").id for n in joint_names])

        # (actuator id, ctrl at open, ctrl at closed) per gripper actuator; the first
        # one's joint reads back the gripper position
        try:
            # AgileX parallel jaw (scene-full-agx.mjcf): two sliding fingers, 0 = closed
            fingers = [model.joint(f"{side}_gripper_finger_{f}") for f in "ab"]
            self.gripper_acts = [(model.actuator(j.name + "_pos").id, model.jnt_range[j.id][1], model.jnt_range[j.id][0]) for j in fingers]
            self.finger_adr = model.jnt_qposadr[fingers[0].id]
        except KeyError:
            # custom gripper (scene-full.mjcf): finger joint1 closes, knuckle joint2 held at 0
            finger = model.joint(f"{side}_gripper_joint1")
            self.gripper_acts = [
                (model.actuator(f"{side}_gripper_joint1_pos").id, 0.0, GRIPPER_CLOSED_FRACTION * model.jnt_range[finger.id][1]),
                (model.actuator(f"{side}_gripper_joint2_pos").id, 0.0, 0.0),
            ]
            self.finger_adr = model.jnt_qposadr[finger.id]
        _, self.finger_open, self.finger_closed = self.gripper_acts[0]

        self.q_cmd = HOME_Q[side].copy()  # interpolated command sent to the actuators
        self.q_goal = HOME_Q[side].copy()
        self.q_vel = np.zeros(7)
        self.gripper_cmd = 1.0
        self.gripper_goal = 1.0
        self.gripper_enabled = True
        rates = _model_numeric(model, "gripper_speed_strokes_s", [GRIPPER_SPEED] * 2)
        opening_rates = _model_numeric(model, "gripper_opening_speed_strokes_s", rates)
        if np.any(rates <= 0) or np.any(opening_rates <= 0):
            raise ValueError("Gripper speeds must be positive")
        self.gripper_speed = float(rates[SIDES.index(side)])
        self.gripper_opening_speed = float(opening_rates[SIDES.index(side)])
        self.last_joint_target = None
        self.last_ik_solved = None
        self.ik_fail_streak = 0

    def set_goal(self, q, duration):
        q = np.asarray(q, dtype=float)
        duration = max(float(duration), float(np.max(np.abs(q - self.q_cmd))) / JOINT_VEL_MAX, 1e-3)
        self.q_goal = q
        self.q_vel = np.abs(q - self.q_cmd) / duration

    def tick(self, data, dt):
        step = np.clip(self.q_goal - self.q_cmd, -self.q_vel * dt, self.q_vel * dt)
        if hasattr(self, "command_acceleration_limit"):
            desired = step / dt
            self.command_velocity += np.clip(
                desired - self.command_velocity,
                -self.command_acceleration_limit * dt,
                self.command_acceleration_limit * dt,
            )
            step = np.sign(step) * np.minimum(
                abs(step), np.maximum(0, self.command_velocity * np.sign(step)) * dt
            )
        self.q_cmd = self.q_cmd + step
        data.ctrl[self.act_ids] = self.q_cmd

        g_step = self.gripper_motion_speed(self.gripper_goal) * dt
        self.gripper_cmd += float(np.clip(self.gripper_goal - self.gripper_cmd, -g_step, g_step))
        for act, q_open, q_closed in self.gripper_acts:
            data.ctrl[act] = q_closed + self.gripper_cmd * (q_open - q_closed)

    def gripper_position(self, data):
        q = data.qpos[self.finger_adr]
        return float(np.clip((q - self.finger_closed) / (self.finger_open - self.finger_closed), 0.0, 1.0))

    def gripper_motion_speed(self, goal, start=None):
        start = self.gripper_cmd if start is None else start
        return self.gripper_opening_speed if goal > start else self.gripper_speed


class YORMujoco:
    def __init__(self, model_path=SIM_MODEL, ik_model_path=None, headless=False):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if ik_model_path is None:
            # A measured robot variant must use matching mounts for Cartesian
            # control. Legacy models continue to use the hardware's original IK.
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXT, "ik_model")
            if index < 0:
                ik_model_path = IK_MODEL
            else:
                start, size = self.model.text_adr[index], self.model.text_size[index]
                filename = self.model.text_data[start:start+size].rstrip(b"\0").decode("utf-8")
                ik_model_path = Path(model_path).resolve().parent / filename
        # Room for the ZED renders. Set before the viewer starts: viewer.sync() copies
        # the viewer's visual settings back into the model, so a later change is lost.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1920)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 1080)
        self.data = mujoco.MjData(self.model)
        self._lock = threading.RLock()  # guards self.data and every command below
        self._initialized = False
        self.no_arms = False

        m = self.model
        mujoco.mj_resetDataKeyframe(m, self.data, m.key("home").id)
        self.arms = {side: _SimArm(side, m, str(ik_model_path)) for side in SIDES}
        for arm in self.arms.values():
            self.data.qpos[arm.qpos_adr] = HOME_Q[arm.side]
            self.data.ctrl[arm.act_ids] = HOME_Q[arm.side]
        mujoco.mj_forward(m, self.data)

        # Robot frame = base body pose at home, so T_world_robot = T_world_base @ inv(T_base_home).
        self._base_id = m.body("base").id
        T_base_home = _pose_matrix(self.data.xpos[self._base_id], self.data.xmat[self._base_id])
        self._T_base_robot = np.linalg.inv(T_base_home)

        # Swerve module positions relative to the chassis centre, robot frame.
        self._steer_adr = np.array([m.jnt_qposadr[m.joint(s).id] for s, _ in MODULES])
        self._drive_dof = np.array([m.jnt_dofadr[m.joint(dr).id] for _, dr in MODULES])
        self._steer_act = np.array([m.actuator(s + "_pos").id for s, _ in MODULES])
        self._drive_act = np.array([m.actuator(dr + "_vel").id for _, dr in MODULES])
        self._steer_range = np.array([m.jnt_range[m.joint(s).id] for s, _ in MODULES])
        anchors = np.array([self.data.xanchor[m.joint(s).id][:2] for s, _ in MODULES])
        self._chassis_center = anchors.mean(axis=0)
        self._module_xy = anchors - self._chassis_center
        wheel_geom = m.geom("MK4_Colson_Hub_4_X_1_5_collision").id
        self._wheel_radius = float(m.geom_aabb[wheel_geom][4])  # half-extent across the tyre
        self.base_target = np.zeros(3)  # [vx, vy, omega], robot frame
        self.base_vel = np.zeros(3)  # after the acceleration limit

        self._lift_adr = m.jnt_qposadr[m.joint("lift_joint").id]
        self._lift_act = m.actuator("Lift").id
        self._lift_range = m.jnt_range[m.joint("lift_joint").id].copy()
        self.lift_setpoint = float(self.data.qpos[self._lift_adr])
        self.lift_dir = 0  # +1 up, -1 down, 0 hold
        self._lift_goal = self.lift_setpoint
        self._lift_request_id = 0
        self._lift_speed = _model_numeric(m, "lift_speed_mps", [LIFT_SPEED_MPS] * 2)
        if np.any(self._lift_speed <= 0):
            raise ValueError("Lift speeds must be positive")

        self._pan_adr = m.jnt_qposadr[m.joint("head_pan_joint").id]
        self._tilt_adr = m.jnt_qposadr[m.joint("head_tilt_joint").id]
        self._pan_act = m.actuator("head_pan_pos").id
        self._tilt_act = m.actuator("head_tilt_pos").id
        # Sophia embeds the hardware degree convention in the model. Older YOR
        # models keep their original zero-centered neck API and home pose.
        def neck_numeric(name, default=(0.0, 0.0)):
            index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
            if index < 0:
                return np.asarray(default, dtype=float).copy()
            start, size = m.numeric_adr[index], m.numeric_size[index]
            values = m.numeric_data[start:start + size].copy()
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain two finite values")
            return values

        self._neck_zero_deg = neck_numeric("neck_zero_deg")
        self._neck_home_deg = neck_numeric("neck_home_deg")
        self._neck_rate = neck_numeric("neck_rate_limit_rad_s", (NECK_SPEED_RAD_S, NECK_SPEED_RAD_S))
        if np.any(self._neck_rate <= 0):
            raise ValueError("neck_rate_limit_rad_s must contain positive rates")
        self._neck_cmd = None  # last commanded [pan, tilt], robot convention
        self._neck_goal = self.data.ctrl[[self._pan_act, self._tilt_act]].copy()
        self._neck_q_cmd = self._neck_goal.copy()
        index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "self_collision_margin_m")
        self._collision_guard = None if index < 0 else SelfCollisionGuard(m, m.numeric_data[m.numeric_adr[index]])
        self._motion_status = {"enabled": self._collision_guard is not None, "accepted": True,
                               "action": "initialization", "last_rejection": None}
        self._motion_stop_serial = 0
        self._guard_holding = False

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.viewer = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(m, self.data, show_left_ui=False, show_right_ui=False)

    # ------------------------------------------------------------- lifecycle
    def init(self):
        if self._initialized:
            print("Warning: YOR already initialized")
            return
        for arm in self.arms.values():
            arm.ik.init(HOME_Q[arm.side])
        self._thread.start()
        self._initialized = True

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.viewer is not None:
            self.viewer.close()

    def is_running(self):
        return not self._stop.is_set() and (self.viewer is None or self.viewer.is_running())

    def _physics_loop(self):
        timestep = self.model.opt.timestep
        max_substeps = max(1, round(0.1 / timestep))  # if we fall >100 ms behind, slow down instead
        rate = RateLimiter(CONTROL_HZ, name="yor_mujoco", warn=False)
        wall0, sim0 = time.monotonic(), self.data.time
        last_view = 0.0
        while not self._stop.is_set():
            # Step as much sim time as wall time has passed: sleep() on Windows only
            # wakes every ~15 ms, so a fixed step per tick would run slower than real time.
            with self._lock:
                behind = (time.monotonic() - wall0) - (self.data.time - sim0)
                substeps = int(np.clip(behind / timestep, 1, max_substeps))
                dt = substeps * timestep
                if self._collision_guard is None:
                    self._tick_controls(dt)
                    mujoco.mj_step(self.model, self.data, nstep=substeps)
                else:
                    # Recheck after every physics step, including catch-up ticks.
                    for _ in range(substeps):
                        self._tick_controls(timestep)
                        mujoco.mj_step(self.model, self.data)
                if substeps == max_substeps:
                    wall0, sim0 = time.monotonic(), self.data.time
                now = time.monotonic()
                if self.viewer is not None and now - last_view >= 1.0 / VIEWER_HZ:
                    self.viewer.sync()
                    last_view = now
            rate.sleep()

    def _motion_plan(self, arms=None, grippers=None, lift=None, neck=None):
        """Build the complete pending trajectory in the live physics frame."""
        goal = self.data.qpos.copy()
        speeds = np.ones(self.model.nq)
        for side, arm in self.arms.items():
            target, grip, duration = (arms or {}).get(side, (arm.q_goal, None, None))
            goal[arm.qpos_adr] = target
            if duration is None:
                speeds[arm.qpos_adr] = np.maximum(arm.q_vel, .05)
            else:
                distance = abs(np.asarray(target)-arm.q_cmd)
                duration = max(float(duration), float(distance.max())/JOINT_VEL_MAX, 1e-3)
                speeds[arm.qpos_adr] = np.maximum(distance/duration, .05)
            opening = arm.gripper_goal if grip is None else grip
            opening = (grippers or {}).get(side, opening)
            grip_rate = arm.gripper_motion_speed(opening, arm.gripper_position(self.data))
            for act, opened, closed in arm.gripper_acts:
                joint = self.model.actuator_trnid[act, 0]
                adr = self.model.jnt_qposadr[joint]
                goal[adr] = closed + opening*(opened-closed)
                speeds[adr] = max(abs(opened-closed)*grip_rate, 1e-6)
        if lift is None:
            lift = self.lift_setpoint if not self.lift_dir else self._lift_goal
        goal[self._lift_adr] = lift
        speeds[self._lift_adr] = self._lift_speed[0 if lift >= self.lift_setpoint else 1]
        goal[[self._pan_adr, self._tilt_adr]] = self._neck_goal if neck is None else neck
        speeds[[self._pan_adr, self._tilt_adr]] = self._neck_rate
        return goal, speeds

    def _preflight(self, action, **changes):
        if self._collision_guard is not None:
            goal, speeds = self._motion_plan(**changes)
            rejection = self._collision_guard.check(self.data.qpos, goal, speeds)
            if rejection is not None:
                self._reject_motion(action, rejection)
                return False
        self._motion_status.update(accepted=True, action=action)
        self._guard_holding = False
        return True

    def _reject_motion(self, action, reason):
        self._motion_status.update(accepted=False, action=action,
                                   last_rejection={"action": action, "sim_time_s": float(self.data.time), **reason})

    def _hold_motion(self):
        """Cancel queued manipulator motion; a blocked command is never retried."""
        if self._guard_holding:
            return
        self._guard_holding = True
        for arm in self.arms.values():
            arm.q_cmd = self.data.qpos[arm.qpos_adr].copy()
            arm.q_goal = arm.q_cmd.copy()
            arm.q_vel[:] = 0
            if hasattr(arm, "command_velocity"):
                arm.command_velocity[:] = 0
            self.data.ctrl[arm.act_ids] = arm.q_cmd
            arm.gripper_cmd = arm.gripper_goal = arm.gripper_position(self.data)
            for act, opened, closed in arm.gripper_acts:
                self.data.ctrl[act] = closed + arm.gripper_cmd*(opened-closed)
        self.lift_dir = 0
        self._lift_request_id += 1
        self.lift_setpoint = float(self.data.qpos[self._lift_adr])
        self._lift_goal = self.lift_setpoint
        self.data.ctrl[self._lift_act] = self.lift_setpoint
        self._neck_q_cmd = self.data.qpos[[self._pan_adr, self._tilt_adr]].copy()
        self._neck_goal = self._neck_q_cmd.copy()
        self.data.ctrl[[self._pan_act, self._tilt_act]] = self._neck_goal
        self._neck_cmd = self._neck_deg()
        self._motion_stop_serial += 1

    def _tick_controls(self, dt):
        for arm in self.arms.values():
            arm.tick(self.data, dt)
        self._tick_base(dt)
        self._tick_lift(dt)
        if self._collision_guard is None:
            return
        self._neck_q_cmd += np.clip(self._neck_goal-self._neck_q_cmd, -self._neck_rate*dt, self._neck_rate*dt)
        self.data.ctrl[[self._pan_act, self._tilt_act]] = self._neck_q_cmd
        candidate = self.data.qpos.copy()
        for act in range(self.model.nu):
            joint = self.model.actuator_trnid[act, 0]
            adr = self.model.jnt_qposadr[joint]
            if adr in self._collision_guard.addresses:
                candidate[adr] = self.data.ctrl[act]
        rejection = self._collision_guard.check(self.data.qpos, candidate)
        # Also inspect a short velocity projection to catch tracking lag while
        # simultaneous controls change. The source gains/braking remain uncalibrated.
        if rejection is None:
            projected = self.data.qpos.copy()
            mujoco.mj_integratePos(self.model, projected, self.data.qvel, .05)
            # Keep base/wheels at the current pose: this guard checks self-contact.
            candidate[:] = self.data.qpos
            candidate[self._collision_guard.addresses] = projected[self._collision_guard.addresses]
            rejection = self._collision_guard.check(self.data.qpos, candidate)
        if rejection is not None:
            self._reject_motion("runtime_hold", rejection)
            self._hold_motion()

    def get_motion_status(self):
        """Simulation-only diagnostics; a rejected target leaves prior goals intact."""
        with self._lock:
            return {**self._motion_status, "last_rejection": None if self._motion_status["last_rejection"] is None
                    else dict(self._motion_status["last_rejection"])}

    # ------------------------------------------------------------------ base
    def _tick_base(self, dt):
        step = BASE_ACCEL_MAX * dt
        self.base_vel = self.base_vel + np.clip(self.base_target - self.base_vel, -step, step)
        vx, vy, w = self.base_vel
        steer = _wrap(self.data.qpos[self._steer_adr])
        if np.allclose(self.base_vel, 0.0, atol=1e-4):
            self.data.ctrl[self._drive_act] = 0.0  # hold the wheels where they point
            return
        # Body velocity at each module: v + omega x r (base_motor.py's swerve IK).
        wx = vx - w * self._module_xy[:, 1]
        wy = vy + w * self._module_xy[:, 0]
        speed = np.hypot(wx, wy)
        angle = np.arctan2(wy, wx)
        err = _wrap(angle - steer)
        flip = np.abs(err) > np.pi / 2  # turn the short way and drive backwards instead
        angle = np.where(flip, _wrap(angle + np.pi), angle)
        speed = np.where(flip, -speed, speed)
        speed *= np.cos(_wrap(angle - steer))  # don't push while still steering
        target = self.data.qpos[self._steer_adr] + _wrap(angle - steer)
        self.data.ctrl[self._steer_act] = np.clip(target, self._steer_range[:, 0], self._steer_range[:, 1])
        self.data.ctrl[self._drive_act] = speed / self._wheel_radius

    def _T_world_robot(self):
        T_world_base = _pose_matrix(self.data.xpos[self._base_id], self.data.xmat[self._base_id])
        return T_world_base @ self._T_base_robot

    # ------------------------------------------------------------------ lift
    def _tick_lift(self, dt):
        if self.lift_dir:
            delta = np.clip(self._lift_goal-self.lift_setpoint,
                            -self._lift_speed[1]*dt, self._lift_speed[0]*dt)
            self.lift_setpoint = float(np.clip(self.lift_setpoint+delta, *self._lift_range))
        self.data.ctrl[self._lift_act] = self.lift_setpoint

    # =================================================== RPC: base (yor.py)
    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        with self._lock:
            self.base_target = np.asarray(velocity, dtype=float).reshape(3)

    @require_initialization
    def get_cmd_vel(self):
        with self._lock:
            return self.base_target.tolist(), time.time()

    @require_initialization
    def get_base_encoders(self) -> dict:
        with self._lock:
            steer = self.data.qpos[self._steer_adr]
            drive = self.data.qvel[self._drive_dof] * self._wheel_radius * DRIVE_VEL_SCALE
            return {
                "timestamp": time.time(),
                "steer_rad": steer.tolist(),
                "steer_deg": np.degrees(steer).tolist(),
                "steer_counts": [0.0] * 4,
                "drive_vel": drive.tolist(),
                "drive_counts": [0.0] * 4,
                "lift_height_m": float(self.data.qpos[self._lift_adr]),
            }

    @require_initialization
    def get_pose(self) -> dict:
        """Ground-truth pose of the chassis centre in the sim world (the real robot gets
        this from the ZED). The base turns about this point."""
        with self._lock:
            T = self._T_world_robot()
        x, y = T[:2, 3] + T[:2, :2] @ self._chassis_center
        return {"x": float(x), "y": float(y), "theta": float(math.atan2(T[1, 0], T[0, 0]))}

    def _not_simulated(self, name):
        print(f"[YORMujoco] {name} is not simulated")
        return None

    def follow_path(self, path=None):
        return self._not_simulated("follow_path")

    def move_to(self, goal=None):
        return self._not_simulated("move_to")

    def move_by(self, deltas=None):
        return self._not_simulated("move_by")

    def get_nav_debug(self):
        return self._not_simulated("get_nav_debug")

    # =================================================== RPC: lift (yor.py)
    @require_initialization
    def lift_up(self) -> bool:
        with self._lock:
            if not self._preflight("lift_up", lift=self._lift_range[1]):
                return False
            self.lift_dir = 1
            self._lift_goal = self._lift_range[1]
            self._lift_request_id += 1
            return True

    @require_initialization
    def lift_down(self) -> bool:
        with self._lock:
            if not self._preflight("lift_down", lift=self._lift_range[0]):
                return False
            self.lift_dir = -1
            self._lift_goal = self._lift_range[0]
            self._lift_request_id += 1
            return True

    @require_initialization
    def lift_stop(self) -> None:
        with self._lock:
            self.lift_dir = 0
            self._lift_request_id += 1
            self.lift_setpoint = float(self.data.qpos[self._lift_adr])
            self._lift_goal = self.lift_setpoint

    @require_initialization
    def lift_home(self) -> bool:
        return self.lift_to_height(0.0)

    @require_initialization
    def get_lift_height(self) -> float:
        with self._lock:
            return float(self.data.qpos[self._lift_adr])

    @require_initialization
    def lift_to_height(self, target_m, tolerance_m=0.002, timeout_s=30.0, min_height_m=0.0, max_height_m=None) -> bool:
        if not np.isfinite(target_m):
            raise ValueError("Lift target must be finite")
        if max_height_m is None:
            max_height_m = self._lift_range[1]
        target_m = float(np.clip(target_m, max(min_height_m, self._lift_range[0]), min(max_height_m, self._lift_range[1])))
        with self._lock:
            if not self._preflight("lift_to_height", lift=target_m):
                return False
            self._lift_goal = target_m
            self._lift_request_id += 1
            request_id = self._lift_request_id
            self.lift_dir = int(np.sign(target_m-self.data.qpos[self._lift_adr]))
            stop_serial = self._motion_stop_serial
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self._lock:
                if stop_serial != self._motion_stop_serial or request_id != self._lift_request_id:
                    return False
                h = float(self.data.qpos[self._lift_adr])
                if abs(target_m - h) <= tolerance_m:
                    self.lift_stop()
                    return True
                self.lift_dir = 1 if target_m > h else -1
            time.sleep(0.01)
        with self._lock:
            if request_id == self._lift_request_id:
                self.lift_stop()
        return False

    @require_initialization
    def lift_delta_height(self, delta_m, tolerance_m=0.002, timeout_s=30.0, min_height_m=0.0, max_height_m=None) -> bool:
        return self.lift_to_height(self.get_lift_height() + float(delta_m), tolerance_m, timeout_s, min_height_m,
                                   max_height_m)

    # =================================================== RPC: arms (yor.py / ArmNode)
    def _set_joint_target(self, side, joint_target, gripper_target, preview_time):
        return self._set_arm_targets({side: (joint_target, gripper_target, preview_time)})

    def _set_arm_targets(self, targets):
        prepared = {}
        for side, (q, grip, duration) in targets.items():
            q = np.asarray(q, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(duration) or duration <= 0:
                raise ValueError("Arm target needs seven finite angles and a positive duration")
            arm = self.arms[side]
            limits = self.model.actuator_ctrlrange[arm.act_ids]
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                with self._lock:
                    self._reject_motion("arm_target", {"reason": "joint_limit", "side": side})
                return False
            if grip is not None:
                if not np.isfinite(grip):
                    raise ValueError("Gripper target must be finite")
                grip = float(np.clip(grip, 0., 1.)) if arm.gripper_enabled else None
            prepared[side] = (q.copy(), grip, duration)
        with self._lock:
            if not self._preflight("arm_target", arms=prepared):
                return False
            for side, (q, grip, duration) in prepared.items():
                arm = self.arms[side]
                arm.set_goal(q, duration)
                arm.last_joint_target = q.copy()
                if grip is not None:
                    arm.gripper_goal = grip
            return True

    def _set_ee_target(self, side, ee_target: mink.SE3, gripper_target, preview_time):
        qd = self._solve_ee_target(side, ee_target)
        if qd is None:
            return False
        accepted = self._set_joint_target(side, qd, gripper_target, preview_time)
        if not accepted:
            self.arms[side].last_ik_solved = False
        return accepted

    def _solve_ee_target(self, side, ee_target):
        # ArmNode.set_ee_target: re-sync IK to the measured joints, solve, and after
        # IK_FAIL_HOLD_THRESHOLD failures in a row hold the last converged command.
        arm = self.arms[side]
        q = self._measured_q(side)
        with arm.ik_lock:
            arm.ik.update_configuration(q)
            qd, solved = arm.ik.solve_ik(ee_target)
        arm.last_ik_solved = bool(solved)
        arm.ik_fail_streak = 0 if solved else arm.ik_fail_streak + 1
        if arm.ik_fail_streak >= IK_FAIL_HOLD_THRESHOLD:
            return None
        return qd

    def _measured_q(self, side):
        with self._lock:
            return self.data.qpos[self.arms[side].qpos_adr].copy()

    def _get_ee_pose(self, side) -> mink.SE3:
        # ArmNode.get_ee_pose: load the measured joints into the IK model, then FK.
        q = self._measured_q(side)
        arm = self.arms[side]
        with arm.ik_lock:
            arm.ik.update_configuration(q)
            return arm.ik.forward_kinematics()

    def _home(self, side, gripper_target):
        # ArmNode.home: reset_to_home, gripper to gripper_target, wait 2 s, re-init IK
        with self._lock:
            if not self._set_joint_target(side, HOME_Q[side], gripper_target, HOME_DURATION_S):
                return False
            stop_serial = self._motion_stop_serial
        time.sleep(HOME_DURATION_S)
        arm = self.arms[side]
        q = self._measured_q(side)
        with arm.ik_lock:
            arm.ik.init(q)
        with self._lock:
            return stop_serial == self._motion_stop_serial and bool(np.max(abs(q-HOME_Q[side])) < .03)

    def _gripper_pose(self, side):
        with self._lock:
            return self.arms[side].gripper_position(self.data)

    def _set_gripper(self, side, value):
        if not np.isfinite(value):
            raise ValueError("Gripper target must be finite")
        with self._lock:
            if self.arms[side].gripper_enabled:
                value = float(np.clip(value, 0.0, 1.0))
                if not self._preflight("gripper_target", grippers={side: value}):
                    return False
                self.arms[side].gripper_goal = value
                return True
            return False

    def _enable_gripper(self, side, on):
        with self._lock:
            self.arms[side].gripper_enabled = bool(on)
        return True

    @require_initialization
    def set_left_joint_target(self, joint_target: np.ndarray, gripper_target: float | None = None,
                              preview_time: float = 1.0):
        return self._set_joint_target("left", joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.3):
        return self._set_ee_target("left", ee_target, gripper_target, preview_time)

    @require_initialization
    def set_left_gripper(self, gripper_pos: float):
        return self._set_gripper("left", gripper_pos)

    @require_initialization
    def set_left_gain(self, kp: np.ndarray, kd: np.ndarray):
        pass  # the sim's position actuators have fixed gains

    @require_initialization
    def home_left_arm(self, gripper_target: float = 1.0):
        return self._home("left", gripper_target)

    @require_initialization
    def enable_left_gripper(self) -> bool:
        return self._enable_gripper("left", True)

    @require_initialization
    def disable_left_gripper(self):
        self._enable_gripper("left", False)

    @require_initialization
    def tuck_left_arm(self):
        return self._set_joint_target("left", np.zeros(7), 1.0, 2.0)

    @require_initialization
    def open_left_gripper(self):
        return self._set_gripper("left", 1.0)

    @require_initialization
    def close_left_gripper(self):
        return self._set_gripper("left", 0.0)

    @require_initialization
    def get_left_ee_pose(self) -> mink.SE3:
        return self._get_ee_pose("left")

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        return self._measured_q("left")

    @require_initialization
    def get_left_gripper_pose(self):
        return self._gripper_pose("left")

    @require_initialization
    def set_right_joint_target(self, joint_target: np.ndarray, gripper_target: float | None = None,
                               preview_time: float = 1.0):
        return self._set_joint_target("right", joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.3):
        return self._set_ee_target("right", ee_target, gripper_target, preview_time)

    @require_initialization
    def set_right_gripper(self, gripper_pos: float):
        return self._set_gripper("right", gripper_pos)

    @require_initialization
    def set_right_gain(self, kp: np.ndarray, kd: np.ndarray):
        pass

    @require_initialization
    def home_right_arm(self, gripper_target: float = 1.0):
        return self._home("right", gripper_target)

    @require_initialization
    def enable_right_gripper(self) -> bool:
        return self._enable_gripper("right", True)

    @require_initialization
    def disable_right_gripper(self):
        self._enable_gripper("right", False)

    @require_initialization
    def tuck_right_arm(self):
        return self._set_joint_target("right", np.zeros(7), 1.0, 2.0)

    @require_initialization
    def open_right_gripper(self):
        return self._set_gripper("right", 1.0)

    @require_initialization
    def close_right_gripper(self):
        return self._set_gripper("right", 0.0)

    @require_initialization
    def get_right_ee_pose(self) -> mink.SE3:
        return self._get_ee_pose("right")

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        return self._measured_q("right")

    @require_initialization
    def get_right_gripper_pose(self):
        return self._gripper_pose("right")

    @require_initialization
    def get_arm_relative_pose(self) -> tuple[mink.SE3, mink.SE3]:
        left_ee_pose = self.get_left_ee_pose()
        right_ee_pose = self.get_right_ee_pose()
        l2r = right_ee_pose.inverse() @ left_ee_pose
        r2l = left_ee_pose.inverse() @ right_ee_pose
        return r2l, l2r

    @require_initialization
    def get_bimanual_state(self) -> list:
        """Same row layout as yor.py: [t, L_ee(7), L_q(7), L_grip, R_ee(7), R_q(7), R_grip, lift].
        Unlike the real robot (no lift feedback, 0.0), the lift entry is the sim height."""
        row = [0.0] * (1 + 7 + 7 + 1 + 7 + 7 + 1 + 1)
        row[0] = time.time()
        row[1:8] = self.get_left_ee_pose().wxyz_xyz.tolist()
        row[8:15] = self.get_left_joint_positions().tolist()
        row[15] = self.get_left_gripper_pose()
        row[16:23] = self.get_right_ee_pose().wxyz_xyz.tolist()
        row[23:30] = self.get_right_joint_positions().tolist()
        row[30] = self.get_right_gripper_pose()
        row[31] = self.get_lift_height()
        return row

    @require_initialization
    def set_bimanual_ee_target(self, L_ee_target: mink.SE3, R_ee_target: mink.SE3,
                               L_gripper_target: float | None = None, L_preview_time: float = 0.1,
                               R_gripper_target: float | None = None, R_preview_time: float = 0.1):
        left = self._solve_ee_target("left", L_ee_target)
        right = self._solve_ee_target("right", R_ee_target)
        if left is None or right is None:
            return False
        accepted = self._set_arm_targets({"left": (left, L_gripper_target, L_preview_time),
                                          "right": (right, R_gripper_target, R_preview_time)})
        if not accepted:
            for arm in self.arms.values():
                arm.last_ik_solved = False
        return accepted

    @require_initialization
    def get_ik_status(self):
        return tuple(self.arms[side].last_ik_solved for side in SIDES)

    @require_initialization
    def get_collection_state(self) -> dict:
        """yor.py's one-call snapshot (vr-teleop's HUD reads it)."""
        with self._lock:
            state = {"timestamp": time.time(), "base_cmd_vel": self.base_target.tolist(),
                     "enc_vel": self.base_vel.tolist(), "neck_deg": self._neck_deg(),
                     "neck_cmd_deg": None if self._neck_cmd is None else list(self._neck_cmd),
                     "motion_status": self.get_motion_status()}
        for side in SIDES:
            arm = self.arms[side]
            state[f"{side}_ee"] = self._get_ee_pose(side).wxyz_xyz.tolist()
            state[f"{side}_joints"] = self._measured_q(side).tolist()
            state[f"{side}_joints_cmd"] = None if arm.last_joint_target is None else arm.last_joint_target.tolist()
            state[f"{side}_gripper"] = self._gripper_pose(side)
            state[f"{side}_gripper_cmd"] = arm.gripper_goal
            state[f"{side}_ik_ok"] = arm.last_ik_solved
        return state

    @require_initialization
    def set_lift_position(self, target_m) -> bool:
        height = float(target_m) if np.isscalar(target_m) else float(np.asarray(target_m).ravel()[0])
        return self.lift_to_height(height)

    # =================================================== RPC: neck (yor.py)
    # Robot convention: degrees, + pan = head right, + tilt = head down. In the model
    # + head_pan_joint turns the camera left and + head_tilt_joint turns it down.
    # Calibrated models add the hardware angles corresponding to geometric zero.
    def _neck_deg(self):
        return (self._neck_zero_deg + [-math.degrees(self.data.qpos[self._pan_adr]),
                                      math.degrees(self.data.qpos[self._tilt_adr])]).tolist()

    @require_initialization
    def set_neck_position(self, pan: float | None = None, tilt: float | None = None):
        if any(value is not None and not np.isfinite(value) for value in (pan, tilt)):
            raise ValueError("Neck target must be finite")
        lim = self.get_neck_limits()
        with self._lock:
            cmd = list(self._neck_cmd or self._neck_deg())
            if pan is not None:
                cmd[0] = float(np.clip(pan, *lim["pan"]))
            if tilt is not None:
                cmd[1] = float(np.clip(tilt, *lim["tilt"]))
            if not np.isfinite(cmd).all():
                raise ValueError("Neck target must be finite")
            goal = np.deg2rad(np.asarray(cmd)-self._neck_zero_deg)*[-1, 1]
            if not self._preflight("neck_target", neck=goal):
                old = self._neck_cmd or self._neck_deg()
                return {"pan": old[0], "tilt": old[1], "accepted": False}
            self._neck_cmd = cmd
            self._neck_goal = goal
            if self._collision_guard is None:
                self.data.ctrl[[self._pan_act, self._tilt_act]] = goal
        return {"pan": cmd[0], "tilt": cmd[1], "accepted": True}

    @require_initialization
    def get_neck_position(self):
        with self._lock:
            pan, tilt = self._neck_deg()
        return {"pan": pan, "tilt": tilt}

    @require_initialization
    def get_neck_limits(self):
        m = self.model
        pan_lo, pan_hi = np.degrees(m.jnt_range[m.joint("head_pan_joint").id])
        tilt_lo, tilt_hi = np.degrees(m.jnt_range[m.joint("head_tilt_joint").id])
        pan_zero, tilt_zero = self._neck_zero_deg
        return {"pan": (float(pan_zero-pan_hi), float(pan_zero-pan_lo)),
                "tilt": (float(tilt_zero+tilt_lo), float(tilt_zero+tilt_hi))}

    @require_initialization
    def home_neck(self) -> bool:
        return self.set_neck_position(*self._neck_home_deg)["accepted"]

    @require_initialization
    def move_to_reference_pose(self, timeout_s=30.0) -> bool:
        """Explicit sequence: bent arms, lift >= 300 mm, then hanging arms.

        Each leg passes the same guard as ordinary commands. No unrelated
        command implicitly moves the lift or the other arm to create clearance.
        """
        if self._collision_guard is None:
            return False
        deadline = time.monotonic() + timeout_s

        def wait_arms(targets):
            while time.monotonic() < deadline:
                with self._lock:
                    if self._motion_stop_serial != stop_serial:
                        return False
                    for side, (q, grip, _) in targets.items():
                        arm = self.arms[side]
                        if not np.array_equal(arm.q_goal, q) or (grip is not None and arm.gripper_goal != grip):
                            return False  # another caller superseded this sequence
                    if all(np.max(abs(self.data.qpos[self.arms[s].qpos_adr]-q)) < .03
                           and (grip is None or abs(self.arms[s].gripper_position(self.data)-grip) < .05)
                           for s, (q, grip, _) in targets.items()):
                        return True
                time.sleep(.01)
            with self._lock:
                self._hold_motion()
            return False

        with self._lock:
            stop_serial = self._motion_stop_serial
            bent = {s: (HOME_Q[s], None, 2.) for s in SIDES}
            if not self._set_arm_targets(bent):
                return False
        if not wait_arms(bent):
            return False
        clearance = float(self.model.numeric("reference_clearance_lift_m").data[0])
        if self.get_lift_height() < clearance-.003:
            if not self.lift_to_height(clearance, timeout_s=max(0., deadline-time.monotonic())):
                return False
        with self._lock:
            if any(not np.array_equal(a.q_goal, HOME_Q[s]) for s, a in self.arms.items()):
                return False
            q = self.model.key("reference").qpos
            hanging = {s: (q[a.qpos_adr].copy(), 1., 2.) for s, a in self.arms.items()}
            if not self._set_arm_targets(hanging):
                return False
        return wait_arms(hanging)

    @require_initialization
    def set_neck_torque(self, on: bool):
        pass  # the sim's neck actuators can't go limp

    # =================================================== RPC: sim-only
    @require_initialization
    def get_sim_pose(self) -> list:
        """Robot frame in the sim world, wxyz_xyz."""
        with self._lock:
            T = self._T_world_robot()
        return mink.SE3.from_matrix(T).wxyz_xyz.tolist()


class StereoPublisher:
    """Renders the head ZED pair and publishes [topic, jpeg (left|right side by side),
    stereo_info json] over ZMQ PUB -- the frames the real ZED would give the Orin."""

    def __init__(self, sim: YORMujoco, address=VIDEO_PUB_ADDR, width=960, height=540, fps=30, jpeg_quality=75):
        self.sim, self.address = sim, address
        self.width, self.height, self.fps, self.quality = width, height, fps, jpeg_quality
        m = sim.model
        cam_l, cam_r = (m.camera(name).id for name in ZED_CAMERAS)
        self.render_width, self.render_height = stereo_render_size(m, ZED_CAMERAS, width, height)
        eyes = {side: camera_intrinsics(m, name, self.render_width, self.render_height, width, height)
                for side, name in zip(SIDES, ZED_CAMERAS)}
        calibrated = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "head_camera_profile") >= 0
        self.stereo_info = {
            "type": "stereo_info", **eyes["left"], "eyes": eyes,
            "image_model": "rectified_pinhole", "distortion": [0., 0., 0., 0., 0.],
            "intrinsics_source": "Orin 2026-10-30 / zed.rectified" if calibrated else "legacy fovy",
            "camera_to_robot_mount_calibrated": False,
            "resize_convention": "full frame; output_pixel + 0.5 = scale * (source_pixel + 0.5)",
            "baseline_m": float(np.linalg.norm(m.cam_pos[cam_l] - m.cam_pos[cam_r])), "model": "ZED 2 (sim)",
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        import json

        import cv2
        import zmq

        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.bind(self.address)
        info = json.dumps(self.stereo_info).encode()
        m = self.sim.model
        scratch = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, self.render_height, self.render_width)
        render_option = mujoco.MjvOption()
        render_option.geomgroup[3] = 0  # collision proxies must not appear in camera images
        rate = RateLimiter(self.fps, name="zed_sim", warn=False)
        print(f"[YORMujoco] ZED stereo {self.width}x{self.height}/eye @ {self.fps} fps on {self.address}")
        try:
            while not self._stop.is_set():
                with self.sim._lock:
                    mujoco.mj_copyData(scratch, m, self.sim.data)  # both eyes from one instant
                eyes = []
                for cam in ZED_CAMERAS:
                    renderer.update_scene(scratch, camera=cam, scene_option=render_option)
                    eye = renderer.render().copy()
                    if (self.render_width, self.render_height) != (self.width, self.height):
                        eye = cv2.resize(eye, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    eyes.append(eye)
                frame = cv2.cvtColor(np.hstack(eyes), cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    sock.send_multipart([b"head_stereo", jpeg.tobytes(), info])
                    self.ready.set()
                rate.sleep()
        finally:
            renderer.close()
            sock.close(linger=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument("--port", type=int, default=YOR_PORT)
    parser.add_argument("--gripper", choices=sorted(SIM_MODELS), default="agx",
                        help="custom: the new grippers (scene-full.mjcf); agx: stock AgileX (scene-full-agx.mjcf)")
    parser.add_argument("--model", default=None, help="physics MJCF (overrides --gripper)")
    parser.add_argument("--no-video", action="store_true", help=f"don't publish ZED frames on {VIDEO_PUB_ADDR}")
    args = parser.parse_args()

    yor = YORMujoco(model_path=args.model or SIM_MODELS[args.gripper], headless=args.headless)
    yor.init()
    video = None if args.no_video else StereoPublisher(yor)
    if video:
        video.start()
    server = RPCServer(yor, port=args.port, threaded=True)

    def graceful_shutdown():
        print("\nRPC Server stopping...")
        server.stop()
        if video:
            video.stop()
        yor.stop()

    atexit.register(graceful_shutdown)
    server.start()
    print(f"[YORMujoco] RPC server on port {args.port}")
    try:
        while yor.is_running():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
