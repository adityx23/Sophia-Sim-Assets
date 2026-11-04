"""Existing VR bridge adapter: accepted simulation targets -> SSH -> Pi YOR RPC."""

import contextlib
import copy
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from robot.sophia_live_common import SCHEMA, SIDES, fingerprint, measured
from robot.sophia_policy import joint_limits
from robot.sophia_profile import encoder_to_lift_extension, lift_extension_to_encoder, load_profile

ROOT = Path(__file__).resolve().parent.parent
INPUT_TIMEOUT_S = 0.20  # observed workcell scheduling stalls exceed 120 ms; Pi lease is 250 ms


def make_contract(sim, profile="yor"):
    mapping = load_profile()["release"]["lift_mapping"]
    pan = np.degrees(sim.model.jnt_range[sim.model.joint("head_pan_joint").id])
    tilt = np.degrees(sim.model.jnt_range[sim.model.joint("head_tilt_joint").id])
    zero = sim._neck_zero_deg
    limits = {
        "pan": (zero[0] - pan[1], zero[0] - pan[0]),
        "tilt": (zero[1] + tilt[0], zero[1] + tilt[1]),
    }
    contract = dict(
        schema=SCHEMA,
        period_s=1 / 30,
        lease_s=0.25,
        joint_speed_rad_s=0.4,
        gripper_speed_s=1.0,
        base_speed_m_s=0.05,
        yaw_speed_rad_s=0.15,
        neck_speed_deg_s=20.0,
        joint_limits={s: joint_limits(s).tolist() for s in SIDES},
        neck_limits_deg=[list(limits[k]) for k in ("pan", "tilt")],
        lift_encoder_bounds_m=[mapping["encoder_lower_m"], mapping["encoder_upper_m"]],
        lift_mapping=mapping,
        hardware_calibrated=False,
    )
    if profile == "yor":
        from robot.yor_mujoco import HOME_Q

        contract.update(
            motion_profile="yor",
            rate_check_policy="driver",
            arm_preview_s=1.0,
            joint_speed_rad_s=3.0,
            joint_acceleration_rad_s2=15.0,
            gripper_speed_s=None,
            base_speed_m_s=1.0,
            yaw_speed_rad_s=1.57,
            base_axis_limits=[1.0, 1.0, 1.57],
            base_acceleration=[1.0, 1.0, 1.57],
            neck_speed_deg_s=2000 * 360 / 4096,
            neck_servo_accel=80,
            arm_home={s: HOME_Q[s].tolist() for s in SIDES},
            neck_home_deg=[2.02, -3.25],
            lift_home_encoder_m=0.648,
        )
    elif profile != "commissioning":
        raise ValueError("Unknown motion profile: " + profile)
    return contract


class LineChannel:
    def __init__(self, process):
        self.process, self.buffer = process, b""

    def receive(self, timeout):
        end = time.monotonic() + timeout
        fd = self.process.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining = end - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("Pi response deadline expired")
            block = os.read(fd, 65536)
            if not block:
                raise ConnectionError("Pi endpoint closed")
            self.buffer += block
            if len(self.buffer) > 1048576:
                raise ValueError("Oversized Pi reply")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def send(self, message):
        data = json.dumps(message, allow_nan=False).encode() + b"\n"
        fd = self.process.stdin.fileno()
        os.set_blocking(fd, False)
        # Bounded messages fit in the pipe; never block the control/stop path.
        if len(data) > 4096 or os.write(fd, data) != len(data):
            raise ConnectionError("Command channel backpressure")


class FollowTransport:
    """One outstanding command; latest frame only, no backlog or reconnect/resume."""

    def __init__(self, argv, contract, journal):
        self.contract, self.journal = contract, journal
        self.stderr = (journal.directory / "pi-stderr.log").open("x")
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0
        )
        self.channel = LineChannel(self.process)
        self.lock = threading.Lock()
        self.latest = None
        try:
            self.response = self.channel.receive(15)
        except Exception:
            self.close_process()
            raise
        if self.response.get("schema") != SCHEMA or self.response.get(
            "contract_sha256"
        ) != fingerprint(contract):
            self.close_process()
            raise ValueError("Pi endpoint contract mismatch")
        self.fault = None
        self.response["acknowledged_source_seq"] = None
        self.last_reply = time.monotonic()
        self.session = str(uuid.uuid4())
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def submit(self, command, sample, source_seq, input_received_monotonic=None):
        with self.lock:
            self.latest = (
                time.monotonic() if input_received_monotonic is None else input_received_monotonic,
                copy.deepcopy(command),
                copy.deepcopy(sample),
                source_seq,
            )

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.response), self.fault, time.monotonic() - self.last_reply

    def run(self):
        seq = 0
        try:
            while not self.stopping.is_set():
                started = time.monotonic()
                with self.lock:
                    frame = self.latest
                    nonce = self.response["nonce"]
                fresh = frame is not None and started - frame[0] <= INPUT_TIMEOUT_S
                command = frame[1] if fresh and not self.fault else {"enabled": False}
                # The producer can still be solving against an ACTIVE reply when
                # this worker has already sent OFF. Never forward that motion as
                # a new enable; require a freshly aligned stationary handshake.
                resync_required = bool(
                    command["enabled"]
                    and command.get("requires_active", False)
                    and not self.response.get("active")
                )
                if resync_required:
                    command = {"enabled": False}
                request = dict(
                    schema=SCHEMA,
                    session=self.session,
                    seq=seq,
                    nonce=nonce,
                    contract_sha256=fingerprint(self.contract),
                    command=command,
                )
                self.journal.write(
                    "request",
                    request=request,
                    source_seq=None if frame is None else frame[3],
                    sim=None if frame is None else frame[2],
                    fresh=fresh,
                )
                self.channel.send(request)
                response = self.channel.receive(0.22)
                received = time.monotonic()
                if response.get("seq") != seq or response.get("contract_sha256") != fingerprint(
                    self.contract
                ):
                    raise ValueError("Pi reply identity mismatch")
                # Associate this reply with the exact submitted simulation target.
                # Acceptance is not evidence that the hardware reached the target.
                response["acknowledged_source_seq"] = (
                    frame[3]
                    if frame is not None
                    and command["enabled"]
                    and response.get("active")
                    and not response.get("fault")
                    else None
                )
                response["source_expired"] = bool(
                    frame is not None and frame[1]["enabled"] and not fresh
                )
                response["resync_required"] = resync_required
                self.journal.write(
                    "response",
                    seq=seq,
                    response=response,
                    send_monotonic=started,
                    receive_monotonic=received,
                    rtt_s=received - started,
                )
                with self.lock:
                    self.response, self.last_reply = response, received
                    if response.get("fault"):
                        self.fault = response["fault"]
                seq += 1
                self.stopping.wait(max(0, self.contract["period_s"] - (time.monotonic() - started)))
        except Exception as e:
            with self.lock:
                self.fault = str(e)
            self.journal.write("transport_fault", error=str(e))
        finally:
            # EOF makes the endpoint stop; its independent receive lease also expires.
            with contextlib.suppress(OSError):
                self.process.stdin.close()

    def close_process(self):
        with contextlib.suppress(OSError, ValueError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.stderr.close()

    def close(self):
        self.stopping.set()
        self.worker.join(timeout=1)
        self.close_process()


def initialize_sim_from_robot(sim, state, contract):
    """One-time simulator reset from measured feedback, before physics starts."""
    import mujoco

    measured(state, contract)
    if sim._initialized:
        raise ValueError("State alignment is only allowed before simulation starts")
    for side in SIDES:
        arm = sim.arms[side]
        q = np.array(state[side + "_joints"])
        grip = state[side + "_gripper"]
        sim.data.qpos[arm.qpos_adr] = q
        sim.data.ctrl[arm.act_ids] = q
        arm.q_cmd = q.copy()
        arm.q_goal = q.copy()
        arm.last_joint_target = q.copy()
        arm.gripper_cmd = arm.gripper_goal = grip
        for act, opened, closed in arm.gripper_acts:
            value = closed + grip * (opened - closed)
            sim.data.ctrl[act] = value
            joint = sim.model.actuator_trnid[act, 0]
            sim.data.qpos[sim.model.jnt_qposadr[joint]] = value
    h = float(
        encoder_to_lift_extension(np.clip(state["lift_height"], *contract["lift_encoder_bounds_m"]))
    )
    sim.data.qpos[sim._lift_adr] = sim.data.ctrl[sim._lift_act] = h
    sim.data.qpos[sim.model.joint("sophia_middle_slide").qposadr[0]] = h / 2
    sim.lift_setpoint = sim._lift_goal = h
    neck = np.deg2rad(np.array(state["neck_deg"]) - sim._neck_zero_deg) * [-1, 1]
    sim.data.qpos[[sim._pan_adr, sim._tilt_adr]] = neck
    sim.data.ctrl[[sim._pan_act, sim._tilt_act]] = neck
    sim._neck_goal = neck.copy()
    sim._neck_q_cmd = neck.copy()
    sim._neck_cmd = list(state["neck_deg"])
    mujoco.mj_forward(sim.model, sim.data)
    sim.init()


class LiveLink:
    """Drop-in link for vr-teleop Runner; its WebXR/controller code stays shared."""

    def __init__(self, sim, transport, contract, journal):
        self.sim, self.transport, self.contract, self.journal = sim, transport, contract, journal
        self.neck_limits = sim.get_neck_limits()
        self.neck_home = dict(zip(("pan", "tilt"), sim._neck_home_deg, strict=False))
        if contract.get("motion_profile") == "yor":
            sim._neck_rate[:] = np.deg2rad(contract["neck_speed_deg_s"])
            for arm in sim.arms.values():
                arm.command_acceleration_limit = contract["joint_acceleration_rad_s2"]
                arm.command_velocity = np.zeros(7)
        self.homing = False
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.latest = None
        self.fault = None
        self.enabled_previous = False
        self.target = {s: sim.arms[s].q_goal.copy() for s in SIDES}
        self.grip = {s: sim.arms[s].gripper_goal for s in SIDES}
        self.neck = np.array(sim._neck_deg())
        self.base = np.zeros(3)
        self.lift_direction = 0
        self.seq = 0
        self.pending = None
        self._alignment_sample = None
        self.input_paused = None
        self.home_phase = None
        self.home_event = None
        self.home_result = None
        self.home_started = None
        self._home_pulse_seen = False
        self.sample()  # cache for bridge/HUD callbacks; they must never wait for physics/IK

    def neck_limits_with_margin(self):
        return {k: (v[0] + 2, v[1] - 2) for k, v in self.neck_limits.items()}

    def submit(self, commands):
        with self.lock:
            self.latest = (time.monotonic(), copy.deepcopy(commands))
            if commands.get("home_all"):
                self.home_event = commands.get("input_received_monotonic") or self.latest[0]

    def robot_state(self):
        snapshot = self._cached_sample
        return dict(
            left_ee=copy.deepcopy(snapshot["left_ee"]),
            right_ee=copy.deepcopy(snapshot["right_ee"]),
            neck_deg=list(snapshot["neck_deg"]),
            lift_height=snapshot["lift_physical_extension_m"],
        )

    def sample(self):
        from robot.yor_mujoco import DRIVE_VEL_SCALE

        with self.sim._lock:
            out = self.sim.get_collection_state()
            # get_collection_state's legacy enc_vel is a target; replace with measured wheels.
            steer = self.sim.data.qpos[self.sim._steer_adr]
            wheel = self.sim.data.qvel[self.sim._drive_dof] / DRIVE_VEL_SCALE
            rows, values = [], []
            for (x, y), angle, speed in zip(self.sim._module_xy, steer, wheel, strict=False):
                rows += [[1, 0, -y], [0, 1, x]]
                values += [speed * np.cos(angle), speed * np.sin(angle)]
            out["enc_vel"] = np.linalg.lstsq(rows, values, rcond=None)[0].tolist()
            out["lift_height"] = float(lift_extension_to_encoder(self.sim.get_lift_height()))
            out["lift_physical_extension_m"] = self.sim.get_lift_height()
            out["lift_homed"], out["lift_ref"], out["lift_homing"] = True, "bottom_hard_stop", False
            out["lift_cmd"] = self.sim.lift_dir
            out["sim_time"] = float(self.sim.data.time)
            out["sample_monotonic"] = time.monotonic()
            out["source"] = "simulation"
            base = self.sim.model.body("base").id
            R = self.sim.data.xmat[base].reshape(3, 3)
            out["base_pose"] = [
                *self.sim.data.xpos[base][:2].tolist(),
                float(np.arctan2(R[1, 0], R[0, 0])),
            ]
            out["pose_source"] = "mujoco_ground_truth"
            if np.any(self.sim.data.warning.number):
                raise ValueError("MuJoCo warning during live follow")
        self._cached_sample = copy.deepcopy(out)
        return out

    def hud(self):
        reply, error, age = self.transport.snapshot()
        return dict(
            dry_run=False,
            rpc_ok=not (error or self.fault) and age < 0.25,
            error=self.fault or error or self.input_paused,
            homing=self.homing,
            home_result=self.home_result,
            follow_active=reply.get("active", False),
            follow_feedback_age_s=age,
            gripper={s: self.grip[s] for s in SIDES},
            base_vel=self.base.tolist(),
            lift={
                "height": self._cached_sample["lift_physical_extension_m"],
                "min": 0,
                "max": 0.5588,
            },
        )

    def monitor(self):
        reply, error, age = self.transport.snapshot()
        return dict(
            status=self.hud(),
            robot_feedback=reply.get("state"),
            sim=copy.deepcopy(self._cached_sample),
            transport_error=error,
            feedback_age_s=age,
        )

    def status(self):
        h = self.hud()
        return (
            "FOLLOW PAUSED: " + self.input_paused
            if self.input_paused and not self.fault
            else "FOLLOW FAULT: " + str(h["error"])
            if h["error"]
            else "HOMING"
            if self.homing
            else "FOLLOW ACTIVE"
            if h["follow_active"]
            else "FOLLOW OFF"
        )

    def hold_sim(self):
        with self.sim._lock:
            stationary_lift = self.sim.lift_dir == 0
            lift_hold = self.sim.lift_setpoint
            self.sim.set_base_velocity(np.zeros(3))
            self.sim.lift_stop()
            self.sim._hold_motion()
            # Stopping stationary actuators must not ratchet gravity sag into
            # a new lift target on every OFF/Home/pause transition.
            if stationary_lift:
                self.sim.lift_setpoint = self.sim._lift_goal = lift_hold
                self.sim.data.ctrl[self.sim._lift_act] = lift_hold
        self.base[:] = 0
        self.lift_direction = 0

    def stop_follow(self):
        # Send OFF before a possibly expensive simulator hold/feedback sample.
        self.transport.submit({"enabled": False}, self._cached_sample, self.seq)
        self.seq += 1
        if self.enabled_previous:
            self.hold_sim()
        self.enabled_previous = False
        self.pending = None

    def pause_input(self, reason, commands=None):
        if not self.input_paused:
            received = (commands or {}).get("input_received_monotonic")
            self.journal.write(
                "input_pause",
                reason=reason,
                input_age_s=None if received is None else time.monotonic() - received,
                homing=self.homing,
            )
        self.input_paused = reason + "; turn teleop OFF, then ON to resume"
        if self.homing:
            self.home_result = "Home cancelled: " + reason
        self.homing = False
        self.home_phase = None
        self.stop_follow()

    def home_reached(self, state):
        from robot.yor_mujoco import HOME_Q

        if (
            "lift_home_encoder_m" in self.contract
            and abs(state["lift_height"] - self.contract["lift_home_encoder_m"]) > 0.003
        ):
            return False
        for side in SIDES:
            if (
                np.max(abs(self.target[side] - HOME_Q[side])) > 1e-6
                or np.max(abs(np.array(state[side + "_joints"]) - HOME_Q[side])) > 0.05
                or np.max(abs(self.sim._measured_q(side) - HOME_Q[side])) > 0.06
                or abs(self.grip[side] - 1.0) > 1e-6
                or abs(state[side + "_gripper"] - 1.0) > 0.05
            ):
                return False
        return bool(
            np.max(abs(self.neck - self.sim._neck_home_deg)) < 1e-6
            and np.max(abs(np.array(state["neck_deg"]) - self.sim._neck_home_deg)) < 1.0
            and np.max(abs(np.array(self.sim._neck_deg()) - self.sim._neck_home_deg)) < 1.0
        )

    def tick(self, commands, fresh=True, home_requested=False):
        c, dt = self.contract, self.contract["period_s"]
        reply, remote_fault, age = self.transport.snapshot()
        fresh = fresh and (commands or {}).get("input_fresh", True)
        received = (commands or {}).get("input_received_monotonic")
        if received is not None:
            fresh = fresh and time.monotonic() - received <= INPUT_TIMEOUT_S
        # Expiry describes an older computed target, not the current headset
        # sample. If input is still fresh, discard that target and establish a
        # new stationary handshake from measured feedback. Actual input loss
        # below still pauses and requires operator OFF -> ON.
        if (
            fresh
            and not reply.get("active")
            and (
                reply.get("source_expired", False)
                or reply.get("resync_required", False)
                or self.pending is not None
                and self.pending[1].get("requires_active", False)
            )
        ):
            self.pending = None
            self.enabled_previous = False
        if remote_fault or age > c["lease_s"]:
            self.fault = remote_fault or "Robot feedback timeout"
        if self.fault:
            self.homing = False
            self.stop_follow()
            return
        if (self.enabled_previous or self.homing) and not fresh:
            self.pause_input("Headset/control stream became stale", commands)
            return
        if self.input_paused:
            if fresh and commands and not commands["enabled"] and not reply.get("active"):
                self.input_paused = None
                self.journal.write("input_rearmed", reason="fresh operator OFF and Pi stopped")
            else:
                self.stop_follow()
                return
        home_pulse = bool(commands and commands.get("home_all"))
        home_requested = home_requested or (home_pulse and not self._home_pulse_seen)
        self._home_pulse_seen = home_pulse
        if home_requested and fresh:
            if self.homing:
                self.pause_input("Home cancelled by operator", commands)
            else:
                self.homing, self.home_phase = True, "stopping"
                self.home_started = time.monotonic()
                self.home_result = "Returning arms and neck home"
                self.journal.write("home_started")
                self.stop_follow()
            return
        if self.homing:
            if commands and commands["enabled"]:
                self.pause_input("Home cancelled by operator", commands)
                return
            if time.monotonic() - self.home_started > 60:
                self.pause_input("Home timed out", commands)
                return
            if self.home_phase == "stopping":
                if reply.get("active"):
                    self.stop_follow()
                    return
                self.home_phase = "moving"
        active = bool(fresh and commands and (commands["enabled"] or self.homing))
        if active and not self.sim.get_motion_status()["accepted"]:
            raise ValueError("Simulation runtime guard rejected motion; restart the session")
        if not active:
            self.stop_follow()
            self.sample()
            return
        if self.pending is not None and reply.get("acknowledged_source_seq") != self.pending[0]:
            # Keep the same accepted sim target while its Pi reply is in flight.
            # A fresh headset frame may refresh the lease, but must not accumulate
            # additional motion steps that turn into a jump when Wi-Fi recovers.
            source_seq, command = self.pending
            self.transport.submit(command, self.sample(), source_seq, received)
            return
        if not self.enabled_previous:
            # First enable only presents the matched stationary pose to the Pi.
            state = measured(reply["state"], c)
            if c.get("motion_profile") == "yor":
                # Allow servo deceleration after OFF before taking a start pose.
                # A snapshot of a moving neck can become >2 degrees old in one RTT.
                pose = np.r_[state["neck_deg"], state["left_joints"], state["right_joints"]]
                previous = self._alignment_sample
                if previous is None or np.any(
                    abs(pose - previous[1]) > np.r_[[0.5] * 2, [0.02] * 14]
                ):
                    self._alignment_sample = (time.monotonic(), pose)
                    self.stop_follow()
                    return
                if time.monotonic() - previous[0] < 0.1:
                    self.stop_follow()
                    return
                self._alignment_sample = None
            self.target = {s: np.array(state[s + "_joints"]) for s in SIDES}
            self.grip = {s: state[s + "_gripper"] for s in SIDES}
            self.neck = np.array(state["neck_deg"])
            self.base[:] = 0
        elif reply.get("active") and self.homing:
            from robot.yor_mujoco import HOME_Q

            if self.home_reached(reply["state"]):
                self.homing, self.home_phase = False, None
                self.home_result = "Home complete"
                self.journal.write("home_completed", state=reply["state"])
                self.stop_follow()
                return
            for side in SIDES:
                if c.get("motion_profile") == "yor":
                    self.target[side] = np.array(c["arm_home"][side])
                    self.grip[side] = 1.0
                    continue
                self.target[side] += np.clip(
                    HOME_Q[side] - self.target[side],
                    -c["joint_speed_rad_s"] * dt,
                    c["joint_speed_rad_s"] * dt,
                )
                self.grip[side] += min(1.0 - self.grip[side], c["gripper_speed_s"] * dt)
            self.neck += np.clip(
                self.sim._neck_home_deg - self.neck,
                -c["neck_speed_deg_s"] * dt,
                c["neck_speed_deg_s"] * dt,
            )
            if c.get("motion_profile") == "yor":
                self.neck = np.array(c["neck_home_deg"])
            self.base[:] = 0
        elif reply.get("active"):
            for side in SIDES:
                arm = commands["arms"][side]
                if arm["target"] is not None:
                    import mink

                    q = self.sim._solve_ee_target(side, mink.SE3(np.array(arm["target"])))
                    if q is None:
                        raise ValueError(side + " IK rejected")
                    self.target[side] += np.clip(
                        q - self.target[side],
                        -c["joint_speed_rad_s"] * dt,
                        c["joint_speed_rad_s"] * dt,
                    )
                if arm.get("gripper") is not None:
                    if c.get("motion_profile") == "yor":
                        self.grip[side] = 1 - float(arm["gripper"])
                        continue
                    self.grip[side] += float(
                        np.clip(
                            1 - float(arm["gripper"]) - self.grip[side],
                            -c["gripper_speed_s"] * dt,
                            c["gripper_speed_s"] * dt,
                        )
                    )
            neck = commands.get("neck")
            if neck is not None:
                goal = np.array([neck["pan"], neck["tilt"]])
                self.neck += np.clip(
                    goal - self.neck, -c["neck_speed_deg_s"] * dt, c["neck_speed_deg_s"] * dt
                )
            goal = np.array(commands["base_velocity"] if commands["base_mode"] else [0.0, 0.0, 0.0])
            norm = np.linalg.norm(goal[:2])
            if "base_axis_limits" in c:
                goal = np.clip(goal, -np.array(c["base_axis_limits"]), c["base_axis_limits"])
            elif norm > c["base_speed_m_s"]:
                goal[:2] *= c["base_speed_m_s"] / norm
            goal[2] = np.clip(goal[2], -c["yaw_speed_rad_s"], c["yaw_speed_rad_s"])
            acceleration = np.array(c.get("base_acceleration", [0.1, 0.1, 0.3]))
            self.base += np.clip(goal - self.base, -acceleration * dt, acceleration * dt)
        direction = (
            {"up": 1, "down": -1, None: 0}[commands.get("lift_manual")]
            if self.enabled_previous and reply.get("active") and not self.homing
            else 0
        )
        if (
            self.homing
            and "lift_home_encoder_m" in c
            and self.enabled_previous
            and reply.get("active")
        ):
            error = c["lift_home_encoder_m"] - reply["state"]["lift_height"]
            direction = int(np.sign(error)) if abs(error) > 0.002 else 0
        with self.sim._lock:
            if not self.sim._set_arm_targets(
                {s: (self.target[s], self.grip[s], c.get("arm_preview_s", dt)) for s in SIDES}
            ):
                raise ValueError("Simulation collision/joint guard rejected arm motion")
            if not self.sim.set_neck_position(*self.neck)["accepted"]:
                raise ValueError("Simulation guard rejected neck motion")
            if direction != self.lift_direction:
                lift_result = {1: self.sim.lift_up, -1: self.sim.lift_down, 0: self.sim.lift_stop}[
                    direction
                ]()
                if lift_result is False:
                    raise ValueError("Simulation guard rejected lift motion")
                self.lift_direction = direction
            self.sim.set_base_velocity(self.base)
            if not self.sim.get_motion_status()["accepted"]:
                raise ValueError("Simulation runtime collision guard stopped motion")
            snapshot = self.sample()
        if received is not None and time.monotonic() - received > INPUT_TIMEOUT_S:
            self.pause_input("Control computation outlived headset sample", commands)
            return
        command = dict(
            enabled=True,
            requires_active=self.enabled_previous,
            base_velocity=self.base.tolist(),
            neck_deg=self.neck.tolist(),
            lift_direction=direction,
            lift_encoder_target_m=float(lift_extension_to_encoder(self.sim.lift_setpoint)),
            preview_time_s=c.get("arm_preview_s", dt),
        )
        for side in SIDES:
            command[side + "_joints"] = self.target[side].tolist()
            command[side + "_gripper"] = self.grip[side]
        self.journal.write(
            "sim_command", source_seq=self.seq, command=command, sim=snapshot, teleop=commands
        )
        self.pending = (self.seq, copy.deepcopy(command))
        self.transport.submit(command, snapshot, self.seq, received)
        self.seq += 1
        self.enabled_previous = True

    def run(self):
        try:
            while not self.stop.is_set():
                t0 = time.monotonic()
                with self.lock:
                    latest = copy.deepcopy(self.latest)
                    home_event, self.home_event = self.home_event, None
                try:
                    self.tick(
                        None if latest is None else latest[1],
                        latest is not None and t0 - latest[0] < INPUT_TIMEOUT_S,
                        home_requested=home_event is not None and t0 - home_event < INPUT_TIMEOUT_S,
                    )
                except Exception as e:
                    self.fault = str(e)
                    self.homing = False
                    self.stop_follow()
                    self.journal.write("sim_fault", error=str(e))
                self.stop.wait(max(0, self.contract["period_s"] - (time.monotonic() - t0)))
        finally:
            self.hold_sim()
            self.transport.submit({"enabled": False}, self.sample(), self.seq)
