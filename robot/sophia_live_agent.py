"""Pi endpoint over an authenticated SSH stdin/stdout channel. Starts disarmed.

This does not initialize/home hardware or change joystick.py. Only explicit
--allow-motion plus a fresh validated enable transition permits motion.
"""

import argparse
import hashlib
import json
import secrets
import select
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.sophia_live_common import (  # noqa: E402
    SCHEMA,
    SIDES,
    BoundedRPC,
    Journal,
    fingerprint,
    measured,
    validate_command,
)


def other_controllers(proc_root=Path("/proc")):
    """Known YOR command writers must be handed off before enabling the endpoint."""
    names = {
        "joystick.py",
        "whole_body_teleop.py",
        "yor_bridge.py",
        "vr_teleop.py",
        "yor_teleop.py",
        "joystick_policy_teleop.py",
        "joystick_policy_teleop_pi0fast.py",
        "policy_teleop.py",
        "nomad_teleop.py",
        "telestick.py",
    }
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / "cmdline").read_bytes().split(b"\0")
            matches = [Path(a.decode(errors="replace")).name for a in args]
            if names.intersection(matches):
                found.append(
                    dict(pid=int(directory.name), controller=sorted(names.intersection(matches)))
                )
        except (OSError, ValueError):
            continue
    return found


class Agent:
    def __init__(
        self, rpc, contract, journal, allow_motion=False, clock=time.monotonic, sleeper=time.sleep
    ):
        self.rpc, self.contract, self.journal = rpc, contract, journal
        self.allow_motion, self.clock = allow_motion, clock
        self.sleeper = sleeper
        self.dispatch_completed = {}
        self.contract_hash = fingerprint(contract)
        self.nonce, self.issued = secrets.token_hex(16), clock()
        self.last_seq, self.session, self.active, self.fault = -1, None, False, None
        self.saw_off, self.previous, self.last_active = False, None, clock()
        self.last_lift = 0
        self.state = None
        self.emergency_check = lambda: False
        self.enable_open_grippers = False

    def read_state(self):
        state = self.rpc.call("get_collection_state")
        if not isinstance(state, dict):
            raise ValueError("YOR get_collection_state did not return a dictionary")
        self.state = state
        return state

    def stop_motion(self, reason, latch=False):
        was_active = self.active
        self.active = False
        if latch:
            self.fault = reason
        outcomes = []
        if was_active:
            # Stop base/lift first. Gripper torque is left unchanged (no dropped object).
            operations = [("set_base_velocity", [np.zeros(3)]), ("lift_stop", [])]
            for name, args in operations:
                try:
                    self.rpc.call(name, *args)
                    outcomes.append({"method": name, "dispatched": True})
                except Exception as e:
                    outcomes.append({"method": name, "dispatched": False, "error": str(e)})
            operations = []
            try:
                current = self.read_state()
            except Exception:
                current = {}  # Never invent a measured arm hold from an old target.
            for side in SIDES:
                q = current.get(side + "_joints")
                if q is not None:
                    operations.append(("set_" + side + "_joint_target", [np.array(q), None, 0.1]))
            neck = current.get("neck_deg")
            if neck is not None:
                operations.append(("set_neck_position", list(neck)))
            for name, args in operations:
                try:
                    self.rpc.call(name, *args)
                    outcomes.append({"method": name, "dispatched": True})
                except Exception as e:
                    outcomes.append({"method": name, "dispatched": False, "error": str(e)})
            self.journal.write("stop", reason=reason, actions=outcomes)
        self.previous = None
        self.dispatch_completed.clear()
        self.last_lift = 0
        return outcomes

    def check_dispatch(self):
        if self.emergency_check():
            raise ValueError("Local joystick stop")
        if (self.journal.directory / "STOP").exists():
            raise ValueError("Local STOP file")
        now = self.clock()
        if now - self.issued > self.contract["lease_s"]:
            raise ValueError("Expired challenge; command too old")
        if self.active and now - self.last_active > self.contract["lease_s"]:
            raise ValueError("command_timeout")

    def pace_command(self, c, seq):
        """Absorb at most 50 ms of dispatch jitter, with the existing rate limits.

        Compare against the last RPC completion for each actuator, not packet
        arrival. Keep the original small rate tolerances; never expand them to
        accommodate networking delays. Large jumps still fault before dispatch.
        """
        if not self.active or self.contract.get("rate_check_policy") == "driver":
            return
        constraints = []
        for side in SIDES:
            method = "set_" + side + "_joint_target"
            constraints.extend(
                [
                    (
                        method,
                        np.max(
                            abs(np.array(c[side + "_joints"]) - self.previous[side + "_joints"])
                        ),
                        self.contract["joint_speed_rad_s"],
                        0.004,
                        "Joint target rate exceeded",
                    ),
                    (
                        method,
                        abs(c[side + "_gripper"] - self.previous[side + "_gripper"]),
                        self.contract["gripper_speed_s"],
                        0.01,
                        "Gripper target rate exceeded",
                    ),
                ]
            )
        constraints.append(
            (
                "set_neck_position",
                np.max(abs(np.array(c["neck_deg"]) - self.previous["neck_deg"])),
                self.contract["neck_speed_deg_s"],
                0.15,
                "Neck target rate exceeded",
            )
        )
        started = self.clock()
        deadline = started
        for method, delta, speed, tolerance, error in constraints:
            due = self.dispatch_completed[method] + max(0.0, (delta - tolerance) / speed)
            if due - started > 0.05:
                raise ValueError(error)
            deadline = max(deadline, due)
        self.check_dispatch()
        while True:
            remaining = deadline - self.clock()
            if remaining <= 1e-9:
                break
            self.sleeper(min(0.005, remaining))
            self.check_dispatch()
        if deadline > started:
            self.journal.write("rate_pacing", seq=seq, waited_s=self.clock() - started)

    def watchdog(self):
        if self.active and self.clock() - self.last_active > self.contract["lease_s"]:
            self.stop_motion("command_timeout", latch=True)

    def reply(self, seq=None, dispatched=None):
        self.nonce, self.issued = secrets.token_hex(16), self.clock()
        return dict(
            schema=SCHEMA,
            seq=seq,
            nonce=self.nonce,
            contract_sha256=self.contract_hash,
            active=self.active,
            fault=self.fault,
            state=self.state,
            pi_monotonic=self.clock(),
            allow_motion=self.allow_motion,
            dispatched=dispatched or [],
        )

    def handle(self, request):
        now = self.clock()
        self.watchdog()
        dispatched = []
        seq = request.get("seq")
        try:
            if (
                request.get("schema") != SCHEMA
                or request.get("contract_sha256") != self.contract_hash
            ):
                raise ValueError("Protocol/contract mismatch")
            if request.get("nonce") != self.nonce:
                raise ValueError("Stale/replayed challenge")
            if type(seq) is not int or seq <= self.last_seq:
                raise ValueError("Repeated/out-of-order sequence")
            if not isinstance(request.get("session"), str) or not request["session"]:
                raise ValueError("Missing session")
            if self.session not in (None, request["session"]):
                raise ValueError("Session changed; restart the endpoint")
            self.last_seq, self.session = seq, request["session"]
            c = validate_command(request["command"], self.contract)
            if not c["enabled"]:
                dispatched = self.stop_motion("operator_off")
                self.saw_off = True
                self.read_state()
                return self.reply(seq, dispatched)
            if now - self.issued > self.contract["lease_s"]:
                raise ValueError("Expired challenge; command too old")
            if not self.allow_motion:
                raise ValueError("Endpoint is read-only; --allow-motion was not supplied")
            if self.fault:
                raise ValueError("Latched fault: " + self.fault)
            if not self.saw_off:
                raise ValueError("Observe teleop OFF before enable")
            self.pace_command(c, seq)
            # Sample after pacing so tracking/start checks use fresh feedback.
            state = measured(self.read_state(), self.contract)
            enable_grippers = []
            for side in SIDES:
                if state.get(side + "_gripper_cmd") is None:
                    if (
                        not self.active
                        and self.enable_open_grippers
                        and state[side + "_gripper"] >= 0.99
                        and c[side + "_gripper"] >= 0.99
                    ):
                        enable_grippers.append(("enable_" + side + "_gripper", []))
                    else:
                        raise ValueError(
                            side + " disabled gripper must already be open before enable"
                        )
            if self.active:
                for side in SIDES:
                    target = state.get(side + "_joints_cmd")
                    if (
                        target is not None
                        and np.max(abs(np.array(target) - self.previous[side + "_joints"])) > 0.002
                    ):
                        raise ValueError("Another controller changed " + side + " arm command")
                base_target = state.get("base_cmd_vel")
                if (
                    base_target is not None
                    and np.max(abs(np.array(base_target) - self.previous["base_velocity"])) > 0.002
                ):
                    raise ValueError("Another controller changed base command")
            for side in SIDES:
                q = np.array(c[side + "_joints"])
                actual = np.array(state[side + "_joints"])
                tolerance = 0.05 if not self.active else 0.25
                if np.max(abs(q - actual)) > tolerance and (
                    not self.active or self.contract.get("rate_check_policy") != "driver"
                ):
                    raise ValueError(side + " arm start/tracking mismatch")
                if not self.active and abs(c[side + "_gripper"] - state[side + "_gripper"]) > 0.1:
                    raise ValueError("Gripper start mismatch")
            if (not self.active or self.contract.get("rate_check_policy") != "driver") and np.max(
                abs(np.array(c["neck_deg"]) - state["neck_deg"])
            ) > (2 if not self.active else 15):
                raise ValueError("Neck start/tracking mismatch")
            if abs(c["lift_encoder_target_m"] - state["lift_height"]) > (
                0.01 if not self.active else 0.05
            ):
                raise ValueError("Lift start/tracking mismatch")
            lo, hi = self.contract["lift_encoder_bounds_m"]
            direction = c["lift_direction"]
            if (
                direction < 0
                and state["lift_height"] < lo + 0.01
                or direction > 0
                and state["lift_height"] > hi - 0.01
            ):
                raise ValueError("Lift is near travel limit")
            if not self.active and (np.linalg.norm(c["base_velocity"]) > 1e-8 or direction != 0):
                raise ValueError("Enable with base and lift stationary")
            # Log the intent before sending any motor-affecting RPC.
            self.journal.write("command", seq=seq, session=self.session, command=c)
            self.check_dispatch()
            self.active = True  # partial dispatch must also trigger stop handling
            self.last_active = self.clock()
            operations = [("set_base_velocity", [np.array(c["base_velocity"])])] + enable_grippers
            for side in SIDES:
                operations.append(
                    (
                        "set_" + side + "_joint_target",
                        [np.array(c[side + "_joints"]), c[side + "_gripper"], c["preview_time_s"]],
                    )
                )
            operations.append(("set_neck_position", c["neck_deg"]))
            if direction != self.last_lift:
                operations.append(({0: "lift_stop", 1: "lift_up", -1: "lift_down"}[direction], []))
            for method, args in operations:
                self.check_dispatch()
                t0 = self.clock()
                result = self.rpc.call(method, *args)
                returned = self.clock()
                dispatched.append(
                    dict(
                        method=method,
                        time_s=t0,
                        returned_time_s=returned,
                        status="rejected"
                        if result is False
                        else "sent_unconfirmed"
                        if result is None
                        else "accepted",
                    )
                )
                if result is False:
                    raise ValueError(method + " rejected")
                self.dispatch_completed[method] = returned
            self.last_lift, self.previous = direction, c
            self.read_state()
            self.journal.write(
                "feedback", seq=seq, session=self.session, state=self.state, dispatched=dispatched
            )
        except Exception as e:
            self.stop_motion(str(e), latch=True)
            self.journal.write("fault", seq=seq, error=str(e), partial_dispatch=dispatched)
        return self.reply(seq, dispatched)


def serve(agent, input_fd, output, stop_requested=lambda: False):
    buffer = b""

    def emit(value):
        output.write(json.dumps(value, allow_nan=False) + "\n")
        output.flush()

    import os

    try:
        agent.read_state()
        emit(agent.reply())
        while not stop_requested():
            agent.watchdog()
            if (agent.journal.directory / "STOP").exists():
                break
            if not select.select([input_fd], [], [], 0.01)[0]:
                continue
            block = os.read(input_fd, 65536)
            if not block:
                break
            buffer += block
            if len(buffer) > 262144:
                raise ValueError("Oversized message")
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                emit(agent.handle(json.loads(line)))
    finally:
        agent.stop_motion("endpoint_exit", latch=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=5557)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--enable-open-grippers",
        action="store_true",
        help="On explicit enable, activate disabled AgileX jaws only if already open",
    )
    parser.add_argument(
        "--joystick-device", type=Path, help="Linux joystick: Back or stick movement cancels follow"
    )
    args = parser.parse_args()
    if args.allow_motion:
        writers = other_controllers()
        if writers:
            parser.error(
                "Stop the existing teleop writer before live follow: " + json.dumps(writers)
            )
    contract = json.loads(args.contract.read_text())
    rpc = BoundedRPC(port=args.rpc_port)
    journal = Journal(
        args.output,
        dict(
            schema=SCHEMA,
            role="pi",
            contract=contract,
            contract_sha256=fingerprint(contract),
            allow_motion=args.allow_motion,
            endpoint_source_sha256={
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("robot/sophia_live_agent.py", "robot/sophia_live_common.py")
            },
        ),
    )
    agent = Agent(rpc, contract, journal, args.allow_motion)
    agent.enable_open_grippers = args.enable_open_grippers
    import os
    import struct

    joystick_fd = None
    joystick_stopped = False
    if args.joystick_device:
        joystick_fd = os.open(args.joystick_device, os.O_RDONLY | os.O_NONBLOCK)

    def joystick_stop():
        nonlocal joystick_stopped
        if joystick_fd is not None and not joystick_stopped:
            try:
                data = os.read(joystick_fd, 8 * 64)
                if not data:
                    joystick_stopped = True
                for start in range(0, len(data), 8):
                    _, value, kind, number = struct.unpack("IhBB", data[start : start + 8])
                    kind &= 0x7F
                    if (
                        kind == 1
                        and number == 6
                        and value
                        or kind == 2
                        and number in (0, 1, 3, 4)
                        and abs(value) > 8192
                    ):
                        joystick_stopped = True
            except BlockingIOError:
                pass
            except OSError:
                joystick_stopped = True
        return joystick_stopped

    import signal

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    agent.emergency_check = lambda: stopping or joystick_stop()
    try:
        serve(agent, sys.stdin.fileno(), sys.stdout, lambda: stopping or joystick_stop())
    finally:
        if joystick_fd is not None:
            os.close(joystick_fd)
        rpc.close()
        journal.close()


if __name__ == "__main__":
    main()
