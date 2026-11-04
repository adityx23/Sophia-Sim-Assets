"""Small shared live-follow contract. No hardware initialization or MuJoCo imports."""

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
SCHEMA = "sophia.live.v1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"Missing/invalid {name}")
    return a


def finite(value, name):
    if value is None or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"Missing/invalid {name}")
    return float(value)


def measured(state, contract):
    """Missing feedback stays missing; commanded values never substitute for it."""
    for side in SIDES:
        q = vector(state.get(side + "_joints"), 7, side + " joints")
        lim = np.array(contract["joint_limits"][side])
        if np.any(q < lim[:, 0] - 0.02) or np.any(q > lim[:, 1] + 0.02):
            raise ValueError(side + " measured joints outside model limits")
        g = finite(state.get(side + "_gripper"), side + " gripper feedback")
        if not 0 <= g <= 1:
            raise ValueError("Gripper feedback outside normalized range")
    vector(state.get("neck_deg"), 2, "neck feedback")
    h = finite(state.get("lift_height"), "lift encoder")
    if state.get("lift_homed") is not True or state.get("lift_homing") is True:
        raise ValueError("Lift needs a valid homed reference; live follow never homes the robot")
    if state.get("lift_ref") != "bottom_hard_stop":
        raise ValueError("Unrecognized lift reference")
    lo, hi = contract["lift_encoder_bounds_m"]
    if not lo - 0.003 <= h <= hi + 0.003:
        raise ValueError("Lift encoder outside configured bounds")
    vector(state.get("enc_vel"), 3, "wheel-derived body velocity")
    finite(state.get("timestamp"), "robot snapshot timestamp")
    return state


def validate_command(c, contract):
    if type(c.get("enabled")) is not bool:
        raise ValueError("Explicit enabled flag required")
    if not c["enabled"]:
        return c
    for side in SIDES:
        q = vector(c.get(side + "_joints"), 7, side + " target")
        lim = np.array(contract["joint_limits"][side])
        if np.any(q < lim[:, 0]) or np.any(q > lim[:, 1]):
            raise ValueError("Joint target outside contract")
        g = finite(c.get(side + "_gripper"), side + " gripper")
        if not 0 <= g <= 1:
            raise ValueError("Gripper target outside [0,1]")
    v = vector(c.get("base_velocity"), 3, "base command")
    if "base_axis_limits" in contract:
        if np.any(abs(v) > np.array(contract["base_axis_limits"]) + 1e-8):
            raise ValueError("Base speed exceeds contract")
    elif (
        np.linalg.norm(v[:2]) > contract["base_speed_m_s"] + 1e-8
        or abs(v[2]) > contract["yaw_speed_rad_s"] + 1e-8
    ):
        raise ValueError("Base speed exceeds contract")
    neck = vector(c.get("neck_deg"), 2, "neck command")
    if np.any(neck < np.array(contract["neck_limits_deg"])[:, 0]) or np.any(
        neck > np.array(contract["neck_limits_deg"])[:, 1]
    ):
        raise ValueError("Neck target outside contract")
    if type(c.get("lift_direction")) is not int or c["lift_direction"] not in (-1, 0, 1):
        raise ValueError("Invalid lift direction")
    h = finite(c.get("lift_encoder_target_m"), "simulated lift encoder")
    if not contract["lift_encoder_bounds_m"][0] <= h <= contract["lift_encoder_bounds_m"][1]:
        raise ValueError("Lift target outside contract")
    if c.get("preview_time_s") != contract.get("arm_preview_s", contract["period_s"]):
        raise ValueError("Wrong interpolation period")
    return c


class Journal:
    """Exclusive per-session log. Flush before dispatch; incomplete lines are never evidence."""

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.stream = (self.directory / "events.jsonl").open("x", buffering=1)
        self.lock = threading.Lock()

    def write(self, kind, **fields):
        with self.lock:
            self.stream.write(
                json.dumps(
                    dict(kind=kind, wall_time=time.time(), monotonic=time.monotonic(), **fields),
                    allow_nan=False,
                )
                + "\n"
            )
            self.stream.flush()

    def close(self):
        self.stream.close()


class BoundedRPC:
    """One socket owned by one thread. A timeout closes it; next call gets a new socket."""

    def __init__(self, host="127.0.0.1", port=5557, timeout_ms=60):
        self.host, self.port, self.timeout = host, port, timeout_ms
        self.client = None

    def close(self):
        if self.client is not None:
            self.client.socket.close(linger=0)
            self.client.context.term()
            self.client = None

    def call(self, method, *args, **kwargs):
        import zmq
        from commlink import RPCClient

        if self.client is None:
            self.client = RPCClient(self.host, self.port)
            self.client.socket.setsockopt(zmq.LINGER, 0)
            self.client.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            self.client.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except Exception:
            self.close()
            raise
