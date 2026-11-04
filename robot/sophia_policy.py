"""Versioned single-arm policy contract shared by simulation and robot adapters.

No hardware imports, connections, homing or background threads. Commands use
joint space, so neither backend's Cartesian IK frame enters this contract.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

DESCRIPTION = Path(__file__).resolve().parent / "yor-description"


def joint_limits(side):
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    root = ET.parse(DESCRIPTION / "sophia.mjcf").getroot()
    return np.array(
        [
            [
                float(v)
                for v in root.find(f".//joint[@name='{side}_arm_joint{i}']")
                .get("range")
                .split()
            ]
            for i in range(1, 8)
        ]
    )


def _vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


@dataclass(frozen=True)
class PolicyContract:
    side: str = "left"
    control_hz: float = 30.0
    joint_speed_rad_s: float = 0.6
    image_width: int = 160
    image_height: int = 90

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not np.isfinite([self.control_hz, self.joint_speed_rad_s]).all():
            raise ValueError("Control frequency and speed must be finite")
        if not 1 <= self.control_hz <= 100 or not 0 < self.joint_speed_rad_s <= 1:
            raise ValueError("Require 1..100 Hz and 0 < joint speed <= 1 rad/s")
        if any(
            type(v) is not int or v < 16 for v in (self.image_width, self.image_height)
        ):
            raise ValueError("Image dimensions must be integers >= 16")

    @property
    def period_s(self):
        return 1.0 / self.control_hz

    @property
    def delta_rad(self):
        return self.joint_speed_rad_s * self.period_s

    def describe(self):
        return {
            "version": "sophia-single-arm-v1",
            "side": self.side,
            "joint_order": [f"{self.side}_arm_joint{i}" for i in range(1, 8)],
            "joint_limits_rad": joint_limits(self.side).tolist(),
            "action": "7 increments in [-1,1] plus absolute opening in [0,1]",
            "increment_reference": "previous dispatched target; reset from measured joints",
            "joint_increment_scale_rad": self.delta_rad,
            "control_hz": self.control_hz,
            "preview_time_s": self.period_s,
            "gripper": "0=closed, 1=open; normalized driver coordinate, not meters",
            "proprio_order": "measured joints[7], measured opening[1], previous target[7]",
            "missing_gripper": "zero placeholder with gripper_valid=0; never command fallback",
            "images": ["head_zed_left", f"{self.side}_wrist"],
            "image_shape_hwc": [self.image_height, self.image_width, 3],
            "image_encoding": "uint8 RGB; aspect-preserving resize and centered black padding",
            "coordinate_frames": "joint coordinates only; no Cartesian pose or world-state input",
            "fixed_subsystems": ["base", "lift", "neck", "other_arm"],
        }

    @cached_property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.describe(), sort_keys=True).encode()
        ).hexdigest()


def preprocess_rgb(image, contract, *, color_order="RGB"):
    """Same preprocessing for camera frames and rendered images. No guessed BGR."""
    import cv2

    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or min(array.shape[:2]) == 0
    ):
        raise ValueError("Camera frame must be a nonempty HxWx3 uint8 array")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("Explicit RGB or BGR color_order required")
    if color_order == "BGR":
        array = array[..., ::-1]
    h, w = contract.image_height, contract.image_width
    scale = min(w / array.shape[1], h / array.shape[0])
    size = (
        max(1, round(array.shape[1] * scale)),
        max(1, round(array.shape[0] * scale)),
    )
    resized = cv2.resize(
        array, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    result = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - size[0]) // 2, (h - size[1]) // 2
    result[y : y + size[1], x : x + size[0]] = resized
    return result


@dataclass(frozen=True)
class JointCommand:
    sequence: int
    joint_target: tuple[float, ...]
    gripper_target: float
    preview_time: float
    joint_limit_clipped: bool
    contract_fingerprint: str


class SophiaPolicyAdapter:
    """Explicit prepare/dispatch/commit, usable with YORMujoco or an existing RPC client.

    This is an interface adapter, not a hardware supervisor. The caller owns
    scheduling, timestamp alignment, interlocks and camera acquisition. A None
    RPC result means sent without acknowledgement, never verified execution.
    """

    def __init__(self, contract=None):
        self.contract = contract or PolicyContract()
        self.limits = joint_limits(self.contract.side)
        self.target = None
        self.sequence = 0

    def reset(self, packet):
        q = _vector(packet[f"{self.contract.side}_joints"], 7, "joint feedback")
        if np.any(q < self.limits[:, 0] - 0.02) or np.any(q > self.limits[:, 1] + 0.02):
            raise ValueError("Joint feedback is outside the model's joint limits")
        self.target = np.clip(q, *self.limits.T)
        self.sequence = 0

    def prepare(self, action):
        if self.target is None:
            raise RuntimeError("Reset the adapter from measured feedback first")
        a = _vector(action, 8, "action")
        if np.any(abs(a[:7]) > 1.0) or not 0 <= a[7] <= 1:
            raise ValueError(
                "Action requires seven values in [-1,1] and opening in [0,1]"
            )
        raw = self.target + a[:7] * self.contract.delta_rad
        q = np.clip(raw, *self.limits.T)
        return JointCommand(
            self.sequence,
            tuple(q),
            float(a[7]),
            self.contract.period_s,
            bool(np.any(q != raw)),
            self.contract.fingerprint,
        )

    def commit(self, command):
        if (
            command.sequence != self.sequence
            or command.contract_fingerprint != self.contract.fingerprint
        ):
            raise ValueError("Stale command or incompatible policy contract")
        self.target = np.asarray(command.joint_target).copy()
        self.sequence += 1

    def dispatch(self, client, action):
        command = self.prepare(action)
        result = getattr(client, f"set_{self.contract.side}_joint_target")(
            joint_target=list(command.joint_target),
            gripper_target=command.gripper_target,
            preview_time=command.preview_time,
        )
        if result is False:
            return command, "rejected"
        self.commit(command)
        return command, "accepted" if result is True else "sent_unconfirmed"

    def observe(self, packet, images=None, *, color_order="RGB"):
        if self.target is None:
            raise RuntimeError("Reset the adapter before constructing observations")
        side = self.contract.side
        q = _vector(packet[f"{side}_joints"], 7, "joint feedback")
        grip = packet.get(f"{side}_gripper")
        valid = (
            grip is not None
            and np.isscalar(grip)
            and np.isfinite(grip)
            and 0 <= grip <= 1
        )
        observation = {
            "proprio": np.r_[q, float(grip) if valid else 0.0, self.target].astype(
                np.float32
            ),
            "gripper_valid": np.array([valid], dtype=np.int8),
        }
        if images is not None:
            for key, name in (
                ("head_rgb", "head_zed_left"),
                ("wrist_rgb", f"{side}_wrist"),
            ):
                if name not in images:
                    raise ValueError(f"Missing camera frame: {name}")
                observation[key] = preprocess_rgb(
                    images[name], self.contract, color_order=color_order
                )
        return observation
