"""Explicit simulation-only mappings adopted for Sophia's approximate release.

These functions do not modify hardware calibration. Encoder zero/scale and
intermediate jaw geometry are engineering assumptions recorded in sophia.json.
"""

import json
from pathlib import Path

import numpy as np

PROFILE = Path(__file__).resolve().parent / "yor-description/sophia.json"


def load_profile():
    return json.loads(PROFILE.read_text())


def _bounded(value, lower, upper):
    array = np.asarray(value, dtype=float)
    if (
        not np.isfinite(array).all()
        or np.any(array < lower - 1e-9)
        or np.any(array > upper + 1e-9)
    ):
        raise ValueError(f"Value outside the assumed [{lower}, {upper}] interval")
    return np.clip(array, lower, upper)


def encoder_to_lift_extension(encoder_m):
    mapping = load_profile()["release"]["lift_mapping"]
    value = _bounded(encoder_m, mapping["encoder_lower_m"], mapping["encoder_upper_m"])
    return (value - mapping["encoder_lower_m"]) * mapping[
        "physical_extension_per_encoder_m"
    ]


def lift_extension_to_encoder(extension_m):
    profile = load_profile()
    measured = profile["measured"]
    travel = (
        measured["shoulder_height_raised_m"] - measured["shoulder_height_lowered_m"]
    )
    value = _bounded(extension_m, 0, travel)
    mapping = profile["release"]["lift_mapping"]
    return (
        mapping["encoder_lower_m"] + value / mapping["physical_extension_per_encoder_m"]
    )


def normalized_gripper_to_gap(opening):
    return (
        _bounded(opening, 0, 1)
        * load_profile()["release"]["gripper_mapping"]["gap_open_m"]
    )
