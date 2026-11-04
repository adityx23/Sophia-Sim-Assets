"""Explicit, provenance-checked historical replay alignment. No hardware access."""

import hashlib
import json
import math
from pathlib import Path

DESC = Path(__file__).resolve().parent / "yor-description"


def recording_lift_offset(native_path, episode, configuration=None):
    """Reject unrelated/changed recordings instead of guessing from encoder value."""
    cfg = json.loads((DESC / "sophia.json").read_text()) if configuration is None else configuration
    entry = cfg.get("base_alignment", {})
    if not entry.get("enabled", False):
        raise ValueError("Reviewed base alignment is not enabled")
    path = DESC / entry["source"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Base alignment checksum mismatch")
    profile = json.loads(path.read_text())
    for dependency in ("arm_reference", "spinal_assembly"):
        if (
            not cfg.get(dependency, {}).get("enabled", False)
            or cfg[dependency]["sha256"] != profile["requires_" + dependency + "_sha256"]
        ):
            raise ValueError("Historical alignment requires its reviewed head and arm frames")
    record = profile["replay_alignment"]["episodes"].get(str(episode))
    if (
        record is None
        or hashlib.sha256(Path(native_path).read_bytes()).hexdigest() != record["native_sha256"]
    ):
        raise ValueError("Historical height alignment does not cover this recording")
    offset = float(record["lift_extension_offset_m"])
    if not math.isfinite(offset) or abs(offset) > 0.15:
        raise ValueError("Historical height alignment is invalid")
    return offset
