"""Compare unchanged replay timing, with explicit missing-data/provenance gates."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from robot.sophia_replay import (
    POSE_FRAME,
    RECORDING_SCHEMA,
    validate_trajectory,
    write_json,
)


def validate_recording(recording, trajectory):
    r = recording
    if r.get("schema") != RECORDING_SCHEMA or r.get("trajectory_sha256") != trajectory["sha256"]:
        raise ValueError("Recording schema/trajectory hash mismatch")
    if r.get("source") not in ("simulation", "robot", "synthetic"):
        raise ValueError("Recording must declare simulation, robot or synthetic source")
    if r.get("clock") != "seconds_from_run_start":
        raise ValueError("Explicit run-relative source timestamps required")
    if type(r.get("complete")) is not bool:
        raise ValueError("Recording complete flag must be boolean")
    if r.get("pose_frame") not in (POSE_FRAME, None):
        raise ValueError("Convert poses to base-center x/y/yaw before comparison")
    samples = r.get("samples", [])
    if not samples:
        raise ValueError("Empty recording is not a measurement")
    times = np.asarray([s["time_s"] for s in samples], dtype=float)
    if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Sample times must be finite, increasing and run-relative")
    sequences = [s["seq"] for s in samples]
    if any(type(v) is not int or v < 0 for v in sequences) or np.any(np.diff(sequences) <= 0):
        raise ValueError("Duplicate/reversed/invalid sample sequences")
    for sample in samples:
        for field, size in (
            ("left_joints", 7),
            ("base_pose", 3),
            ("base_velocity_body", 3),
            ("steer_rad", 4),
            ("drive_m_s", 4),
        ):
            value = sample.get(field)
            if value is not None and (np.shape(value) != (size,) or not np.isfinite(value).all()):
                raise ValueError(f"Invalid {field}; use null for unavailable feedback")
        grip = sample.get("left_gripper")
        if grip is not None and (
            not np.isscalar(grip) or not np.isfinite(grip) or not 0 <= grip <= 1
        ):
            raise ValueError("Invalid measured gripper opening")
        if sample.get("base_pose") is not None:
            stamp = sample.get("base_pose_time_s")
            if (
                stamp is None
                or not np.isfinite(stamp)
                or stamp < 0
                or stamp > sample["time_s"] + 0.02
            ):
                raise ValueError("Base pose needs its own run-relative source timestamp")
    return r


def initial_conditions(reference, actual):
    first, second = reference.get("initial_state"), actual.get("initial_state")
    if not isinstance(first, dict) or not isinstance(second, dict):
        return {"verified": False, "issues": ["measured_initial_state_missing"]}
    differences, issues = {}, []
    for key, size, tolerance in (
        ("left_joints", 7, 0.05),
        ("right_joints", 7, 0.05),
        ("left_gripper", 1, 0.1),
        ("lift_extension_m", 1, 0.005),
        ("neck_deg", 2, 2.0),
        ("steer_rad", 4, 0.05),
    ):
        try:
            a, b = (
                np.asarray(first[key]).reshape(size),
                np.asarray(second[key]).reshape(size),
            )
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Missing measurement")
            delta = b - a
            if key == "steer_rad":
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
            error = float(np.max(abs(delta)))
            differences[key] = {
                "max_absolute_difference": error,
                "tolerance": tolerance,
            }
            if error > tolerance:
                issues.append(key + "_initial_mismatch")
        except (KeyError, ValueError, TypeError):
            issues.append(key + "_initial_unavailable")
    return {
        "verified": not issues,
        "issues": issues,
        "differences": differences,
        "note": "Engineering setup tolerances, not established hardware accuracy limits.",
    }


def command_parity(recording, trajectory):
    actual, expected = recording.get("commands", []), trajectory["frames"]
    reasons, delays = [], []
    if len(actual) != len(expected):
        reasons.append("command_count_mismatch")
    previous = -np.inf
    for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
        try:
            good = a["seq"] == i and abs(a["time_s"] - e["time_s"]) < 1e-7
            good &= np.shape(a["left_joints"]) == (7,) and np.allclose(
                a["left_joints"], e["left_joints"], atol=1e-6, rtol=0
            )
            good &= np.shape(a["base_velocity"]) == (3,) and np.allclose(
                a["base_velocity"], e["base_velocity"], atol=1e-7, rtol=0
            )
            good &= abs(a["left_gripper"] - e["left_gripper"]) < 1e-6
            good &= abs(a["preview_time_s"] - e["preview_time_s"]) < 1e-7
            dispatch = float(a["dispatch_time_s"])
            good &= np.isfinite(dispatch) and dispatch >= 0 and dispatch > previous
            good &= a.get("status") in ("accepted", "sent_unconfirmed")
            if not good:
                reasons.append(f"command_{i}_mismatch")
            delays.append(dispatch - e["time_s"])
            previous = dispatch
        except (ValueError, TypeError, KeyError):
            reasons.append(f"command_{i}_invalid")
    timing = None
    if delays and np.isfinite(delays).all():
        timing = {
            "median_ms": float(np.median(delays) * 1000),
            "p95_absolute_ms": float(np.percentile(np.abs(delays), 95) * 1000),
            "max_absolute_ms": float(np.max(np.abs(delays)) * 1000),
        }
    return {
        "verified": not reasons,
        "issues": reasons[:20],
        "issue_count": len(reasons),
        "actual_commands": len(actual),
        "expected_commands": len(expected),
        "dispatch_minus_planned_time": timing,
    }


def series(recording, field, size):
    times, values = [], []
    for sample in recording["samples"]:
        value = sample.get(field)
        if value is None:
            continue
        stamp = sample["base_pose_time_s"] if field == "base_pose" else sample["time_s"]
        v = np.asarray(value, dtype=float).reshape(size)
        if times and stamp <= times[-1]:
            if stamp == times[-1] and np.array_equal(v, values[-1]):
                continue  # unchanged asynchronous pose, not a new measurement
            raise ValueError(f"{field} source time reversed or reused with changed data")
        times.append(stamp)
        values.append(v)
    return np.asarray(times), np.asarray(values).reshape(-1, size)


def interpolate(times, values, query, max_gap_s=0.1):
    result = np.full((len(query), values.shape[1]), np.nan)
    if len(times) < 2:
        return result
    right = np.searchsorted(times, query, side="right")
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    exact = np.isclose(query, times[left], atol=1e-9, rtol=0)
    valid = (query >= times[0] - 1e-9) & (query <= times[-1] + 1e-9)
    valid &= exact | (times[right] - times[left] <= max_gap_s + 1e-9)
    for col in range(values.shape[1]):
        result[valid, col] = np.interp(query[valid], times, values[:, col])
    return result


def normalize_pose(values):
    """Only initial translation/heading alignment; no fitted scale, drift or delay."""
    if not len(values):
        return values
    delta = values[:, :2] - values[0, :2]
    yaw0 = values[0, 2]
    rotation = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
    return np.column_stack([delta @ rotation, np.unwrap(values[:, 2]) - yaw0])


def _errors(reference, actual, scale=1.0):
    valid = np.isfinite(reference).all(axis=1) & np.isfinite(actual).all(axis=1)
    if not valid.any():
        return {"available": False, "coverage": 0.0, "samples": 0}
    error = (actual[valid] - reference[valid]) * scale
    return {
        "available": True,
        "coverage": float(valid.mean()),
        "samples": int(valid.sum()),
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "max_abs": np.max(abs(error), axis=0).tolist(),
        "p95_abs": np.percentile(abs(error), 95, axis=0).tolist(),
    }


def compare(trajectory, reference, actual, output=None):
    validate_trajectory(trajectory)
    validate_recording(reference, trajectory)
    validate_recording(actual, trajectory)
    if reference["source"] != "simulation" or not reference.get("complete"):
        raise ValueError("Reference must be a complete simulation replay")
    if not command_parity(reference, trajectory)["verified"]:
        raise ValueError("Reference command stream does not match the trajectory")
    query = np.array([s["time_s"] for s in reference["samples"]])
    arrays = {"time_s": query}
    metrics = {}
    for field, n, unit in (
        ("left_joints", 7, "rad"),
        ("left_gripper", 1, "normalized_opening"),
    ):
        for label, recording in (("reference", reference), ("observed", actual)):
            times, values = series(recording, field, n)
            arrays[label + "_" + field] = interpolate(times, values, query)
        metrics[field] = {
            "unit": unit,
            **_errors(arrays["reference_" + field], arrays["observed_" + field]),
        }
    pose_sources = {
        "simulation": {"mujoco_ground_truth"},
        "robot": {"zed_base_pose", "external_base_tracking"},
        "synthetic": {"synthetic_base_pose"},
    }
    pose_available = all(
        r.get("pose_frame") == POSE_FRAME and r.get("pose_source") in pose_sources[r["source"]]
        for r in (reference, actual)
    )
    pose_reason = None
    if pose_available:
        for label, recording in (("reference", reference), ("observed", actual)):
            times, values = series(recording, "base_pose", 3)
            # Both origins must be observed near run start; don't silently move
            # the origin to the middle of an incomplete/stale capture.
            if len(times) < 2 or times[0] > 0.1:
                pose_available = False
                pose_reason = "missing_initial_pose_or_insufficient_pose_samples"
                break
            arrays[label + "_base_pose"] = interpolate(times, normalize_pose(values), query)
    else:
        pose_reason = "independent_base_pose_unavailable_or_unverified"
    if pose_available:
        ref, obs = arrays["reference_base_pose"], arrays["observed_base_pose"]
        error = obs - ref
        error[:, 2] = (error[:, 2] + np.pi) % (2 * np.pi) - np.pi
        metrics["base_pose"] = {
            "unit": ["m", "m", "rad"],
            **_errors(np.zeros_like(error), error),
        }
        valid = np.flatnonzero(np.isfinite(error).all(axis=1))
        if len(valid):
            end = valid[-1]
            metrics["base_pose"].update(
                {
                    "comparison_end_time_s": float(query[end]),
                    "covers_run_end": bool(end == len(query) - 1),
                    "endpoint_error": error[end].tolist(),
                    "reference_displacement": ref[end].tolist(),
                    "observed_displacement": obs[end].tolist(),
                }
            )
    else:
        metrics["base_pose"] = {
            "available": False,
            "coverage": 0.0,
            "reason": pose_reason,
        }
        arrays["reference_base_pose"] = arrays["observed_base_pose"] = np.full(
            (len(query), 3), np.nan
        )
    for field, n in (("steer_rad", 4), ("drive_m_s", 4), ("base_velocity_body", 3)):
        values = []
        for recording in (reference, actual):
            ts, vs = series(recording, field, n)
            values.append(interpolate(ts, vs, query))
        if field == "base_velocity_body":
            (
                arrays["reference_base_velocity_body"],
                arrays["observed_base_velocity_body"],
            ) = values
        if field == "steer_rad":
            difference = (values[1] - values[0] + np.pi) % (2 * np.pi) - np.pi
            metrics[field] = _errors(np.zeros_like(difference), difference)
        else:
            metrics[field] = _errors(*values)
    parity = command_parity(actual, trajectory)
    initial = initial_conditions(reference, actual)
    issues = []
    if not initial["verified"]:
        issues.append("initial_conditions_not_matched")
    if not actual.get("complete") or actual.get("end_reason") != "complete":
        issues.append("run_incomplete_or_interrupted")
    if any(
        e.get("event") in ("aborted", "manual_takeover", "fault") for e in actual.get("events", [])
    ):
        issues.append("operator_intervention_or_fault")
    if not parity["verified"]:
        issues.append("same_commands_not_verified")
    seq = [s["seq"] for s in actual["samples"]]
    missing = int(seq[0] + np.sum(np.maximum(0, np.diff(seq) - 1)))
    declared = actual.get("declared_sample_count")
    if declared is not None:
        missing += max(0, int(declared) - seq[-1] - 1)
    if missing:
        issues.append("missing_sample_sequences")
    for name in ("left_joints", "left_gripper", "base_pose"):
        if metrics[name]["coverage"] < 0.95:
            issues.append(name + "_coverage_below_95_percent")
    reference_end = float(query[-1])
    if actual["samples"][-1]["time_s"] < reference_end - 0.1:
        issues.append("run_end_not_observed")
    is_hardware = actual["source"] == "robot"
    report = {
        "schema": "sophia.replay.comparison.v1",
        "trajectory_sha256": trajectory["sha256"],
        "reference_run": reference["run_id"],
        "observed_run": actual["run_id"],
        "observed_source": actual["source"],
        "hardware_validated": False,
        "status": "measurement_ready_for_review"
        if is_hardware and not issues
        else "diagnostic_only"
        if is_hardware
        else "software_self_check",
        "matched_command_data_eligible": bool(not issues),
        "quality_issues": issues,
        "missing_sample_sequences": missing,
        "command_parity": parity,
        "metrics": metrics,
        "initial_conditions": initial,
        "alignment": (
            "Source run-relative time, initial base translation/heading only; "
            "no fitted time shift, scale, or warping."
        ),
        "limits": (
            "Unloaded motion only. No contact, grip-force, payload or policy-transfer validation. "
            "The 95% gate is data coverage, not an accuracy pass threshold."
        ),
    }
    if output is not None:
        write_report(output, report, arrays)
    return report


def write_report(output, report, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", report)
    names, columns = ["time_s"], [arrays["time_s"]]
    for field, label in (
        ("left_joints", "joint"),
        ("left_gripper", "opening"),
        ("base_pose", "base"),
        ("base_velocity_body", "wheel_body_velocity"),
    ):
        for source in ("reference", "observed"):
            for i, column in enumerate(arrays[source + "_" + field].T):
                names.append(f"{source}_{label}_{i}")
                columns.append(column)
    with (output / "aligned-samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(
            [[float(v) if np.isfinite(v) else "" for v in row] for row in np.array(columns).T]
        )
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, layout="constrained")
    t = arrays["time_s"]
    for source, style in (("reference", "-"), ("observed", "--")):
        for axis, col in ((axes[0], 0), (axes[1], 1)):
            axis.plot(t, arrays[source + "_base_pose"][:, col], style, label=source)
        axes[3].plot(t, arrays[source + "_left_gripper"][:, 0], style, label=source)
        axes[4].plot(t, arrays[source + "_base_velocity_body"][:, 0], style, label=source)
    for joint, error in enumerate(
        (arrays["observed_left_joints"] - arrays["reference_left_joints"]).T
    ):
        axes[2].plot(t, np.rad2deg(error), label=f"J{joint + 1}")
    for axis, label in zip(
        axes,
        (
            "Forward (m)",
            "Lateral (m)",
            "Joint error (deg)",
            "Gripper opening",
            "Wheel-derived vx (m/s)",
        ),
        strict=True,
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Elapsed source time (s)")
    if not report["metrics"]["base_pose"]["available"]:
        for axis in axes[:2]:
            axis.text(
                0.5,
                0.5,
                "Independent base pose unavailable",
                transform=axis.transAxes,
                ha="center",
            )
    fig.suptitle(f"Sophia replay — {report['observed_source']} — {report['status']}")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    title = (
        "Software self-check — no robot measurements"
        if report["observed_source"] != "robot"
        else "Robot replay comparison"
    )
    details = html.escape(json.dumps(report, indent=2))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>{title}</title>
<style>
body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#192b36}}
img{{width:100%}}pre{{overflow:auto;background:#f0f4f6;padding:20px;font-size:13px}}
.status{{padding:15px;background:#fff0c5}}
</style>
<h1>{title}</h1><p class="status">{html.escape(report["status"])}. Hardware validated: false.</p>
<p>{html.escape(report["alignment"])}</p><p>{html.escape(report["limits"])}</p>
<img src="comparison.png"
alt="Base displacement, joint errors and gripper opening over unchanged elapsed time">
<p><a href="comparison.json">Machine-readable report</a> ·
<a href="aligned-samples.csv">Aligned samples</a></p><pre>{details}</pre></html>"""
    (output / "report.html").write_text(page)
