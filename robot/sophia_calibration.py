"""Offline Sophia calibration and comparison tools. No hardware is imported or commanded.

Generate blank measurement templates with `templates --output DIR`. Fits write
review proposals, never modify robot configuration, and report held-out errors
separately from training errors. See docs/SOPHIA_OFFLINE_TOOLS.md for conventions.
"""

import argparse
import csv
import hashlib
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

DESCRIPTION = Path(__file__).resolve().parent / "yor-description"
JOINTS = [f"joint{i}_rad" for i in range(1, 8)]
TEMPLATES = {
    "lift.csv": ["sample_id", "split", "encoder_value", "shoulder_height_m"],
    "gripper.csv": ["sample_id", "split", "side", "raw_width_um", "jaw_gap_m"],
    "poses.csv": [
        "sample_id",
        "side",
        "lift_m",
        *JOINTS,
        "x_m",
        "y_m",
        "z_m",
        "qw",
        "qx",
        "qy",
        "qz",
    ],
    "points.csv": [
        "sample_id",
        "split",
        "source_x_m",
        "source_y_m",
        "source_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
    ],
    "trajectory.csv": [
        "time_s",
        "lift_m",
        "left_gripper_gap_m",
        "right_gripper_gap_m",
        "pan_deg",
        "tilt_deg",
    ],
    "gripper_can.csv": ["time_s", "side", "arbitration_id", "data_hex"],
}


def load_rows(path, required):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(
                f"{path}: missing columns {sorted(set(required) - set(reader.fieldnames or []))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no measurements; blank templates are not calibration data")
    if "sample_id" in required:
        ids = [row["sample_id"] for row in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("sample_id values must be nonempty and unique")
    return rows


def values(rows, columns):
    try:
        array = np.asarray([[float(row[name]) for name in columns] for row in rows])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Expected numeric columns: {columns}") from exc
    if not np.isfinite(array).all():
        raise ValueError("Measurements must be finite; NaN and infinity are not accepted")
    return array


def partition(rows):
    if any(row["split"] not in ("train", "validation") for row in rows):
        raise ValueError("split must be train or validation")
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) < 3:
        raise ValueError("At least three training measurements are required")
    return train, validation


def errors(array):
    array = np.asarray(array)
    return {
        "count": int(array.size),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "mean_absolute": float(np.mean(abs(array))),
        "max_absolute": float(np.max(abs(array))),
    }


def fit_linear(rows, x_name, y_name):
    train, validation = partition(rows)
    xy = values(train, [x_name, y_name])
    x, y = xy.T
    if len(np.unique(x)) < 3 or np.ptp(x) < 1e-12:
        raise ValueError("Need at least three distinct encoder/feedback positions")
    dx, dy = x - x.mean(), y - y.mean()
    slope = float(dx @ dy / (dx @ dx))
    if abs(slope) < 1e-15:
        raise ValueError("Zero response: measurements do not identify a useful mapping")
    intercept = float(y.mean() - slope * x.mean())
    holdout = values(validation, [x_name, y_name]) if validation else None
    result = {
        "equation": f"{y_name} = slope * {x_name} + intercept",
        "slope": slope,
        "intercept": intercept,
        "training_error_m": errors(slope * x + intercept - y),
        "validation_error_m": None
        if holdout is None
        else errors(slope * holdout[:, 0] + intercept - holdout[:, 1]),
        "input_range": [float(x.min()), float(x.max())],
        "validation_outside_training_range": bool(
            holdout is not None and np.any((holdout[:, 0] < x.min()) | (holdout[:, 0] > x.max()))
        ),
        "status": "proposal_requires_review"
        if validation
        else "fit_only_no_independent_validation",
        "hardware_configuration_modified": False,
    }
    return result


def fit_grippers(rows):
    if any(row["side"] not in ("left", "right") for row in rows):
        raise ValueError("side must be left or right")
    if np.any(values(rows, ["jaw_gap_m"]) < 0):
        raise ValueError("Physical jaw gaps must be nonnegative")
    output = {}
    for side in sorted({row["side"] for row in rows}):
        result = fit_linear(
            [row for row in rows if row["side"] == side], "raw_width_um", "jaw_gap_m"
        )
        result["feedback_direction"] = "inverted" if result["slope"] < 0 else "increasing"
        result["note"] = (
            "Raw width is fitted without normalization or clamping; "
            "inspect residuals before assuming a linear linkage."
        )
        output[side] = result
    return output


def fit_transform(rows):
    train, validation = partition(rows)
    source_columns = [f"source_{axis}_m" for axis in "xyz"]
    target_columns = [f"target_{axis}_m" for axis in "xyz"]
    source, target = values(train, source_columns), values(train, target_columns)
    centered_source, centered_target = source - source.mean(axis=0), target - target.mean(axis=0)
    if (
        min(
            np.linalg.matrix_rank(centered_source, tol=1e-8),
            np.linalg.matrix_rank(centered_target, tol=1e-8),
        )
        < 2
    ):
        raise ValueError("Point sets must not be coincident or collinear")
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation

    def residual(subset):
        return errors(
            np.linalg.norm(
                values(subset, source_columns) @ rotation.T
                + translation
                - values(subset, target_columns),
                axis=1,
            )
        )

    return {
        "convention": "p_target = T_target_source @ p_source; positions in meters",
        "T_target_source": transform.tolist(),
        "training_position_error_m": residual(train),
        "validation_position_error_m": residual(validation) if validation else None,
        "status": "proposal_requires_review"
        if validation
        else "fit_only_no_independent_validation",
    }


def compare_series(reference, candidate, signals, max_gap_s=0.1, candidate_time_offset_s=0.0):
    if not signals or max_gap_s <= 0 or not np.isfinite([max_gap_s, candidate_time_offset_s]).all():
        raise ValueError("Signals and a positive finite maximum interpolation gap are required")
    actual = values(reference, ["time_s", *signals])
    predicted = values(candidate, ["time_s", *signals])
    if min(len(actual), len(predicted)) < 2:
        raise ValueError("At least two timestamps are required in each series")
    predicted[:, 0] += candidate_time_offset_s
    for series in (actual, predicted):
        if np.any(np.diff(series[:, 0]) <= 0):
            raise ValueError("Timestamps must be strictly increasing; duplicates are not averaged")
    if actual[0, 0] < predicted[0, 0] or actual[-1, 0] > predicted[-1, 0]:
        raise ValueError(
            "Reference timestamps extend outside candidate data; extrapolation is forbidden"
        )
    for t in actual[:, 0]:
        index = np.searchsorted(predicted[:, 0], t)
        if index < len(predicted) and predicted[index, 0] == t:
            continue
        if predicted[index, 0] - predicted[index - 1, 0] > max_gap_s:
            raise ValueError("Candidate data contains an interpolation gap larger than max_gap_s")
    return {
        "candidate_time_offset_s": candidate_time_offset_s,
        "automatic_time_alignment": False,
        "signal_errors_in_input_units": {
            name: errors(np.interp(actual[:, 0], predicted[:, 0], predicted[:, i]) - actual[:, i])
            for i, name in enumerate(signals, 1)
        },
        "note": (
            "Signals must use identical units and coordinate frames; "
            "angular signals must be unwrapped consistently."
        ),
    }


def check_camera(config):
    width, height = config["width"], config["height"]
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("Camera width and height must be positive integers")
    intrinsic = np.asarray(config["K"], dtype=float)
    distortion = np.asarray(config["distortion"], dtype=float)
    transform = np.asarray(config["T_parent_camera_optical"], dtype=float)
    parent = config["parent_frame"]
    if parent not in ("head_tilt", "chassis_ground"):
        raise ValueError("parent_frame must be head_tilt or chassis_ground")
    if parent == "chassis_ground":
        values([config["calibration_pose"]], ["lift_m", "pan_deg", "tilt_deg"])
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or not np.allclose(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError("K must be a finite 3x3 camera intrinsic matrix")
    if min(intrinsic[0, 0], intrinsic[1, 1]) <= 0 or not (
        0 <= intrinsic[0, 2] < width and 0 <= intrinsic[1, 2] < height
    ):
        raise ValueError("Focal lengths must be positive and the principal point inside the image")
    if (
        config["distortion_model"] != "opencv"
        or distortion.ndim != 1
        or len(distortion) not in (4, 5, 8, 12, 14)
        or not np.isfinite(distortion).all()
    ):
        raise ValueError("Expected a finite OpenCV distortion vector of length 4, 5, 8, 12 or 14")
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
    ):
        raise ValueError("Camera transform must be a finite homogeneous 4x4 matrix")
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or abs(np.linalg.det(rotation) - 1) > 1e-6
    ):
        raise ValueError("Camera rotation must be orthonormal with determinant +1")
    return {
        "structurally_valid": True,
        "hardware_accuracy_verified": False,
        "image_size": [width, height],
        "camera_frame": "optical: +X right, +Y down, +Z forward",
        "parent_frame": parent,
        "transform_convention": "p_parent = T_parent_camera_optical @ p_camera_optical",
        "pose_specific": parent == "chassis_ground",
        "note": (
            "MuJoCo camera +Y is up and its viewing direction is -Z. "
            "Convert optical axes before applying a mount transform. "
            "A chassis transform is valid only at its recorded lift/neck pose."
        ),
    }


def compare_poses(rows, site_kind):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(DESCRIPTION / "scene-sophia.mjcf"))
    data = mujoco.MjData(model)
    results = []
    for row in rows:
        if row["side"] not in ("left", "right"):
            raise ValueError("side must be left or right")
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        q = values([row], ["lift_m", *JOINTS])[0]
        names = ["lift_joint", *[f"{row['side']}_arm_joint{i}" for i in range(1, 8)]]
        for name, value in zip(names, q, strict=True):
            joint = model.joint(name)
            if value < joint.range[0] - 1e-8 or value > joint.range[1] + 1e-8:
                raise ValueError(f"{row['sample_id']}: {name} outside model travel")
            data.qpos[joint.qposadr[0]] = value
        data.qpos[model.joint("sophia_middle_slide").qposadr[0]] = q[0] / 2
        mujoco.mj_forward(model, data)
        steer = [
            model.joint(f"{end}_{side}_steer").id
            for end in ("front", "back")
            for side in ("left", "right")
        ]
        origin = np.r_[data.xanchor[steer, :2].mean(axis=0), 0.0]
        site = model.site(f"{row['side']}_{site_kind}").id
        expected = values([row], ["x_m", "y_m", "z_m"])[0]
        position = data.site_xpos[site] - origin
        quaternion = values([row], ["qw", "qx", "qy", "qz"])[0]
        if abs(np.linalg.norm(quaternion) - 1) > 1e-3:
            raise ValueError("Measured quaternions must be normalized, in qw qx qy qz order")
        quaternion /= np.linalg.norm(quaternion)
        predicted = np.empty(4)
        mujoco.mju_mat2Quat(predicted, data.site_xmat[site])
        angle = 2 * np.arccos(np.clip(abs(predicted @ quaternion), 0, 1))
        results.append(
            {
                "sample_id": row["sample_id"],
                "side": row["side"],
                "predicted_position_m": position.tolist(),
                "position_error_m": float(np.linalg.norm(position - expected)),
                "orientation_error_deg": float(np.rad2deg(angle)),
            }
        )
    return {
        "frame": (
            "chassis_ground: X forward, Y left, Z up; "
            "floor origin under mean steering-axis XY position"
        ),
        "site_kind": site_kind,
        "gravity_settling_applied": False,
        "samples": results,
        "position_error_m": errors([row["position_error_m"] for row in results]),
        "orientation_error_deg": errors([row["orientation_error_deg"] for row in results]),
    }


def decode_gripper(rows):
    output = []
    for row in rows:
        if row["side"] not in ("left", "right"):
            raise ValueError("side must be left or right")
        timestamp = float(values([row], ["time_s"])[0, 0])
        if int(row["arbitration_id"], 0) != 0x2A8:
            continue
        raw = bytes.fromhex(row["data_hex"])
        if len(raw) != 8:
            raise ValueError("0x2A8 gripper feedback must contain exactly 8 bytes")
        if raw[7] != 0:
            continue  # angle-mode feedback is not a width measurement
        output.append(
            {
                "time_s": timestamp,
                "side": row["side"],
                "raw_width_um": int.from_bytes(raw[:4], "big", signed=True),
                "effort_i16_raw": int.from_bytes(raw[4:6], "big", signed=True),
                "status_raw": raw[6],
                "data_hex": raw.hex(),
            }
        )
    if not output:
        raise ValueError("No width-mode gripper feedback frames found")
    return output


def write_templates(directory):
    directory.mkdir(parents=True, exist_ok=True)
    existing = [name for name in (*TEMPLATES, "camera.json") if (directory / name).exists()]
    if existing:
        raise FileExistsError(f"Template destination already contains: {', '.join(existing)}")
    for name, fields in TEMPLATES.items():
        with (directory / name).open("x", newline="") as stream:
            csv.writer(stream).writerow(fields)
    camera = {
        "width": None,
        "height": None,
        "K": None,
        "distortion_model": "opencv",
        "distortion": None,
        "parent_frame": "head_tilt",
        "T_parent_camera_optical": None,
        "source": "Fill from actual camera calibration; this is a blank template.",
    }
    with (directory / "camera.json").open("x") as stream:
        json.dump(camera, stream, indent=2)


def record_states(get_state, output, duration, period, source_label):
    """Record an existing server's getter; do not initialize or command it."""
    if not np.isfinite([duration, period]).all() or duration <= 0 or period <= 0:
        raise ValueError("Recording duration and period must be positive and finite")
    start = time.monotonic()
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "metadata",
                    "source_label": source_label,
                    "clock": "client monotonic elapsed seconds; request start/end recorded",
                    "method": "get_collection_state",
                    "data_semantics": (
                        "Values preserved as returned; "
                        "command fallback is not promoted to measured feedback."
                    ),
                }
            )
            + "\n"
        )
        try:
            while time.monotonic() - start < duration:
                before = time.monotonic()
                state = get_state()
                if not isinstance(state, dict):
                    raise ValueError("get_collection_state must return a dictionary")
                stream.write(
                    json.dumps(
                        {
                            "type": "state",
                            "request_start_s": before - start,
                            "request_end_s": time.monotonic() - start,
                            "state": state,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                count += 1
                remaining = min(
                    period - (time.monotonic() - before), duration - (time.monotonic() - start)
                )
                if remaining > 0:
                    time.sleep(remaining)
        except BaseException:
            stream.write(json.dumps({"type": "end", "complete": False, "samples": count}) + "\n")
            raise
        stream.write(json.dumps({"type": "end", "complete": True, "samples": count}) + "\n")
    return count


def record_gripper(interface, side, output, duration):
    """Receive classic SocketCAN feedback only; this function never transmits."""
    if not hasattr(socket, "AF_CAN"):
        raise ValueError("Passive CAN recording requires Linux SocketCAN")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration must be positive and finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", 0x2A8, 0x7FF))
        bus.settimeout(0.1)
        bus.bind((interface,))
        with output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TEMPLATES["gripper_can.csv"])
            writer.writeheader()
            start = time.monotonic()
            while time.monotonic() - start < duration:
                try:
                    packet = bus.recv(16)
                except TimeoutError:
                    continue
                if len(packet) != 16:
                    continue
                identifier, length, payload = struct.unpack("=IB3x8s", packet)
                if identifier != 0x2A8 or length != 8:
                    continue
                writer.writerow(
                    {
                        "time_s": time.monotonic() - start,
                        "side": side,
                        "arbitration_id": hex(identifier),
                        "data_hex": payload.hex(),
                    }
                )
                stream.flush()
                count += 1
    return count


def record_remote_states(host, port, output, duration, period, source_label, timeout_s=5.0):
    """Bound getter waits and close the local client without stopping the server."""
    import zmq
    from commlink import RPCClient

    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("RPC timeout must be positive and finite")
    client = RPCClient(host, port)
    try:
        timeout_ms = max(1, int(timeout_s * 1000))
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        # Resolve the getter inside the capture so a failed RPC handshake also
        # leaves an incomplete footer instead of hanging before recording starts.
        return record_states(
            lambda: client.get_collection_state(), output, duration, period, source_label
        )
    except zmq.Again as exc:
        raise TimeoutError(f"Robot state getter timed out after {timeout_s:g} seconds") from exc
    finally:
        client.socket.close(linger=0)
        client.context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("templates")
    template.add_argument("--output", type=Path, required=True)
    state = sub.add_parser("record-state")
    state.add_argument("--host", required=True)
    state.add_argument("--port", type=int, default=5557)
    state.add_argument("--source-label", required=True)
    state.add_argument("--duration", type=float, default=10.0)
    state.add_argument("--period", type=float, default=0.05)
    state.add_argument("--timeout-s", type=float, default=5.0)
    state.add_argument("--output", type=Path, required=True)
    can = sub.add_parser("record-gripper")
    can.add_argument("--interface", required=True)
    can.add_argument("--side", choices=("left", "right"), required=True)
    can.add_argument("--duration", type=float, default=10.0)
    can.add_argument("--output", type=Path, required=True)
    for name in (
        "fit-lift",
        "fit-gripper",
        "fit-transform",
        "compare-poses",
        "check-camera",
        "decode-gripper",
    ):
        command = sub.add_parser(name)
        command.add_argument("input", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "fit-lift":
            command.add_argument(
                "--encoder-unit",
                required=True,
                help="Unit of the recorded encoder_value; no inferred conversion",
            )
        if name == "compare-poses":
            command.add_argument("--site", choices=("arm_ee", "gripper_center"), required=True)
    series = sub.add_parser("compare-series")
    series.add_argument("reference", type=Path)
    series.add_argument("candidate", type=Path)
    series.add_argument("--signals", nargs="+", required=True)
    series.add_argument("--max-gap-s", type=float, default=0.1)
    series.add_argument("--candidate-time-offset-s", type=float, default=0.0)
    series.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "record-state":
            count = record_remote_states(
                args.host,
                args.port,
                args.output,
                args.duration,
                args.period,
                args.source_label,
                args.timeout_s,
            )
            print(f"Recorded {count} getter responses; no robot commands sent.")
            return
        if args.command == "record-gripper":
            count = record_gripper(args.interface, args.side, args.output, args.duration)
            print(f"Recorded {count} CAN feedback frames; no CAN frames transmitted.")
            if not count:
                parser.exit(1, "No feedback received; the empty log is not calibration data.\n")
            return
        if args.command == "templates":
            write_templates(args.output)
            print(f"Blank templates written to {args.output}; no measurements fabricated.")
            return
        if args.command == "compare-series":
            required = ["time_s", *args.signals]
            report = compare_series(
                load_rows(args.reference, required),
                load_rows(args.candidate, required),
                args.signals,
                args.max_gap_s,
                args.candidate_time_offset_s,
            )
            sources = [args.reference, args.candidate]
        else:
            sources = [args.input]
            if args.command == "check-camera":
                report = check_camera(json.loads(args.input.read_text()))
            else:
                filename = {
                    "fit-lift": "lift.csv",
                    "fit-gripper": "gripper.csv",
                    "fit-transform": "points.csv",
                    "compare-poses": "poses.csv",
                    "decode-gripper": "gripper_can.csv",
                }[args.command]
                rows = load_rows(args.input, TEMPLATES[filename])
                if args.command == "fit-lift":
                    report = fit_linear(rows, "encoder_value", "shoulder_height_m")
                    report["encoder_unit"] = args.encoder_unit
                    report["note"] = (
                        "This fits absolute shoulder height. Establish the measured "
                        "lowered-height zero before converting to lift extension."
                    )
                elif args.command == "fit-gripper":
                    report = fit_grippers(rows)
                elif args.command == "fit-transform":
                    report = fit_transform(rows)
                elif args.command == "compare-poses":
                    report = compare_poses(rows, args.site)
                else:
                    decoded = decode_gripper(rows)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("x", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(decoded[0]))
                        writer.writeheader()
                        writer.writerows(decoded)
                    print(f"Decoded {len(decoded)} frames without normalization or clamping.")
                    return
        report = {
            "command": args.command,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "results": report,
            "hardware_configuration_modified": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Review results written to {args.output}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Calibration input error: {exc}\n")


if __name__ == "__main__":
    main()
