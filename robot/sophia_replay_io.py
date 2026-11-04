"""Passive ZMQ capture and offline import. No RPC clients or command sockets."""

import base64
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import zmq
from robot.sophia_replay import RECORDING_SCHEMA, write_json

TOPICS = ("sophia/replay/", "teleop/episode", "episode/signal", "teleop/episode_result")


def capture(endpoint, output, duration, should_stop=lambda: False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Capture duration must be positive and finite")
    if not endpoint.startswith(("tcp://", "ipc://")):
        raise ValueError("A TCP or IPC subscription endpoint is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVHWM, 10000)
    for topic in TOPICS:
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    packets = bad_packets = 0
    started = time.monotonic()
    reason, last_status = "error", -math.inf

    def status(running):
        result = {
            "mode": "passive_ZMQ_SUB_only",
            "endpoint": endpoint,
            "topics": list(TOPICS),
            "running": running,
            "packets": packets,
            "malformed_packets": bad_packets,
            "elapsed_s": time.monotonic() - started,
            "received_any_packet": packets > 0,
            "stop_reason": None if running else reason,
            "clock_note": "Receiver times are diagnostic, never substituted for source timestamps.",
            "hardware_commands_sent": False,
        }
        temporary = output / "status.tmp"
        write_json(temporary, result)
        temporary.replace(output / "status.json")
        return result

    try:
        subscriber.connect(endpoint)
        status(True)
        with (output / "packets.jsonl").open("x") as stream:
            while True:
                if should_stop() or (output / "STOP").exists():
                    reason = "requested"
                    break
                if time.monotonic() - started >= duration:
                    reason = "duration_limit"
                    break
                if subscriber.poll(100):
                    parts = subscriber.recv_multipart()
                    packets += 1
                    row = {
                        "received_unix_ns": time.time_ns(),
                        "received_monotonic_ns": time.monotonic_ns(),
                        "parts_base64": [base64.b64encode(part).decode() for part in parts],
                    }
                    try:
                        if len(parts) != 2:
                            raise ValueError("Expected topic and JSON payload")
                        row["topic"] = parts[0].decode("utf-8")
                        row["payload"] = json.loads(parts[1])
                        if not isinstance(row["payload"], dict):
                            raise TypeError("JSON object required")
                    except (ValueError, TypeError, UnicodeDecodeError) as error:
                        row["parse_error"] = str(error)
                        bad_packets += 1
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                if time.monotonic() - last_status >= 1:
                    status(True)
                    last_status = time.monotonic()
    finally:
        subscriber.close(linger=0)
        context.term()
        result = status(False)
        write_json(output / "summary.json", result)
    return result


def json_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def import_capture(path, run_id, trajectory):
    from robot.sophia_replay_compare import validate_recording

    metadata, commands, samples, events = None, [], [], []
    for row in json_rows(path):
        if row.get("parse_error"):
            continue
        payload, topic = row.get("payload", {}), row.get("topic")
        if payload.get("run_id") != run_id:
            continue
        if topic == "sophia/replay/meta":
            if metadata is not None and metadata != payload:
                raise ValueError("Conflicting run metadata")
            metadata = payload
        elif topic == "sophia/replay/command":
            commands.append(payload)
        elif topic == "sophia/replay/state":
            samples.append(payload)
        elif topic == "sophia/replay/event":
            events.append(payload)
    if metadata is None:
        raise ValueError("No matching replay metadata; use import-teleop for legacy observations")
    finished = [e for e in events if e.get("event") == "complete"]
    interrupted = any(e.get("event") in ("aborted", "manual_takeover", "fault") for e in events)
    result = {
        **metadata,
        "commands": commands,
        "samples": samples,
        "events": events,
        "complete": bool(finished) and not interrupted,
        "end_reason": "interrupted"
        if interrupted
        else "complete"
        if finished
        else "end_event_missing",
        "declared_sample_count": finished[-1].get("sample_count") if finished else None,
        "capture_path": str(Path(path).resolve()),
    }
    return validate_recording(result, trajectory)


def import_teleop(path, episode_id, start_timestamp, trajectory):
    """Legacy observations are useful diagnostics, never proof of replay dispatch."""
    from robot.sophia_replay_compare import validate_recording

    if not math.isfinite(start_timestamp):
        raise ValueError("Explicit publisher-clock run start required")
    samples = []

    def vector(value, size):
        try:
            a = np.asarray(value, dtype=float)
            return a.tolist() if a.shape == (size,) and np.isfinite(a).all() else None
        except (TypeError, ValueError):
            return None

    for raw in json_rows(path):
        frame = raw.get("payload", raw)
        if frame.get("episode_id") != episode_id:
            continue
        stamp = frame.get("timestamp")
        if not isinstance(stamp, (float, int)) or not math.isfinite(stamp):
            raise ValueError("Selected episode has invalid source timestamps")
        elapsed = stamp - start_timestamp
        if elapsed < 0 or elapsed > trajectory["duration_s"] + 1:
            continue
        grip = frame.get("left_gripper")
        if (
            frame.get("left_gripper_measured") is False
            or not isinstance(grip, (float, int))
            or not math.isfinite(grip)
            or not 0 <= grip <= 1
        ):
            grip = None
        samples.append(
            {
                "seq": frame.get("seq"),
                "time_s": elapsed,
                "left_joints": vector(frame.get("left_joints"), 7),
                "left_gripper": grip,
                "base_pose": None,
                "base_velocity_body": vector(frame.get("enc_vel"), 3),
                "observed_joint_target": vector(frame.get("left_joints_cmd"), 7),
                "observed_base_target": vector(frame.get("cmd_vel", frame.get("base_cmd_vel")), 3),
            }
        )
    result = {
        "schema": RECORDING_SCHEMA,
        "run_id": "legacy-" + episode_id,
        "trajectory_sha256": trajectory["sha256"],
        "source": "robot",
        "clock": "seconds_from_run_start",
        "pose_frame": None,
        "pose_source": None,
        "base_velocity_source": "wheel_kinematics",
        "complete": False,
        "end_reason": "legacy_observations_only",
        "commands": [],
        "events": [],
        "samples": samples,
        "initial_state": None,
        "source_path": str(Path(path).resolve()),
        "start_timestamp_supplied": start_timestamp,
        "association_note": (
            "Trajectory association supplied by caller, not verified from dispatch logs."
        ),
    }
    return validate_recording(result, trajectory)
