"""Passively record Sophia's two-part ZMQ episode stream for later sim replay.

No robot SDK, RPC or command sockets are used. Receive times are diagnostic;
replay must use the unchanged publisher timestamps. An empty log is not data.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import signal
import time
from collections import Counter
from pathlib import Path

import zmq
from zmq.utils.monitor import recv_monitor_message

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ("teleop/episode", "episode/signal", "teleop/episode_result")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_vector(value, size):
    return (
        isinstance(value, list)
        and len(value) == size
        and all(map(finite_number, value))
    )


class Quality:
    """Track observed completeness separately from sensor availability."""

    def __init__(self):
        self.packets = 0
        self.bad_packets = 0
        self.episodes = {}

    def observe(self, topic, payload):
        self.packets += 1
        if topic not in TOPICS or not isinstance(payload, dict):
            self.bad_packets += 1
            return
        identifier = payload.get("episode_id")
        if not isinstance(identifier, str) or not identifier:
            self.bad_packets += 1
            return
        ep = self.episodes.setdefault(
            identifier,
            {
                "frames": 0,
                "seen": set(),
                "last_seq": None,
                "sequence_gap_events": 0,
                "duplicate_sequences": 0,
                "out_of_order_sequences": 0,
                "invalid_sequences": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "invalid_timestamps": 0,
                "nonincreasing_timestamps": 0,
                "max_timestamp_gap_s": 0.0,
                "start_signal_seen": False,
                "end_signal_seen": False,
                "publisher_num_frames": None,
                "invalid_or_missing_fields": Counter(),
                "available_optional_fields": Counter(),
            },
        )
        if topic == "episode/signal":
            if payload.get("active") is True:
                ep["start_signal_seen"] = True
            elif payload.get("active") is False:
                ep["end_signal_seen"] = True
                count = payload.get("num_frames")
                if type(count) is int and count >= 0:
                    ep["publisher_num_frames"] = count
            return
        if topic != "teleop/episode":
            return
        ep["frames"] += 1
        seq = payload.get("seq")
        if type(seq) is not int or seq < 0:
            ep["invalid_sequences"] += 1
        else:
            if seq in ep["seen"]:
                ep["duplicate_sequences"] += 1
            last = ep["last_seq"]
            if last is not None:
                ep["sequence_gap_events"] += seq > last + 1
                ep["out_of_order_sequences"] += seq < last
            ep["last_seq"] = seq
            ep["seen"].add(seq)
        stamp = payload.get("timestamp")
        if not finite_number(stamp):
            ep["invalid_timestamps"] += 1
        else:
            last = ep["last_timestamp"]
            if last is None:
                ep["first_timestamp"] = stamp
            else:
                ep["nonincreasing_timestamps"] += stamp <= last
                ep["max_timestamp_gap_s"] = max(ep["max_timestamp_gap_s"], stamp - last)
            ep["last_timestamp"] = stamp
        for side in ("left", "right"):
            for suffix in ("joints", "joints_cmd"):
                key = f"{side}_{suffix}"
                if not valid_vector(payload.get(key), 7):
                    ep["invalid_or_missing_fields"][key] += 1
            for suffix in ("gripper", "gripper_cmd"):
                key = f"{side}_{suffix}"
                value = payload.get(key)
                if not finite_number(value) or not 0 <= value <= 1:
                    ep["invalid_or_missing_fields"][key] += 1
            if payload.get(f"{side}_gripper_measured") is not True:
                ep["invalid_or_missing_fields"][
                    f"{side}_gripper_measured_not_true"
                ] += 1
        for key in ("cmd_vel", "enc_vel"):
            if not valid_vector(payload.get(key), 3):
                ep["invalid_or_missing_fields"][key] += 1
        for key in (
            "neck_deg",
            "neck_cmd_deg",
            "lift_height",
            "lift_cmd",
            "base_pose",
            "cam_pose",
            "zed_timestamp",
            "zed_age_s",
            "left_joint_velocities",
            "right_joint_velocities",
            "left_joint_torques",
            "right_joint_torques",
        ):
            if payload.get(key) is not None:
                ep["available_optional_fields"][key] += 1

    def summary(self):
        episodes = {}
        for identifier, ep in self.episodes.items():
            item = {k: v for k, v in ep.items() if k != "seen"}
            seen = ep["seen"]
            expected = ep["publisher_num_frames"]
            observed_span = max(seen, default=-1) + 1
            span = expected if expected is not None else observed_span
            item["missing_sequences"] = span - sum(seq < span for seq in seen)
            item["sequences_outside_publisher_count"] = sum(seq >= span for seq in seen)
            item["unique_sequences"] = len(seen)
            item["sequence_capture_complete"] = bool(
                ep["end_signal_seen"]
                and expected is not None
                and expected > 0
                and ep["frames"] == expected
                and len(seen) == expected
                and observed_span == expected
                and not ep["invalid_sequences"]
            )
            item["sequence_capture_note"] = (
                "Sequence completeness only; review timestamps and field availability before replay."
                if expected is not None
                else "End frame count unavailable; missing trailing packets cannot be assessed."
            )
            episodes[identifier] = item
        return {
            "packets": self.packets,
            "bad_packets": self.bad_packets,
            "episodes": episodes,
        }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(endpoint, output, duration, should_stop=lambda: False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "sim-baseline"
    snapshots.mkdir()
    hashes = {}
    for name in (
        "robot/yor-description/sophia.json",
        "robot/yor-description/sophia.mjcf",
        "robot/yor-description/scene-sophia.mjcf",
        "robot/yor-description/sophia-ik.mjcf",
        "robot/yor_mujoco.py",
        "scripts/listen_sophia_episodes.py",
    ):
        source = ROOT / name
        if source.exists():
            content = source.read_bytes()
            target = snapshots / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    write_json(
        output / "metadata.json",
        {
            "endpoint": endpoint,
            "topics": TOPICS,
            "started_unix_ns": time.time_ns(),
            "pid": os.getpid(),
            "duration_limit_s": duration,
            "mode": "ZMQ SUB only",
            "baseline_sha256": hashes,
            "wire_format": "multipart [UTF-8 topic, JSON object]",
            "clock_note": "Publisher timestamp is preserved. Receiver times do not establish one-way latency.",
            "feedback_note": "Command fallback and missing sensors are not measurements.",
            "capture_note": "PUB/SUB may drop frames. Keep the Pi episode recorder as the authoritative copy.",
        },
    )
    quality = Quality()
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVHWM, 10000)
    for topic in TOPICS:
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    monitor = subscriber.get_monitor_socket(
        events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED
    )
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    poller.register(monitor, zmq.POLLIN)
    start = time.monotonic()
    last_packet = None
    last_report = -math.inf
    connected = False
    reason = "error"

    def report(running):
        value = {
            "pid": os.getpid(),
            "endpoint": endpoint,
            "running": running,
            "updated_unix_ns": time.time_ns(),
            "elapsed_s": time.monotonic() - start,
            "tcp_connected": connected,
            "received_any_packet": last_packet is not None,
            "last_packet_age_s": None
            if last_packet is None
            else time.monotonic() - last_packet,
            "stop_reason": None if running else reason,
            **quality.summary(),
        }
        write_json(output / "status.json", value)
        return value

    try:
        subscriber.connect(endpoint)
        report(True)
        print(f"Listening: {endpoint}; output: {output.resolve()}", flush=True)
        with (
            (output / "packets.jsonl").open("x") as raw,
            (output / "frames.jsonl").open("x") as frames,
        ):
            while True:
                if should_stop() or (output / "STOP").exists():
                    reason = "requested"
                    break
                if time.monotonic() - start >= duration:
                    reason = "duration_limit"
                    break
                ready = dict(poller.poll(200))
                if monitor in ready:
                    event = recv_monitor_message(monitor)["event"]
                    connected = event == zmq.EVENT_CONNECTED
                if subscriber in ready:
                    parts = subscriber.recv_multipart()
                    wall, monotonic = time.time_ns(), time.monotonic_ns()
                    last_packet = time.monotonic()
                    raw.write(
                        json.dumps(
                            {
                                "received_unix_ns": wall,
                                "received_monotonic_ns": monotonic,
                                "parts_base64": [
                                    base64.b64encode(part).decode("ascii")
                                    for part in parts
                                ],
                            }
                        )
                        + "\n"
                    )
                    raw.flush()
                    try:
                        if len(parts) != 2:
                            raise ValueError("Expected two multipart frames")
                        topic = parts[0].decode("utf-8")
                        payload = json.loads(parts[1])
                        quality.observe(topic, payload)
                        if topic == "teleop/episode" and isinstance(payload, dict):
                            # Preserve values/flags; exact wire bytes remain in packets.jsonl.
                            frames.write(json.dumps(payload) + "\n")
                            frames.flush()
                    except (ValueError, UnicodeDecodeError):
                        quality.bad_packets += 1
                        quality.packets += 1
                if time.monotonic() - last_report >= 5:
                    status = report(True)
                    print(
                        f"packets={status['packets']} tcp_connected={connected} "
                        f"episodes={len(status['episodes'])}",
                        flush=True,
                    )
                    last_report = time.monotonic()
    finally:
        subscriber.disable_monitor()
        monitor.close(linger=0)
        subscriber.close(linger=0)
        context.term()
        final = report(False)
        write_json(output / "summary.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.123.55:5558")
    parser.add_argument(
        "--output", type=Path, required=True, help="New session directory"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=7200,
        help="Maximum seconds, including waiting",
    )
    args = parser.parse_args()
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    result = capture(args.endpoint, args.output, args.duration, lambda: stopping)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
