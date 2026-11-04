"""Import a stopped episode capture as portable simulation calibration evidence.

Keeps exact compressed logs, checks raw packets against decoded frames, and
exports nullable native-time arrays. Does not fit or change sim parameters.
"""

import argparse
import base64
import gzip
import hashlib
import json
import shutil
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
from listen_sophia_episodes import Quality

SIGNALS = {
    "left_joints": (7, "rad"),
    "right_joints": (7, "rad"),
    "left_joints_cmd": (7, "rad"),
    "right_joints_cmd": (7, "rad"),
    "neck_deg": (2, "deg"),
    "neck_cmd_deg": (2, "deg"),
    "cmd_vel": (3, "publisher body velocity: m/s, m/s, rad/s"),
    "enc_vel": (3, "wheel-derived body velocity: m/s, m/s, rad/s"),
    "lift_height": (1, "publisher lift coordinate; no physical-height conversion"),
    "lift_cmd": (1, "publisher command; semantics not inferred"),
    "left_gripper": (
        1,
        "normalized opening; measured/fallback provenance may be unknown",
    ),
    "right_gripper": (
        1,
        "normalized opening; measured/fallback provenance may be unknown",
    ),
    "left_gripper_cmd": (1, "normalized opening; null commands remain missing"),
    "right_gripper_cmd": (1, "normalized opening; null commands remain missing"),
    "left_ee": (7, "joint-derived position/quaternion; source frame retained"),
    "right_ee": (7, "joint-derived position/quaternion; source frame retained"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def matrix(rows, key, width):
    values = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            continue
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (ValueError, TypeError):
            continue
        if len(vector) == width:
            values[index] = vector
    return values


def signal_stats(values):
    valid = np.isfinite(values).all(axis=1)
    clean = values[valid]
    return {
        "valid_rows": int(valid.sum()),
        "missing_or_invalid_rows": int((~valid).sum()),
        "min": clean.min(axis=0).tolist() if len(clean) else None,
        "max": clean.max(axis=0).tolist() if len(clean) else None,
        "range": np.ptp(clean, axis=0).tolist() if len(clean) else None,
    }


def wire_rows(source, quality, events):
    """Pair every data packet with its decoded log entry, preserving receipt clocks."""
    with (
        (source / "packets.jsonl").open() as packets,
        (source / "frames.jsonl").open() as frames,
    ):
        for line in packets:
            packet = json.loads(line)
            parts = [
                base64.b64decode(part, validate=True) for part in packet["parts_base64"]
            ]
            if len(parts) != 2:
                raise ValueError("Malformed multipart packet; capture requires review")
            topic, payload = parts[0].decode("utf-8"), json.loads(parts[1])
            quality.observe(topic, payload)
            if topic == "teleop/episode":
                decoded = frames.readline()
                if not decoded or json.loads(decoded) != payload:
                    raise ValueError("Raw packet / decoded frame mismatch")
                yield (
                    payload,
                    packet["received_unix_ns"],
                    packet["received_monotonic_ns"],
                )
            else:
                events.append(
                    {
                        "topic": topic,
                        "payload": payload,
                        "received_unix_ns": packet["received_unix_ns"],
                        "received_monotonic_ns": packet["received_monotonic_ns"],
                    }
                )
        if frames.readline():
            raise ValueError("Decoded frames have no corresponding raw packets")


def export_episode(records, target, identifier):
    rows = [record[0] for record in records]
    timestamp = np.asarray([row["timestamp"] for row in rows], dtype=float)
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"Invalid source timestamps: {identifier}")
    sequences = [row.get("seq") for row in rows]
    if any(type(seq) is not int or seq < 0 for seq in sequences):
        raise ValueError(f"Invalid source sequence: {identifier}")
    arrays = {
        "time_s": timestamp - timestamp[0],
        "robot_timestamp": timestamp,
        "seq": np.asarray(sequences, dtype=np.int64),
        "recv_ns": np.asarray([record[1] for record in records], dtype=np.int64),
        "recv_monotonic_ns": np.asarray(
            [record[2] for record in records], dtype=np.int64
        ),
    }
    stats = {}
    for key, (width, _) in SIGNALS.items():
        values = matrix(rows, key, width)
        arrays[key] = values
        arrays[f"{key}_valid"] = np.isfinite(values).all(axis=1)
        stats[key] = signal_stats(values)
    for side in ("left", "right"):
        arrays[f"{side}_target_active"] = np.asarray(
            [row.get(f"{side}_ee_target") is not None for row in rows]
        )
        arrays[f"{side}_gripper_measured_flag"] = np.asarray(
            [
                1
                if row.get(f"{side}_gripper_measured") is True
                else 0
                if row.get(f"{side}_gripper_measured") is False
                else -1
                for row in rows
            ],
            dtype=np.int8,
        )
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "episode_id": identifier,
        "file": f"native-replay/{target.name}",
        "frames": len(rows),
        "duration_s": float(timestamp[-1] - timestamp[0]),
        "first_timestamp": float(timestamp[0]),
        "last_timestamp": float(timestamp[-1]),
        "max_timestamp_gap_s": float(np.diff(timestamp).max())
        if len(rows) > 1
        else None,
        "signals": stats,
        "gripper_feedback_flags": {
            side: {
                label: int((arrays[f"{side}_gripper_measured_flag"] == code).sum())
                for code, label in (
                    (-1, "unknown"),
                    (0, "command_fallback"),
                    (1, "measured"),
                )
            }
            for side in ("left", "right")
        },
    }


def import_capture(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    final = json.loads((source / "summary.json").read_text())
    live = json.loads((source / "status.json").read_text())
    if final.get("running") is not False or live.get("running") is not False:
        raise ValueError("Stop the listener before importing a capture")
    if output == source or source in output.parents:
        raise ValueError("Import destination must be outside the original capture")
    source_files = [
        p for p in sorted(source.rglob("*")) if p.is_file() and p.name != "STOP"
    ]
    if any(path.is_symlink() for path in source_files):
        raise ValueError("Capture source must contain regular files, not symlinks")
    source_hashes = {
        str(path.relative_to(source)): sha256(path) for path in source_files
    }
    output.mkdir(parents=True, exist_ok=False)
    exports = output / "native-replay"
    exports.mkdir()
    quality, events, episodes = Quality(), [], []
    seen = set()
    rows = wire_rows(source, quality, events)
    for index, (identifier, records) in enumerate(
        groupby(rows, key=lambda item: item[0]["episode_id"])
    ):
        if identifier in seen:
            raise ValueError(
                "Interleaved/restarted episode ID requires explicit segmentation"
            )
        seen.add(identifier)
        episodes.append(
            export_episode(
                list(records), exports / f"episode_{index:06d}.npz", identifier
            )
        )
    if not episodes:
        raise ValueError("Capture contains no robot frames")
    checked = quality.summary()
    if any(
        checked[key] != final[key] for key in ("packets", "bad_packets", "episodes")
    ):
        raise ValueError(
            "Recomputed packet quality differs from the stopped listener summary"
        )
    if checked["bad_packets"] or not all(
        ep["sequence_capture_complete"] for ep in checked["episodes"].values()
    ):
        raise ValueError(
            "Capture is incomplete; resolve sequence/packet issues before importing"
        )
    for episode in episodes:
        episode_events = [
            event
            for event in events
            if event["payload"].get("episode_id") == episode["episode_id"]
        ]
        starts = [
            event["payload"]
            for event in episode_events
            if event["topic"] == "episode/signal"
            and event["payload"].get("active") is True
        ]
        ends = [
            event["payload"]
            for event in episode_events
            if event["topic"] == "episode/signal"
            and event["payload"].get("active") is False
        ]
        episode["start_metadata"] = starts
        episode["end_metadata"] = ends
    with (output / "episode-events.jsonl").open("x") as stream:
        for event in events:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    # Store all original evidence. gzip has deterministic headers and is lossless.
    for path in source_files:
        relative = path.relative_to(source)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name in ("frames.jsonl", "packets.jsonl"):
            destination = destination.with_suffix(destination.suffix + ".gz")
            with (
                path.open("rb") as src,
                destination.open("xb") as dst,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=dst, mtime=0, compresslevel=6
                ) as compressed,
            ):
                shutil.copyfileobj(src, compressed)
        else:
            shutil.copyfile(path, destination)
        if sha256(path) != source_hashes[str(relative)]:
            raise ValueError(f"Source changed during import: {relative}")
    total = sum(ep["frames"] for ep in episodes)
    summary = {
        "format_version": 1,
        "status": "captured_and_imported_not_parameter_fitted",
        "source_session": source.name,
        "episodes": episodes,
        "episode_count": len(episodes),
        "frames": total,
        "active_recording_seconds": sum(ep["duration_s"] for ep in episodes),
        "packets_including_episode_events": checked["packets"],
        "quality": {
            key: sum(ep[key] for ep in checked["episodes"].values())
            for key in (
                "missing_sequences",
                "duplicate_sequences",
                "out_of_order_sequences",
                "invalid_sequences",
                "invalid_timestamps",
                "nonincreasing_timestamps",
            )
        },
        "bad_packets": checked["bad_packets"],
        "task_counts": dict(
            Counter(
                start.get("task", "unspecified")
                for ep in episodes
                for start in ep["start_metadata"]
            )
        ),
        "signal_schema": {
            key: {"width": width, "units_and_semantics": units}
            for key, (width, units) in SIGNALS.items()
        },
        "missing_values": "NaN with explicit *_valid masks; no filling or resampling. Gripper measured flag: -1 unknown, 0 fallback, 1 measured.",
        "hardware_calibrated": False,
        "simulation_parameters_changed": False,
        "use": "Native-time command/feedback evidence for later replay and held-out validation. Original packet and frame archives retain every source field.",
        "limitations": [
            "Sequence completeness does not establish sensor validity or physical fidelity.",
            "Gripper command nulls are not reconstructed; confirm source command-hold semantics before replay.",
            "Absent gripper provenance flags remain unknown, not measured or confirmed fallback.",
            "Lift coordinates remain in publisher units; constant lift data cannot identify speed or stroke.",
            "Joint-derived end-effector poses and wheel-derived velocity are not independent pose measurements.",
            "No video, force, payload mass or exact command dispatch/interpolation timings are supplied by this stream.",
            "Task/outcome labels are publisher annotations, not independently verified task success.",
        ],
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Sophia robot capture\n\n"
        "This is calibration evidence, not an applied parameter fit. See summary.json "
        "for coverage, missing fields and limitations.\n\n"
        "- source/ contains the stopped capture and its original sim-baseline snapshots. "
        "The two large JSONL files are compressed losslessly with gzip.\n"
        "- native-replay/ contains one NPZ per episode in source order. Load with "
        "numpy.load(path, allow_pickle=False). Keep validity masks and NaNs.\n"
        "- episode-events.jsonl retains task, cycle, outcome and episode boundary messages.\n"
        "- MANIFEST.json hashes all imported files and the original uncompressed sources.\n\n"
        "The older arm benchmark requires finite gripper commands; these nullable exports "
        "must not be passed to it without resolving command/feedback semantics first.\n"
    )
    manifest = {
        "format_version": 1,
        "source_session": source.name,
        "source_uncompressed_sha256": source_hashes,
        "importer_sha256": sha256(Path(__file__)),
        "files": {
            str(path.relative_to(output)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "MANIFEST.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = import_capture(args.source, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "episode_count",
                    "frames",
                    "active_recording_seconds",
                    "quality",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
