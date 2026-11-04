"""Audit downloaded real-robot telemetry before using it for sim calibration.

Requires numpy and pyarrow in the analysis environment; leaves the simulator's
locked environment and parameters untouched. Exports native timestamps and
nullable measurements for replay without reusing resampled frames as new data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATASET = "adityx23/Pi-Atlas_Sanity_Check_Dataset_1080p"
REVISION = "ed7c9f2f1a78ceec46543229aacb0387120ea8f7"
SIGNALS = {
    "left_joints": 7,
    "right_joints": 7,
    "left_joints_cmd": 7,
    "right_joints_cmd": 7,
    "neck_deg": 2,
    "neck_cmd_deg": 2,
    "cmd_vel": 3,
    "enc_vel": 3,
    "lift_height": 1,
    "lift_cmd": 1,
    "left_gripper": 1,
    "right_gripper": 1,
    "left_gripper_cmd": 1,
    "right_gripper_cmd": 1,
    "left_ee": 7,
    "right_ee": 7,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(rows, key, width):
    result = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            continue
        values = np.asarray(value, dtype=float).reshape(-1)
        if len(values) != width:
            raise ValueError(f"Unexpected width: {key}, row {i}")
        result[i] = values
    return result


def stats(values):
    finite = np.isfinite(values)
    valid_rows = finite.all(axis=1)
    clean = values[valid_rows]
    return {
        "valid_rows": int(valid_rows.sum()),
        "total_rows": len(values),
        "min": clean.min(axis=0).tolist() if len(clean) else None,
        "max": clean.max(axis=0).tolist() if len(clean) else None,
        "range": np.ptp(clean, axis=0).tolist() if len(clean) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source, exports = root / "source", root / "native-replay"
    exports.mkdir(exist_ok=True)
    tree = json.loads((root / "tree.json").read_text())
    files = {x["path"]: x for x in tree if x["type"] == "file"}
    info = json.loads((source / "meta/info.json").read_text())
    labels = [json.loads(s) for s in (source / "meta/episodes.jsonl").read_text().splitlines()]
    declared = {r["episode_index"] for r in labels}
    available = {
        int(Path(name).stem.split("_")[-1])
        for name in files
        if name.startswith("data/") and name.endswith(".parquet")
    }
    hashes = {}
    # Verify downloaded bytes against the pinned remote file inventory.
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(source))
        meta = files[name]
        data = path.read_bytes()
        assert len(data) == meta["size"], name
        if "lfs" in meta:
            assert sha(path) == meta["lfs"]["oid"], name
        else:
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            assert digest == meta["oid"], name
        hashes[name] = sha(path)

    episodes, all_signals, total_masks = [], {key: [] for key in SIGNALS}, {}
    pose_counts = dict.fromkeys(
        ("base_pose", "cam_pose", "zed_timestamp", "left_ee_target", "right_ee_target"), 0
    )
    all_keys = set()
    for index in sorted(available):
        name = f"episode_{index:06d}"
        table = pq.read_table(source / f"native/chunk-000/{name}/teleop.parquet")
        rows = [json.loads(s) for s in table["raw"].to_pylist()]
        t = np.array([r["timestamp"] for r in rows])
        dt = np.diff(t)
        all_keys.update(set().union(*(r.keys() for r in rows)))
        arrays = {key: matrix(rows, key, width) for key, width in SIGNALS.items()}
        for key, array in arrays.items():
            all_signals[key].append(array)
        for key in pose_counts:
            pose_counts[key] += sum(r.get(key) is not None for r in rows)
        targeted = np.array([r.get("right_ee_target") is not None for r in rows])
        np.savez_compressed(
            exports / f"{name}.npz",
            time_s=t - t[0],
            robot_timestamp=t,
            recv_ns=np.asarray(table["recv_ns"]),
            right_target_active=targeted,
            **arrays,
        )
        converted = pq.read_table(source / f"data/chunk-000/{name}.parquet")
        masks = {}
        for column in converted.column_names:
            if column.startswith("annotation.") and column.endswith(
                ("valid", "inferred", "filled")
            ):
                arr = np.asarray(converted[column].to_pylist(), dtype=bool)
                masks[column] = {"true": int(arr.sum()), "elements": int(arr.size)}
                current = total_masks.setdefault(column, {"true": 0, "elements": 0})
                current["true"] += int(arr.sum())
                current["elements"] += int(arr.size)
        robot_indices = np.asarray(converted["source.robot_row_index"])
        valid_indices = robot_indices[robot_indices >= 0]
        error = arrays["right_joints_cmd"] - arrays["right_joints"]
        valid = targeted & np.isfinite(error).all(axis=1)
        episodes.append(
            {
                "episode_index": index,
                "native_rows": len(rows),
                "resampled_rows": converted.num_rows,
                "duration_s": float(t[-1] - t[0]),
                "timestamp_nonincreasing": int((dt <= 0).sum()),
                "native_median_dt_s": float(np.median(dt)),
                "native_max_gap_s": float(dt.max()),
                "native_gaps_over_100ms": int((dt > 0.1).sum()),
                "reused_robot_rows_in_resampled_data": int(
                    len(valid_indices) - len(np.unique(valid_indices))
                ),
                "signals": {key: stats(array) for key, array in arrays.items()},
                "right_target_active_rows": int(targeted.sum()),
                "right_joint_command_feedback_rmse_rad": np.sqrt(
                    np.mean(error[valid] ** 2, axis=0)
                ).tolist(),
                "masks": masks,
            }
        )
    all_signals = {k: np.concatenate(v) for k, v in all_signals.items()}
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "hardware_configuration": (
            "User confirms Oct 26 recordings match Oct 29 reference configuration."
        ),
        "metadata_episodes": info["total_episodes"],
        "available_episodes": len(available),
        "missing_episode_indices": sorted(declared - available),
        "native_rows": sum(e["native_rows"] for e in episodes),
        "resampled_rows": sum(e["resampled_rows"] for e in episodes),
        "native_duration_s": sum(e["duration_s"] for e in episodes),
        "signals": {key: stats(array) for key, array in all_signals.items()},
        "non_null_pose_target_counts": pose_counts,
        "raw_keys": sorted(all_keys),
        "masks": total_masks,
        "episodes": episodes,
        "limits": [
            "Five metadata episodes have no files in the pinned repository revision.",
            "Lift command/height are constant within episodes; no physical height pairs.",
            "Left arm never targeted; left gripper never commanded. Only right side varies.",
            "Base/camera pose are null; enc_vel is derived wheel kinematics, not ground motion.",
            "Raw wheel channels, raw gripper widths/forces and joint torques are absent.",
            "EE poses are FK outputs in the supplied driver, not external pose measurements.",
            "No valid IMU; camera/robot matching allows 100 ms and can repeat source frames.",
            "No per-recording camera calibration or mounting transforms in this repository.",
        ],
        "completed": [
            "Verified/downloaded available telemetry",
            "Profiled signal coverage",
            "Exported native-timestamp replay inputs",
        ],
        "simulation_parameters_changed": False,
        "source_sha256": hashes,
        "analysis_script_sha256": sha(Path(__file__)),
    }
    (root / "dataset-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(available)} available episodes, {report['native_rows']} native samples.")
    print(f"Missing episodes: {report['missing_episode_indices']}")
    print(f"Report: {root / 'dataset-audit.json'}")


if __name__ == "__main__":
    main()
