"""Replay recorded neck commands against Sophia, without fitting or robot access.

Input NPZs come from audit_sophia_dataset.py. Preserve native Pi timestamps;
hold each command until the next recorded command. This subsystem benchmark
holds the base, arms and lift at simulation home (not the complete real task).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import YORMujoco  # noqa: E402


def replay(path, output, tilt_rate=None):
    record = np.load(path, allow_pickle=False)
    t = record["time_s"]
    measured, commands = record["neck_deg"], record["neck_cmd_deg"]
    if np.any(np.diff(t) <= 0) or not np.isfinite(np.c_[t, measured, commands]).all():
        raise ValueError(f"Invalid neck/timestamp input: {path}")
    sim = YORMujoco(ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
    if tilt_rate is not None:
        sim._neck_rate[1] = tilt_rate
    m, d = sim.model, sim.data
    adrs, acts = [sim._pan_adr, sim._tilt_adr], [sim._pan_act, sim._tilt_act]
    sign = np.array([-1, 1])
    q0 = sign * np.deg2rad(measured[0] - sim._neck_zero_deg)
    d.qpos[adrs] = q0
    d.ctrl[acts] = q0
    sim._neck_q_cmd[:] = sim._neck_goal[:] = q0
    mujoco.mj_forward(m, d)
    sampled = np.full_like(measured, np.nan)
    sampled[0] = measured[0]
    index, target = 0, 1
    guard_holds = 0
    previous_time, previous_angles = 0.0, measured[0].copy()
    limits = m.jnt_range[[m.joint("head_pan_joint").id, m.joint("head_tilt_joint").id]]
    joint_commands = sign * np.deg2rad(commands - sim._neck_zero_deg)
    if not ((joint_commands >= limits[:, 0]) & (joint_commands <= limits[:, 1])).all():
        raise ValueError("Recorded command exceeds modeled neck limits")
    try:
        while d.time < t[-1]:
            while index + 1 < len(t) and t[index + 1] <= d.time:
                index += 1
            sim._neck_goal[:] = joint_commands[index]
            sim._tick_controls(m.opt.timestep)
            guard_holds += int(sim._guard_holding)
            mujoco.mj_step(m, d)
            if d.warning.number.any() or not np.isfinite(d.qpos).all():
                raise RuntimeError("Non-finite state or MuJoCo warning during replay")
            angles = sim._neck_zero_deg + sign * np.rad2deg(d.qpos[adrs])
            while target < len(t) and t[target] <= d.time:
                alpha = (t[target] - previous_time) / (d.time - previous_time)
                sampled[target] = previous_angles + alpha * (angles - previous_angles)
                target += 1
            previous_time, previous_angles = float(d.time), angles.copy()
        assert np.isfinite(sampled).all()
        # Exclude only the explicitly initialized first second, consistently.
        valid = t >= 1.0
        error = sampled[valid] - measured[valid]
        hardware_tracking = commands[valid] - measured[valid]
        np.savez_compressed(
            output / path.name,
            time_s=t,
            command_deg=commands,
            hardware_deg=measured,
            simulation_deg=sampled,
        )
        return (
            {
                "episode": path.stem,
                "samples": int(valid.sum()),
                "duration_s": float(t[-1]),
                "guard_hold_steps": guard_holds,
                "sim_vs_hardware_rmse_deg": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                "sim_vs_hardware_bias_deg": np.mean(error, axis=0).tolist(),
                "sim_vs_hardware_p95_abs_deg": np.percentile(np.abs(error), 95, axis=0).tolist(),
                "sim_vs_hardware_max_abs_deg": np.max(np.abs(error), axis=0).tolist(),
                "hardware_command_tracking_rmse_deg": np.sqrt(
                    np.mean(hardware_tracking**2, axis=0)
                ).tolist(),
                "mujoco_warnings": d.warning.number.tolist(),
            },
            np.sum(error**2, axis=0),
            int(valid.sum()),
        )
    finally:
        sim.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--label", default="neck-replay-current")
    parser.add_argument("--tilt-rate-rad-s", type=float)
    args = parser.parse_args()
    if Path(args.label).name != args.label:
        raise ValueError("Label must be a directory name")
    if args.tilt_rate_rad_s is not None and (
        args.tilt_rate_rad_s <= 0 or not np.isfinite(args.tilt_rate_rad_s)
    ):
        raise ValueError("Tilt rate must be finite and positive")
    output = args.root / args.label
    output.mkdir(exist_ok=True)
    paths = sorted((args.root / "native-replay").glob("*.npz"))
    if args.episodes is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in args.episodes]
    if not paths:
        raise ValueError("No native replay inputs found")
    reports, squared_error, samples = [], np.zeros(2), 0
    for path in paths:
        row, err, count = replay(path, output, args.tilt_rate_rad_s)
        reports.append(row)
        squared_error += err
        samples += count
        print(json.dumps(row), flush=True)
    sources = [
        Path(__file__),
        ROOT / "robot/yor_mujoco.py",
        ROOT / "robot/yor-description/sophia.mjcf",
        *paths,
    ]
    report = {
        "axis_order": ["pan", "tilt"],
        "episodes": reports,
        "pooled_sim_vs_hardware_rmse_deg": np.sqrt(squared_error / samples).tolist(),
        "samples_after_initial_second": samples,
        "parameters_modified": False,
        "temporary_tilt_rate_override_rad_s": args.tilt_rate_rad_s,
        "hardware_calibrated": False,
        "method": (
            "Native-time zero-order command hold; 5 ms simulation outputs interpolated "
            "to Pi timestamps; no fitted clock shift; first second excluded."
        ),
        "limits": (
            "Neck-only replay with other subsystems held at simulation home; missing physical "
            "lift scale prevents full measured-pose replay. Aggregate robot snapshots are "
            "not per-servo timestamps."
        ),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (args.root / f"{args.label}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("Pooled RMSE degrees:", report["pooled_sim_vs_hardware_rmse_deg"], flush=True)


if __name__ == "__main__":
    main()
