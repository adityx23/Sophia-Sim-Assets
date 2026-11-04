"""Compare available hardware summaries with simulation without fitting missing data.

The recorded neck commands are approximated as linear ramps from their reported
endpoints and durations. These are summary-level diagnostics, not raw-log replay
or independent validation. No robot connection or parameter mutation occurs.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from sophia_sources import CAMERA_SOURCE, HERE, MOTION_SOURCE, sha256

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import YORMujoco  # noqa: E402


def make_sim():
    return YORMujoco(model_path=HERE / "scene-sophia.mjcf", headless=True)


def tick(sim, seconds):
    for _ in range(round(seconds / sim.model.opt.timestep)):
        sim._tick_controls(sim.model.opt.timestep)
        mujoco.mj_step(sim.model, sim.data)
        if sim._guard_holding or sim.data.warning.number.any():
            raise RuntimeError(f"Simulation comparison failed: {sim.get_motion_status()}")


def lift_comparison(summary):
    trials = []
    mapping = json.loads((HERE / "sophia.json").read_text())["release"]["lift_mapping"]
    for direction, start, target in (("up", 0.14, 0.29), ("down", 0.29, 0.14)):
        sim = make_sim()
        try:
            sim.data.qpos[sim._lift_adr] = start
            sim.data.qpos[sim.model.joint("sophia_middle_slide").qposadr[0]] = start / 2
            sim.lift_setpoint = sim._lift_goal = start
            tick(sim, 1.0)
            sim._lift_goal, sim.lift_dir = target, 1 if direction == "up" else -1
            times, heights = [], []
            origin = sim.data.time
            for _ in range(round(2.5 / sim.model.opt.timestep)):
                tick(sim, sim.model.opt.timestep)
                times.append(sim.data.time - origin)
                heights.append(sim.data.qpos[sim._lift_adr])
            mask = (np.array(times) > 0.4) & (np.array(times) < 2.2)
            rate = np.polyfit(np.asarray(times)[mask], np.asarray(heights)[mask], 1)[0]
            trials.append(
                {
                    "direction": direction,
                    "sim_physical_rate_mps": float(rate),
                    "hardware_reported_encoder_rate_mps": summary[f"mean_{direction}_rate_mps"],
                    "assumed_physical_rate_mps": summary[f"mean_{direction}_rate_mps"]
                    * mapping["physical_extension_per_encoder_m"],
                    "direct_physical_scale_comparison_valid": False,
                }
            )
        finally:
            sim.stop()
    return {
        "simulation_command_speed_mps_up_down": sim._lift_speed.tolist(),
        "assumed_encoder_mapping": mapping,
        "trials": trials,
        "physical_tape_pairs": len(summary["tape_pairs"]),
        "finding": (
            "Simulation rates use the user-authorized assumed affine encoder mapping. "
            "Physical travel remains measured; the inferred mapping is not independently calibrated."
        ),
    }


def neck_comparison(summary):
    trials = []
    for axis in ("pan", "tilt"):
        for move in summary[axis]["moves"]:
            if move["cmd_duration_s"] < 0.5 or move["meas_rate_dps"] is None:
                continue
            sim = make_sim()
            try:
                k = 0 if axis == "pan" else 1
                sign = -1 if axis == "pan" else 1
                adr = [sim._pan_adr, sim._tilt_adr][k]
                actuator = [sim._pan_act, sim._tilt_act][k]
                start = move["cmd_from_deg"]
                end = move["cmd_to_deg"]
                duration = move["cmd_duration_s"]
                initial = sign * np.deg2rad(start - sim._neck_zero_deg[k])
                sim.data.qpos[adr] = initial
                sim.data.ctrl[actuator] = initial
                sim._neck_q_cmd[k] = sim._neck_goal[k] = initial
                tick(sim, 0.5)
                times, angles, commands = [], [], []
                origin = sim.data.time
                for n in range(round((duration + 0.5) / sim.model.opt.timestep)):
                    t = (n + 1) * sim.model.opt.timestep
                    commanded = start + (end - start) * min(t / duration, 1.0)
                    sim._neck_goal[k] = sign * np.deg2rad(commanded - sim._neck_zero_deg[k])
                    tick(sim, sim.model.opt.timestep)
                    times.append(sim.data.time - origin)
                    angles.append(sim._neck_zero_deg[k] + sign * np.rad2deg(sim.data.qpos[adr]))
                    commands.append(commanded)
                times, angles, commands = map(np.asarray, (times, angles, commands))
                mask = (times > 0.2 * duration) & (times < 0.8 * duration)
                rate = float(np.polyfit(times[mask], angles[mask], 1)[0])
                lag = float(np.median((commands[mask] - angles[mask]) / ((end - start) / duration)))
                trials.append(
                    {
                        "axis": axis,
                        "hardware_move_start": move["t_start"],
                        "from_deg": start,
                        "to_deg": end,
                        "duration_s": duration,
                        "hardware_measured_rate_dps": move["meas_rate_dps"],
                        "sim_reconstructed_ramp_rate_dps": rate,
                        "hardware_reported_tracking_lag_s": move["tracking_lag_s"],
                        "sim_mid_ramp_error_equivalent_lag_s": lag,
                        "sim_final_error_after_half_second_deg": float(angles[-1] - end),
                    }
                )
            finally:
                sim.stop()
    return {
        "simulation_rate_limit_deg_s": np.rad2deg(sim._neck_rate).tolist(),
        "rate_limit_axis_order": ["pan", "tilt"],
        "trials": trials,
        "method": (
            "Independent single-axis linear-ramp reconstructions, other axis at sim home; "
            "mid-ramp slope by least squares. Reported hardware lag estimator is unknown, "
            "so sim equivalent lag is diagnostic, not the same metric."
        ),
        "raw_trace_replay": False,
        "finding": (
            "Use raw commands and feedback on a shared clock for latency/response fitting. "
            "Do not infer camera mounting orientation or change joint limits from motion extrema."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/asset-review/2026-10-30-update"
    )
    args = parser.parse_args()
    data = json.loads(MOTION_SOURCE.read_text())
    report = {
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (MOTION_SOURCE, CAMERA_SOURCE, HERE / "sophia.mjcf", Path(__file__))
        },
        "simulation_configuration_modified": False,
        "hardware_calibrated": False,
        "source_samples": data["run"]["samples"],
        "source_duration_s": data["run"]["duration_s"],
        "lift": lift_comparison(data["lift"]),
        "neck": neck_comparison(data["neck"]),
        "base": {
            "status": "not_fit",
            "reason": (
                "Third-wheel feedback was reconstructed and the report flags an IK/FK mismatch; "
                "no independent ground displacement is present."
            ),
            "reported_reconstruction": data["base_reconstruction"]["replicated_module"],
            "reported_fk_mismatch": data["base_reconstruction"]["fk_mismatch"],
        },
        "grippers": {
            "physical_gap_feedback_pairs": len(data["calibration"]["gripper"]["marker_pairs"]),
            "raw_closed_zero_reported": data["calibration"]["gripper"]["raw_zero"],
            "status": "Intermediate gap mapping and grasp/slip response remain unmeasured.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "hardware-comparison.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Compared {len(report['neck']['trials'])} neck ramps and both lift directions: {path}")


if __name__ == "__main__":
    main()
