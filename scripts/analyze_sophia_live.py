"""Audit live-follow logs and export measured responses for offline identification.

Never changes model parameters. Samples preserve separate Pi/workstation clocks.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(directory, output):
    directory, output = Path(directory), Path(output)
    metadata = json.loads((directory / "metadata.json").read_text())
    rows, damaged = [], []
    for number, line in enumerate((directory / "events.jsonl").read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except ValueError:
            damaged.append(number)
    requests = {r["request"]["seq"]: r for r in rows if r["kind"] == "request"}
    responses = [r for r in rows if r["kind"] == "response"]
    accepted, issues = [], []
    for row in responses:
        request = requests.get(row["seq"])
        if request is None:
            issues.append("response_without_request")
            continue
        response = row["response"]
        if (
            response.get("active")
            and not response.get("fault")
            and request["request"]["command"]["enabled"]
        ):
            methods = {
                x["method"] for x in response.get("dispatched", []) if x["status"] != "rejected"
            }
            if {
                "set_left_joint_target",
                "set_right_joint_target",
                "set_base_velocity",
                "set_neck_position",
            } <= methods:
                accepted.append((request, row))
    if damaged:
        issues.append("incomplete_or_invalid_log_lines")
    if not accepted:
        issues.append("no_dispatched_motion_samples")
    faults = [r for r in rows if r["kind"] in ("transport_fault", "sim_fault")]
    if faults or any(r["response"].get("fault") for r in responses):
        issues.append("session_interrupted_by_fault")
    output.mkdir(parents=True, exist_ok=False)
    fields = {}
    if accepted:
        fields["sequence"] = np.array([r["seq"] for _, r in accepted])
        fields["robot_timestamp"] = np.array(
            [r["response"]["state"]["timestamp"] for _, r in accepted]
        )
        fields["workstation_send_monotonic"] = np.array([r["send_monotonic"] for _, r in accepted])
        fields["workstation_receive_monotonic"] = np.array(
            [r["receive_monotonic"] for _, r in accepted]
        )
        fields["pi_reply_monotonic"] = np.array(
            [r["response"]["pi_monotonic"] for _, r in accepted]
        )
        fields["sim_sample_monotonic"] = np.array(
            [q["sim"]["sample_monotonic"] for q, _ in accepted]
        )
        for key, width in (
            ("left_joints", 7),
            ("right_joints", 7),
            ("left_gripper", 1),
            ("right_gripper", 1),
            ("neck_deg", 2),
            ("lift_height", 1),
            ("enc_vel", 3),
        ):
            for prefix in ("robot", "sim"):
                values = []
                for q, r in accepted:
                    state = r["response"]["state"] if prefix == "robot" else q["sim"]
                    value = state.get(key)
                    values.append(
                        np.full(width, np.nan)
                        if value is None
                        else np.asarray(value).reshape(width)
                    )
                fields[prefix + "_" + key] = np.array(values)
        for key in (
            "left_joints",
            "right_joints",
            "left_gripper",
            "right_gripper",
            "neck_deg",
            "base_velocity",
            "lift_direction",
            "lift_encoder_target_m",
        ):
            fields["command_" + key] = np.array([q["request"]["command"][key] for q, _ in accepted])
        for side in ("left", "right"):
            fields[side + "_dispatch_pi_monotonic"] = np.array(
                [
                    next(
                        d["time_s"]
                        for d in r["response"]["dispatched"]
                        if d["method"] == "set_" + side + "_joint_target"
                    )
                    for _, r in accepted
                ]
            )
        times = fields["robot_timestamp"]
        if np.any(np.diff(times) <= 0):
            issues.append("non_monotonic_robot_snapshot_clock")
        if len(times) > 1 and np.max(np.diff(times)) > 0.1:
            issues.append("robot_sample_gaps_over_100ms")
    np.savez_compressed(output / "identification-input.npz", **fields)
    rtts = [r["rtt_s"] for r in responses]
    metrics = {}
    for side in ("left", "right"):
        if accepted:
            error = fields["robot_" + side + "_joints"] - fields["command_" + side + "_joints"]
            metrics[side + "_robot_tracking_rmse_rad"] = np.sqrt(np.mean(error**2, axis=0)).tolist()
    report = dict(
        schema="sophia.live.audit.v1",
        source=metadata.get("source"),
        hardware_validated=False,
        requests=len(requests),
        replies=len(responses),
        dispatched_samples=len(accepted),
        damaged_lines=damaged,
        issues=sorted(set(issues)),
        rtt_median_ms=None if not rtts else float(np.median(rtts) * 1000),
        rtt_p95_ms=None if not rtts else float(np.percentile(rtts, 95) * 1000),
        metrics=metrics,
        contract_sha256=metadata["contract_sha256"],
        clock_note=(
            "Pi dispatch times, Pi sensor snapshot timestamps, sim sample times and SSH "
            "round trips remain separate. No unmeasured time shift is fitted."
        ),
        calibration_note=(
            "Unloaded tracking data. Keep complete sessions for held-out validation; "
            "force, friction, payload and camera extrinsics are not identified "
            "by this recording."
        ),
        independent_base_pose=False,
        model_parameters_modified=False,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.session, args.output), indent=2))
