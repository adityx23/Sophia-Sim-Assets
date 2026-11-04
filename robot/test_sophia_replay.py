"""Synthetic comparison/transport regression tests. Never connects to a robot."""

import base64
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.sophia_replay import (  # noqa: E402
    POSE_FRAME,
    RECORDING_SCHEMA,
    build_trajectory,
    trajectory_hash,
    validate_trajectory,
)
from robot.sophia_replay_compare import compare, interpolate, normalize_pose  # noqa: E402
from robot.sophia_replay_io import capture, import_capture, import_teleop  # noqa: E402


def fixture(full):
    """Analytical test data, not an actual MuJoCo or hardware observation."""
    trajectory = copy.deepcopy(full)
    trajectory["frames"] = trajectory["frames"][:61]
    trajectory["duration_s"] = trajectory["frames"][-1]["time_s"]
    trajectory["sha256"] = trajectory_hash(trajectory)
    commands = [
        {**f, "dispatch_time_s": f["time_s"], "status": "accepted"} for f in trajectory["frames"]
    ]
    initial = {
        k: trajectory["initial"][k]
        for k in (
            "left_joints",
            "right_joints",
            "left_gripper",
            "lift_extension_m",
            "neck_deg",
        )
    }
    initial["steer_rad"] = [0.0] * 4
    samples = [
        {
            "seq": f["seq"],
            "time_s": f["time_s"],
            "left_joints": f["left_joints"],
            "left_gripper": 1.0,
            "base_pose": [0.1 * f["time_s"], 0.0, 0.0],
            "base_pose_time_s": f["time_s"],
            "base_velocity_body": [0.1, 0, 0],
            "drive_m_s": [0.1] * 4,
            "steer_rad": [0.0] * 4,
        }
        for f in trajectory["frames"]
    ]
    reference = {
        "schema": RECORDING_SCHEMA,
        "source": "simulation",
        "run_id": "analytical-fixture",
        "trajectory_sha256": trajectory["sha256"],
        "clock": "seconds_from_run_start",
        "pose_frame": POSE_FRAME,
        "pose_source": "mujoco_ground_truth",
        "initial_state": initial,
        "complete": True,
        "end_reason": "complete",
        "events": [],
        "commands": commands,
        "samples": samples,
        "synthetic_fixture": True,
    }
    actual = copy.deepcopy(reference)
    actual.update(
        source="synthetic",
        pose_source="synthetic_base_pose",
        run_id="synthetic-observed",
    )
    return trajectory, reference, actual


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = build_trajectory()

    def setUp(self):
        self.trajectory, self.reference, self.actual = fixture(self.full)

    def result(self):
        return compare(self.trajectory, self.reference, self.actual)

    def test_reference_generation_is_deterministic_and_bounded(self):
        second = build_trajectory()
        self.assertEqual(second["sha256"], self.full["sha256"])
        self.assertTrue(self.full["no_objects"])
        frames = self.full["frames"]
        self.assertAlmostEqual(sum(f["base_velocity"][0] / 30 for f in frames), 0.25, places=8)
        validate_trajectory(self.full)

    def test_tampered_hash_and_preview_are_rejected(self):
        self.trajectory["frames"][1]["left_gripper"] = 0
        with self.assertRaises(ValueError):
            validate_trajectory(self.trajectory)
        self.trajectory["sha256"] = trajectory_hash(self.trajectory)
        self.trajectory["frames"][1]["preview_time_s"] = 1
        self.trajectory["sha256"] = trajectory_hash(self.trajectory)
        with self.assertRaises(ValueError):
            validate_trajectory(self.trajectory)

    def test_identical_runs_have_zero_error_without_hardware_claim(self):
        r = self.result()
        self.assertTrue(r["matched_command_data_eligible"])
        self.assertFalse(r["hardware_validated"])
        self.assertEqual(r["status"], "software_self_check")
        np.testing.assert_allclose(r["metrics"]["left_joints"]["rmse"], 0, atol=1e-12)
        np.testing.assert_allclose(r["metrics"]["base_pose"]["endpoint_error"], 0, atol=1e-12)

    def test_known_joint_bias_and_lateral_drift_are_recovered(self):
        for row in self.actual["samples"]:
            row["left_joints"][0] += 0.02
            row["base_pose"][1] = 0.03 * row["time_s"]
        r = self.result()
        self.assertAlmostEqual(r["metrics"]["left_joints"]["rmse"][0], 0.02)
        self.assertAlmostEqual(r["metrics"]["base_pose"]["endpoint_error"][1], 0.06)

    def test_initial_pose_alignment_handles_rotation_and_translation_only(self):
        for row in self.actual["samples"]:
            x = row["base_pose"][0]
            row["base_pose"] = [3, 4 + x, np.pi / 2]
        np.testing.assert_allclose(self.result()["metrics"]["base_pose"]["rmse"], 0, atol=1e-12)
        pose = normalize_pose(np.array([[0, 0, 3.13], [0, 0, -3.13]]))
        self.assertLess(abs(pose[1, 2]), 0.03)

    def test_missing_pose_cannot_be_replaced_with_wheel_odometry(self):
        self.actual["pose_source"] = "wheel_odometry"
        r = self.result()
        self.assertFalse(r["metrics"]["base_pose"]["available"])
        self.assertTrue(r["metrics"]["drive_m_s"]["available"])
        self.assertFalse(r["matched_command_data_eligible"])

    def test_missing_commands_are_not_assumed_to_match(self):
        self.actual["commands"] = []
        self.assertIn("same_commands_not_verified", self.result()["quality_issues"])

    def test_simulator_pose_cannot_be_declared_an_independent_robot_measurement(self):
        self.actual.update(source="robot", pose_source="mujoco_ground_truth")
        r = self.result()
        self.assertFalse(r["metrics"]["base_pose"]["available"])
        self.assertEqual(r["status"], "diagnostic_only")

    def test_changed_command_is_detected(self):
        self.actual["commands"][15]["left_joints"][1] += 0.1
        self.assertFalse(self.result()["command_parity"]["verified"])

    def test_initial_setup_mismatch_is_flagged(self):
        self.actual["initial_state"]["lift_extension_m"] += 0.02
        self.assertIn("initial_conditions_not_matched", self.result()["quality_issues"])

    def test_interruption_remains_a_failed_completeness_gate(self):
        self.actual["events"] = [{"event": "manual_takeover", "time_s": 1.0}]
        r = self.result()
        self.assertIn("operator_intervention_or_fault", r["quality_issues"])
        self.assertFalse(r["matched_command_data_eligible"])

    def test_delay_is_reported_without_time_warping(self):
        for row in self.actual["commands"]:
            row["dispatch_time_s"] += 0.15
        for row in self.actual["samples"]:
            row["time_s"] += 0.15
            row["base_pose_time_s"] += 0.15
        r = self.result()
        self.assertAlmostEqual(r["command_parity"]["dispatch_minus_planned_time"]["median_ms"], 150)
        self.assertLess(r["metrics"]["left_joints"]["coverage"], 0.95)
        self.assertFalse(r["metrics"]["base_pose"]["available"])

    def test_gaps_duplicates_and_trailing_loss(self):
        del self.actual["samples"][10]
        r = self.result()
        self.assertEqual(r["missing_sample_sequences"], 1)
        self.actual["samples"].insert(1, self.actual["samples"][0])
        with self.assertRaises(ValueError):
            self.result()
        self.actual = copy.deepcopy(self.reference)
        self.actual["declared_sample_count"] = len(self.actual["samples"])
        self.actual["samples"] = self.actual["samples"][:-5]
        self.assertEqual(self.result()["missing_sample_sequences"], 5)

    def test_no_interpolation_across_long_gaps_or_extrapolation(self):
        result = interpolate(
            np.array([0.0, 0.03, 1.0]),
            np.array([[0.0], [3.0], [100.0]]),
            np.array([-0.1, 0.015, 0.5, 1.0, 1.1]),
        )
        self.assertAlmostEqual(result[1, 0], 1.5)
        self.assertEqual(result[3, 0], 100.0)
        self.assertTrue(np.isnan(result[[0, 2, 4]]).all())

    def test_stale_pose_samples_do_not_create_fresh_pose_coverage(self):
        for row in self.actual["samples"]:
            row["base_pose"] = [0, 0, 0]
            row["base_pose_time_s"] = 0.0
        self.assertFalse(self.result()["metrics"]["base_pose"]["available"])

    def test_wrong_trajectory_hash_is_rejected(self):
        self.actual["trajectory_sha256"] = "unrelated"
        with self.assertRaises(ValueError):
            self.result()

    def test_empty_capture_never_reports_measurement_and_outputs_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            result = capture("tcp://127.0.0.1:1", path, 0.01)
            self.assertFalse(result["received_any_packet"])
            self.assertFalse(result["hardware_commands_sent"])
            with self.assertRaises(FileExistsError):
                capture("tcp://127.0.0.1:1", path, 0.01)

    def test_legacy_import_is_observations_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            rows = [
                {
                    "episode_id": "legacy",
                    "seq": r["seq"],
                    "timestamp": 100 + r["time_s"],
                    "left_joints": r["left_joints"],
                    "left_gripper": 1.0,
                    "left_gripper_measured": False,
                    "enc_vel": [0.1, 0, 0],
                }
                for r in self.actual["samples"]
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            actual = import_teleop(path, "legacy", 100.0, self.trajectory)
            self.assertIsNone(actual["samples"][0]["left_gripper"])
            self.assertFalse(actual["complete"])
            r = compare(self.trajectory, self.reference, actual)
            self.assertEqual(r["status"], "diagnostic_only")
            self.assertFalse(r["matched_command_data_eligible"])

    def test_json_csv_png_and_html_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report"
            r = compare(self.trajectory, self.reference, self.actual, target)
            self.assertEqual(json.loads((target / "comparison.json").read_text()), r)
            self.assertGreater((target / "comparison.png").stat().st_size, 1000)
            self.assertIn("no robot measurements", (target / "report.html").read_text())
            self.assertEqual(len((target / "aligned-samples.csv").read_text().splitlines()), 62)

    def test_loopback_capture_preserves_raw_bytes_and_imports_one_run(self):
        with tempfile.TemporaryDirectory() as directory, zmq.Context() as context:
            pub = context.socket(zmq.XPUB)
            pub.setsockopt(zmq.LINGER, 0)
            pub.setsockopt(zmq.RCVTIMEO, 5000)
            port = pub.bind_to_random_port("tcp://127.0.0.1")
            out = Path(directory) / "capture"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/sophia_replay.py"),
                    "listen",
                    "--endpoint",
                    f"tcp://127.0.0.1:{port}",
                    "--duration",
                    "2",
                    "--output",
                    str(out),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                subscriptions = {pub.recv() for _ in range(4)}
                self.assertIn(b"\x01sophia/replay/", subscriptions)
                meta = {
                    k: v
                    for k, v in self.actual.items()
                    if k not in ("samples", "commands", "events")
                }
                messages = [("meta", meta)]
                messages.extend(
                    ("command", {**r, "run_id": meta["run_id"]}) for r in self.actual["commands"]
                )
                messages.extend(
                    ("state", {**r, "run_id": meta["run_id"]}) for r in self.actual["samples"]
                )
                messages.append(
                    (
                        "event",
                        {
                            "run_id": meta["run_id"],
                            "event": "complete",
                            "time_s": 2.0,
                            "sample_count": 61,
                        },
                    )
                )
                wire = []
                for topic, value in messages:
                    parts = [
                        f"sophia/replay/{topic}".encode(),
                        json.dumps(value, indent=2).encode(),
                    ]
                    wire.append(parts)
                    pub.send_multipart(parts)
                pub.send_multipart([b"sophia/replay/state", b"malformed", b"extra"])
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                captured = [
                    json.loads(line) for line in (out / "packets.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [base64.b64decode(x) for x in captured[0]["parts_base64"]], wire[0]
                )
                self.assertEqual(
                    json.loads((out / "summary.json").read_text())["malformed_packets"],
                    1,
                )
                actual = import_capture(out / "packets.jsonl", meta["run_id"], self.trajectory)
                self.assertTrue(
                    compare(self.trajectory, self.reference, actual)[
                        "matched_command_data_eligible"
                    ]
                )
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                pub.close(linger=0)


if __name__ == "__main__":
    unittest.main()
