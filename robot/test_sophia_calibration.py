"""Synthetic regression fixtures for offline calibration math and input handling.

These fixtures are not robot measurements or evidence of hardware calibration.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot import sophia_calibration as calibration  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def linear_rows(self):
        return [
            {
                "sample_id": str(i),
                "split": "train" if i < 3 else "validation",
                "encoder_value": str(x),
                "shoulder_height_m": str(0.8128 + 0.8 * x),
            }
            for i, x in enumerate((0.0, 0.2, 0.7, 0.3, 0.5))
        ]

    def test_affine_lift_mapping_and_independent_validation(self):
        rows = self.linear_rows()
        rows[3]["shoulder_height_m"] = str(float(rows[3]["shoulder_height_m"]) + 0.01)
        result = calibration.fit_linear(rows, "encoder_value", "shoulder_height_m")
        self.assertAlmostEqual(result["slope"], 0.8)
        self.assertAlmostEqual(result["intercept"], 0.8128)
        self.assertLess(result["training_error_m"]["max_absolute"], 1e-12)
        self.assertAlmostEqual(result["validation_error_m"]["max_absolute"], 0.01)

    def test_inverted_gripper_raw_feedback_is_not_clamped(self):
        rows = [
            {
                "sample_id": str(i),
                "split": "train" if i < 3 else "validation",
                "side": "left",
                "raw_width_um": str(raw),
                "jaw_gap_m": str(0.1016 * (77000 - raw) / 84000),
            }
            for i, raw in enumerate((77000, 35000, -7000, 14000))
        ]
        result = calibration.fit_grippers(rows)["left"]
        self.assertEqual(result["feedback_direction"], "inverted")
        self.assertLess(result["validation_error_m"]["max_absolute"], 1e-12)
        self.assertEqual(result["input_range"], [-7000.0, 77000.0])

    def test_missing_holdout_is_not_labeled_validated(self):
        result = calibration.fit_linear(
            self.linear_rows()[:3], "encoder_value", "shoulder_height_m"
        )
        self.assertEqual(result["status"], "fit_only_no_independent_validation")
        self.assertIsNone(result["validation_error_m"])

    def test_degenerate_and_nonfinite_fits_rejected(self):
        for bad in ("nan", "inf"):
            rows = self.linear_rows()
            rows[0]["encoder_value"] = bad
            with self.assertRaises(ValueError):
                calibration.fit_linear(rows, "encoder_value", "shoulder_height_m")
        rows = self.linear_rows()
        for row in rows:
            row["encoder_value"] = "5"
        with self.assertRaises(ValueError):
            calibration.fit_linear(rows, "encoder_value", "shoulder_height_m")

    def test_rigid_transform_recovers_rotation_and_translation(self):
        source = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4], [0.1, 0.1, 0.1]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([0.3, -0.1, 0.2])
        rows = []
        for i, (a, b) in enumerate(zip(source, source @ rotation.T + translation, strict=True)):
            rows.append(
                {
                    "sample_id": str(i),
                    "split": "train" if i < 4 else "validation",
                    **{f"source_{axis}_m": a[j] for j, axis in enumerate("xyz")},
                    **{f"target_{axis}_m": b[j] for j, axis in enumerate("xyz")},
                }
            )
        result = calibration.fit_transform(rows)
        actual = np.array(result["T_target_source"])
        np.testing.assert_allclose(actual[:3, :3], rotation, atol=1e-12)
        np.testing.assert_allclose(actual[:3, 3], translation, atol=1e-12)
        self.assertLess(result["validation_position_error_m"]["max_absolute"], 1e-12)
        for row in rows:
            row["source_y_m"] = row["source_z_m"] = 0
        with self.assertRaises(ValueError):
            calibration.fit_transform(rows)

    def test_series_alignment_and_bias(self):
        candidate = [{"time_s": t, "lift_m": 0.2 * t + 0.003} for t in (0.0, 0.05, 0.1)]
        actual = [{"time_s": t, "lift_m": 0.2 * t} for t in (0.025, 0.075)]
        result = calibration.compare_series(actual, candidate, ["lift_m"])
        self.assertAlmostEqual(result["signal_errors_in_input_units"]["lift_m"]["rmse"], 0.003)

    def test_series_rejects_extrapolation_gaps_and_duplicates(self):
        for times in ((0.0, 0.0), (0.0, 1.0), (0.05, 0.1)):
            candidate = [{"time_s": t, "lift_m": t} for t in times]
            actual = [{"time_s": t, "lift_m": t} for t in (0.025, 0.075)]
            with self.assertRaises(ValueError):
                calibration.compare_series(actual, candidate, ["lift_m"], max_gap_s=0.1)

    def test_raw_can_decoding_preserves_overtravel_and_filters_mode(self):
        payload = (
            (-7000).to_bytes(4, "big", signed=True)
            + (-12).to_bytes(2, "big", signed=True)
            + bytes([3, 0])
        )
        row = {"time_s": 0.0, "side": "right", "arbitration_id": "0x2a8", "data_hex": payload.hex()}
        result = calibration.decode_gripper(
            [row, {**row, "data_hex": (payload[:-1] + b"\x01").hex()}]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["raw_width_um"], -7000)
        self.assertEqual(result[0]["effort_i16_raw"], -12)
        with self.assertRaises(ValueError):
            calibration.decode_gripper([{**row, "data_hex": "0011"}])

    def test_camera_validation_distinguishes_frame_and_pose(self):
        config = {
            "width": 640,
            "height": 480,
            "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            "distortion_model": "opencv",
            "distortion": [0] * 5,
            "parent_frame": "head_tilt",
            "T_parent_camera_optical": np.eye(4).tolist(),
        }
        self.assertFalse(calibration.check_camera(config)["hardware_accuracy_verified"])
        config["parent_frame"] = "chassis_ground"
        with self.assertRaises(KeyError):
            calibration.check_camera(config)
        config["calibration_pose"] = {"lift_m": 0.3, "pan_deg": 2.02, "tilt_deg": -3.25}
        self.assertTrue(calibration.check_camera(config)["pose_specific"])
        config["T_parent_camera_optical"][0][0] = -1
        with self.assertRaises(ValueError):
            calibration.check_camera(config)

    def test_pose_comparison_frame_translation_and_quaternion_sign(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(calibration.DESCRIPTION / "scene-sophia.mjcf"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        site = model.site("left_gripper_center").id
        anchors = [
            data.xanchor[model.joint(f"{end}_{side}_steer").id][:2]
            for end in ("front", "back")
            for side in ("left", "right")
        ]
        position = data.site_xpos[site] - np.r_[np.mean(anchors, axis=0), 0.0]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        row = {
            "sample_id": "synthetic",
            "side": "left",
            "lift_m": 0.3,
            **{
                f"joint{i}_rad": data.qpos[model.joint(f"left_arm_joint{i}").qposadr[0]]
                for i in range(1, 8)
            },
            **dict(zip(("x_m", "y_m", "z_m"), position, strict=True)),
            **dict(zip(("qw", "qx", "qy", "qz"), -quat, strict=True)),
        }
        row["x_m"] += 0.01
        result = calibration.compare_poses([row], "gripper_center")
        self.assertAlmostEqual(result["position_error_m"]["rmse"], 0.01)
        self.assertLess(result["orientation_error_deg"]["max_absolute"], 1e-5)

    def test_templates_are_empty_and_existing_data_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            calibration.write_templates(directory)
            with self.assertRaises(ValueError):
                calibration.load_rows(directory / "lift.csv", calibration.TEMPLATES["lift.csv"])
            with self.assertRaises(FileExistsError):
                calibration.write_templates(directory)

    def test_state_recorder_preserves_data_and_marks_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.jsonl"
            count = calibration.record_states(
                lambda: {"left_gripper": 0.5}, path, 0.015, 0.01, "synthetic test"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreater(count, 0)
            self.assertTrue(records[-1]["complete"])
            self.assertEqual(records[1]["state"]["left_gripper"], 0.5)

            def fail():
                raise ValueError("synthetic capture failure")

            failed = Path(temp) / "failed.jsonl"
            with self.assertRaises(ValueError):
                calibration.record_states(fail, failed, 0.01, 0.01, "synthetic test")
            self.assertFalse(json.loads(failed.read_text().splitlines()[-1])["complete"])

    def test_can_recorder_only_receives(self):
        import struct

        payload = (35000).to_bytes(4, "big") + bytes(4)
        packet = struct.pack("=IB3x8s", 0x2A8, 8, payload)
        delivered = False

        def receive(_):
            nonlocal delivered
            if delivered:
                raise TimeoutError
            delivered = True
            return packet

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(calibration.socket, "socket") as factory,
        ):
            bus = factory.return_value.__enter__.return_value
            bus.recv.side_effect = receive
            path = Path(temp) / "feedback.csv"
            self.assertEqual(calibration.record_gripper("synthetic0", "left", path, 0.005), 1)
            bus.send.assert_not_called()
            bus.sendto.assert_not_called()
            rows = calibration.load_rows(path, calibration.TEMPLATES["gripper_can.csv"])
            self.assertEqual(calibration.decode_gripper(rows)[0]["raw_width_um"], 35000)

    def test_rpc_timeout_marks_incomplete_and_closes_only_the_client(self):
        import zmq

        with tempfile.TemporaryDirectory() as temp, patch("commlink.RPCClient") as factory:
            client = factory.return_value
            client.get_collection_state.side_effect = zmq.Again()
            path = Path(temp) / "state.jsonl"
            with self.assertRaises(TimeoutError):
                calibration.record_remote_states(
                    "synthetic.invalid", 5557, path, 1, 0.01, "synthetic test", timeout_s=0.1
                )
            self.assertFalse(json.loads(path.read_text().splitlines()[-1])["complete"])
            client.socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 100)
            client.socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 100)
            client.socket.close.assert_called_once_with(linger=0)
            client.context.term.assert_called_once()
            client.stop_server.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
