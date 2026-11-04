"""Offline contract regression tests, including real captured observation rows."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.sophia_policy import (
    DESCRIPTION,
    PolicyContract,
    SophiaPolicyAdapter,
    preprocess_rgb,
)


class FakeRPC:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def set_left_joint_target(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class TestPolicyContract(unittest.TestCase):
    def setUp(self):
        self.adapter = SophiaPolicyAdapter()
        self.packet = {"left_joints": np.zeros(7), "left_gripper": 0.25}
        self.adapter.reset(self.packet)

    def test_delta_units_integration_and_gripper_direction(self):
        rpc = FakeRPC(True)
        self.adapter.dispatch(rpc, [0.5] * 7 + [0])
        self.adapter.dispatch(rpc, [0.5] * 7 + [1])
        np.testing.assert_allclose(rpc.calls[0]["joint_target"], 0.01)
        np.testing.assert_allclose(rpc.calls[1]["joint_target"], 0.02)
        self.assertEqual([p["gripper_target"] for p in rpc.calls], [0, 1])
        self.assertAlmostEqual(rpc.calls[0]["preview_time"], 1 / 30)
        obs = self.adapter.observe(self.packet)
        np.testing.assert_allclose(obs["proprio"][:7], 0)
        np.testing.assert_allclose(obs["proprio"][8:], 0.02)
        self.assertAlmostEqual(obs["proprio"][7], 0.25)

    def test_identical_rpc_payload_and_no_false_hardware_acknowledgement(self):
        packets = []
        for reply, expected in ((True, "accepted"), (None, "sent_unconfirmed")):
            adapter, rpc = SophiaPolicyAdapter(), FakeRPC(reply)
            adapter.reset(self.packet)
            _, status = adapter.dispatch(rpc, [0.2] * 7 + [0.7])
            packets.append(rpc.calls[0])
            self.assertEqual(status, expected)
        self.assertEqual(*packets)

    def test_rejection_exception_and_stale_commit_do_not_advance_target(self):
        command, status = self.adapter.dispatch(FakeRPC(False), [1.0] * 8)
        self.assertEqual(status, "rejected")
        np.testing.assert_array_equal(self.adapter.target, 0)
        self.assertEqual(self.adapter.sequence, 0)
        self.adapter.commit(command)
        with self.assertRaises(ValueError):
            self.adapter.commit(command)

        class BrokenRPC:
            def set_left_joint_target(self, **kwargs):
                raise ConnectionError("disconnected")

        old = self.adapter.target.copy()
        with self.assertRaises(ConnectionError):
            self.adapter.dispatch(BrokenRPC(), [1.0] * 8)
        np.testing.assert_array_equal(old, self.adapter.target)

    def test_limits_and_invalid_actions(self):
        self.adapter.reset({"left_joints": self.adapter.limits[:, 1] - 0.001})
        command = self.adapter.prepare([1.0] * 8)
        self.assertTrue(command.joint_limit_clipped)
        np.testing.assert_allclose(command.joint_target, self.adapter.limits[:, 1])
        for invalid in (
            [0] * 7,
            [np.nan] * 8,
            [np.inf] * 8,
            [0] * 7 + [-0.1],
            [1.1] * 8,
        ):
            with self.assertRaises(ValueError):
                self.adapter.prepare(invalid)

    def test_missing_feedback_never_becomes_commanded_feedback(self):
        for missing in (None, np.nan, -1, 1.1):
            packet = dict(self.packet, left_gripper=missing, left_gripper_cmd=1.0)
            obs = self.adapter.observe(packet)
            self.assertEqual(obs["gripper_valid"].tolist(), [0])
            self.assertEqual(obs["proprio"][7], 0)

    def test_rgb_bgr_and_aspect_preservation(self):
        c = PolicyContract(image_width=32, image_height=18)
        rgb = np.zeros((8, 8, 3), np.uint8)
        rgb[:, :, 0] = 255
        a = preprocess_rgb(rgb, c)
        b = preprocess_rgb(rgb[:, :, ::-1], c, color_order="BGR")
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[9, 16].tolist(), [255, 0, 0])
        self.assertEqual(a[9, 0].tolist(), [0, 0, 0])
        with self.assertRaises(ValueError):
            preprocess_rgb(rgb.astype(float), c)
        with self.assertRaises(ValueError):
            self.adapter.observe(self.packet, {"head_zed_left": rgb})

    def test_real_capture_joint_order_and_gripper_feedback(self):
        path = (
            DESCRIPTION
            / "calibration/teleop_2026-11-02/native-replay/episode_000000.npz"
        )
        with np.load(path) as capture:
            indices = np.flatnonzero(
                capture["left_joints_valid"] & capture["left_gripper_valid"]
            )
            self.assertGreater(len(indices), 10)
            for i in indices[:: max(1, len(indices) // 25)]:
                packet = {
                    "left_joints": capture["left_joints"][i],
                    "left_gripper": float(capture["left_gripper"][i, 0]),
                }
                self.adapter.reset(packet)
                obs = self.adapter.observe(packet)
                np.testing.assert_allclose(
                    obs["proprio"][:7], capture["left_joints"][i], atol=1e-6
                )
                self.assertAlmostEqual(
                    float(obs["proprio"][7]), packet["left_gripper"], places=6
                )
                self.assertEqual(obs["gripper_valid"].tolist(), [1])

    def test_contract_changes_are_detectable(self):
        c = PolicyContract()
        self.assertEqual(c.fingerprint, PolicyContract().fingerprint)
        for other in (
            PolicyContract(side="right"),
            PolicyContract(control_hz=20),
            PolicyContract(image_width=320),
        ):
            self.assertNotEqual(c.fingerprint, other.fingerprint)


if __name__ == "__main__":
    unittest.main()
