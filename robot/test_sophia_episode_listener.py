"""Synthetic stream tests; no robot connection or hardware commands."""

import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import zmq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from listen_sophia_episodes import Quality, capture


def frame(seq, episode="synthetic"):
    value = {
        "episode_id": episode,
        "seq": seq,
        "timestamp": 100.0 + seq / 30,
        "cmd_vel": [0, 0, 0],
        "enc_vel": [0, 0, 0],
    }
    for side in ("left", "right"):
        value.update(
            {
                f"{side}_joints": [0.0] * 7,
                f"{side}_joints_cmd": [0.0] * 7,
                f"{side}_gripper": 1.0,
                f"{side}_gripper_cmd": 1.0,
                f"{side}_gripper_measured": True,
            }
        )
    return value


class EpisodeListenerTests(unittest.TestCase):
    def test_gap_and_trailing_loss_are_counted(self):
        quality = Quality()
        for seq in (0, 2):
            quality.observe("teleop/episode", frame(seq))
        quality.observe(
            "episode/signal",
            {"episode_id": "synthetic", "active": False, "num_frames": 4},
        )
        result = quality.summary()["episodes"]["synthetic"]
        self.assertEqual(result["missing_sequences"], 2)
        self.assertEqual(result["sequence_gap_events"], 1)
        self.assertFalse(result["sequence_capture_complete"])

    def test_complete_sequence_still_reports_bad_feedback(self):
        quality = Quality()
        value = frame(0)
        value["left_gripper_measured"] = False
        value["right_joints"] = [float("nan")] * 7
        quality.observe("teleop/episode", value)
        quality.observe(
            "episode/signal",
            {"episode_id": "synthetic", "active": False, "num_frames": 1},
        )
        result = quality.summary()["episodes"]["synthetic"]
        self.assertTrue(result["sequence_capture_complete"])
        self.assertEqual(
            result["invalid_or_missing_fields"]["left_gripper_measured_not_true"], 1
        )
        self.assertEqual(result["invalid_or_missing_fields"]["right_joints"], 1)

    def test_duplicates_clock_reversal_and_episode_reset(self):
        quality = Quality()
        for seq in (0, 1, 1, 0):
            quality.observe("teleop/episode", frame(seq))
        quality.observe("teleop/episode", frame(0, episode="second"))
        episodes = quality.summary()["episodes"]
        self.assertEqual(episodes["synthetic"]["duplicate_sequences"], 2)
        self.assertEqual(episodes["synthetic"]["out_of_order_sequences"], 1)
        self.assertEqual(episodes["synthetic"]["nonincreasing_timestamps"], 2)
        self.assertEqual(episodes["second"]["duplicate_sequences"], 0)
        self.assertFalse(episodes["second"]["sequence_capture_complete"])

    def test_existing_directory_is_never_overwritten(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileExistsError),
        ):
            capture("tcp://127.0.0.1:1", directory, 0.1)

    def test_empty_capture_is_not_reported_as_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            result = capture("tcp://127.0.0.1:1", Path(directory) / "empty", 0.05)
            self.assertEqual(result["packets"], 0)
            self.assertFalse(result["received_any_packet"])
            self.assertEqual(result["episodes"], {})

    def test_real_zmq_transport_preserves_wire_and_all_topics(self):
        with tempfile.TemporaryDirectory() as directory, zmq.Context() as context:
            publisher = context.socket(zmq.XPUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.RCVTIMEO, 5000)
            port = publisher.bind_to_random_port("tcp://127.0.0.1")
            output = Path(directory) / "capture"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/listen_sophia_episodes.py"),
                    "--endpoint",
                    f"tcp://127.0.0.1:{port}",
                    "--duration",
                    "3",
                    "--output",
                    str(output),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                # Wait for all subscriptions, not an arbitrary slow-joiner sleep.
                subscriptions = {publisher.recv() for _ in range(3)}
                self.assertEqual(
                    subscriptions,
                    {
                        b"\x01teleop/episode",
                        b"\x01episode/signal",
                        b"\x01teleop/episode_result",
                    },
                )
                messages = [
                    [
                        b"episode/signal",
                        json.dumps(
                            {"episode_id": "synthetic", "active": True}
                        ).encode(),
                    ],
                    [b"teleop/episode", json.dumps(frame(0), indent=2).encode()],
                    [b"teleop/episode", json.dumps(frame(2)).encode()],
                    [b"teleop/episode", b"malformed", b"third frame"],
                    [
                        b"episode/signal",
                        json.dumps(
                            {
                                "episode_id": "synthetic",
                                "active": False,
                                "num_frames": 4,
                            }
                        ).encode(),
                    ],
                    [
                        b"teleop/episode_result",
                        b'{"episode_id":"synthetic","result":"success"}',
                    ],
                ]
                for parts in messages:
                    publisher.send_multipart(parts)
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["packets"], 6)
                self.assertEqual(summary["bad_packets"], 1)
                ep = summary["episodes"]["synthetic"]
                self.assertEqual(ep["missing_sequences"], 2)
                self.assertTrue(ep["start_signal_seen"])
                self.assertTrue(ep["end_signal_seen"])
                raw = [
                    json.loads(line)
                    for line in (output / "packets.jsonl").read_text().splitlines()
                ]
                recovered = [
                    [base64.b64decode(part) for part in packet["parts_base64"]]
                    for packet in raw
                ]
                self.assertEqual(recovered, messages)
                frames = [
                    json.loads(line)
                    for line in (output / "frames.jsonl").read_text().splitlines()
                ]
                self.assertEqual(frames, [frame(0), frame(2)])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                publisher.close()


if __name__ == "__main__":
    unittest.main()
