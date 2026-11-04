"""Synthetic coverage and provenance tests for portable robot capture imports."""

import base64
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from import_sophia_capture import import_capture, sha256
from listen_sophia_episodes import Quality


def make_capture(source, sequences=(0, 1)):
    source.mkdir()
    frames = []
    for seq in sequences:
        frame = {
            "episode_id": "synthetic",
            "seq": seq,
            "timestamp": 100.0 + seq / 30,
            "cmd_vel": [0, 0, 0],
            "enc_vel": [0, 0, 0],
            "lift_height": 0.6738,
            "lift_cmd": 0.0,
            "neck_deg": [0, 0],
            "neck_cmd_deg": [0, 0],
        }
        for side in ("left", "right"):
            frame.update(
                {
                    f"{side}_joints": [0.1 * seq] * 7,
                    f"{side}_joints_cmd": [0.2 * seq] * 7,
                    f"{side}_gripper": 0.5,
                    f"{side}_gripper_cmd": None if seq == 0 else 1.0,
                }
            )
        frames.append(frame)
    messages = [
        ("episode/signal", {"episode_id": "synthetic", "active": True, "task": "test"}),
        *[("teleop/episode", frame) for frame in frames],
        (
            "episode/signal",
            {
                "episode_id": "synthetic",
                "active": False,
                "success": True,
                "num_frames": max(sequences) + 1,
            },
        ),
    ]
    quality = Quality()
    with (source / "packets.jsonl").open("w") as stream:
        for index, (topic, payload) in enumerate(messages):
            quality.observe(topic, payload)
            stream.write(
                json.dumps(
                    {
                        "received_unix_ns": 200000000000 + index * 33333333,
                        "received_monotonic_ns": 1000000 + index * 33333333,
                        "parts_base64": [
                            base64.b64encode(part).decode()
                            for part in (topic.encode(), json.dumps(payload).encode())
                        ],
                    }
                )
                + "\n"
            )
    (source / "frames.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames)
    )
    (source / "metadata.json").write_text('{"source":"synthetic test"}\n')
    for name in ("summary.json", "status.json"):
        (source / name).write_text(json.dumps({"running": False, **quality.summary()}))


class CaptureImportTests(unittest.TestCase):
    def test_lossless_logs_masks_timestamps_and_task_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "import"
            make_capture(source)
            summary = import_capture(source, output)
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["episode_count"], 1)
            self.assertFalse(summary["simulation_parameters_changed"])
            self.assertFalse(summary["hardware_calibrated"])
            self.assertEqual(summary["task_counts"], {"test": 1})
            with np.load(
                output / "native-replay/episode_000000.npz", allow_pickle=False
            ) as arrays:
                np.testing.assert_allclose(arrays["time_s"], [0, 1 / 30])
                np.testing.assert_array_equal(
                    arrays["left_gripper_cmd_valid"], [False, True]
                )
                self.assertTrue(np.isnan(arrays["left_gripper_cmd"][0, 0]))
                np.testing.assert_array_equal(
                    arrays["left_gripper_measured_flag"], [-1, -1]
                )
                np.testing.assert_allclose(arrays["lift_height"], [[0.6738], [0.6738]])
                self.assertEqual(arrays["left_joints"].shape, (2, 7))
                self.assertEqual(arrays["recv_ns"].dtype, np.int64)
            for name in ("frames.jsonl", "packets.jsonl"):
                with gzip.open(output / "source" / (name + ".gz"), "rb") as stream:
                    self.assertEqual(stream.read(), (source / name).read_bytes())
            manifest = json.loads((output / "MANIFEST.json").read_text())
            for name, info in manifest["files"].items():
                self.assertEqual(sha256(output / name), info["sha256"])
            for name, digest in manifest["source_uncompressed_sha256"].items():
                self.assertEqual(sha256(source / name), digest)
            with self.assertRaises(FileExistsError):
                import_capture(source, output)

    def test_live_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            make_capture(source)
            (source / "status.json").write_text('{"running":true}')
            with self.assertRaisesRegex(ValueError, "Stop the listener"):
                import_capture(source, Path(directory) / "import")

    def test_raw_decoded_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            make_capture(source)
            lines = (source / "frames.jsonl").read_text().splitlines()
            changed = json.loads(lines[0])
            changed["left_gripper_cmd"] = 0.0
            (source / "frames.jsonl").write_text(
                json.dumps(changed) + "\n" + lines[1] + "\n"
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                import_capture(source, Path(directory) / "import")

    def test_missing_sequence_is_not_imported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            make_capture(source, sequences=(0, 2))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                import_capture(source, Path(directory) / "import")


if __name__ == "__main__":
    unittest.main()
