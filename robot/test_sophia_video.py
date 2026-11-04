"""Native MuJoCo -> JPEG -> ZMQ -> OpenCV stereo smoke test; no hardware/network.

MUJOCO_GL=egl python robot/test_sophia_video.py

Uses in-process ZMQ transport and a temporary scene with a visible target to
check nonblank frames, channel order, stereo disparity and metadata.
"""

import json
import sys
import tempfile
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.yor_mujoco import StereoPublisher, YORMujoco  # noqa: E402

DESCRIPTION = Path(__file__).parent / "yor-description"


class StereoVideoTests(unittest.TestCase):
    def test_render_publish_decode_stereo_target(self):
        self.check_stereo_target(320, 180)

    def test_non_native_aspect_preserves_calibrated_projection(self):
        self.check_stereo_target(320, 240)

    def test_separate_process_preserves_calibrated_stereo_projection(self):
        self.check_stereo_target(320, 180, isolated=True)

    def check_stereo_target(self, width, height, isolated=False):
        reference = YORMujoco(model_path=DESCRIPTION / "scene-sophia.mjcf", headless=True)
        left = reference.model.camera("head_zed_left").id
        right = reference.model.camera("head_zed_right").id
        rotation = reference.data.cam_xmat[left].reshape(3, 3)
        center = (reference.data.cam_xpos[left] + reference.data.cam_xpos[right]) / 2
        target = center + rotation @ np.array([0.08, 0.0, -0.7])
        scene = ET.parse(DESCRIPTION / "scene-sophia.mjcf").getroot()
        scene.find("include").set("file", "model.mjcf")
        ET.SubElement(
            scene.find("worldbody"),
            "geom",
            name="stereo_test_target",
            type="sphere",
            pos=" ".join(map(str, target)),
            size=".04",
            rgba="1 0 0 1",
            contype="0",
            conaffinity="0",
        )
        reference.stop()
        with tempfile.TemporaryDirectory() as temp:
            physics = ET.parse(DESCRIPTION / "sophia.mjcf")
            physics.getroot().find("compiler").set(
                "meshdir", str((DESCRIPTION / "assets_nero").resolve())
            )
            physics.write(Path(temp) / "model.mjcf", encoding="unicode")
            path = Path(temp) / "scene.mjcf"
            ET.ElementTree(scene).write(path, encoding="unicode")
            sim = YORMujoco(
                model_path=path, ik_model_path=DESCRIPTION / "sophia-ik.mjcf", headless=True
            )
            endpoint = (
                "ipc://" + str(Path(temp) / "stereo")
                if isolated
                else f"inproc://sophia-stereo-{uuid.uuid4().hex}"
            )
            subscriber = zmq.Context.instance().socket(zmq.SUB)
            subscriber.setsockopt(zmq.SUBSCRIBE, b"head_stereo")
            subscriber.connect(endpoint)
            if isolated:
                from robot.sophia_live_video import ProcessStereoPublisher

                publisher = ProcessStereoPublisher(
                    sim, path, address=endpoint, width=width, height=height, fps=10
                )
            else:
                publisher = StereoPublisher(
                    sim, address=endpoint, width=width, height=height, fps=10
                )
            try:
                publisher.start()
                self.assertTrue(
                    subscriber.poll(10000), "No stereo frame received within 10 seconds"
                )
                topic, payload, metadata = subscriber.recv_multipart()
                self.assertEqual(topic, b"head_stereo")
                info = json.loads(metadata)
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape, (height, width * 2, 3))
                self.assertEqual((info["width"], info["height"]), (width, height))
                source = json.loads(
                    (DESCRIPTION / "calibration/camera_intrinsics_2026-10-30.json").read_text()
                )["zed"]["rectified"]
                self.assertAlmostEqual(info["baseline_m"], source["baseline_m"], places=8)
                self.assertFalse(info["camera_to_robot_mount_calibrated"])
                for name, scale, offset in (
                    ("fx", width / 1920, 0),
                    ("fy", height / 1080, 0),
                    ("cx", width / 1920, 0.5),
                    ("cy", height / 1080, 0.5),
                ):
                    expected = (source["left"][name] + offset) * scale - offset
                    self.assertAlmostEqual(info[name], expected, places=4)
                self.assertGreater(info["fx"], 0)
                centers = []
                for eye in np.split(frame, 2, axis=1):
                    # OpenCV decoded BGR: the target must remain red, not blue.
                    red = (
                        (eye[:, :, 2] > 100)
                        & (eye[:, :, 2] > 1.8 * eye[:, :, 1])
                        & (eye[:, :, 2] > 1.8 * eye[:, :, 0])
                    )
                    yy, xx = np.nonzero(red)
                    self.assertGreater(len(xx), 30)
                    centers.append((float(xx.mean()), float(yy.mean())))
                self.assertGreater(centers[0][0] - centers[1][0], 5.0)
                self.assertLess(abs(centers[0][1] - centers[1][1]), 2.0)
                for side, observed in zip(("left", "right"), centers, strict=True):
                    camera_id = sim.model.camera(f"head_zed_{side}").id
                    local = sim.data.cam_xmat[camera_id].reshape(3, 3).T @ (
                        target - sim.data.cam_xpos[camera_id]
                    )
                    k = info["eyes"][side]
                    projected = [
                        k["fx"] * local[0] / (-local[2]) + k["cx"],
                        k["fy"] * local[1] / local[2] + k["cy"],
                    ]
                    # This tests the rendered JPEG against an independent pinhole
                    # projection, catching principal-point signs and resize errors.
                    np.testing.assert_allclose(observed, projected, atol=1.0, rtol=0)
            finally:
                publisher.stop()
                subscriber.close(linger=0)
                sim.stop()
            self.assertFalse(
                publisher.process.is_alive() if isolated else publisher._thread.is_alive()
            )

    def test_legacy_camera_keeps_fovy_projection(self):
        sim = YORMujoco(model_path=DESCRIPTION / "scene-full-agx.mjcf", headless=True)
        try:
            publisher = StereoPublisher(sim, width=320, height=240)
            info = publisher.stereo_info
            expected = 120 / np.tan(np.deg2rad(70) / 2)
            self.assertAlmostEqual(info["fx"], expected)
            self.assertAlmostEqual(info["fy"], expected)
            self.assertEqual((info["cx"], info["cy"]), (159.5, 119.5))
            self.assertEqual(info["intrinsics_source"], "legacy fovy")
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
