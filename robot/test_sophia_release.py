"""Check assumed mappings, controller integration and calibrated camera rays."""

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.sim_camera import SophiaCameraRenderer
from robot.sophia_profile import (
    encoder_to_lift_extension,
    lift_extension_to_encoder,
    normalized_gripper_to_gap,
)
from robot.yor_mujoco import YORMujoco

HERE = ROOT / "robot/yor-description"


class ReleaseTests(unittest.TestCase):
    def test_mapping_endpoints_and_invalid_inputs(self):
        # Independent user tape endpoints and supplied summary hard stop.
        np.testing.assert_allclose(
            encoder_to_lift_extension([0.0018, 0.3381, 0.6744]),
            [0, 0.2794, 0.5588],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            lift_extension_to_encoder([0, 0.2794, 0.5588]),
            [0.0018, 0.3381, 0.6744],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            normalized_gripper_to_gap([0, 0.5, 1]), [0, 0.0508, 0.1016]
        )
        for function in (
            encoder_to_lift_extension,
            lift_extension_to_encoder,
            normalized_gripper_to_gap,
        ):
            for invalid in (np.nan, np.inf, -1, 2):
                with self.assertRaises(ValueError):
                    function(invalid)

    def test_rates_match_runtime_and_collision_prediction(self):
        sim = YORMujoco(HERE / "scene-sophia.mjcf", headless=True)
        try:
            for direction, index in ((1, 0), (-1, 1)):
                start = 0.3
                sim.lift_setpoint = start
                sim._lift_goal = 0.5 if direction == 1 else 0.1
                sim.lift_dir = direction
                for _ in range(100):
                    sim._tick_lift(0.01)
                self.assertAlmostEqual(
                    sim.lift_setpoint - start,
                    direction * sim._lift_speed[index],
                    places=9,
                )
                _, rates = sim._motion_plan(lift=sim._lift_goal)
                self.assertAlmostEqual(rates[sim._lift_adr], sim._lift_speed[index])
            for side, opening_rate in (("left", 5.0), ("right", 6.0)):
                arm = sim.arms[side]
                for start, goal, expected_rate in ((0.0, 1.0, opening_rate), (1.0, 0.0, 5.0)):
                    arm.gripper_cmd, arm.gripper_goal = start, goal
                    for act, opened, closed in arm.gripper_acts:
                        joint = sim.model.actuator_trnid[act, 0]
                        sim.data.qpos[sim.model.jnt_qposadr[joint]] = closed + start * (opened - closed)
                    arm.tick(sim.data, 0.01)
                    self.assertAlmostEqual(abs(arm.gripper_cmd - start), 0.01 * expected_rate)
                    _, rates = sim._motion_plan(grippers={side: goal})
                    self.assertAlmostEqual(
                        rates[arm.finger_adr], expected_rate * (arm.finger_open - arm.finger_closed)
                    )
        finally:
            sim.stop()

    def test_default_wrist_output_is_flat_with_no_invalid_border(self):
        model = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        original = model.cam_intrinsic.copy()
        camera = SophiaCameraRenderer(model, 640, 360)
        try:
            for name in ("left_wrist", "right_wrist"):
                with self.subTest(camera=name):
                    rgb, depth, valid, meta = camera.render(data, name)
                    self.assertEqual(rgb.shape, (360, 640, 3))
                    self.assertEqual(meta["image_model"], "pinhole_rectified")
                    self.assertTrue(valid.all())
                    self.assertTrue(np.isfinite(depth).all())
                    self.assertFalse(np.any(meta["distortion"]))
                    np.testing.assert_array_equal(model.cam_intrinsic, original)
                    _, _, _, raw = camera.render(data, name, raw=True)
                    self.assertEqual(raw["image_model"], "opencv_distorted")
                    self.assertTrue(np.any(raw["distortion"]))
        finally:
            camera.close()

    def test_four_camera_projection_depth_and_mounts(self):
        scene = ET.parse(HERE / "scene-sophia.mjcf").getroot()
        scene.find("include").set("file", str(HERE / "sophia.mjcf"))
        ET.SubElement(scene, "compiler", meshdir=str(HERE / "assets_nero"))
        world = scene.find("worldbody")
        if world is None:
            world = ET.SubElement(scene, "worldbody")
        ET.SubElement(
            world,
            "geom",
            name="projection_probe",
            type="sphere",
            size=".012",
            pos="10 10 10",
            rgba="1 0 0 1",
            contype="0",
            conaffinity="0",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        camera = SophiaCameraRenderer(model, 640, 360)
        original = model.cam_intrinsic.copy()
        probe = model.geom("projection_probe").id
        try:
            for name in camera.names:
                for raw in (False, True):
                    with self.subTest(camera=name, raw=raw):
                        cid = model.camera(name).id
                        optical = data.cam_xmat[cid].reshape(3, 3) @ np.diag(
                            [1, -1, -1]
                        )
                        point = np.array([0.045, -0.03, 0.35])
                        model.geom_pos[probe] = data.cam_xpos[cid] + optical @ point
                        mujoco.mj_forward(model, data)
                        rgb, depth, valid, meta = camera.render(data, name, raw=raw)
                        k = np.array(meta["K"])
                        projected, _ = cv2.projectPoints(
                            point,
                            np.zeros(3),
                            np.zeros(3),
                            k,
                            np.array(meta["distortion"]),
                        )
                        u, v = projected.ravel()
                        yy, xx = np.indices(valid.shape)
                        red = (
                            (rgb[..., 0] > 150)
                            & (rgb[..., 1] < 70)
                            & (rgb[..., 2] < 70)
                        )
                        red &= (abs(xx - u) < 20) & (abs(yy - v) < 20)
                        self.assertGreater(red.sum(), 20)
                        self.assertLess(
                            np.hypot(xx[red].mean() - u, yy[red].mean() - v), 2.0
                        )
                        self.assertLess(abs(float(np.median(depth[red])) - 0.35), 0.025)
                        self.assertGreater(valid.mean(), 0.85)
                        self.assertTrue(np.isfinite(depth[valid]).all())
                        np.testing.assert_array_equal(model.cam_intrinsic, original)
                        transform = np.array(meta["T_parent_optical"])
                        np.testing.assert_allclose(
                            transform[:3, :3].T @ transform[:3, :3],
                            np.eye(3),
                            atol=1e-8,
                        )
                        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)
                        if name.endswith("wrist"):
                            side = name.split("_")[0]
                            data.qpos[model.joint(f"{side}_arm_joint7").qposadr[0]] += (
                                0.05
                            )
                            mujoco.mj_forward(model, data)
                            _, _, _, moved = camera.render(data, name, raw=raw)
                            np.testing.assert_allclose(
                                moved["T_parent_optical"], transform, atol=1e-8
                            )
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
