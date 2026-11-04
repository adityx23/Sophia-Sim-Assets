"""Hardware Home survives independent geometric reference and rigid rig changes."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import scene  # noqa: E402

from scripts import calibrate_sophia_head as calibration  # noqa: E402


class NeckReferenceTests(unittest.TestCase):
    def test_independent_reference_keeps_home_feedback_and_hardware_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            scene(folder)
            model = mujoco.MjModel.from_xml_path(str(folder / "scene.mjcf"))
            ids = [model.camera("head_zed_" + s).id for s in ("left", "right")]
            r = np.empty(9)
            mujoco.mju_quat2Mat(r, model.cam_quat[ids[0]])
            home = np.array([2.02, -3.25])
            source_zero = np.array(model.numeric("neck_zero_deg").data)
            x = np.array([9.16, 2.0, 3.0, 1.0, 0.01])
            with patch.object(calibration, "OUT", folder):
                path, zero = calibration.candidate(
                    folder, x, home, source_zero, r.reshape(3, 3), model.cam_pos[ids].mean(0)
                )
            sim = YORMujoco(path, ROOT / "robot/yor-description/sophia-ik.mjcf", headless=True)
            try:
                np.testing.assert_allclose(sim._neck_home_deg, home)
                np.testing.assert_allclose(sim._neck_zero_deg, source_zero + [2, -9.16])
                np.testing.assert_allclose(sim._neck_deg(), home, atol=1e-7)
                # Exercise the same public API used by the Pi bridge, without stepping.
                sim._initialized = True
                sim._collision_guard = None
                limits = sim.get_neck_limits()
                calib = json.loads((ROOT / "robot/neck/neck_calibration.json").read_text())
                for name in ("pan", "tilt"):
                    motor = calib["motors"][name]
                    half = (motor["range_max"] - motor["range_min"]) * 180 / 4095
                    np.testing.assert_allclose(limits[name], [-half, half], atol=1e-6)
                self.assertTrue(sim.home_neck())
                sim.data.qpos[[sim._pan_adr, sim._tilt_adr]] = sim._neck_goal
                np.testing.assert_allclose(list(sim.get_neck_position().values()), home, atol=1e-7)
                baseline = np.linalg.norm(sim.model.cam_pos[ids[0]] - sim.model.cam_pos[ids[1]])
                self.assertAlmostEqual(baseline, 0.1199918518066, places=9)
                for name in ("head_zed2", "zed_housing_collision"):
                    gid = sim.model.geom(name).id
                    other = sim.model.geom(
                        "zed_housing_collision" if name == "head_zed2" else "head_zed2"
                    ).id
                    np.testing.assert_allclose(sim.model.geom_pos[gid], sim.model.geom_pos[other])
            finally:
                sim.stop()

    def test_reference_validation_and_home_are_independent(self):
        spec = importlib.util.spec_from_file_location(
            "sophia_sources", ROOT / "robot/yor-description/sophia_sources.py"
        )
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        c = json.loads((ROOT / "robot/neck/neck_calibration.json").read_text())
        h, z = source.neck_reference(
            {"neck_reference": {"geometric_zero_hardware_deg": [2.02, -12.41]}}, c
        )
        np.testing.assert_equal(h, [2.02, -3.25])
        np.testing.assert_equal(z, [2.02, -12.41])
        for value in ([1], [1, float("nan")], [1, 200]):
            with self.assertRaises(ValueError):
                source.neck_reference({"neck_reference": {"geometric_zero_hardware_deg": value}}, c)


if __name__ == "__main__":
    unittest.main()
