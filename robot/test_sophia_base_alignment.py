"""Regress evidence binding and preserve live-vs-historical lift semantics."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "robot/yor-description"))
from robot.sophia_profile import encoder_to_lift_extension  # noqa: E402
from robot.sophia_recording_alignment import recording_lift_offset  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.render_continuous_teleop import measured_pose  # noqa: E402
from sophia_sources import apply_base_alignment  # noqa: E402

DESC = ROOT / "robot/yor-description"


class BaseAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((DESC / "sophia.json").read_text())

    def test_changed_dependencies_and_dimensions_fail_before_geometry_mutation(self):
        for key in ("base_frame_width_m", "base_frame_height_m"):
            cfg = copy.deepcopy(self.cfg)
            cfg["measured"][key] += 0.01
            with self.assertRaisesRegex(ValueError, "dimensions"):
                apply_base_alignment(None, cfg, 0.1)
        cfg = copy.deepcopy(self.cfg)
        cfg["arm_reference"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "head and arm"):
            apply_base_alignment(None, cfg, 0.1)

    def test_historical_offset_requires_exact_recording_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.npz"
            path.write_bytes(b"fixture recording")
            profile = json.loads((DESC / self.cfg["base_alignment"]["source"]).read_text())
            profile["replay_alignment"]["episodes"] = {
                "0": dict(
                    native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    lift_extension_offset_m=-0.036,
                )
            }
            target = Path(tmp) / "profile.json"
            target.write_text(json.dumps(profile))
            cfg = copy.deepcopy(self.cfg)
            cfg["base_alignment"].update(
                source=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()
            )
            self.assertEqual(recording_lift_offset(path, 0, cfg), -0.036)
            with self.assertRaisesRegex(ValueError, "does not cover"):
                recording_lift_offset(path, 1, cfg)
            path.write_bytes(b"changed recording")
            with self.assertRaisesRegex(ValueError, "does not cover"):
                recording_lift_offset(path, 0, cfg)
            target.write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                recording_lift_offset(path, 0, cfg)

    def test_opt_in_alignment_moves_torso_and_preserves_arm_camera_pose(self):
        sim = YORMujoco(DESC / "scene-sophia.mjcf", headless=True)
        try:
            m, d = sim.model, sim.data
            mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
            a = dict(lift_height=np.array([[0.59153]]), neck_deg=np.array([[2.02, -3.25]]))
            for side, arm in sim.arms.items():
                a[side + "_joints"] = d.qpos[arm.qpos_adr][None, :].copy()
                a[side + "_gripper"] = np.array([0.5])

            def capture():
                cid = m.camera("head_zed_left").id
                ee = d.site_xpos[m.site("right_arm_ee").id]
                return d.qpos[sim._lift_adr], d.cam_xpos[cid].copy(), ee - d.cam_xpos[cid]

            measured_pose(sim, d, a, 0)
            h, cam, relative = capture()
            self.assertAlmostEqual(h, float(encoder_to_lift_extension(0.59153)))
            measured_pose(sim, d, a, 0, lift_extension_offset_m=-0.036)
            hn, cn, rn = capture()
            self.assertAlmostEqual(hn, h - 0.036)
            np.testing.assert_allclose(cn - cam, [0, 0, -0.036], atol=1e-10)
            np.testing.assert_allclose(rn, relative, atol=1e-10)
            with self.assertRaisesRegex(ValueError, "outside physical travel"):
                measured_pose(sim, d, a, 0, lift_extension_offset_m=1.0)
        finally:
            sim.stop()

    def test_compiled_frame_retains_tape_dimensions_and_floor_clearance(self):
        m = mujoco.MjModel.from_xml_path(str(DESC / "scene-sophia.mjcf"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        ids = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(("upper_rail_", "lower_rail_"))
            and not m.geom(g).name.endswith("_collision")
        ]
        lower = np.min([d.geom_xpos[g] - m.geom_size[g] for g in ids], axis=0)
        upper = np.max([d.geom_xpos[g] + m.geom_size[g] for g in ids], axis=0)
        np.testing.assert_allclose(upper - lower, [0.1905, 0.4318, 0.2032], atol=1e-9)
        self.assertAlmostEqual(lower[2], 0.0635)
        self.assertGreater((upper + lower)[0] / 2, 0.15)


if __name__ == "__main__":
    unittest.main()
