"""Preserve user-confirmed concentric mounting through lift and neck motion."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_sophia_head_camera import candidate_model  # noqa: E402


class HeadAlignmentTests(unittest.TestCase):
    def test_supplied_lattice_keeps_source_dimensions_and_mounting_contact(self):
        m = mujoco.MjModel.from_xml_path(str(ROOT / "robot/yor-description/scene-sophia.mjcf"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        gid = m.geom("spinal_neck_lattice").id
        mid = m.geom_dataid[gid]
        start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        world = (
            m.mesh_vert[start : start + count] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        )
        neck = m.body("head_neck").id
        local = (world - d.xpos[neck]) @ d.xmat[neck].reshape(3, 3)
        low, high = local.min(0), local.max(0)
        np.testing.assert_allclose(high - low, [0.075, 0.09, 0.10895000194], atol=1e-7)
        np.testing.assert_allclose((high + low)[:2] / 2, 0, atol=1e-7)
        self.assertAlmostEqual(low[2], 0, places=7)
        profile = json.loads((ROOT / "robot/yor-description/sophia.json").read_text())
        expected = high[2]
        if profile.get("spinal_assembly", {}).get("enabled", False):
            cad = json.loads(
                (
                    ROOT / "robot/yor-description/calibration/spinal_assembly_2026-11-04.json"
                ).read_text()
            )
            expected = cad["pan_position_neck_m"][2]
        self.assertAlmostEqual(m.body_pos[m.body("head_pan").id, 2], expected, places=7)
        self.assertEqual(int(m.geom_contype[gid]), 0)
        self.assertEqual(int(m.mesh_facenum[mid]), 12983)

    def test_neutral_lens_midpoint_matches_confirmed_plate_front(self):
        m = mujoco.MjModel.from_xml_path(str(ROOT / "robot/yor-description/scene-sophia.mjcf"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[m.joint("head_pan_joint").qposadr[0]] = 0
        d.qpos[m.joint("head_tilt_joint").qposadr[0]] = 0
        mujoco.mj_forward(m, d)
        torso = m.body("Lift_Top").id
        r = d.xmat[torso].reshape(3, 3)
        eyes = [m.camera("head_zed_" + side).id for side in ("left", "right")]
        midpoint = r.T @ (d.cam_xpos[eyes].mean(axis=0) - d.xpos[torso])
        np.testing.assert_allclose(midpoint[:2], [-0.0508, 0], atol=1e-10)
        plate = m.geom("shoulder_bridge").id
        front_x = m.geom_pos[plate, 0] - m.geom_size[plate, 0]
        self.assertAlmostEqual(midpoint[0], front_x)
        housing = m.geom("head_zed2").id
        collider = m.geom("zed_housing_collision").id
        np.testing.assert_allclose(m.geom_pos[housing], m.geom_pos[collider], atol=1e-12)
        baseline = np.linalg.norm(d.cam_xpos[eyes[0]] - d.cam_xpos[eyes[1]])
        self.assertAlmostEqual(baseline, 0.1199918518066, places=10)

    def test_pillar_plate_and_neck_share_vertical_centreline(self):
        m = mujoco.MjModel.from_xml_path(str(ROOT / "robot/yor-description/scene-sophia.mjcf"))
        d = mujoco.MjData(m)
        neck = m.body("head_neck").id
        column, plate, bridge = [
            m.geom(name).id for name in ("upper_column", "neck_mount_plate", "shoulder_bridge")
        ]
        for height in (0, 0.3, 0.5588):
            for pan, tilt in ((0, 0), (0.6, 0.8), (-0.6, -0.3)):
                with self.subTest(lift=height, pan=pan, tilt=tilt):
                    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
                    for name, value in (
                        ("lift_joint", height),
                        ("sophia_middle_slide", height / 2),
                        ("head_pan_joint", pan),
                        ("head_tilt_joint", tilt),
                    ):
                        d.qpos[m.joint(name).qposadr[0]] = value
                    mujoco.mj_forward(m, d)
                    for point in (d.geom_xpos[plate], d.geom_xpos[bridge], d.xpos[neck]):
                        np.testing.assert_allclose(point[:2], d.geom_xpos[column, :2], atol=1e-9)
                    self.assertAlmostEqual(
                        d.geom_xpos[column, 2] + m.geom_size[column, 2],
                        d.geom_xpos[bridge, 2] - m.geom_size[bridge, 2],
                    )
                    gap = d.xpos[neck, 2] - (d.geom_xpos[plate, 2] + m.geom_size[plate, 2])
                    self.assertLessEqual(abs(gap), 0.002)

    def test_superseded_offset_fit_cannot_be_activated(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "concentric neck/pillar"),
        ):
            candidate_model(Path(temporary))


if __name__ == "__main__":
    unittest.main()
