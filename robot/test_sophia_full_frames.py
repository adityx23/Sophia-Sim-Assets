"""Regress supplied CAD axes, effective arm reference and saved-pose semantics."""

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_collision import SelfCollisionGuard  # noqa: E402

DESC = ROOT / "robot/yor-description"


class FullFrameTests(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_path(str(DESC / "scene-sophia.mjcf"))
        self.d = mujoco.MjData(self.m)

    def test_supplied_part_hashes_and_recovered_axes(self):
        c = json.loads((DESC / "sophia.json").read_text())
        source = json.loads((DESC / c["spinal_assembly"]["source"]).read_text())
        for part in source["parts"]:
            self.assertEqual(
                hashlib.sha256((DESC / "assets_nero" / part["asset"]).read_bytes()).hexdigest(),
                part["sha256"],
            )
        for axis in source["axis_recovery"].values():
            self.assertGreater(axis["inliers"], 40)
            self.assertLess(axis["radial_rms_mm"], 0.005)
        np.testing.assert_allclose(
            self.m.body_pos[self.m.body("head_pan").id], source["pan_position_neck_m"]
        )
        np.testing.assert_allclose(
            self.m.body_pos[self.m.body("head_tilt").id], source["tilt_position_pan_m"]
        )
        np.testing.assert_allclose(
            self.m.jnt_axis[self.m.joint("head_tilt_joint").id], source["tilt_axis_direction_local"]
        )

    def test_physics_and_ik_use_identical_calibrated_arm_mounts(self):
        ik = mujoco.MjModel.from_xml_path(str(DESC / "sophia-ik.mjcf"))
        di = mujoco.MjData(ik)
        for side in ["left", "right"]:
            name = side + "_arm_base_link"
            np.testing.assert_allclose(
                self.m.body_pos[self.m.body(name).id], ik.body_pos[ik.body(name).id], atol=1e-9
            )
            np.testing.assert_allclose(
                self.m.body_quat[self.m.body(name).id], ik.body_quat[ik.body(name).id], atol=1e-9
            )
            for q in [[0] * 7, [0.2, 1.1, -0.8, 1.2, 0.1, 0.2, -0.3]]:
                for model, data in [(self.m, self.d), (ik, di)]:
                    for j, value in enumerate(q, 1):
                        data.qpos[model.joint(f"{side}_arm_joint{j}").qposadr[0]] = value
                    mujoco.mj_forward(model, data)

                def relative(model, data, name=name, side=side):
                    root = model.body(name).id
                    site = model.site(side + "_arm_ee").id
                    return (data.site_xpos[site] - data.xpos[root]) @ data.xmat[root].reshape(3, 3)

                np.testing.assert_allclose(relative(self.m, self.d), relative(ik, di), atol=1e-6)

    def test_calibrated_reference_is_clear_and_old_angles_change_ee_pose(self):
        m = self.m
        guard = SelfCollisionGuard(m, 0.005)
        q = m.key("reference").qpos.copy()
        q[m.joint("lift_joint").qposadr[0]] = 0
        q[m.joint("sophia_middle_slide").qposadr[0]] = 0
        self.assertFalse(any(v < -0.001 for v in guard._contacts(q).values()))
        stale = q.copy()
        for i, v in enumerate([-np.pi / 2, np.pi / 2, np.pi / 2, 0, 0, 0, 0], 1):
            stale[m.joint(f"right_arm_joint{i}").qposadr[0]] = v
        # The corrected base frame no longer occupies the old rail location.
        # An obsolete joint reference is a pose error, not inherently a collision.
        self.d.qpos[:] = q
        mujoco.mj_forward(m, self.d)
        expected = self.d.site_xpos[m.site("right_arm_ee").id].copy()
        self.d.qpos[:] = stale
        mujoco.mj_forward(m, self.d)
        self.assertGreater(np.linalg.norm(self.d.site_xpos[m.site("right_arm_ee").id] - expected), .01)

    def test_reference_preserves_intended_ee_pose_and_home_readings(self):
        m = self.m
        c = json.loads((DESC / "sophia.json").read_text())
        profile = json.loads((DESC / c["arm_reference"]["source"]).read_text())
        old = copy.copy(m)
        b = old.body("right_arm_base_link").id
        delta = Rotation.from_rotvec(
            np.deg2rad(profile["right_mount_rotvec_parent_deg"])
        ).as_matrix()
        r = Rotation.from_quat(old.body_quat[b], scalar_first=True).as_matrix()
        old.body_quat[b] = Rotation.from_matrix(delta.T @ r).as_quat(scalar_first=True)
        d0 = mujoco.MjData(old)
        d0.qpos[:] = old.key("reference").qpos
        for i, v in enumerate([-np.pi / 2, np.pi / 2, np.pi / 2, 0, 0, 0, 0], 1):
            d0.qpos[old.joint(f"right_arm_joint{i}").qposadr[0]] = v
        mujoco.mj_forward(old, d0)
        mujoco.mj_resetDataKeyframe(m, self.d, m.key("reference").id)
        mujoco.mj_forward(m, self.d)
        site = m.site("right_arm_ee").id
        np.testing.assert_allclose(self.d.site_xpos[site], d0.site_xpos[site], atol=1e-6)
        np.testing.assert_allclose(self.d.site_xmat[site], d0.site_xmat[site], atol=1e-6)
        mujoco.mj_resetDataKeyframe(m, self.d, m.key("home").id)
        a = [m.joint("head_" + s + "_joint").qposadr[0] for s in ["pan", "tilt"]]
        feedback = m.numeric("neck_zero_deg").data + np.rad2deg(self.d.qpos[a]) * [-1, 1]
        np.testing.assert_allclose(feedback, [2.02, -3.25], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
