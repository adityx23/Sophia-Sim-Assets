"""Check native rollout equivalence and subsystem frames before adopting fits."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import identify_sophia_capture as identification

from robot.yor_mujoco import _SimArm


class IdentificationTests(unittest.TestCase):
    def test_side_gain_overrides_and_invalid_vectors(self):
        sys.path.insert(0, str(identification.HERE))
        from build_sophia import arm_position_gains

        profile = {
            "arm_position_kp": 500,
            "arm_position_kv": 20,
            "arm_position_gains": {"left": {"kp": [300] * 7, "kv": [15] * 7}},
        }
        np.testing.assert_array_equal(
            arm_position_gains(profile, "left")["kp"], [300] * 7
        )
        np.testing.assert_array_equal(
            arm_position_gains(profile, "right")["kp"], [500] * 7
        )
        for invalid in ([1] * 6, [1] * 6 + [np.nan], [1] * 6 + [-1]):
            profile["arm_position_gains"]["left"]["kp"] = invalid
            with self.assertRaises(ValueError):
                arm_position_gains(profile, "left")

    def test_left_and_right_rollout_match_runtime_control_and_stepping(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                experiment = identification.ArmExperiment(side, [2])
                record, model = experiment.records[2], experiment.model
                expected = experiment.predict(2)
                vector = experiment.predict(2, kp=[500] * 7, kv=[20] * 7)
                np.testing.assert_allclose(vector, expected, atol=1e-12)
                data = mujoco.MjData(model)
                mujoco.mj_setState(model, data, record["initial"], identification.STATE)
                mujoco.mj_forward(model, data)
                arm = _SimArm(side, model, str(identification.HERE / "sophia-ik.mjcf"))
                arm.gripper_speed = experiment.arm.gripper_speed
                arm.gripper_opening_speed = experiment.arm.gripper_opening_speed
                arm.q_cmd = arm.q_goal = record[f"{side}_joints"][0].copy()
                arm.gripper_cmd = arm.gripper_goal = float(
                    record[f"{side}_gripper"][0, 0]
                )
                values = [np.r_[data.qpos[arm.qpos_adr], arm.gripper_position(data)]]
                times, last, index = [0.0], None, 0
                while data.time < record["time_s"][-1]:
                    while (
                        index + 1 < len(record["time_s"])
                        and record["time_s"][index + 1] <= data.time
                    ):
                        index += 1
                    command = record[f"{side}_joints_cmd"][index]
                    if last is None or not np.array_equal(last, command):
                        arm.set_goal(command, 2 / 30)
                        last = command
                    arm.gripper_goal = float(record[f"{side}_gripper_cmd"][index, 0])
                    arm.tick(data, model.opt.timestep)
                    mujoco.mj_step(model, data)
                    times.append(float(data.time))
                    values.append(
                        np.r_[data.qpos[arm.qpos_adr], arm.gripper_position(data)]
                    )
                values = np.asarray(values)
                actual = np.column_stack(
                    [np.interp(record["time_s"], times, values[:, j]) for j in range(8)]
                )
                np.testing.assert_allclose(actual, expected, atol=1e-9)
                self.assertFalse(data.warning.number.any())

    def test_neck_fixture_preserves_mass_matrix_and_gravity(self):
        full = mujoco.MjModel.from_xml_path(
            str(identification.HERE / "scene-sophia.mjcf")
        )
        small = identification.neck_fixture()
        original, isolated = mujoco.MjData(full), mujoco.MjData(small)
        mujoco.mj_resetDataKeyframe(full, original, full.key("home").id)
        joints = [full.joint(name).id for name in ("head_pan_joint", "head_tilt_joint")]
        adrs, dofs = full.jnt_qposadr[joints], full.jnt_dofadr[joints]
        for angles in ([0.0, 0.0], [0.3, -0.2], [-0.5, 0.6]):
            original.qpos[adrs] = isolated.qpos[:] = angles
            mujoco.mj_forward(full, original)
            mujoco.mj_forward(small, isolated)
            mass_full, mass_small = (
                np.empty((full.nv, full.nv)),
                np.empty((small.nv, small.nv)),
            )
            mujoco.mj_fullM(full, original, mass_full)
            mujoco.mj_fullM(small, isolated, mass_small)
            np.testing.assert_allclose(
                mass_full[np.ix_(dofs, dofs)], mass_small, atol=1e-10
            )
            np.testing.assert_allclose(
                original.qfrc_bias[dofs], isolated.qfrc_bias, atol=1e-10
            )

    def test_disabled_gripper_commands_are_not_forward_filled(self):
        record = identification.load_episode(2)
        record["left_gripper_cmd"][0] = np.nan
        with (
            patch.object(identification, "load_episode", return_value=record),
            self.assertRaisesRegex(ValueError, "Disabled or missing"),
        ):
            identification.ArmExperiment("left", [2])


if __name__ == "__main__":
    unittest.main()
