"""Gymnasium semantics, reset isolation, observability and task failure gates."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.sophia_env import OBJECT_Z, SophiaPickPlaceEnv


class TestSophiaEnvironment(unittest.TestCase):
    def test_gymnasium_contract(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            check_env(env, skip_render_check=True)

    def test_seeded_reset_isolates_controller_history_and_step_timing(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            obs0, info0 = env.reset(seed=17)
            q0 = env.data.qpos.copy()
            for _ in range(30):
                env.step(np.r_[np.zeros(7), 0.7])
            self.assertAlmostEqual(env.data.time, 1.0, places=9)
            obs1, info1 = env.reset(seed=17)
            np.testing.assert_array_equal(env.data.qpos, q0)
            for key in obs0:
                np.testing.assert_array_equal(obs0[key], obs1[key])
            self.assertEqual(info0, info1)
            env.reset(seed=18)
            self.assertFalse(
                np.array_equal(
                    env.data.qpos[env.object_qadr : env.object_qadr + 2],
                    q0[env.object_qadr : env.object_qadr + 2],
                )
            )
            self.assertFalse(env.sim._thread.is_alive())

    def test_rgb_has_no_privileged_object_or_camera_transform_fields(self):
        with SophiaPickPlaceEnv(image_width=96, image_height=54) as env:
            obs, _ = env.reset(seed=2)
            self.assertEqual(
                set(obs), {"proprio", "gripper_valid", "head_rgb", "wrist_rgb"}
            )
            self.assertTrue(env.observation_space.contains(obs))
            self.assertGreater(np.std(obs["head_rgb"]), 5)
            self.assertGreater(np.std(obs["wrist_rgb"]), 5)
            image = env.render()
            self.assertEqual(image.shape, (360, 640, 3))

    def test_time_limit_is_truncation_and_requires_reset(self):
        with SophiaPickPlaceEnv(observation_mode="proprio", max_episode_steps=1) as env:
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(8))
            env.reset(seed=0)
            _, _, terminated, truncated, info = env.step(np.r_[np.zeros(7), 1])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["is_success"])
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(8))

    def test_independent_environments_and_registered_entry_point(self):
        import gymnasium as gym

        with (
            gym.make(
                "robot.sophia_env:SophiaPickPlace-v0", observation_mode="proprio"
            ) as a,
            SophiaPickPlaceEnv(observation_mode="proprio") as b,
        ):
            a.reset(seed=4)
            b.reset(seed=5)
            before = b.data.qpos.copy()
            a.step(np.r_[np.zeros(7), 0.5])
            np.testing.assert_array_equal(b.data.qpos, before)
            self.assertEqual(b.data.time, 0)

    def test_transient_collision_ends_episode_at_physics_substep(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            env.reset(seed=0)
            with patch.object(
                env, "_substep_failure", return_value="robot_table_collision"
            ):
                _, _, terminated, truncated, info = env.step(np.r_[np.zeros(7), 1])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["failure_reason"], "robot_table_collision")
            self.assertAlmostEqual(env.data.time, env.model.opt.timestep)

    def test_invalid_action_is_atomic_and_rejection_is_failure(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            env.reset(seed=0)
            q, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
            with self.assertRaises(ValueError):
                env.step(np.full(8, np.nan))
            np.testing.assert_array_equal(env.data.qpos, q)
            np.testing.assert_array_equal(env.data.ctrl, ctrl)
            self.assertEqual(env.steps, 0)
            with patch.object(env.sim, "set_left_joint_target", return_value=False):
                _, _, terminated, truncated, info = env.step(np.r_[np.zeros(7), 1])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["failure_reason"], "command_rejected")
            self.assertEqual(env.adapter.sequence, 0)

    def test_pushing_to_goal_and_airborne_object_do_not_count_as_pick_place(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            env.reset(seed=0)
            # Adversarial state injection ONLY in this test, never in rollout.
            env.data.qpos[env.object_qadr : env.object_qadr + 3] = [
                *env.goal_xy,
                OBJECT_Z - 0.00005,
            ]
            mujoco.mj_forward(env.model, env.data)
            self.assertTrue(env._task_state()["table_supported"])
            for _ in range(12):
                _, _, terminated, _, info = env.step(np.r_[np.zeros(7), 1])
                self.assertFalse(terminated)
                self.assertFalse(info["is_success"])
            env.lifted = True
            env.data.qpos[env.object_qadr + 2] = OBJECT_Z + 0.05
            mujoco.mj_forward(env.model, env.data)
            self.assertFalse(env._task_state()["placed"])

    def test_drop_terminates_with_failure(self):
        with SophiaPickPlaceEnv(observation_mode="proprio") as env:
            env.reset(seed=0)
            env.data.qpos[env.object_qadr + 2] = 0.2
            _, reward, terminated, truncated, info = env.step(np.r_[np.zeros(7), 1])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["is_success"])
            self.assertEqual(info["failure_reason"], "object_dropped")
            self.assertLess(reward, 0)


if __name__ == "__main__":
    unittest.main()
