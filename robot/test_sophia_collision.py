"""Controller regressions for Sophia self-collision rejection, without hardware.

    python robot/test_sophia_collision.py

Deterministically advances the normal controller and MuJoCo on one thread.
"""

import sys
import json
import threading
import time
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.yor_mujoco import HOME_Q, YORMujoco  # noqa: E402

MODEL = Path(__file__).parent / "yor-description" / "scene-sophia.mjcf"


class SophiaCollisionTests(unittest.TestCase):
    def setUp(self):
        self.sim = YORMujoco(model_path=MODEL, headless=True)
        # Exercise public API decorators without launching a realtime thread.
        self.sim._initialized = True
        for side, arm in self.sim.arms.items():
            arm.ik.init(HOME_Q[side])

    def tearDown(self):
        self.sim.stop()

    def pose(self, height=0.3, hanging=False, opening=1.0):
        s = self.sim
        q = s.model.key("reference" if hanging else "home").qpos
        s.data.qpos[:] = q
        s.data.qvel[:] = 0
        s.data.qpos[s._lift_adr] = height
        s.data.qpos[s.model.joint("sophia_middle_slide").qposadr[0]] = height / 2
        s.lift_setpoint = s._lift_goal = height
        s.lift_dir = 0
        s.data.ctrl[s._lift_act] = height
        for arm in s.arms.values():
            arm.q_cmd = arm.q_goal = s.data.qpos[arm.qpos_adr].copy()
            arm.q_vel[:] = 0
            s.data.ctrl[arm.act_ids] = arm.q_goal
            arm.gripper_goal = arm.gripper_cmd = opening
            for act, opened, closed in arm.gripper_acts:
                value = closed + opening * (opened - closed)
                s.data.qpos[s.model.jnt_qposadr[s.model.actuator_trnid[act, 0]]] = value
                s.data.ctrl[act] = value
        s._neck_q_cmd[:] = s._neck_goal[:] = 0
        s._neck_cmd = None
        s.data.ctrl[[s._pan_act, s._tilt_act]] = 0
        s._guard_holding = False
        mujoco.mj_forward(s.model, s.data)

    def advance(self, seconds):
        s = self.sim
        worst = 0.0
        for _ in range(round(seconds / s.model.opt.timestep)):
            s._tick_controls(s.model.opt.timestep)
            mujoco.mj_step(s.model, s.data)
            for c in s.data.contact:
                bodies = {int(s.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)}
                if 0 not in bodies and frozenset(bodies) not in s._collision_guard.ignored:
                    worst = max(worst, -float(c.dist))
        self.assertFalse(s.data.warning.number.any())
        self.assertLess(worst, 0.001)

    def hanging_targets(self):
        return {
            side: (self.sim.model.key("reference").qpos[a.qpos_adr], 1.0, 2.0)
            for side, a in self.sim.arms.items()
        }

    def test_neck_source_alignment_preserved(self):
        cfg = json.loads((MODEL.parent/"sophia.json").read_text())
        if cfg.get("spinal_assembly",{}).get("enabled",False):
            cad = json.loads((MODEL.parent/cfg["spinal_assembly"]["source"]).read_text())
            m = self.sim.model
            np.testing.assert_allclose(m.body_pos[m.body("head_pan").id],cad["pan_position_neck_m"])
            np.testing.assert_allclose(m.body_pos[m.body("head_tilt").id],cad["tilt_position_pan_m"])
            np.testing.assert_allclose(m.jnt_axis[m.joint("head_tilt_joint").id],cad["tilt_axis_direction_local"])
            self.assertGreater(m.geom("spinal_fixed_adapter_collision").id,0)
            return
        source = mujoco.MjModel.from_xml_path(str(MODEL.parent / "scene-full-agx.mjcf"))
        old = next(
            i
            for i in range(source.ngeom)
            if source.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            and source.mesh(source.geom_dataid[i]).name == "head_topbase2"
        )
        current = self.sim.model
        a = source.geom_pos[old] - source.body_pos[source.body("head_pan").id]
        b = (
            current.geom_pos[current.geom("neck_fixed_bracket_collision").id]
            - current.body_pos[current.body("head_pan").id]
        )
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_guard_does_not_change_physics_margins(self):
        self.assertEqual(float(self.sim.model.geom_margin.max()), 0.0)
        self.assertEqual(float(self.sim._collision_guard.model.geom_margin.max()), 0.005)
        bit = int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.assertFalse(self.sim.model.opt.disableflags & bit)
        self.assertTrue(self.sim._collision_guard.model.opt.disableflags & bit)

    def test_housing_components_preserve_source_triangles_and_world_vertices(self):
        def triangles(path):
            raw = path.read_bytes()
            records = np.frombuffer(raw, offset=84, dtype=[
                ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
            ])
            return records["vertices"].copy()

        def rows(value):
            return np.sort(np.ascontiguousarray(value.reshape(-1, 9)).view("V36").ravel())

        assets = MODEL.parent / "assets_nero"
        source = triangles(assets / "gripper_base.stl")
        parts = sorted((assets / "sophia_collision").glob("sophia_gripper_housing_*.stl"))
        self.assertEqual(len(parts), 5)
        np.testing.assert_array_equal(rows(source), rows(np.concatenate([triangles(p) for p in parts])))
        m, d = self.sim.model, self.sim.data
        self.pose()

        def world_vertices(geom):
            mesh = m.geom_dataid[geom]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            return m.mesh_vert[start:start+count] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]

        for side in ("left", "right"):
            body = m.body(f"{side}_gripper_base").id
            visual = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == body
                          and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2)
            geoms = [m.geom(f"{side}_gripper_housing_{i}_collision").id for i in range(5)]
            before = world_vertices(visual)
            after = np.concatenate([world_vertices(g) for g in geoms])
            self.assertLess(float(cKDTree(before).query(after)[0].max()), 2e-7)
            self.assertLess(float(cKDTree(after).query(before)[0].max()), 2e-7)

    def test_hull_bridge_removed_but_submargin_gap_still_rejected(self):
        self.pose(.1, hanging=True)
        m, d = self.sim.model, self.sim.data
        rail = m.geom("upper_rail_x_1_collision").id
        # Synthetic rail fixture: retain the reproduced 2.9mm mesh/BVH query
        # case independently of the production base's stereo-calibrated location.
        m.geom_pos[rail, :2] = [0, .2059]
        from robot.sim_collision import SelfCollisionGuard
        self.sim._collision_guard = SelfCollisionGuard(m, .005)
        mujoco.mj_forward(m, d)
        pieces = [m.geom(f"left_gripper_housing_{i}_collision").id for i in range(5)]
        distances = [mujoco.mj_geomDistance(m, d, rail, g, .5, None) for g in pieces]
        self.assertGreater(min(distances), .0028)
        self.assertLess(min(distances), .0030)
        contacts = self.sim._collision_guard._contacts(d.qpos)
        distances = [v for pair, v in contacts.items() if rail in pair and any(g in pair for g in pieces)]
        self.assertTrue(distances)
        self.assertTrue(all(0 < distance < .005 for distance in distances))

    def test_reviewed_neck_clearance_replaces_legacy_hull_collision(self):
        self.pose()
        before = self.sim._neck_goal.copy()
        result = self.sim.set_neck_position(-115.08, 72.527)
        cfg = json.loads((MODEL.parent/"sophia.json").read_text())
        if cfg.get("spinal_assembly",{}).get("enabled",False):
            self.assertTrue(result["accepted"])
            self.advance(3)
        else:
            self.assertFalse(result["accepted"])
            np.testing.assert_array_equal(self.sim._neck_goal, before)
            self.assertEqual(self.sim.get_motion_status()["last_rejection"]["reason"], "self_collision")

    def test_clear_neck_extremes_remain_available(self):
        for pan, tilt in ((2.02, 72.527), (125.5, -3.25), (-125.5, -3.25)):
            with self.subTest(pan=pan, tilt=tilt):
                self.pose()
                self.assertTrue(self.sim.set_neck_position(pan, tilt)["accepted"])
                self.advance(3)
                position = self.sim.get_neck_position()
                self.assertLess(abs(position["pan"] - pan), 1.0)
                self.assertLess(abs(position["tilt"] - tilt), 1.0)
        self.assertGreater(self.sim.get_neck_limits()["tilt"][1], 72.0)

    def test_low_lift_hanging_paths_rejected_atomically(self):
        for height in (0.0, 0.1):
            with self.subTest(height=height):
                self.pose(height)
                before = {side: a.q_goal.copy() for side, a in self.sim.arms.items()}
                self.assertFalse(self.sim._set_arm_targets(self.hanging_targets()))
                for side, a in self.sim.arms.items():
                    np.testing.assert_array_equal(a.q_goal, before[side])
                self.advance(0.3)

    def test_hanging_path_executes_at_clearance_height(self):
        self.pose(0.3)
        self.assertTrue(self.sim._set_arm_targets(self.hanging_targets()))
        self.advance(3)
        self.assertEqual(self.sim._motion_stop_serial, 0)
        for arm in self.sim.arms.values():
            self.assertLess(np.max(abs(self.sim.data.qpos[arm.qpos_adr] - arm.q_goal)), 0.03)

    def test_lift_descent_rejects_collision_between_clear_endpoints(self):
        self.pose(0.3, hanging=True, opening=0.0)
        guard = self.sim._collision_guard
        start = self.sim.data.qpos.copy()
        end = start.copy()
        end[self.sim._lift_adr] = 0.0
        self.assertFalse(guard._contacts(start))
        self.assertFalse(guard._contacts(end))
        self.assertFalse(self.sim.lift_down())
        self.assertFalse(self.sim.set_lift_position(0.0))
        self.assertEqual(self.sim.lift_dir, 0)

    def test_gripper_opening_rejected_at_constructed_rail_clearance(self):
        self.pose(0.0, hanging=True, opening=0.0)
        # Place a real rail collider at the known closed-jaw clearance to test
        # opening rejection. The canonical frame has moved away from this pose.
        m = self.sim.model
        m.geom_pos[m.geom("lower_rail_x_1_collision").id, :2] = [0, .2059]
        from robot.sim_collision import SelfCollisionGuard
        self.sim._collision_guard = SelfCollisionGuard(m, .005)
        mujoco.mj_forward(m, self.sim.data)
        self.assertFalse(self.sim.open_left_gripper())
        self.assertEqual(self.sim.arms["left"].gripper_goal, 0.0)

    def test_pending_arm_target_blocks_lift_descent(self):
        self.pose(0.3)
        self.assertTrue(self.sim._set_arm_targets(self.hanging_targets()))
        # Neither arm has moved yet; their future posture must still count.
        self.assertFalse(self.sim.lift_down())
        self.assertEqual(self.sim.lift_dir, 0)

    def test_pending_lift_descent_blocks_arm_target(self):
        self.pose(0.3)
        self.assertTrue(self.sim.lift_down())
        self.assertFalse(self.sim._set_arm_targets(self.hanging_targets()))

    def test_bimanual_cartesian_rejection_does_not_move_either_goal(self):
        self.pose(0.1)
        poses = {}
        before = {side: a.q_goal.copy() for side, a in self.sim.arms.items()}
        for side, arm in self.sim.arms.items():
            q = (
                self.sim.model.key("reference" if side == "right" else "home")
                .qpos[arm.qpos_adr]
                .copy()
            )
            if side == "left":
                q[0] += 0.02
            arm.ik.update_configuration(q)
            poses[side] = arm.ik.forward_kinematics()
        self.assertFalse(self.sim.set_bimanual_ee_target(poses["left"], poses["right"]))
        self.assertEqual(self.sim.get_motion_status()["last_rejection"]["reason"], "self_collision")
        for side, arm in self.sim.arms.items():
            np.testing.assert_array_equal(arm.q_goal, before[side])

    def test_reference_sequence_from_lowered_lift(self):
        self.pose(0.0)
        self.sim._initialized = False
        self.sim.init()
        self.assertTrue(self.sim.move_to_reference_pose(timeout_s=20))
        with self.sim._lock:
            self.assertGreater(self.sim.get_lift_height(), 0.29)
            self.assertEqual(self.sim._motion_stop_serial, 0)
            for arm in self.sim.arms.values():
                expected = self.sim.model.key("reference").qpos[arm.qpos_adr]
                self.assertLess(np.max(abs(self.sim.data.qpos[arm.qpos_adr] - expected)), 0.03)

    def test_lift_stop_cancels_blocking_request_without_restart(self):
        self.pose(0.3)
        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.sim.set_lift_position(0.5)), daemon=True
        )
        worker.start()
        deadline = time.monotonic() + 1.0
        while self.sim.lift_dir == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.sim.lift_dir, 1)
        self.sim.lift_stop()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self.assertEqual(self.sim.lift_dir, 0)

    def test_runtime_guard_stops_bypassed_target_before_contact(self):
        self.pose(0.1)
        # Deliberately bypass command preflight to exercise the feedback guard.
        for side, (q, _, duration) in self.hanging_targets().items():
            self.sim.arms[side].set_goal(q, duration)
        self.advance(4)
        self.assertEqual(self.sim.get_motion_status()["action"], "runtime_hold")
        self.assertGreater(self.sim._motion_stop_serial, 0)
        self.assertEqual(self.sim._motion_stop_serial, 1)

    def test_reference_sequence_yields_to_a_new_arm_command(self):
        self.pose(0.3)
        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.sim.move_to_reference_pose(timeout_s=3)),
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + 2.0
        while np.array_equal(self.sim.arms["left"].q_goal, HOME_Q["left"]):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        self.assertTrue(self.sim.set_left_joint_target(HOME_Q["left"]))
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        np.testing.assert_array_equal(self.sim.arms["left"].q_goal, HOME_Q["left"])

    def test_joint_limits_and_nonfinite_targets_rejected(self):
        self.pose()
        q = HOME_Q["left"].copy()
        q[0] = 10
        self.assertFalse(self.sim.set_left_joint_target(q))
        with self.assertRaises(ValueError):
            self.sim.set_neck_position(float("nan"), 0)
        with self.assertRaises(ValueError):
            self.sim.set_left_gripper(float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
