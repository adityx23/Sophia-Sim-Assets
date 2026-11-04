"""Live-follow regression tests; fake Pi hardware only, never robot endpoints."""

import copy
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.sophia_live import (  # noqa: E402
    FollowTransport,
    LineChannel,
    LiveLink,
    initialize_sim_from_robot,
    make_contract,
)
from robot.sophia_live_agent import Agent, other_controllers  # noqa: E402
from robot.sophia_live_common import SCHEMA, BoundedRPC, Journal, fingerprint  # noqa: E402
from robot.sophia_profile import lift_extension_to_encoder  # noqa: E402
from robot.yor_mujoco import HOME_Q, YORMujoco  # noqa: E402


class FakeRobot:
    """Instantaneous deterministic response, explicitly not a physical plant model."""

    def __init__(self, contract):
        self.calls = []
        self.state = dict(
            timestamp=time.time(),
            enc_vel=[0.0, 0.0, 0.0],
            base_cmd_vel=[0.0, 0.0, 0.0],
            lift_height=float(lift_extension_to_encoder(0.3)),
            lift_homed=True,
            lift_ref="bottom_hard_stop",
            lift_homing=False,
            lift_cmd=0,
            neck_deg=[2.02, -3.25],
            neck_cmd_deg=[2.02, -3.25],
        )
        for s in ("left", "right"):
            self.state.update(
                {
                    s + "_joints": HOME_Q[s].tolist(),
                    s + "_joints_cmd": HOME_Q[s].tolist(),
                    s + "_gripper": 1.0,
                    s + "_gripper_cmd": 1.0,
                }
            )

    def get_collection_state(self):
        self.calls.append(("get_collection_state", []))
        self.state["timestamp"] = time.time()
        return copy.deepcopy(self.state)

    def set_base_velocity(self, v):
        self.calls.append(("set_base_velocity", np.array(v).tolist()))
        self.state["base_cmd_vel"] = self.state["enc_vel"] = np.array(v).tolist()

    def set_left_joint_target(self, q, g=None, preview=0.1):
        self.arm("left", q, g, preview)

    def set_right_joint_target(self, q, g=None, preview=0.1):
        self.arm("right", q, g, preview)

    def arm(self, s, q, g, preview):
        self.calls.append(("set_" + s + "_joint_target", np.array(q).tolist()))
        self.state[s + "_joints"] = self.state[s + "_joints_cmd"] = np.array(q).tolist()
        if g is not None:
            self.state[s + "_gripper"] = self.state[s + "_gripper_cmd"] = float(g)

    def enable_left_gripper(self):
        self.calls.append(("enable_left_gripper", []))
        self.state["left_gripper_cmd"] = self.state["left_gripper"] = 1.0
        return True

    def enable_right_gripper(self):
        self.calls.append(("enable_right_gripper", []))
        self.state["right_gripper_cmd"] = self.state["right_gripper"] = 1.0
        return True

    def set_neck_position(self, pan, tilt):
        self.calls.append(("set_neck_position", [pan, tilt]))
        self.state["neck_deg"] = self.state["neck_cmd_deg"] = [pan, tilt]

    def lift_up(self):
        self.calls.append(("lift_up", []))
        self.state["lift_cmd"] = 1

    def lift_down(self):
        self.calls.append(("lift_down", []))
        self.state["lift_cmd"] = -1

    def lift_stop(self):
        self.calls.append(("lift_stop", []))
        self.state["lift_cmd"] = 0

    def call(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)


def frame(fake):
    s = fake.state
    return dict(
        enabled=True,
        left_joints=list(s["left_joints"]),
        right_joints=list(s["right_joints"]),
        left_gripper=s["left_gripper"],
        right_gripper=s["right_gripper"],
        base_velocity=[0.0, 0.0, 0.0],
        neck_deg=list(s["neck_deg"]),
        lift_direction=0,
        lift_encoder_target_m=s["lift_height"],
        preview_time_s=1 / 30,
    )


class ManualTransport:
    """Hold acknowledgements until a test releases them; no networking/hardware."""

    def __init__(self, state):
        self.response = dict(state=state, active=False, acknowledged_source_seq=None)
        self.latest = None

    def snapshot(self):
        return copy.deepcopy(self.response), None, 0.0

    def submit(self, command, sample, source_seq, input_received_monotonic=None):
        self.latest = (copy.deepcopy(command), copy.deepcopy(sample), source_seq)

    def acknowledge(self):
        command, _, source_seq = self.latest
        self.response.update(
            active=command["enabled"],
            acknowledged_source_seq=source_seq if command["enabled"] else None,
        )
        if command["enabled"]:
            for key in ("left_joints", "right_joints", "left_gripper", "right_gripper", "neck_deg"):
                self.response["state"][key] = copy.deepcopy(command[key])


class FollowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sim = YORMujoco(ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
        cls.contract = make_contract(sim, profile="commissioning")
        cls.yor_contract = make_contract(sim)
        sim.stop()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake = FakeRobot(self.contract)
        self.journal = Journal(self.root / "pi", dict(source="synthetic"))
        self.now = 100.0
        self.agent = Agent(
            self.fake, self.contract, self.journal, True, lambda: self.now, self.advance
        )
        self.sequence = 0

    def advance(self, seconds):
        self.now += seconds

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def send(self, command, **overrides):
        self.now += 1 / 30
        req = dict(
            schema=SCHEMA,
            contract_sha256=fingerprint(self.contract),
            nonce=self.agent.nonce,
            seq=self.sequence,
            session="test",
            command=command,
        )
        req.update(overrides)
        self.sequence += 1
        return self.agent.handle(req)

    def arm(self):
        self.send({"enabled": False})
        reply = self.send(frame(self.fake))
        self.assertIsNone(reply["fault"])
        self.assertTrue(reply["active"])

    def test_off_is_read_only_and_no_homing(self):
        self.send({"enabled": False})
        self.assertEqual([c[0] for c in self.fake.calls], ["get_collection_state"])

    def test_existing_controller_detection(self):
        proc = self.root / "proc"
        (proc / "123").mkdir(parents=True)
        (proc / "123/cmdline").write_bytes(b"python\0robot/whole_body_teleop.py\0")
        self.assertEqual(
            other_controllers(proc), [dict(pid=123, controller=["whole_body_teleop.py"])]
        )

    def test_second_webxr_controller_is_rejected(self):
        import asyncio

        from aiohttp import WSServerHandshakeError
        from aiohttp.test_utils import TestClient, TestServer

        sys.path.insert(0, str(ROOT / "vr-teleop/server"))
        import server

        async def check():
            app = server.make_app(asyncio.get_running_loop(), lambda p: {"enabled": False})
            async with TestClient(TestServer(app)) as client:
                first = await client.ws_connect("/ws")
                try:
                    with self.assertRaises(WSServerHandshakeError) as raised:
                        await client.ws_connect("/ws")
                    self.assertEqual(raised.exception.status, 409)
                finally:
                    await first.close()

        asyncio.run(check())

    def test_rpc_timeout_does_not_block_forever(self):
        import zmq

        with socket.socket() as port_holder:
            port_holder.bind(("127.0.0.1", 0))
            port = port_holder.getsockname()[1]
        rpc = BoundedRPC(port=port, timeout_ms=20)
        start = time.monotonic()
        try:
            with self.assertRaises(zmq.Again):
                rpc.call("get_collection_state")
            self.assertLess(time.monotonic() - start, 0.5)
            self.assertIsNone(rpc.client)
        finally:
            rpc.close()

    def test_explicit_off_before_enable(self):
        r = self.send(frame(self.fake))
        self.assertIn("OFF", r["fault"])
        self.assertFalse(any(c[0].startswith("set_") for c in self.fake.calls))

    def test_read_only_cannot_arm(self):
        self.agent.allow_motion = False
        self.send({"enabled": False})
        self.assertIn("read-only", self.send(frame(self.fake))["fault"])

    def test_all_six_subsystems_and_actual_dispatch_log(self):
        self.arm()
        c = frame(self.fake)
        c["left_joints"][0] += 0.005
        c["right_joints"][0] -= 0.005
        c["left_gripper"] = c["right_gripper"] = 0.98
        c["base_velocity"] = [0.01, 0, 0]
        c["neck_deg"][0] += 0.2
        c["lift_direction"] = 1
        r = self.send(c)
        self.assertIsNone(r["fault"])
        self.assertEqual(len(r["dispatched"]), 5)
        self.assertEqual(r["state"]["lift_cmd"], 1)
        self.assertTrue(all(d["status"] == "sent_unconfirmed" for d in r["dispatched"]))
        events = [
            json.loads(line)
            for line in (self.journal.directory / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(events[-1]["kind"], "feedback")

    def test_manual_off_stops_base_lift_before_sampling_hold(self):
        self.arm()
        self.fake.calls.clear()
        self.send({"enabled": False})
        self.assertEqual(
            [c[0] for c in self.fake.calls[:3]],
            ["set_base_velocity", "lift_stop", "get_collection_state"],
        )
        self.assertFalse(self.agent.active)

    def test_watchdog_latches_and_never_resumes(self):
        self.arm()
        self.now += 0.3
        self.agent.watchdog()
        self.assertEqual(self.agent.fault, "command_timeout")
        self.send({"enabled": False})
        self.assertFalse(self.send(frame(self.fake))["active"])

    def test_duplicate_and_stale_nonce(self):
        self.arm()
        self.assertIsNotNone(self.send(frame(self.fake), nonce="old")["fault"])

    def test_duplicate_sequence(self):
        self.arm()
        self.assertIsNotNone(self.send(frame(self.fake), seq=0)["fault"])

    def test_delayed_challenge_is_rejected(self):
        self.send({"enabled": False})
        self.now += 0.3
        self.assertIn("Expired", self.send(frame(self.fake))["fault"])

    def test_initial_mismatch_not_homed_automatically(self):
        self.send({"enabled": False})
        c = frame(self.fake)
        c["left_joints"][0] += 0.1
        self.assertIn("mismatch", self.send(c)["fault"])

    def test_missing_measured_gripper_blocks_motion(self):
        self.send({"enabled": False})
        c = frame(self.fake)
        self.fake.state["left_gripper"] = None
        self.assertIn("feedback", self.send(c)["fault"])

    def test_handoff_enables_only_already_open_grippers(self):
        self.agent.enable_open_grippers = True
        self.fake.state["left_gripper_cmd"] = None
        self.fake.state["right_gripper_cmd"] = None
        self.send({"enabled": False})
        self.assertFalse(any(c[0].startswith("enable_") for c in self.fake.calls))
        reply = self.send(frame(self.fake))
        self.assertIsNone(reply["fault"])
        self.assertEqual(sum(c[0].startswith("enable_") for c in self.fake.calls), 2)

    def test_disabled_closed_gripper_is_not_opened(self):
        self.agent.enable_open_grippers = True
        self.fake.state["left_gripper_cmd"] = None
        self.fake.state["left_gripper"] = 0.2
        self.send({"enabled": False})
        self.assertIn("already be open", self.send(frame(self.fake))["fault"])
        self.assertFalse(any(c[0].startswith("enable_") for c in self.fake.calls))

    def test_unhomed_lift_blocks_motion(self):
        self.send({"enabled": False})
        self.fake.state["lift_homed"] = False
        self.assertIn("homed", self.send(frame(self.fake))["fault"])

    def test_excess_speed_is_rejected(self):
        self.arm()
        c = frame(self.fake)
        c["left_joints"][0] += 0.08
        self.assertIn("rate", self.send(c)["fault"])

    def use_yor_contract(self):
        self.contract = copy.deepcopy(self.yor_contract)
        self.agent = Agent(
            self.fake, self.contract, self.journal, True, lambda: self.now, self.advance
        )

    def test_yor_defaults_use_driver_interpolation_without_packet_rate_faults(self):
        self.use_yor_contract()
        self.send({"enabled": False})
        c = frame(self.fake)
        c["preview_time_s"] = 1.0
        self.assertTrue(self.send(c)["active"])
        c = copy.deepcopy(c)
        c["left_joints"][0] += 0.5
        c["neck_deg"][0] += 30
        c["left_gripper"] = 0.0
        c["base_velocity"] = [1.0, 1.0, 1.57]
        reply = self.send(c)
        self.assertIsNone(reply["fault"])
        self.assertTrue(reply["active"])
        self.assertEqual(self.contract["joint_speed_rad_s"], 3.0)
        self.assertEqual(self.contract["neck_home_deg"], [2.02, -3.25])
        np.testing.assert_allclose(self.contract["arm_home"]["left"], HOME_Q["left"])

    def test_yor_defaults_reject_wrong_preview_and_out_of_range_targets(self):
        self.use_yor_contract()
        self.send({"enabled": False})
        self.assertIn("interpolation", self.send(frame(self.fake))["fault"])

    def test_yor_defaults_home_in_sim(self):
        self.use_yor_contract()
        self.fake.state["lift_height"] = 0.648
        self.exercise_home()

    def test_yor_start_waits_for_neck_to_settle_before_alignment(self):
        self.use_yor_contract()
        link, transport, commands = self.pending_link()
        for tilt in (7.12, 3.95, 1.05, -1.23, -2.9):
            transport.response["state"]["neck_deg"] = [2.46, tilt]
            link.tick(commands)
            self.assertFalse(transport.latest[0]["enabled"])
            time.sleep(0.02)
        time.sleep(0.11)
        link.tick(commands)
        self.assertTrue(transport.latest[0]["enabled"])
        self.assertFalse(transport.latest[0]["requires_active"])
        np.testing.assert_allclose(transport.latest[0]["neck_deg"], [2.46, -2.9])

    def test_neck_dispatch_jitter_is_paced_with_existing_limits(self):
        self.arm()
        c = frame(self.fake)
        # Same targets as the reported session's last accepted and rejected steps.
        self.fake.state["neck_deg"] = self.agent.previous["neck_deg"] = [
            4.312612924150347,
            -6.236258521768813,
        ]
        c["neck_deg"] = [4.8975605264437005, -5.569591855102146]
        previous_return = self.now
        # 15 ms between dispatches, although the source ran at 30 Hz.
        self.now += 0.015 - 1 / 30
        r = self.send(c)
        self.assertIsNone(r["fault"])
        neck = next(d for d in r["dispatched"] if d["method"] == "set_neck_position")
        self.assertGreater(neck["time_s"] - previous_return, 0.015)
        self.assertLess(neck["time_s"] - previous_return, 0.05)
        self.assertLessEqual(2 / 3, 20 * (neck["time_s"] - previous_return) + 0.15 + 1e-9)
        events = [json.loads(s) for s in (self.root / "pi/events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["kind"] == "rate_pacing" for e in events))

    def test_pacing_uses_actual_rpc_completion_not_arrival(self):
        self.arm()
        c = frame(self.fake)
        c["neck_deg"][0] += 2 / 3
        # Previous RPC returned 20 ms after its request arrived.
        self.agent.dispatch_completed["set_neck_position"] += 0.02
        prior = self.agent.dispatch_completed["set_neck_position"]
        r = self.send(c)
        self.assertIsNone(r["fault"])
        t = next(d["time_s"] for d in r["dispatched"] if d["method"] == "set_neck_position")
        self.assertGreaterEqual(20 * (t - prior) + 0.15 + 1e-9, 2 / 3)

    def test_large_neck_jump_still_faults_without_dispatch(self):
        self.arm()
        c = frame(self.fake)
        c["neck_deg"][0] += 5
        r = self.send(c)
        self.assertIn("Neck target rate exceeded", r["fault"])
        self.assertEqual(r["dispatched"], [])

    def test_stops_and_expiry_interrupt_pacing_without_dispatch(self):
        # Each case needs an independent endpoint because faults remain latched.
        for cause in ("joystick", "file", "expiry"):
            with self.subTest(cause=cause):
                self.agent = Agent(
                    self.fake, self.contract, self.journal, True, lambda: self.now, self.advance
                )
                self.arm()
                c = frame(self.fake)
                c["neck_deg"][0] += 2 / 3
                self.now += 0.015 - 1 / 30

                def interrupted_sleep(seconds, cause=cause):
                    self.advance(seconds)
                    if cause == "joystick":
                        self.agent.emergency_check = lambda: True
                    elif cause == "file":
                        (self.root / "pi/STOP").touch()
                    else:
                        self.advance(0.3)

                self.agent.sleeper = interrupted_sleep
                r = self.send(c)
                expected = {"joystick": "joystick stop", "file": "STOP file", "expiry": "Expired"}
                self.assertIn(expected[cause], r["fault"])
                self.assertEqual(r["dispatched"], [])
                self.assertFalse(r["active"])
                (self.root / "pi/STOP").unlink(missing_ok=True)

    def test_nan_and_base_overspeed(self):
        self.send({"enabled": False})
        c = frame(self.fake)
        c["base_velocity"][0] = float("nan")
        self.assertIsNotNone(self.send(c)["fault"])

    def test_other_writer_aborts_following(self):
        self.arm()
        c = frame(self.fake)
        self.fake.state["base_cmd_vel"] = [0.1, 0, 0]
        self.assertIn("Another controller", self.send(c)["fault"])

    def test_local_stop_during_dispatch(self):
        self.arm()
        self.agent.emergency_check = lambda: True
        self.assertIn("joystick", self.send(frame(self.fake))["fault"])

    def test_lift_travel_guard(self):
        self.arm()
        self.fake.state["lift_height"] = self.contract["lift_encoder_bounds_m"][1] - 0.005
        c = frame(self.fake)
        c["lift_direction"] = 1
        self.assertIn("travel limit", self.send(c)["fault"])

    def pending_link(self, scene=None):
        sim = YORMujoco(scene or ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
        self.addCleanup(sim.stop)
        initialize_sim_from_robot(sim, self.fake.state, self.contract)
        transport = ManualTransport(self.fake.state)
        link = LiveLink(sim, transport, self.contract, self.journal)
        commands = dict(
            enabled=True,
            arms={s: dict(target=None, gripper=0.5) for s in ("left", "right")},
            neck=dict(pan=10.0, tilt=-10.0),
            base_mode=True,
            base_velocity=[0.01, 0, 0],
        )
        link.tick(commands)
        return link, transport, commands

    def test_targets_wait_for_matching_ack_without_accumulating_steps(self):
        link, transport, commands = self.pending_link()
        first = copy.deepcopy(transport.latest)
        for _ in range(5):
            link.tick(commands)
            self.assertEqual(transport.latest[0], first[0])
            self.assertEqual(transport.latest[2], first[2])
        transport.acknowledge()
        link.tick(commands)
        second = copy.deepcopy(transport.latest)
        self.assertGreater(second[2], first[2])
        self.assertAlmostEqual(second[0]["neck_deg"][0] - first[0]["neck_deg"][0], 2 / 3)
        self.assertAlmostEqual(first[0]["left_gripper"] - second[0]["left_gripper"], 1 / 30)
        # Duplicate acknowledgement of the old target must not release the new one.
        for _ in range(5):
            link.tick(commands)
            self.assertEqual(transport.latest[0], second[0])
            self.assertEqual(transport.latest[2], second[2])
        transport.acknowledge()
        link.tick(commands)
        self.assertAlmostEqual(transport.latest[0]["neck_deg"][0] - second[0]["neck_deg"][0], 2 / 3)

    def test_operator_off_bypasses_pending_ack(self):
        link, transport, commands = self.pending_link()
        commands["enabled"] = False
        link.tick(commands)
        self.assertEqual(transport.latest[0], {"enabled": False})
        self.assertIsNone(link.pending)

    def test_stale_headset_bypasses_pending_ack(self):
        link, transport, commands = self.pending_link()
        link.tick(commands, fresh=False)
        self.assertEqual(transport.latest[0], {"enabled": False})
        self.assertIn("stale", link.input_paused)
        self.assertIsNone(link.fault)
        self.assertIsNone(link.pending)

    def test_stale_pause_requires_fresh_off_then_on_without_restart(self):
        link, transport, commands = self.pending_link()
        transport.acknowledge()
        link.tick(commands, fresh=False)
        transport.acknowledge()
        link.tick(commands)  # Headset recovered, but the old enable is not permission.
        self.assertFalse(transport.latest[0]["enabled"])
        commands["enabled"] = False
        link.tick(commands)
        self.assertIsNone(link.input_paused)
        self.assertIsNone(link.fault)
        commands["enabled"] = True
        link.tick(commands)
        self.assertTrue(transport.latest[0]["enabled"])

    def test_raw_headset_age_cannot_be_refreshed_by_bridge_heartbeat(self):
        link, transport, commands = self.pending_link()
        commands["input_received_monotonic"] = time.monotonic() - 0.25
        commands["input_fresh"] = True
        link.tick(commands, fresh=True)
        self.assertFalse(transport.latest[0]["enabled"])
        self.assertIn("stale", link.input_paused)

    def test_expired_previous_target_does_not_poison_fresh_headset_input(self):
        link, transport, commands = self.pending_link()
        transport.response.update(source_expired=True, active=False)
        commands["input_received_monotonic"] = time.monotonic()
        # This case tests reply identity, not the host's scheduling latency.
        with patch(
            "robot.sophia_live.time.monotonic", return_value=commands["input_received_monotonic"]
        ):
            link.tick(commands)
        self.assertIsNone(link.input_paused)
        self.assertTrue(transport.latest[0]["enabled"])
        self.assertEqual(transport.latest[0]["base_velocity"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(transport.latest[0]["neck_deg"], self.fake.state["neck_deg"])

    def test_headset_base_mode_moves_chassis_in_furnished_scene(self):
        from scripts.render_sophia_demo import make_scene

        sys.path.insert(0, str(ROOT / "vr-teleop/server"))
        from whole_body import WholeBodyConfig, WholeBodyTeleop

        link, transport, _ = self.pending_link(make_scene(self.root))
        tele = WholeBodyTeleop(WholeBodyConfig(start_enabled=True, max_base_linear_mps=0.05))
        controller = dict(
            pos=[0, 1.2, 0],
            quat=[0, 0, 0, 1],
            axes=[0, 0, 0, -1],
            buttons={"5": dict(pressed=True, value=1)},
        )
        payload = dict(
            head=dict(pos=[0, 1.6, 0], quat=[0, 0, 0, 1]),
            left=controller,
            right=dict(controller, axes=[0, 0, 0, 0]),
        )
        start = link.sample()["base_pose"][:2]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            command = tele.update(payload, link.robot_state(), time.monotonic())
            link.tick(command)
            transport.acknowledge()
            time.sleep(1 / 30)
        displacement = float(np.linalg.norm(np.array(link.sample()["base_pose"][:2]) - start))
        self.assertIsNone(link.fault)
        self.assertGreater(displacement, 0.04, f"Chassis only moved {displacement} m")
        artifact = os.environ.get("SOPHIA_LIVE_TEST_ARTIFACTS")
        if artifact:
            destination = Path(artifact) / "base-motion.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    dict(
                        chassis_displacement_m=displacement,
                        duration_s=3,
                        speed_limit_m_s=0.05,
                        hardware_motion_executed=False,
                    ),
                    indent=2,
                )
            )

    def test_bridge_callbacks_do_not_wait_for_physics_or_ik(self):
        link, _, _ = self.pending_link()
        with (
            patch.object(link, "sample", side_effect=AssertionError("blocking sample")),
            patch.object(link.sim, "get_left_ee_pose", side_effect=AssertionError("blocking IK")),
            patch.object(link.sim, "get_lift_height", side_effect=AssertionError("physics lock")),
        ):
            self.assertIn("left_ee", link.robot_state())
            self.assertIn("lift", link.hud())
            self.assertIn("sim", link.monitor())

    def test_home_returns_measured_arms_neck_and_grippers_through_agent(self):
        self.exercise_home()

    def test_recorded_workcell_home_after_repeated_stops_and_reenable(self):
        self.exercise_home(recorded=True)

    def exercise_home(self, recorded=False):
        scene = None
        if recorded:
            from scripts.render_sophia_demo import make_scene

            self.fake.state = json.loads(
                (
                    ROOT / "robot/yor-description/calibration/live_home_start_2026-11-03.json"
                ).read_text()
            )["state"]
            scene = make_scene(self.root)
        else:
            self.fake.state["left_joints"][0] += 0.08
            self.fake.state["right_joints"][0] -= 0.08
            self.fake.state["neck_deg"] = [7.02, 16.75]
            self.fake.state["left_gripper"] = self.fake.state["right_gripper"] = 0.8
        link, transport, commands = self.pending_link(scene)
        self.send({"enabled": False})

        def acknowledge():
            command, _, seq = transport.latest
            reply = self.send(command)
            self.assertIsNone(reply["fault"])
            reply["acknowledged_source_seq"] = seq if reply["active"] else None
            transport.response = reply

        acknowledge()
        if recorded:
            original_hold = link.sim.lift_setpoint
            for _ in range(30):
                # Controlled settling disturbance, recreating the old ratchet.
                with link.sim._lock:
                    link.sim.data.qpos[link.sim._lift_adr] = link.sim.lift_setpoint - 0.002
                commands["enabled"] = False
                link.tick(commands)
                acknowledge()
                self.assertEqual(link.sim.lift_setpoint, original_hold)
                commands["enabled"] = True
                link.tick(commands)
                acknowledge()
        commands.update(enabled=False, home_all=True)
        link.tick(commands)
        self.assertTrue(link.homing)
        self.assertFalse(transport.latest[0]["enabled"])
        acknowledge()  # First stop base/lift and hold before returning home.
        commands["home_all"] = False
        deadline = time.monotonic() + (15 if recorded else 6)
        while link.homing and time.monotonic() < deadline:
            link.tick(commands)
            if transport.latest[0]["enabled"]:
                self.assertEqual(transport.latest[0]["base_velocity"], [0.0, 0.0, 0.0])
                self.assertEqual(transport.latest[0]["lift_direction"], 0)
            acknowledge()
            time.sleep(1 / 30)
        self.assertFalse(link.homing, link.home_result)
        self.assertEqual(link.home_result, "Home complete")
        self.assertFalse(transport.latest[0]["enabled"])
        for side in ("left", "right"):
            np.testing.assert_allclose(self.fake.state[side + "_joints"], HOME_Q[side], atol=0.05)
            self.assertAlmostEqual(self.fake.state[side + "_gripper"], 1.0)
        np.testing.assert_allclose(self.fake.state["neck_deg"], [2.02, -3.25], atol=1.0)
        self.assertFalse(any(name.startswith("home_") for name, _ in self.fake.calls))
        if recorded:
            commands.update(enabled=True, home_all=False)
            link.tick(commands)
            acknowledge()
            link.tick(commands)
            acknowledge()
            self.assertIsNone(link.fault)
            commands["enabled"] = False
            link.tick(commands)
            acknowledge()
            artifact = os.environ.get("SOPHIA_LIVE_TEST_ARTIFACTS")
            if artifact:
                destination = Path(artifact) / "recorded-home"
                destination.mkdir(parents=True)
                shutil.copy(self.root / "pi/events.jsonl", destination / "events.jsonl")
                (destination / "summary.json").write_text(
                    json.dumps(
                        dict(
                            source="synthetic plant initialized from real recorded pose",
                            source_session="20261103-184446",
                            furnished_workcell=True,
                            repeated_stop_reenable_cycles=30,
                            home_completed=True,
                            reenable_after_home_passed=True,
                            hardware_motion_executed=False,
                            lift_hold_m=original_hold,
                        ),
                        indent=2,
                    )
                    + "\n"
                )

    def test_home_cancels_on_input_loss(self):
        link, transport, commands = self.pending_link()
        commands.update(enabled=False, home_all=True)
        link.tick(commands)
        self.assertTrue(link.homing)
        commands["home_all"] = False
        link.tick(commands, fresh=False)
        self.assertFalse(link.homing)
        self.assertFalse(transport.latest[0]["enabled"])
        self.assertIn("cancelled", link.home_result)

    def test_home_pulse_is_latched_and_not_repeated(self):
        link, transport, commands = self.pending_link()
        commands.update(enabled=False, home_all=True)
        link.submit(commands)
        link.submit(dict(commands, home_all=False))
        self.assertIsNotNone(link.home_event)
        link.tick(commands)
        link.tick(commands)  # Same one-frame event sampled twice must not cancel Home.
        self.assertTrue(link.homing)
        transport.acknowledge()

    def test_home_cancel_button_does_not_enable_follow(self):
        link, transport, commands = self.pending_link()
        commands.update(enabled=False, home_all=True)
        link.tick(commands)
        commands.update(enabled=True, home_all=False)
        link.tick(commands)
        self.assertFalse(link.homing)
        self.assertFalse(transport.latest[0]["enabled"])
        self.assertIn("cancelled", link.input_paused)

    def test_workcell_render_load_preserves_input_freshness(self):
        from robot.sophia_live_video import ProcessStereoPublisher
        from scripts.render_sophia_demo import make_scene

        sys.path.insert(0, str(ROOT / "vr-teleop/server"))
        from whole_body import WholeBodyConfig
        from yor_bridge import Runner

        class ImmediateTransport(ManualTransport):
            def submit(self, *args):
                super().submit(*args)
                self.acknowledge()

        scene = make_scene(self.root)
        sim = YORMujoco(scene, headless=True)
        initialize_sim_from_robot(sim, self.fake.state, self.contract)
        link = LiveLink(sim, ImmediateTransport(self.fake.state), self.contract, self.journal)
        cfg = WholeBodyConfig(
            start_enabled=True,
            neck_relative_pitch=True,
            neck_pan_sign=-1,
            neck_tilt_sign=-1,
            neck_tilt_offset_deg=26.75,
        )
        runner = Runner(30, 3600, cfg, link)
        publisher = ProcessStereoPublisher(sim, scene, address="ipc://" + str(self.root / "video"))
        threads = [threading.Thread(target=f, daemon=True) for f in (runner.control_loop, link.run)]
        try:
            publisher.start()
            for thread in threads:
                thread.start()
            start = time.monotonic()
            while time.monotonic() - start < 4:
                elapsed = time.monotonic() - start
                runner.on_payload(
                    dict(
                        head=dict(pos=[0, 1.6, 0], quat=[0, 0, 0, 1]),
                        left=dict(
                            pos=[-0.2, 1.2, -0.3 - min(elapsed, 2) * 0.004],
                            quat=[0, 0, 0, 1],
                            axes=[0, 0, 0, 0],
                            buttons={"1": dict(pressed=True, value=1.0)},
                        ),
                        right=None,
                        view_pitch_deg=0.0,
                    )
                )
                time.sleep(1 / 90)
            self.assertIsNone(link.fault)
            events = [
                json.loads(s) for s in (self.root / "pi/events.jsonl").read_text().splitlines()
            ]
            pauses = [e for e in events if e["kind"] == "input_pause"]
            if link.input_paused:
                # A heavily loaded host must stop rather than forward old input.
                self.assertTrue(pauses)
                self.assertFalse(link.transport.latest[0]["enabled"])
                self.assertTrue(all(e["input_age_s"] > 0.20 for e in pauses))
            commands = [e for e in events if e["kind"] == "sim_command"]
            self.assertTrue(commands or pauses)
            self.assertTrue(
                all(
                    e["monotonic"] - e["teleop"]["input_received_monotonic"] <= 0.20
                    for e in commands
                )
            )
            self.assertTrue(all("input_received_monotonic" in e["teleop"] for e in commands))
            artifact = os.environ.get("SOPHIA_LIVE_TEST_ARTIFACTS")
            if artifact:
                destination = Path(artifact) / "workcell-render-load"
                destination.mkdir(parents=True)
                (destination / "events.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in events) + "\n"
                )
                ages = [e["monotonic"] - e["teleop"]["input_received_monotonic"] for e in commands]
                (destination / "summary.json").write_text(
                    json.dumps(
                        dict(
                            source="synthetic",
                            hardware_motion_executed=False,
                            command_samples=len(commands),
                            input_pause_count=len(pauses),
                            max_input_age_at_command_s=max(ages, default=None),
                            stereo_fps=30,
                            stereo_pixels_per_eye=[960, 540],
                        ),
                        indent=2,
                    )
                    + "\n"
                )
        finally:
            runner.stop.set()
            link.stop.set()
            for thread in threads:
                thread.join(timeout=3)
            publisher.stop()
            sim.stop()

    def test_end_to_end_sim_vr_transport_agent_and_export(self):
        self.exercise_end_to_end()

    def test_end_to_end_delayed_packets_do_not_accumulate_target_jumps(self):
        self.exercise_end_to_end(delayed=True)

    def test_yor_defaults_end_to_end_transport(self):
        self.exercise_end_to_end(native=True)

    def test_yor_defaults_end_to_end_delayed_transport(self):
        self.exercise_end_to_end(native=True, delayed=True)

    def test_yor_transport_off_race_realigns_neck_before_reenable(self):
        self.exercise_end_to_end(native=True, resync_race=True)

    def exercise_end_to_end(self, delayed=False, native=False, resync_race=False):
        if native:
            self.use_yor_contract()
            self.fake.state["lift_height"] = 0.648
        from commlink import RPCServer
        from scripts.analyze_sophia_live import analyze

        sys.path.insert(0, str(ROOT / "vr-teleop/server"))
        from whole_body import WholeBodyConfig, WholeBodyTeleop

        if delayed:
            original_send = LineChannel.send
            # Abrupt 130/100 ms outbound delays followed by a fast packet reproduce
            # the observed compression. Both endpoints and MuJoCo are real code;
            # only the robot and the network delay are synthetic.
            delays = iter([0.0, 0.13, 0.1, 0.0, 0.015, 0.1, 0.0] * 30)

            def delayed_send(channel, message):
                time.sleep(next(delays, 0.0))
                return original_send(channel, message)

            self.enterContext(patch.object(LineChannel, "send", delayed_send))

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = RPCServer(self.fake, port=port)
        server.start()
        contract_path = self.root / "contract.json"
        contract_path.write_text(json.dumps(self.contract))
        journal = Journal(
            self.root / "workstation",
            dict(
                schema=SCHEMA,
                source="synthetic",
                contract=self.contract,
                contract_sha256=fingerprint(self.contract),
            ),
        )
        transport, sim = None, None
        try:
            transport = FollowTransport(
                [
                    sys.executable,
                    str(ROOT / "robot/sophia_live_agent.py"),
                    "--contract",
                    str(contract_path),
                    "--output",
                    str(self.root / "endpoint"),
                    "--rpc-port",
                    str(port),
                    "--allow-motion",
                ],
                self.contract,
                journal,
            )
            sim = YORMujoco(ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
            initialize_sim_from_robot(sim, transport.snapshot()[0]["state"], self.contract)
            link = LiveLink(sim, transport, self.contract, journal)
            # Use the real VR controller state machine; synthetic headset only.
            tele = WholeBodyTeleop(WholeBodyConfig(start_enabled=True, max_ee_linear_mps=0.1))

            def payload(z):
                return dict(
                    head=dict(pos=[0, 1.6, 0], quat=[0, 0, 0, 1]),
                    left=dict(
                        pos=[-0.2, 1.2, z],
                        quat=[0, 0, 0, 1],
                        axes=[0, 0, 0, 0],
                        buttons={"1": dict(pressed=True, value=1.0)},
                    ),
                    right=None,
                )

            for i in range(120 if delayed else 35):
                command = tele.update(
                    payload(-0.3 - min(i, 20) * 0.0005), link.robot_state(), time.monotonic()
                )
                if delayed:
                    command["neck"] = dict(pan=15 if i < 65 else -5, tilt=-15 if i < 65 else -5)
                link.tick(command)
                time.sleep(1 / 30)
            self.assertIsNone(link.fault)
            self.assertIsNone(transport.snapshot()[1], transport.snapshot())
            self.assertTrue(transport.snapshot()[0]["active"])
            if resync_race:
                # Reproduce session 20261103-190633: a target computed while
                # ACTIVE arrives after the transport has independently sent OFF.
                old_command = copy.deepcopy(transport.latest[1])
                old_command["requires_active"] = True
                old_command["neck_deg"][1] += 20
                transport.submit({"enabled": False}, link.sample(), 20000)
                deadline = time.monotonic() + 1
                while transport.snapshot()[0]["active"] and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertFalse(transport.snapshot()[0]["active"])
                transport.submit(old_command, link.sample(), 20001)
                deadline = time.monotonic() + 1
                while (
                    not transport.snapshot()[0].get("resync_required")
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                reply, fault, _ = transport.snapshot()
                self.assertTrue(reply.get("resync_required"))
                self.assertIsNone(fault)
                self.assertFalse(reply["active"])
                for _ in range(15):
                    link.tick(command)
                    time.sleep(1 / 30)
                self.assertIsNone(transport.snapshot()[1])
                self.assertTrue(transport.snapshot()[0]["active"])
            # Tracking loss cancels and is latched even when the operator reconnects.
            command["input_fresh"] = False
            link.tick(command)
            self.assertIn("stale", link.input_paused)
            deadline = time.monotonic() + 0.5
            while transport.snapshot()[0]["active"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(transport.snapshot()[0]["active"])
            transport.close()
            transport = None
            journal.close()
            report = analyze(journal.directory, self.root / "audit")
            self.assertGreater(report["dispatched_samples"], 15)
            self.assertEqual(report["source"], "synthetic")
            self.assertFalse(report["hardware_validated"])
            self.assertTrue((self.root / "audit/identification-input.npz").exists())
            if delayed:
                events = [
                    json.loads(s)
                    for s in (journal.directory / "events.jsonl").read_text().splitlines()
                ]
                replies = [e for e in events if e["kind"] == "response"]
                self.assertGreater(max(e["rtt_s"] for e in replies), 0.1)
                self.assertTrue(all(e["response"]["fault"] is None for e in replies))
                requests = {
                    e["request"]["seq"]: e["request"]["command"]
                    for e in events
                    if e["kind"] == "request"
                }
                previous = None
                distinct = 0
                for reply in replies:
                    if not reply["response"]["active"]:
                        continue
                    target = np.array(requests[reply["seq"]]["neck_deg"])
                    if previous is not None:
                        self.assertLessEqual(
                            np.max(abs(target - previous)),
                            self.contract["neck_speed_deg_s"] * self.contract["period_s"] + 1e-9,
                        )
                        distinct += int(np.max(abs(target - previous)) > 1e-6)
                    previous = target
                # Native neck speed reaches these short sweeps in fewer steps.
                self.assertGreaterEqual(distinct, 6 if native else 11)
            artifact = os.environ.get("SOPHIA_LIVE_TEST_ARTIFACTS")
            if artifact:
                destination = Path(artifact) / "delayed" if delayed else Path(artifact)
                if native:
                    destination = Path(artifact) / ("yor-delayed" if delayed else "yor-transport")
                if resync_race:
                    destination = Path(artifact) / "off-transition-race"
                for name in ("workstation", "endpoint", "audit"):
                    shutil.copytree(self.root / name, destination / name)
        finally:
            if transport:
                transport.close()
            if sim:
                sim.stop()
            if not journal.stream.closed:
                journal.close()
            server.stop()


if __name__ == "__main__":
    unittest.main()
