"""Validate live following locally; optionally check the Pi with motion disabled."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    # Match the live launcher's control-process math threading configuration.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pi-host")
    parser.add_argument("--pi-dir", default="/home/yor/sophia-live-2026-11-03-r4")
    parser.add_argument("--pi-python", default="/home/yor/miniforge3/envs/yor-nero/bin/python")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MUJOCO_GL="egl", SOPHIA_LIVE_TEST_ARTIFACTS=str(output / "loopback"))
    tested = {}
    for name, path in (
        ("live_follow", "robot/test_sophia_live.py"),
        ("vr_controller", "vr-teleop/tests/test_whole_body.py"),
        ("vr_cycles", "vr-teleop/tests/test_cycles.py"),
    ):
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(
                [sys.executable, path],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=True,
            )
        text = (output / (name + ".log")).read_text()
        matches = re.search(r"Ran (\d+) tests", text) or re.search(r"(\d+)/(\d+) passed", text)
        tested[name] = {"passed": True, "count": int(matches.group(1)) if matches else None}
    remote = None
    if args.pi_host:
        from robot.sophia_live import (
            FollowTransport,
            LiveLink,
            initialize_sim_from_robot,
            make_contract,
        )
        from robot.sophia_live_common import SCHEMA, Journal, fingerprint, measured
        from robot.yor_mujoco import YORMujoco

        sim = YORMujoco(ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
        contract = make_contract(sim)
        journal = Journal(
            output / "pi-read-only",
            dict(
                schema=SCHEMA,
                source="robot",
                role="readiness_only",
                contract=contract,
                contract_sha256=fingerprint(contract),
            ),
        )
        remote_output = args.pi_dir + "/readiness-" + time.strftime("%Y%m%d-%H%M%S")
        command = [
            args.pi_python,
            args.pi_dir + "/robot/sophia_live_agent.py",
            "--contract",
            args.pi_dir + "/contract.json",
            "--output",
            remote_output,
        ]
        transport = None
        try:
            transport = FollowTransport(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=5",
                    args.pi_host,
                    shlex.join(command),
                ],
                contract,
                journal,
            )
            state = measured(transport.snapshot()[0]["state"], contract)
            initialize_sim_from_robot(sim, state, contract)
            link = LiveLink(sim, transport, contract, journal)
            for _ in range(60):
                link.tick(None)
                time.sleep(1 / 30)
            reply, fault, age = transport.snapshot()
            assert not reply["active"] and not reply["allow_motion"] and not fault, (reply, fault)
            remote = dict(
                read_only_check_passed=True,
                real_motion_enabled=False,
                host=args.pi_host,
                pi_logs=remote_output,
                latest_feedback_age_s=age,
                lift_homed=state["lift_homed"],
                measured_start=state,
                sim_start=link.sample(),
            )
        finally:
            if transport:
                transport.close()
            sim.stop()
            journal.close()
        events = [
            json.loads(line)
            for line in (journal.directory / "events.jsonl").read_text().splitlines()
        ]
        requests = [r for r in events if r["kind"] == "request"]
        assert requests and all(not r["request"]["command"]["enabled"] for r in requests)
        remote["read_only_requests"] = len(requests)
        rtt = [r["rtt_s"] for r in events if r["kind"] == "response"]
        import numpy as np

        remote["rtt_p95_ms"] = float(np.percentile(rtt, 95) * 1000)
    paths = [
        "robot/sophia_live.py",
        "robot/sophia_live_video.py",
        "robot/sophia_live_common.py",
        "robot/sophia_live_agent.py",
        "robot/test_sophia_live.py",
        "scripts/live_sophia_teleop.py",
        "scripts/check_sophia_live.py",
        "scripts/analyze_sophia_live.py",
        "vr-teleop/server/yor_bridge.py",
        "vr-teleop/server/cameras.py",
        "vr-teleop/camera_config.sim.json",
        "requirements-sophia-live.txt",
    ]
    result = dict(
        passed=True,
        tests=tested,
        remote=remote,
        hardware_motion_tested=False,
        hardware_parameters_modified=False,
        joystick_sha256=hashlib.sha256(
            (ROOT / "robot/teleop/joystick.py").read_bytes()
        ).hexdigest(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    )
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "tests": tested,
                "pi_read_only": remote is not None,
                "hardware_motion_tested": False,
                "report": str(output / "validation.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
