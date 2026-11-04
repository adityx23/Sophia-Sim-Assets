"""Launch existing VR teleop with a live Sophia follower over SSH.

Use prepare to create the Pi bundle. run starts disarmed; A+X in VR enables.
No automatic homing or driver initialization occurs on startup.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    # Tiny IK/collision matrices are latency-sensitive; 32-worker BLAS dispatch
    # adds scheduling stalls. Configure this process before NumPy is imported.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="build a self-contained Pi source/contract bundle")
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="live VR session; real commands only with --allow-motion")
    run.add_argument("--host", required=True, help="SSH destination, e.g. yor@192.168.123.55")
    run.add_argument("--remote-dir", required=True, help="absolute extracted bundle directory")
    run.add_argument("--remote-python", required=True, help="absolute Python with numpy/commlink")
    run.add_argument("--remote-output", required=True, help="new absolute directory for Pi logs")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--model",
        type=lambda p: Path(p).resolve(),
        default=ROOT / "robot/yor-description/scene-sophia.mjcf",
        help="Sophia scene MJCF; default is the empty floor",
    )
    run.add_argument("--allow-motion", action="store_true")
    run.add_argument("--headless", action="store_true")
    run.add_argument("--no-video", action="store_true")
    run.add_argument("--ip", help="workstation address used by headset video")
    run.add_argument(
        "--view-down-deg",
        type=float,
        default=0.0,
        help="downward view offset from yor.py neck home; default 0 degrees",
    )
    run.add_argument(
        "--joystick-device", help="Pi joystick device; Back or stick motion aborts following"
    )
    args = parser.parse_args()
    if args.mode == "run" and not 0 <= args.view_down_deg <= 60:
        parser.error("--view-down-deg must be between 0 and 60")
    from robot.sophia_live import (
        INPUT_TIMEOUT_S,
        FollowTransport,
        LiveLink,
        initialize_sim_from_robot,
        make_contract,
    )
    from robot.sophia_live_common import SCHEMA, Journal, fingerprint
    from robot.sophia_live_video import ProcessStereoPublisher
    from robot.yor_mujoco import YORMujoco

    model = args.model if args.mode == "run" else ROOT / "robot/yor-description/scene-sophia.mjcf"
    sim = YORMujoco(model_path=model, headless=args.mode == "prepare" or args.headless)
    contract = make_contract(sim)
    if args.mode == "prepare":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        names = ["robot/sophia_live_common.py", "robot/sophia_live_agent.py"]
        payload = {n: (ROOT / n).read_bytes() for n in names}
        payload["robot/__init__.py"] = b""
        payload["contract.json"] = (json.dumps(contract, indent=2) + "\n").encode()
        payload["README.txt"] = (
            b"Pi live-follow endpoint. Existing numpy/commlink environment required. "
            b"Starts disarmed; never initializes/homes robot. "
            b"See workstation docs/SOPHIA_LIVE_FOLLOW.md.\n"
        )
        manifest = {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}
        payload["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in payload.items():
                bundle.writestr(name, data)
        sim.stop()
        print(
            json.dumps(
                dict(
                    bundle=str(args.output),
                    sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    contract_sha256=fingerprint(contract),
                ),
                indent=2,
            )
        )
        return

    if args.host.startswith("-") or any(ch.isspace() for ch in args.host):
        parser.error("Invalid SSH destination")
    for name in ("remote_dir", "remote_python", "remote_output"):
        if not getattr(args, name).startswith("/"):
            parser.error(name + " must be absolute")
    journal = Journal(
        args.output,
        dict(
            schema=SCHEMA,
            role="workstation",
            source="robot",
            contract=contract,
            contract_sha256=fingerprint(contract),
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_path=str(model),
            allow_motion=args.allow_motion,
            neutral_view_down_deg=args.view_down_deg,
            neck_pitch_reference="captured_on_enable",
            control_math_threads=1,
            input_timeout_s=INPUT_TIMEOUT_S,
            robot=args.host,
            pi_output=args.remote_output,
            source_sha256={
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in (
                    "robot/yor-description/sophia.mjcf",
                    "robot/yor-description/sophia.json",
                    "robot/sophia_live.py",
                    "robot/sophia_live_agent.py",
                    "robot/sophia_live_common.py",
                    "robot/sophia_live_video.py",
                    "vr-teleop/server/yor_bridge.py",
                )
            },
            hardware_validated=False,
        ),
    )
    remote = [
        args.remote_python,
        args.remote_dir + "/robot/sophia_live_agent.py",
        "--contract",
        args.remote_dir + "/contract.json",
        "--output",
        args.remote_output,
    ]
    if args.allow_motion:
        remote += ["--allow-motion", "--enable-open-grippers"]
    if args.joystick_device:
        remote += ["--joystick-device", args.joystick_device]
    transport, video, video_server = None, None, None
    try:
        transport = FollowTransport(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "ServerAliveInterval=1",
                "-o",
                "ServerAliveCountMax=2",
                args.host,
                shlex.join(remote),
            ],
            contract,
            journal,
        )
        initialize_sim_from_robot(sim, transport.snapshot()[0]["state"], contract)
        sys.path.insert(0, str(ROOT / "vr-teleop/server"))
        import yor_bridge
        from robot.sim_vr_teleop import lan_ip, wait_for_port

        ip = args.ip or lan_ip()
        if not args.no_video:
            yor_bridge.ensure_cert([ip])
            video = ProcessStereoPublisher(sim, model)
            video.start()
            video_server = subprocess.Popen(
                [
                    sys.executable,
                    "server/server.py",
                    "--video-only",
                    "--port",
                    "8444",
                    "--camera-config",
                    "camera_config.sim.json",
                ],
                cwd=ROOT / "vr-teleop",
            )
            wait_for_port(8444, video_server, "simulated stereo video")
        link = LiveLink(sim, transport, contract, journal)
        # Disable the legacy episode reader: paired follow logs already record every tick.
        sys.argv = [
            "yor_bridge.py",
            "--no-episodes",
            "--cycle-source",
            "none",
            "--arm-speed",
            ".5",
            "--arm-preview",
            str(contract["arm_preview_s"]),
            "--base-speed",
            str(contract["base_speed_m_s"]),
            "--base-turn",
            str(contract["yaw_speed_rad_s"]),
            "--neck-speed",
            str(contract["neck_speed_deg_s"]),
            "--ip",
            ip,
            "--video",
            f"{ip}:8444",
        ] + (["--no-video"] if args.no_video else [])
        print(
            f"Live follow starts OFF. Headset: https://{ip}:8443/?tilt=0 ; monitor: https://{ip}:8443/monitor"
        )
        print(f"Paired recording: {args.output.resolve()}")
        yor_bridge.main(
            neck_calib=str(ROOT / "robot/neck/neck_calibration.json"),
            cycle_source=None,
            link_factory=lambda *a, **kw: link,
            neck_relative_pitch=True,
            neck_view_down_deg=args.view_down_deg,
        )
    finally:
        if transport:
            transport.close()
        if video_server:
            video_server.terminate()
            try:
                video_server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                video_server.kill()
                video_server.wait()
        if video:
            video.stop()
        sim.stop()
        journal.close()
        from scripts.analyze_sophia_live import analyze

        report = analyze(journal.directory, journal.directory / "analysis")
        print(f"Saved session audit and identification data: {journal.directory / 'analysis'}")
        print(f"Dispatched motion samples: {report['dispatched_samples']}")


if __name__ == "__main__":
    main()
