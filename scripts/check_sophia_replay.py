"""Build and validate the entire workstation replay pipeline, without a robot."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.sophia_replay import build_trajectory, simulate, write_json  # noqa: E402
from robot.sophia_replay_compare import compare  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [
        "robot/teleop/joystick.py",
        "robot/yor.py",
        "robot/base.py",
        "robot/base_motor.py",
        "robot/arm/can_gripper.py",
    ]
    # Standalone simulation bundles deliberately omit most hardware drivers.
    # Check every present file without claiming that absent files were verified.
    protected_present = [p for p in protected if (ROOT / p).is_file()]
    protected_absent = [p for p in protected if p not in protected_present]
    before = {p: sha(ROOT / p) for p in protected_present}
    versions = {}
    for name, expected in re.findall(
        r"^([\w.-]+)==([^\s\\]+)",
        (ROOT / "requirements-sophia-replay.txt").read_text(),
        re.MULTILINE,
    ):
        actual = importlib.metadata.version(name)
        versions[name] = actual
        if actual != expected:
            raise RuntimeError(f"Install the replay lock: {name} expected {expected}, got {actual}")
    environment = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    with (args.output / "unit-tests.log").open("w") as stream:
        subprocess.run(
            [sys.executable, "robot/test_sophia_replay.py", "-v"],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=True,
        )
    print("Replay regression and loopback tests passed", flush=True)
    unit_tests = int(
        re.search(r"Ran (\d+) tests", (args.output / "unit-tests.log").read_text()).group(1)
    )
    trajectory = build_trajectory()
    write_json(args.output / "trajectory.json", trajectory)
    reference = simulate(trajectory, args.output / "simulation", video=args.video)
    print(f"Simulation passed: {trajectory['duration_s']:.2f}s", flush=True)
    identity = compare(trajectory, reference, reference, args.output / "self-check")
    assert identity["matched_command_data_eligible"]
    assert np.max(identity["metrics"]["left_joints"]["rmse"]) < 1e-12
    assert np.max(identity["metrics"]["base_pose"]["rmse"]) < 1e-12
    perturbed = copy.deepcopy(reference)
    perturbed.update(
        source="synthetic",
        pose_source="synthetic_base_pose",
        run_id="synthetic-perturbation",
        synthetic_note="Artificial 0.02 rad J1 bias and a 20 mm sideways drift; not robot data.",
    )
    duration = perturbed["samples"][-1]["time_s"]
    yaw = perturbed["samples"][0]["base_pose"][2]
    perturbed["initial_state"]["left_joints"][0] += 0.02
    for row in perturbed["samples"]:
        row["left_joints"][0] += 0.02
        drift = 0.02 * row["time_s"] / duration
        row["base_pose"][0] -= np.sin(yaw) * drift
        row["base_pose"][1] += np.cos(yaw) * drift
    write_json(args.output / "synthetic-perturbation.json", perturbed)
    injected = compare(trajectory, reference, perturbed, args.output / "injected-error-check")
    assert abs(injected["metrics"]["left_joints"]["rmse"][0] - 0.02) < 1e-9
    assert abs(injected["metrics"]["base_pose"]["endpoint_error"][1] - 0.02) < 1e-9
    assert injected["matched_command_data_eligible"] and not injected["hardware_validated"]
    video_report = None
    raw_video = args.output / "simulation/replay-preview-raw.mp4"
    if args.video and shutil.which("ffmpeg") and shutil.which("ffprobe"):
        video = args.output / "replay-preview.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-n",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw_video),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,width,height,nb_frames:format=duration,size",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )
        video_report = {
            "path": video.name,
            "sha256": sha(video),
            "decode_passed": True,
            "probe": probe,
        }
        write_json(args.output / "video-verification.json", video_report)
    after = {p: sha(ROOT / p) for p in protected_present}
    assert before == after
    sources = [
        "robot/sophia_replay.py",
        "robot/sophia_replay_io.py",
        "robot/sophia_replay_compare.py",
        "robot/test_sophia_replay.py",
        "scripts/sophia_replay.py",
        "scripts/check_sophia_replay.py",
        "requirements-sophia-replay.txt",
        "robot/yor-description/sophia.mjcf",
        "robot/yor-description/sophia.json",
        "robot/yor_mujoco.py",
    ]
    status = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_checks_passed": True,
        "unit_tests_passed": unit_tests,
        "simulation_duration_s": trajectory["duration_s"],
        "commands": len(trajectory["frames"]),
        "nominal_forward_m": 0.25,
        "simulated_displacement": identity["metrics"]["base_pose"]["reference_displacement"],
        "identity_comparison_passed": True,
        "known_error_recovery_passed": True,
        "robot_commands_sent": False,
        "robot_connection_used": False,
        "hardware_validated": False,
        "joystick_and_hardware_sources_unchanged": (
            before == after if not protected_absent else None
        ),
        "present_hardware_sources_unchanged": before == after,
        "hardware_sources_not_present": protected_absent,
        "protected_source_sha256": after,
        "source_sha256": {p: sha(ROOT / p) for p in sources},
        "versions": versions,
        "trajectory_sha256": trajectory["sha256"],
        "video": video_report,
        "remaining": (
            "Import a real recording to measure hardware response; "
            "no Pi control integration included."
        ),
    }
    write_json(args.output / "pipeline-status.json", status)
    print(
        json.dumps(
            {
                k: v
                for k, v in status.items()
                if k not in ("source_sha256", "protected_source_sha256", "versions", "video")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
