"""Create a deterministic standalone Sophia source/asset ZIP with checksums.

python scripts/package_sophia.py --output output/asset-review/sophia-sim.zip

Includes the source model assets needed to regenerate and audit Sophia, pinned
calibration evidence and imported robot captures. Does not include environments,
hardware libraries, unrelated runtime logs or photographs.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bundle_files():
    files = set()
    for path in (ROOT / "robot/yor-description").rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() not in (".pyc", ".log", ".bak")
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(ROOT).parts
            )
        ):
            files.add(path)
    for relative in (
        "LICENSE",
        "requirements-sophia.in",
        "requirements-sophia.txt",
        "requirements-sophia-training.in",
        "requirements-sophia-training.txt",
        "requirements-sophia-replay.in",
        "requirements-sophia-replay.txt",
        "requirements-sophia-live.in",
        "requirements-sophia-live.txt",
        "docs/SOPHIA_SIM_ASSETS.md",
        "docs/SOPHIA_OFFLINE_TOOLS.md",
        "docs/SOPHIA_TRAINING.md",
        "docs/SOPHIA_REPLAY.md",
        "docs/SOPHIA_LIVE_FOLLOW.md",
        "robot/yor_mujoco.py",
        "robot/sim_collision.py",
        "robot/sim_camera.py",
        "robot/sophia_calibration.py",
        "robot/sophia_profile.py",
        "robot/sophia_policy.py",
        "robot/sophia_env.py",
        "robot/test_sophia_policy.py",
        "robot/test_sophia_env.py",
        "robot/sophia_replay.py",
        "robot/sophia_replay_compare.py",
        "robot/sophia_replay_io.py",
        "robot/test_sophia_replay.py",
        "robot/sophia_live.py",
        "robot/sophia_live_video.py",
        "robot/sophia_live_common.py",
        "robot/sophia_live_agent.py",
        "robot/test_sophia_live.py",
        "robot/test_yor_mujoco.py",
        "robot/test_sophia_collision.py",
        "robot/test_sophia_calibration.py",
        "robot/test_sophia_episode_listener.py",
        "robot/test_sophia_capture_import.py",
        "robot/test_sophia_identification.py",
        "robot/test_sophia_video.py",
        "robot/test_sophia_head_alignment.py",
        "robot/test_sophia_neck_reference.py",
        "robot/test_sophia_full_frames.py",
        "robot/test_sophia_base_alignment.py",
        "robot/sophia_recording_alignment.py",
        "robot/test_sophia_release.py",
        "robot/arm/ik_solver.py",
        "robot/arm/can_gripper.py",
        "robot/teleop/joystick.py",
        "robot/neck/neck_calibration.json",
        "scripts/check_sophia.py",
        "scripts/package_sophia.py",
        "scripts/verify_sophia_bundle.py",
        "scripts/render_sophia_demo.py",
        "scripts/run_sophia_pick_place.py",
        "scripts/sophia_replay.py",
        "scripts/check_sophia_replay.py",
        "scripts/live_sophia_teleop.py",
        "scripts/check_sophia_live.py",
        "scripts/analyze_sophia_live.py",
        "scripts/run_sophia_live.sh",
        "scripts/audit_sophia_dataset.py",
        "scripts/audit_sophia_neck_recording.py",
        "scripts/extract_sophia_spinal_neck.py",
        "scripts/extract_sophia_spinal_assembly.py",
        "scripts/audit_sophia_arm_frames.py",
        "scripts/run_sophia_full_frame_calibration.py",
        "scripts/diagnose_sophia_neck_reference.py",
        "scripts/prepare_sophia_head_calibration.py",
        "scripts/calibrate_sophia_head.py",
        "scripts/calibrate_sophia_base.py",
        "scripts/fit_sophia_base.py",
        "scripts/check_sophia_base.py",
        "scripts/check_sophia_head_calibration.py",
        "scripts/run_sophia_head_calibration.sh",
        "scripts/fit_sophia_head_camera.py",
        "scripts/render_continuous_teleop.py",
        "scripts/replay_sophia_dataset_neck.py",
        "scripts/replay_sophia_dataset_arm.py",
        "scripts/render_sophia_cameras.py",
        "scripts/review_sophia_collisions.py",
        "scripts/review_sophia_mounts.py",
        "scripts/identify_sophia_response.py",
        "scripts/listen_sophia_episodes.py",
        "scripts/import_sophia_capture.py",
        "scripts/identify_sophia_capture.py",
        "scripts/identify_sophia_joint_gains.py",
    ):
        files.add(ROOT / relative)
    # Keep the exact source copies used to establish gripper mesh provenance.
    for directory in ("robot/arm/mujoco/assets", "robot/arm/urdf/assets"):
        for filename in ("gripper_base.stl", "link7.stl", "link8.stl"):
            files.add(ROOT / directory / filename)
    for relative in ("robot/__init__.py", "robot/arm/__init__.py"):
        if (ROOT / relative).is_file():
            files.add(ROOT / relative)
    # Preserve the reviewed Pi bridge plus local simulation/live-follow patches.
    # Certificates/private keys and run recordings are deliberately excluded.
    for directory in ("server", "static", "tests", "docs"):
        for path in (ROOT / "vr-teleop" / directory).rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix.lower() not in (".pyc", ".key", ".pem", ".log", ".yep")
            ):
                files.add(path)
    for name in ("README.md", "AGENTS.md", "requirements.txt", "camera_config.sim.json"):
        files.add(ROOT / "vr-teleop" / name)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Include acceptance JSON and check logs from the release audit",
    )
    args = parser.parse_args()
    files = bundle_files()
    payload = {str(path.relative_to(ROOT)): path.read_bytes() for path in files}
    release = json.loads((ROOT / "robot/yor-description/sophia.json").read_text())["release"]
    if args.evidence_dir:
        report = json.loads((args.evidence_dir / "acceptance.json").read_text())
        if not report.get("simulation_checks_passed"):
            raise ValueError("Cannot package a release with failed acceptance")
        for name, checksum in report.get("source_sha256", {}).items():
            source = ROOT / name
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
                raise ValueError("Acceptance evidence is stale: " + name)
        # Include reports/logs only; recordings, credentials and binary captures
        # are outside this evidence contract.
        for path in sorted(args.evidence_dir.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".log"):
                name = "release-evidence/" + str(path.relative_to(args.evidence_dir))
                payload[name] = path.read_bytes()
    payload["README.md"] = (
        f"# {release.get('id', 'Sophia simulation bundle')}\n\n".encode()
        + b"Start with docs/SOPHIA_OFFLINE_TOOLS.md for the pinned Python 3.11 setup, "
        b"acceptance checks, and offline calibration tools.\n\n"
        b"Launch: python robot/yor_mujoco.py "
        b"--model robot/yor-description/scene-sophia.mjcf --no-video\n\n"
        b"Manifest checksums cover the bundled files. This is the finalized simulation "
        b"baseline with documented calibration scope and accepted physical assumptions. "
        b"Record its release ID and archive checksum with training/replay runs.\n"
    )
    manifest = {
        "format_version": 1,
        "release": {
            key: release[key] for key in ("version", "id", "status", "scope") if key in release
        },
        "hardware_calibrated": False,
        "entry_model": "robot/yor-description/scene-sophia.mjcf",
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(payload.items())
        },
    }
    payload["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compresslevel=6)
    print(
        f"Wrote {args.output}: {len(payload)} files, {args.output.stat().st_size / 1024**2:.1f} MiB"
    )
    print(f"SHA256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
