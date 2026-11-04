"""Run the standalone Sophia checks and write a machine-readable acceptance report.

MUJOCO_GL=egl python scripts/check_sophia.py --render

Only explicitly listed simulation/offline tests run; hardware tests in robot/
are never discovered or imported. Review findings remain separate from pass/fail.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTION = ROOT / "robot/yor-description"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/asset-review/acceptance")
    parser.add_argument(
        "--render",
        action="store_true",
        help="also save the multi-view robot review renders",
    )
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument(
        "--training",
        action="store_true",
        help="also check the optional Gymnasium training environment",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    checks = []

    def run(name, arguments):
        started = time.monotonic()
        command = [sys.executable, *arguments]
        logfile = output / f"{name}.log"
        print(f"[RUN] {name}", flush=True)
        try:
            with logfile.open("w") as stream:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
            code = result.returncode
            error = None
        except subprocess.TimeoutExpired:
            code, error = 124, "Timed out after 180 seconds"
        check = {
            "name": name,
            "passed": code == 0,
            "exit_code": code,
            "error": error,
            "elapsed_s": round(time.monotonic() - started, 3),
            "command": command,
            "log": logfile.name,
        }
        print(
            f"[{'PASS' if check['passed'] else 'FAIL'}] {name} ({check['elapsed_s']} s)",
            flush=True,
        )
        return check

    versions = {}
    lock = ROOT / (
        "requirements-sophia-training.txt" if args.training else "requirements-sophia.txt"
    )
    for name, expected in re.findall(r"^([\w.-]+)==([^\s\\]+)", lock.read_text(), re.MULTILINE):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {"expected": expected, "actual": actual}
    checks.append(
        {
            "name": "locked_dependencies",
            "passed": bool(versions)
            and all(v["actual"] == v["expected"] for v in versions.values()),
            "packages": versions,
        }
    )
    generated = [
        DESCRIPTION / filename
        for filename in ("sophia.mjcf", "scene-sophia.mjcf", "sophia-ik.mjcf")
    ]
    generated.extend(sorted((DESCRIPTION / "assets_nero/sophia").glob("*.stl")))
    generated.extend(sorted((DESCRIPTION / "assets_nero/sophia_collision").glob("*.stl")))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in generated if p.exists()}
    build = run("rebuild", ["robot/yor-description/build_sophia.py"])
    checks.append(build)
    changed = (
        [name for name, checksum in before.items() if sha256(ROOT / name) != checksum]
        if build["passed"]
        else []
    )
    checks.append(
        {
            "name": "generated_files_reproducible",
            "passed": build["passed"] and bool(before) and not changed,
            "checked_files": len(before),
            "changed_files": changed,
        }
    )
    if build["passed"] and checks[0]["passed"]:
        asset_args = [
            "robot/yor-description/validate_sophia.py",
            "--output",
            str(output),
        ]
        if args.render:
            asset_args.append("--render")
        tasks = {
            "asset_audit": [
                "robot/yor-description/audit_sophia.py",
                "--output",
                str(output),
            ],
            "asset_validation": asset_args,
            "calibration_tools": ["robot/test_sophia_calibration.py"],
            "episode_listener": ["robot/test_sophia_episode_listener.py"],
            "capture_import": ["robot/test_sophia_capture_import.py"],
            "capture_identification": ["robot/test_sophia_identification.py"],
            "hardware_summary_comparison": [
                "robot/yor-description/compare_sophia_hardware.py",
                "--output",
                str(output),
            ],
            "mobile_manipulation": [
                "scripts/render_sophia_demo.py",
                "--simulate-only",
                "--output",
                str(output / "mobile-manipulation"),
            ],
            "stereo_video": ["robot/test_sophia_video.py"],
            "release_profile_and_cameras": ["robot/test_sophia_release.py"],
            "calibrated_frames": [
                "-m",
                "unittest",
                "robot.test_sophia_full_frames",
                "robot.test_sophia_head_alignment",
                "robot.test_sophia_neck_reference",
                "robot.test_sophia_base_alignment",
            ],
            "collision_regressions": ["robot/test_sophia_collision.py"],
            "collision_shape_review": [
                "scripts/review_sophia_collisions.py",
                "--expect-reviewed-geometry",
                "--output",
                str(output / "collision-shapes"),
            ],
            "mount_frame_review": [
                "scripts/review_sophia_mounts.py",
                "--output",
                str(output / "mount-frames"),
            ],
            "controller_sophia": [
                "robot/test_yor_mujoco.py",
                "--model",
                "robot/yor-description/scene-sophia.mjcf",
            ],
            "controller_legacy": ["robot/test_yor_mujoco.py", "--gripper", "agx"],
            "extended_evaluation": [
                "robot/yor-description/evaluate_sophia.py",
                "--output",
                str(output),
            ],
        }
        if args.training:
            tasks.update(
                {
                    "policy_contract": ["robot/test_sophia_policy.py"],
                    "training_environment": ["robot/test_sophia_env.py"],
                    "training_task_oracle": [
                        "scripts/run_sophia_pick_place.py",
                        "--episodes",
                        "3",
                        "--rgb-observations",
                        "--output",
                        str(output / "training-task"),
                    ],
                }
            )
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run, name, arguments) for name, arguments in tasks.items()]
            checks.extend(future.result() for future in as_completed(futures))
    else:
        checks.append(
            {
                "name": "remaining_suites",
                "passed": False,
                "reason": (
                    "Install the exact lock and resolve build failures before accepting results."
                ),
            }
        )
    audit_path = output / "asset-audit.json"
    audit = (
        json.loads(audit_path.read_text())
        if any(c["name"] == "asset_audit" and c["passed"] for c in checks)
        else {}
    )
    sources = [
        ROOT / "requirements-sophia.txt",
        ROOT / "requirements-sophia-training.txt",
        *generated,
        *sorted((ROOT / "robot").glob("*sophia*.py")),
        ROOT / "robot/yor_mujoco.py",
        ROOT / "robot/sim_collision.py",
        ROOT / "robot/sim_camera.py",
        DESCRIPTION / "sophia_sources.py",
        DESCRIPTION / "sophia_collision_geometry.py",
        DESCRIPTION / "compare_sophia_hardware.py",
        ROOT / "scripts/render_sophia_demo.py",
        ROOT / "scripts/audit_sophia_dataset.py",
        ROOT / "scripts/replay_sophia_dataset_neck.py",
        ROOT / "scripts/replay_sophia_dataset_arm.py",
        ROOT / "scripts/render_sophia_cameras.py",
        ROOT / "scripts/review_sophia_collisions.py",
        ROOT / "scripts/review_sophia_mounts.py",
        ROOT / "scripts/identify_sophia_response.py",
        ROOT / "scripts/listen_sophia_episodes.py",
        ROOT / "scripts/import_sophia_capture.py",
        ROOT / "scripts/identify_sophia_capture.py",
        ROOT / "scripts/identify_sophia_joint_gains.py",
        ROOT / "scripts/run_sophia_pick_place.py",
        *sorted((DESCRIPTION / "calibration").glob("*.json")),
        *sorted(p for p in (DESCRIPTION / "vendor").rglob("*") if p.is_file()),
        DESCRIPTION / "sophia.json",
        DESCRIPTION / "audit_sophia.py",
        DESCRIPTION / "build_sophia.py",
        DESCRIPTION / "validate_sophia.py",
        DESCRIPTION / "evaluate_sophia.py",
        ROOT / "robot/test_yor_mujoco.py",
        Path(__file__),
    ]
    report = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "system_site_packages_inherited": sys.prefix == sys.base_prefix,
        "simulation_checks_passed": all(check["passed"] for check in checks),
        "hardware_calibrated": False,
        "training_checks_enabled": args.training,
        "release_status": audit.get("release_status"),
        "remaining_required_work": [c["name"] for c in checks if not c["passed"]],
        "hardware_review_findings": audit.get("findings", []),
        "checks": sorted(checks, key=lambda row: row["name"]),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
    }
    # venv configuration is more specific than the interpreter prefix for
    # environments created with --system-site-packages.
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    if configuration.exists():
        report["system_site_packages_inherited"] = bool(
            re.search(
                r"include-system-site-packages\s*=\s*true",
                configuration.read_text(),
                re.IGNORECASE,
            )
        )
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'acceptance.json'}", flush=True)
    raise SystemExit(0 if report["simulation_checks_passed"] else 1)


if __name__ == "__main__":
    main()
