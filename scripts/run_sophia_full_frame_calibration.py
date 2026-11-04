"""Run the full CAD/frame calibration offline, with optional validated promotion.

Uses cached robot recordings only. Never connects to the Pi or Orin. The source
STL extraction is a separate pinned step; generated part assets are sufficient.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
DESC = ROOT / "robot/yor-description"
OUT = ROOT / "output/asset-review/full-neck-cad-calibration"
sys.path.insert(0, str(DESC))
from build_sophia import build  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "data").exists():
        (OUT / "data").symlink_to(
            Path("../head-calibration-pipeline/data"), target_is_directory=True
        )
    cfg = json.loads((DESC / "sophia.json").read_text())
    neutral = copy.deepcopy(cfg)
    neutral["spinal_assembly"]["enabled"] = True
    neutral.setdefault("arm_reference", {})["enabled"] = False
    hardware = json.loads((ROOT / "robot/neck/neck_calibration.json").read_text())
    neutral["neck_reference"]["geometric_zero_hardware_deg"] = [
        hardware["home_deg"][s] for s in ["pan", "tilt"]
    ]
    build(neutral, OUT / "source")
    commands = [
        [
            "scripts/calibrate_sophia_head.py",
            "--output",
            str(OUT),
            "--scene",
            str(OUT / "source/scene-sophia.mjcf"),
        ],
        ["scripts/audit_sophia_arm_frames.py"],
        [
            "scripts/check_sophia_head_calibration.py",
            "--output",
            str(OUT),
            "--fit-report",
            "combined-fit-report.json",
        ],
    ]
    if args.no_video:
        commands[-1].append("--no-video")
    for index, command in enumerate(commands):
        with (OUT / f"pipeline-stage-{index + 1}.log").open("w") as log:
            subprocess.run(
                [sys.executable, *command],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    fit = json.loads((OUT / "combined-fit-report.json").read_text())
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    if not acceptance["accepted"]:
        raise SystemExit("Calibration was not accepted; defaults unchanged.")
    if args.promote:
        profile = dict(
            method=(
                "Full supplied neck CAD plus training-only effective arm/neck "
                "reference alignment; no hardware changes."
            ),
            requires_spinal_assembly_sha256=cfg["spinal_assembly"]["sha256"],
            geometric_zero_hardware_deg=fit["geometric_zero_hardware_deg"],
            right_mount_rotvec_parent_deg=fit["right_mount_rotvec_parent_deg"],
            right_reference_q=fit["right_reference_q"],
            hardware_home_deg=fit["hardware_home_deg"],
            labels_sha256=fit["labels_sha256"],
            validation_scores=fit["fits"][fit["selected"]]["scores"],
            stereo_point_validation_scores=fit["stereo_point_scores"],
            reference_ee_error_m=fit["reference_ee_error_m"],
            scope=(
                "Development validation on five episodes, observed tilt about30..46deg; "
                "effective arm reference is not measured mechanical mounting. "
                "Broad-workspace accuracy and dynamics remain separate."
            ),
        )
        target = DESC / "calibration/image_frame_alignment_2026-11-04.json"
        target.write_text(json.dumps(profile, indent=2) + "\n")
        cfg["spinal_assembly"]["enabled"] = True
        cfg["neck_reference"]["geometric_zero_hardware_deg"] = profile[
            "geometric_zero_hardware_deg"
        ]
        cfg["arm_reference"] = dict(
            enabled=True,
            source="calibration/" + target.name,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            status="effective_reference_validated_in_recorded_workspace",
        )
        (DESC / "sophia.json").write_text(json.dumps(cfg, indent=2) + "\n")
        build(cfg)
        acceptance.update(
            promoted=True,
            promoted_model_sha256=hashlib.sha256((DESC / "sophia.mjcf").read_bytes()).hexdigest(),
        )
        (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        (OUT / "promotion.json").write_text(
            json.dumps(
                dict(
                    profile_sha256=cfg["arm_reference"]["sha256"],
                    model_sha256=acceptance["promoted_model_sha256"],
                    hardware_modified=False,
                ),
                indent=2,
            )
            + "\n"
        )
        fit.update(changes_promoted=True, promoted_model_sha256=acceptance["promoted_model_sha256"])
        (OUT / "combined-fit-report.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(dict(accepted=True, promoted=args.promote, output=str(OUT)), indent=2))


if __name__ == "__main__":
    main()
