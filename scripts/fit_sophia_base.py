"""Fit base landmarks only; the accepted head and arm calibration stays fixed.

Run calibrate_sophia_base.py first. This builds a candidate, leaving defaults alone.
Historical height offsets are tied to exact native recording files, never to live
encoder readings. Static settings do not identify a lift gain or speed.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/asset-review/base-calibration"
DESC = ROOT / "robot/yor-description"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence_path = OUT / "stereo-base-points.json"
    evidence = json.loads(evidence_path.read_text())
    records = evidence["records"]
    cfg = json.loads((DESC / "sophia.json").read_text())
    dims = cfg["measured"]
    top = dims["base_frame_floor_clearance_m"] + dims["base_frame_height_m"]
    half_width = dims["base_frame_width_m"] / 2
    half_depth = dims["base_frame_depth_m"] / 2
    base_y = cfg["cad"]["base_center_y_m"]
    train = [r for r in records if r["split"] == "train"]

    def group(r):
        return 0 if r["episode"] in (0, 1) else 1

    def point(r, x):
        return np.array(
            [
                half_depth + x[0],
                base_y + x[1] + (half_width if r["corner"] == "front_left" else -half_width),
                top,
            ]
        )

    def residual(x):
        return np.concatenate(
            [np.array(r["world_m"]) + [0, 0, x[2 + group(r)]] - point(r, x) for r in train]
        )

    result = least_squares(
        residual,
        [0.16, 0.045, -0.035, -0.1],
        bounds=([0, -0.1, -0.15, -0.15], [0.25, 0.1, 0.05, 0.05]),
        loss="soft_l1",
        f_scale=0.008,
    )
    x = result.x
    k = evidence["intrinsics"]

    def projection(r, p, offset):
        cam = np.array(r["camera_world"]) + [0, 0, offset]
        p = np.array(r["camera_rotation"]).T @ (p - cam)
        return np.array([k["cx"] + k["fx"] * p[0] / -p[2], k["cy"] - k["fy"] * p[1] / -p[2]])

    metrics = {}
    for split in ("train", "validation"):
        selected = [r for r in records if r["split"] == split]
        metrics[split] = dict(n=len(selected), episodes=sorted({r["episode"] for r in selected}))
        for label, v, align in [
            ("before", np.zeros(4), False),
            ("geometry_only", x, False),
            ("aligned", x, True),
        ]:
            errs3, errs2 = [], []
            for r in selected:
                off = v[2 + group(r)] if align else 0
                p = point(r, v)
                errs3.append(np.linalg.norm(np.array(r["world_m"]) + [0, 0, off] - p))
                errs2.append(np.linalg.norm(projection(r, p, off) - r["left_px"]))
            metrics[split][label] = dict(
                rmse_m=float(np.sqrt(np.mean(np.square(errs3)))),
                rmse_px=float(np.sqrt(np.mean(np.square(errs2)))),
                max_m=float(max(errs3)),
            )
    episodes = {}
    for ep in range(18):
        native = ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
        raw = np.load(native)["lift_height"][:, 0]
        g = 0 if ep < 2 else 1
        # Only the two documented, static configurations are covered.
        expected = 0.59153 if g == 0 else 0.65579
        if np.max(abs(raw - expected)) > 0.0001:
            raise ValueError(f"Unexpected lift setting in episode {ep}")
        episodes[str(ep)] = dict(
            native_sha256=digest(native),
            lift_extension_offset_m=float(x[2 + g]),
            raw_lift_m=float(np.median(raw)),
            status="training_setting"
            if ep in (0, 3)
            else (
                "held_out_base_landmarks"
                if ep in (1, 2)
                else "same_static_recording_setting_unvalidated"
            ),
        )
    profile = dict(
        method=(
            "Training-only rigid base-frame XY fit to stereo line intersections; "
            "confirmed dimensions fixed."
        ),
        frame_translation_chassis_m=[float(x[0]), float(x[1]), 0.0],
        measured_dimensions_m={
            n: dims[n]
            for n in (
                "base_frame_width_m",
                "base_frame_depth_m",
                "base_frame_height_m",
                "base_frame_floor_clearance_m",
            )
        },
        requires_arm_reference_sha256=cfg["arm_reference"]["sha256"],
        requires_spinal_assembly_sha256=cfg["spinal_assembly"]["sha256"],
        evidence_sha256=digest(evidence_path),
        metrics=metrics,
        # Accessories have weaker edge visibility than the rail. Keep their
        # uncertainties explicit; do not treat these approximations as metrology.
        accessories=dict(
            estop_plate_center_chassis_xy_m=[0.04, -0.115],
            estop_plate_world_z_m=0.250,
            antenna_body_chassis_y_m=0.049,
            antenna_rods_world_chassis_xy_m=[
                [[0.083, 0.057, 0.480], [0.258, 0.067, 0.480]],
                [[0.083, 0.030, 0.465], [0.240, -0.050, 0.465]],
            ],
            board_center_chassis_xy_world_z_m=[0.191, 0.141, 0.241],
            board_half_size_m=[0.026, 0.027, 0.002],
            finned_module_center_chassis_xy_world_z_m=[0.117, 0.172, 0.213],
            finned_module_half_size_m=[0.025, 0.035, 0.015],
            uncertainty_m=0.020,
            basis=(
                "Stereo-reviewed ep0/frame930 antenna and board, ep1/frame855 blue plate; "
                "hidden faces and thickness are photo estimates. Antenna rods have aperture "
                "ambiguity; do not use as independent metric validation."
            ),
        ),
        replay_alignment=dict(
            scope=(
                "Pi-Atlas October26 native recordings only; exact file hash required. "
                "Effective historical torso height alignment, not a live encoder "
                "calibration."
            ),
            fit_offsets_m=[float(x[2]), float(x[3])],
            episodes=episodes,
        ),
        locked=[
            "neck CAD axes",
            "camera pose and intrinsics",
            "arm roots",
            "wheel/module poses",
            "live lift mapping",
            "hardware Home",
            "source assembly masses",
        ],
        limitations=[
            (
                "Only three training frames and five accepted held-out frames; not an "
                "untouched final test set."
            ),
            (
                "Static lift readings cannot identify encoder scale, speed or the cause of "
                "historical height offsets."
            ),
            (
                "About 10mm residual rail-plane roll/edge uncertainty remains; no camera or "
                "chassis tilt fitted."
            ),
            (
                "Electronics, rail section, hidden structure and antenna thickness remain "
                "approximations."
            ),
        ],
    )
    serialized = (json.dumps(profile, indent=2) + "\n").encode()
    (OUT / "candidate-profile.json").write_bytes(serialized)
    target = DESC / "calibration/base_alignment_2026-11-04.json"
    # Do not invalidate an enabled profile when new evidence changes the fit.
    if target.exists() and target.read_bytes() != serialized:
        suffix = hashlib.sha256(serialized).hexdigest()[:12]
        target = target.with_name(f"base_alignment_2026-11-04_{suffix}.json")
    target.write_bytes(serialized)
    cfg["base_alignment"] = dict(
        enabled=True,
        source=str(target.relative_to(DESC)),
        sha256=digest(target),
        status="stereo_fit_pending_validation",
    )
    (OUT / "candidate-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    sys.path.insert(0, str(DESC))
    from build_sophia import build

    build(cfg, OUT / "candidate")
    print(json.dumps(dict(parameters=x.tolist(), metrics=metrics), indent=2))


if __name__ == "__main__":
    main()
