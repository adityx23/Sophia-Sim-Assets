"""Reconcile existing model frames without claiming measured hardware extrinsics.

python scripts/review_sophia_mounts.py --output output/asset-review/mount-review
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTION = ROOT / "robot/yor-description"
sys.path.insert(0, str(DESCRIPTION))
from sophia_collision_geometry import read_binary_stl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def body_transform(model, data, name):
    index = model.body(name).id
    return transform(data.xpos[index], data.xmat[index])


def relative_body(model, data, parent, child):
    return np.linalg.solve(
        body_transform(model, data, parent), body_transform(model, data, child)
    )


def relative_camera(model, data, parent, camera):
    index = model.camera(camera).id
    optical = transform(
        data.cam_xpos[index],
        data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1]),
    )
    return np.linalg.solve(body_transform(model, data, parent), optical)


def relative_site(model, data, parent, site):
    index = model.site(site).id
    return np.linalg.solve(
        body_transform(model, data, parent),
        transform(data.site_xpos[index], data.site_xmat[index]),
    )


def review():
    wrist_mount = json.loads((DESCRIPTION / "sophia.json").read_text())["wrist_cameras"]
    files = [
        DESCRIPTION / name
        for name in (
            "robot.mjcf",
            "arm-upright.mjcf",
            "yor-full.mjcf",
            "yor-full-agx.mjcf",
            "sophia.mjcf",
            "make_agx_variant.py",
            "sophia.json",
            "calibration/camera_intrinsics_2026-10-30.json",
        )
    ]
    loaded, inventory = {}, []
    for path in files[:5]:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        loaded[path.name] = model, data
        tree = ET.parse(path).getroot()
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)),
                "arm_joint_counts": {
                    side: sum(
                        model.joint(j).name.startswith(f"{side}_arm_joint")
                        for j in range(model.njnt)
                    )
                    for side in ("left", "right")
                },
                "cameras": [model.camera(i).name for i in range(model.ncam)],
                "gripper_base_mesh_referenced_by_geom": bool(
                    tree.findall('.//geom[@mesh="gripper_base"]')
                ),
            }
        )

    mesh_evidence = []
    for relative, scale in (
        ("assets_nero/gripper_base.stl", 1),
        ("meshes/piper/gripper_base.stl", 1),
        ("meshes/piper/gripper_body.stl", 0.001),
        ("meshes/piper/iphone.stl", 0.001),
        ("assets_nero/left_gripper/L_Camera_Mount.stl", 1),
        ("assets_nero/right_gripper/R_Camera_Mount.stl", 1),
    ):
        path = DESCRIPTION / relative
        files.append(path)
        triangles = read_binary_stl(path).astype(float) * scale
        mesh_evidence.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "triangles": len(triangles),
                "bounds_m": [
                    triangles.min(axis=(0, 1)).tolist(),
                    triangles.max(axis=(0, 1)).tolist(),
                ],
            }
        )

    legacy = []
    for filename in ("robot.mjcf", "arm-upright.mjcf"):
        model, data = loaded[filename]
        for side in ("left", "right"):
            camera = f"{side}_arm_egocentric"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                continue
            palm = f"{side}_arm_gripper_base"
            legacy.append(
                {
                    "source": filename,
                    "side": side,
                    "chain": [
                        f"{side}_arm_link6",
                        palm,
                        f"{side}_arm_iphone_body",
                        f"{side}_arm_camera_body",
                        camera,
                    ],
                    "T_link6_gripper": relative_body(
                        model, data, f"{side}_arm_link6", palm
                    ).tolist(),
                    "T_gripper_optical": relative_camera(
                        model, data, palm, camera
                    ).tolist(),
                }
            )

    source, source_data = loaded["yor-full-agx.mjcf"]
    model, data = loaded["sophia.mjcf"]
    current, errors = [], []
    for side in ("left", "right"):
        link, ee, palm = (
            f"{side}_arm_link7",
            f"{side}_arm_end_effector",
            f"{side}_gripper_base",
        )
        stock = relative_body(source, source_data, link, palm)
        mounted = relative_body(model, data, link, palm)
        errors.append(float(np.max(abs(stock - mounted))))
        current.append(
            {
                "side": side,
                "T_link7_end_effector": relative_body(model, data, link, ee).tolist(),
                "T_end_effector_gripper": relative_body(model, data, ee, palm).tolist(),
                "T_link7_gripper": mounted.tolist(),
                "T_link7_grasp": relative_site(
                    model, data, link, f"{side}_gripper_center"
                ).tolist(),
                "T_gripper_grasp": relative_site(
                    model, data, palm, f"{side}_gripper_center"
                ).tolist(),
                "T_gripper_optical": relative_camera(
                    model, data, palm, f"{side}_wrist"
                ).tolist(),
                "T_link7_optical": relative_camera(
                    model, data, link, f"{side}_wrist"
                ).tolist(),
                "finger_opening_axes_in_gripper": [
                    (
                        data.xmat[model.body(palm).id].reshape(3, 3).T
                        @ data.xaxis[model.joint(f"{side}_gripper_finger_{finger}").id]
                    ).tolist()
                    for finger in "ab"
                ],
            }
        )

    # Use varied poses to check the complete compiled parent chains, including
    # both arms and lift, rather than compare XML offsets with unlike origins.
    rng = np.random.default_rng(31)
    rigid_errors = []
    for _ in range(20):
        for j in range(model.njnt):
            if model.joint(j).name.startswith(("left_arm_joint", "right_arm_joint")):
                data.qpos[model.jnt_qposadr[j]] = rng.uniform(*model.jnt_range[j])
        lift = rng.uniform(0, 0.5588)
        data.qpos[model.jnt_qposadr[model.joint("lift_joint").id]] = lift
        data.qpos[model.jnt_qposadr[model.joint("sophia_middle_slide").id]] = lift / 2
        mujoco.mj_forward(model, data)
        for entry in current:
            side = entry["side"]
            for observed, expected in (
                (
                    relative_body(
                        model, data, f"{side}_arm_link7", f"{side}_gripper_base"
                    ),
                    entry["T_link7_gripper"],
                ),
                (
                    relative_camera(
                        model, data, f"{side}_gripper_base", f"{side}_wrist"
                    ),
                    entry["T_gripper_optical"],
                ),
            ):
                rigid_errors.append(float(np.max(abs(observed - expected))))

    checks = {
        "stock_source_to_sophia_mounts_match": max(errors) < 1e-10,
        "left_right_local_mounts_agree": bool(
            np.allclose(
                current[0]["T_link7_gripper"], current[1]["T_link7_gripper"], atol=1e-10
            )
        ),
        "legacy_camera_compositions_agree": all(
            np.allclose(
                row["T_gripper_optical"], legacy[0]["T_gripper_optical"], atol=1e-10
            )
            for row in legacy
        ),
        "shared_stock_mesh_is_unused_in_legacy_camera_assembly": (
            mesh_evidence[0]["sha256"] == mesh_evidence[1]["sha256"]
            and not inventory[0]["gripper_base_mesh_referenced_by_geom"]
        ),
        "compiled_mounts_remain_rigid_across_20_joint_lift_poses": max(rigid_errors)
        < 1e-10,
    }
    files.append(Path(__file__))
    return {
        "review_status": "completed_with_accepted_assembly_estimates",
        "checks_passed": all(checks.values()),
        "checks": checks,
        "user_confirmed_mounts_unchanged": True,
        "hardware_extrinsics_measured": False,
        "physical_geometry_changed": False,
        "required_measurements": [],
        "required_robot_sessions": [],
        "model_inventory": inventory,
        "mesh_evidence": mesh_evidence,
        "legacy_camera_frames": legacy,
        "adopted_current_frames": current,
        "maximum_source_mount_matrix_error": max(errors),
        "maximum_rigid_mount_matrix_error": max(rigid_errors),
        "decisions": {
            "gripper": "Retain the stock AgileX assembly from yor-full-agx.mjcf. "
            "Its link7-to-palm and grasp transforms are preserved in Sophia. "
            "make_agx_variant.py explicitly labels the installed adapter offset as an "
            "estimate; source inheritance does not make it a hardware measurement.",
            "legacy_camera": "Do not transplant the Piper/iPhone camera transform. "
            "It is attached to a different 37342-triangle gripper_body mesh on a "
            "six-joint arm. The shared 12988-triangle stock housing is declared but "
            "unused in that camera assembly, so its matching filename is not a frame bridge.",
            "custom_camera_mounts": "L/R_Camera_Mount meshes belong to the custom "
            "articulated gripper assembly in yor-full.mjcf; that model defines no "
            "wrist optical camera. Retain these as alternative-assembly evidence.",
            "wrist": {
                "configuration": wrist_mount,
                "interpretation": "Upper-side placement is user confirmed. Offset magnitudes "
                "and toe-in remain photo estimates; supplied per-camera intrinsics are retained. "
                "Camera, lens and visible board use the same mount orientation.",
            },
            "completion": "Frame/source reconciliation is complete. No supported "
            "numeric mounting correction was found; remaining uncertainty is an "
            "accepted simulation assumption, not a request for further measurements.",
        },
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = review()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "mount-review.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Mount review: {sum(report['checks'].values())}/{len(report['checks'])} checks passed; {path}"
    )
    raise SystemExit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
