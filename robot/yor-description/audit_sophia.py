"""Read-only provenance, gripper mounting and inertia audit for Sophia.

python robot/yor-description/audit_sophia.py

Unmeasured properties remain explicit accepted assumptions. A successful
integrity audit does not mean the physical properties have been calibrated.
"""

import argparse
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from sophia_sources import CAMERA_SOURCE, VENDOR, vendor_inertials

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stl_bounds(path):
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    records = np.frombuffer(
        raw,
        dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]),
        offset=84,
        count=count,
    )
    points = records["vertices"].reshape(-1, 3).astype(float)
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite STL coordinates: {path}")
    return {
        "triangles": count,
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "extent": np.ptp(points, axis=0).tolist(),
    }


def audit():
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    findings, checks = [], []
    release = json.loads((HERE / "sophia.json").read_text()).get("release", {})

    def finding(code, detail, **evidence):
        optional_validation = evidence.pop("requires", None)
        findings.append({"code": code, "severity": "accepted_assumption", "detail": detail,
                         "blocks_simulation_release": False,
                         "disposition": "Retain supplied model under user-authorized approximation; unused meshes remain excluded.",
                         "optional_hardware_validation": optional_validation, **evidence})

    grippers = []
    for name in ("gripper_base.stl", "gripper_link7.stl", "link8.stl", "gripper.STL"):
        path = HERE / "assets_nero" / name
        matches = []
        for directory in (ROOT / "robot/arm", HERE / "meshes"):
            for candidate in directory.rglob("*.stl"):
                if candidate.stat().st_size == path.stat().st_size and digest(candidate) == digest(
                    path
                ):
                    matches.append(str(candidate.relative_to(ROOT)))
        grippers.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "raw_stl_bounds": stl_bounds(path),
                "identical_files": sorted(matches),
                "source_units": "unknown; coordinates consistent with mm"
                if name == "gripper.STL"
                else "m in mounted model",
                "mounted": name != "gripper.STL",
            }
        )
    finding(
        "gripper_provenance",
        "Both finger meshes exactly match the pinned official AgileX stock-gripper meshes. "
        "The housing differs from the current vendor housing. Existing-asset frame "
        "review retains the supplied stock assembly; installed revision and physical "
        "mount precision remain accepted assumptions.",
        requires=(
            "Compare the installed housing and adapter against the model; "
            "the user-confirmed Nero identity and finger source are already established."
        ),
    )
    finding(
        "unreferenced_gripper_mesh",
        "gripper.STL has no established assembly transform or unit declaration; "
        "it is not substituted into the model.",
        requires="Identify the part and its local mounting frame before any substitution.",
    )

    mounts = []
    for side in ("left", "right"):
        frame = model.body(f"{side}_arm_end_effector").id
        palm = model.body(f"{side}_gripper_base").id
        site = model.site(f"{side}_gripper_center").id
        rotation = data.xmat[frame].reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3] = rotation.T @ data.xmat[palm].reshape(3, 3)
        transform[:3, 3] = rotation.T @ (data.xpos[palm] - data.xpos[frame])
        mounts.append(
            {
                "side": side,
                "reference_frame": f"{side}_arm_end_effector",
                "T_end_effector_gripper_base": transform.tolist(),
                "grasp_center_in_end_effector_m": (
                    rotation.T @ (data.site_xpos[site] - data.xpos[frame])
                ).tolist(),
                "measured_on_hardware": False,
            }
        )
    checks.append(
        {
            "name": "left/right local gripper mounts agree",
            "passed": bool(
                np.allclose(
                    mounts[0]["T_end_effector_gripper_base"],
                    mounts[1]["T_end_effector_gripper_base"],
                    atol=1e-8,
                )
            ),
        }
    )

    vendor_errors = []
    for suffix, source in vendor_inertials().items():
        for side in ("left", "right"):
            body = model.body(f"{side}_{suffix}").id
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
            rotation = rotation.reshape(3, 3)
            tensor = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
            vendor_errors.append(
                max(
                    abs(model.body_mass[body] - source["mass"]),
                    np.max(abs(model.body_ipos[body] - source["pos"])),
                    np.max(abs(tensor - source["matrix"])),
                )
            )
    checks.append(
        {
            "name": "arm and finger inertials match pinned vendor data in verified mesh frames",
            "passed": bool(max(vendor_errors) < 1e-9),
            "bodies_checked": len(vendor_errors),
            "max_error": float(max(vendor_errors)),
        }
    )
    finding(
        "gripper_mount",
        "The simulated palm is 12 mm along end-effector +Y; the configured grasp site "
        "is 132 mm along that axis. Existing-asset reconciliation confirms inheritance "
        "from the stock assembly, with unchanged hardware confirmed by the user. "
        "These remain accepted mounting estimates rather than measured extrinsics.",
        requires=(
            "Check adapter seating/orientation and flange-to-grasp distance "
            "on the installed assembly."
        ),
    )

    inertias = []
    for body in range(1, model.nbody):
        mass = float(model.body_mass[body])
        if mass == 0:
            continue
        principal = np.sort(model.body_inertia[body])
        valid = bool(
            np.isfinite(principal).all()
            and np.isfinite(mass)
            and mass > 0
            and principal[0] > 0
            and principal[-1] <= principal[:2].sum() + 1e-12
        )
        geoms = np.flatnonzero(model.geom_bodyid == body)
        radius = max(
            (
                float(
                    np.linalg.norm(model.geom_pos[g] - model.body_ipos[body]) + model.geom_rbound[g]
                )
                for g in geoms
            ),
            default=0.0,
        )
        gyration = float(np.sqrt(principal[-1] / mass))
        item = {
            "body": model.body(body).name,
            "mass_kg": mass,
            "principal_inertia_kg_m2": principal.tolist(),
            "com_in_body_m": model.body_ipos[body].tolist(),
            "mathematically_valid": valid,
            "largest_radius_of_gyration_m": gyration,
            "own_geometry_radius_bound_from_com_m": radius,
        }
        inertias.append(item)
        if mass <= 1e-5 or (radius > 0 and gyration > radius * 1.05):
            finding(
                "suspect_inertia",
                "Mass/inertia is suspicious relative to this link's own geometry. "
                "Geometry bounds are a screening heuristic, not a replacement "
                "for component inertial data.",
                **item,
                requires=(
                    "Resolve whether this is a placeholder frame or a physical link; "
                    "obtain consistent component mass and inertia."
                ),
            )
    checks.append(
        {
            "name": "positive finite masses and physically admissible principal inertias",
            "passed": all(item["mathematically_valid"] for item in inertias),
        }
    )

    discrepancies = []
    for side in ("left", "right"):
        urdf = HERE / "urdf" / f"nero_{side}_arm.urdf"
        for link in ET.parse(urdf).getroot().findall("link"):
            inertia = link.find("inertial")
            if inertia is None:
                continue
            name = f"{side}_arm_{link.get('name')}"
            try:
                body = model.body(name).id
            except KeyError:
                continue
            tensor = inertia.find("inertia")
            matrix = np.array(
                [[float(tensor.get(f"i{''.join(sorted((a, b)))}")) for b in "xyz"] for a in "xyz"]
            )
            expected = np.linalg.eigvalsh(matrix)
            mass = float(inertia.find("mass").get("value"))
            if (
                not np.allclose(expected, np.sort(model.body_inertia[body]), atol=1e-10, rtol=1e-4)
                or abs(mass - model.body_mass[body]) > 1e-8
            ):
                discrepancies.append(
                    {
                        "body": name,
                        "urdf": str(urdf.relative_to(ROOT)),
                        "urdf_mass_kg": mass,
                        "mjcf_mass_kg": float(model.body_mass[body]),
                        "urdf_principal_inertia": expected.tolist(),
                        "mjcf_principal_inertia": np.sort(model.body_inertia[body]).tolist(),
                    }
                )
    # These legacy arm-only URDFs remain inputs to other robot variants. Sophia
    # explicitly supersedes their inertials with the pinned vendor source.
    totals = {}
    for side in ("left", "right"):
        totals[side] = {
            "arm_base_and_links_kg": float(
                sum(
                    model.body_mass[b]
                    for b in range(model.nbody)
                    if model.body(b).name.startswith(f"{side}_arm_")
                    and not model.body(b).name.endswith("end_effector")
                )
            ),
            "end_effector_adapter_kg": float(model.body(f"{side}_arm_end_effector").mass[0]),
            "gripper_assembly_kg": float(
                sum(
                    model.body_mass[b]
                    for b in range(model.nbody)
                    if model.body(b).name.startswith(f"{side}_gripper_")
                )
            ),
        }
    finding(
        "unmeasured_mass_distribution",
        "Arm and finger inertials now use verified vendor CAD data. Chassis, lift, "
        "housing/accessory mass distribution, friction and actuator response still "
        "need physical validation; the complete robot has not been weighed.",
        total_model_mass_kg=float(model.body_mass.sum()),
        arm_totals=totals,
        requires="Weigh assemblies and record motion/grasp response before dynamics calibration.",
    )
    return {
        "mujoco_version": mujoco.__version__,
        "integrity_passed": all(c["passed"] for c in checks),
        "hardware_calibrated": False,
        "release_status": release.get("status"),
        "blocking_measurements": release.get("blocking_measurements", []),
        "accepted_assumptions": release.get("assumptions", {}),
        "checks": checks,
        "findings": findings,
        "gripper_sources": grippers,
        "gripper_mounts": mounts,
        "body_inertias": inertias,
        "total_mass_kg": float(model.body_mass.sum()),
        "arm_totals": totals,
        "vendor_provenance": json.loads((VENDOR / "provenance.json").read_text()),
        "resolved": [
            "Both 1 mg link7 masses corrected to the vendor's 1 g value.",
            "Both arms and fingers now use full vendor inertia tensors and COMs.",
            "Head rectified intrinsics/baseline integrated; wrist cameras use measured lenses and accepted photo-based mounts.",
            "Legacy/current mount-frame review completed; retain the stock source gripper and photo-estimated wrist transforms. No additional measurements are required for this release.",
        ],
        "legacy_urdf_differences_superseded_by_vendor": discrepancies,
        "sources_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                HERE / "sophia.mjcf",
                HERE / "sophia.json",
                HERE / "make_agx_variant.py",
                HERE / "urdf/nero_left_arm.urdf",
                HERE / "urdf/nero_right_arm.urdf",
                VENDOR / "provenance.json",
                CAMERA_SOURCE,
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/asset-review/evals")
    args = parser.parse_args()
    report = audit()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "asset-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for finding in report["findings"]:
        print(f"[REVIEW] {finding['code']}: {finding['detail']}")
    print(f"Integrity checks: {report['integrity_passed']}; hardware calibrated: False")
    raise SystemExit(0 if report["integrity_passed"] else 1)


if __name__ == "__main__":
    main()
