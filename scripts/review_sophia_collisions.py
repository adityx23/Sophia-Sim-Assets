"""Inspect the identified gripper/frame and camera/neck collision poses.

Clip original CAD triangles against each box, inset by 0.1 mm, to find actual
surface crossings independently of MuJoCo's convex-hull collision result.
Absent surface crossings alone are not a general solid-containment test;
positive distances to the source-preserving component hulls establish the
specific clear gripper/rail case. No claim of hardware clearance is made.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "robot/yor-description"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from robot.sim_collision import SelfCollisionGuard  # noqa: E402
from validate_sophia import qadr, reset  # noqa: E402


def mesh_vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return (
        model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
        + data.geom_xpos[geom]
    )


def clip_polygon(polygon, axis, sign, limit):
    result = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        a, b = sign * start[axis] - limit, sign * end[axis] - limit
        if a <= 0:
            result.append(start)
        if (a <= 0) != (b <= 0):
            result.append(start + (end - start) * a / (a - b))
    return np.asarray(result)


def surface_crossing(model, data, source_geom, box, inset=0.0001):
    vertices = mesh_vertices(model, data, source_geom)
    local = (vertices - data.geom_xpos[box]) @ data.geom_xmat[box].reshape(3, 3)
    mesh = model.geom_dataid[source_geom]
    start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    triangles = local[model.mesh_face[start : start + count]]
    half = model.geom_size[box] - inset
    candidates = triangles[((triangles.min(1) <= half) & (triangles.max(1) >= -half)).all(1)]
    witnesses = []
    for triangle in candidates:
        polygon = triangle.copy()
        for axis in range(3):
            for sign in (-1, 1):
                polygon = clip_polygon(polygon, axis, sign, half[axis])
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            area = sum(
                np.linalg.norm(np.cross(polygon[i] - polygon[0], polygon[i + 1] - polygon[0])) / 2
                for i in range(1, len(polygon) - 1)
            )
            if area > 1e-12:
                point = polygon.mean(0)
                witnesses.append(
                    {
                        "depth_inside_box_m": float(np.min(model.geom_size[box] - abs(point))),
                        "point_in_box_frame_m": point.tolist(),
                        "clipped_triangle_area_m2": float(area),
                    }
                )
    witnesses.sort(key=lambda w: -w["depth_inside_box_m"])
    return {
        "source_triangles": len(triangles),
        "inset_m": inset,
        "crossing_triangles": len(witnesses),
        "deepest_witnesses": witnesses[:3],
    }


def housing_geoms(model, side, group):
    body = model.body(f"{side}_gripper_base").id
    return [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_group[g] == group
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]


def set_arm_pose(model, data, lift, fraction):
    data.qpos[qadr(model, "lift_joint")] = lift
    data.qpos[qadr(model, "sophia_middle_slide")] = lift / 2
    adrs = [qadr(model, f"{side}_arm_joint{i}") for side in ("left", "right") for i in range(1, 8)]
    data.qpos[adrs] = (1 - fraction) * model.key("home").qpos[adrs] + fraction * model.key(
        "reference"
    ).qpos[adrs]
    mujoco.mj_forward(model, data)


def render_pair(model, data, source, colliders, box, directory, name):
    """Actual source surface versus the convex shapes used for collision."""
    m = copy.copy(model)
    d = mujoco.MjData(m)
    d.qpos[:] = data.qpos
    mujoco.mj_forward(m, d)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (d.geom_xpos[source] + d.geom_xpos[box]) / 2
    camera.distance = 0.22 if name.startswith("neck") else 0.34
    camera.azimuth, camera.elevation = 130, -15
    with mujoco.Renderer(m, height=480, width=640) as renderer:
        for label, selected in (("cad", [source]), ("collision-hulls", colliders)):
            m.geom_group[:] = 5
            m.geom_group[[*selected, box]] = 0
            m.geom_matid[[*selected, box]] = -1
            m.geom_rgba[selected] = [0.15, 0.55, 0.95, 1]
            m.geom_rgba[box] = [1, 0.25, 0.12, 0.65]
            option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = label == "collision-hulls"
            renderer.update_scene(d, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(directory / f"{name}-{label}.png")


def review(model, output, render=False):
    guard = SelfCollisionGuard(model, 0.005)
    cases = []
    definitions = [
        ("lowered-transition", 0.0, 0.87, "corner_post_1_1_collision"),
        ("100mm-hanging-endpoint", 0.1, 1.0, "upper_rail_x_1_collision"),
        ("100mm-transition", 0.1, 0.83, "upper_rail_x_1_collision"),
        ("neck-extreme", None, None, "zed_housing_collision"),
    ]
    for name, lift, fraction, box_name in definitions:
        data = reset(model)
        box = model.geom(box_name).id
        if lift is None:
            for axis in ("pan", "tilt"):
                joint = model.joint(f"head_{axis}_joint")
                data.qpos[joint.qposadr[0]] = joint.range[1]
            mujoco.mj_forward(model, data)
            names = {model.geom(g).name for g in range(model.ngeom)}
            source = model.geom(
                "neck_fixed_bracket_collision"
                if "neck_fixed_bracket_collision" in names
                else "spinal_fixed_adapter_collision"
            ).id
            colliders = [source]
        else:
            set_arm_pose(model, data, lift, fraction)
            source = housing_geoms(model, "left", 2)[0]
            colliders = housing_geoms(model, "left", 3)
        distances = [
            float(mujoco.mj_geomDistance(model, data, box, g, 0.5, None)) for g in colliders
        ]
        nearby = guard._contacts(data.qpos)
        contacts = [
            float(distance)
            for pair, distance in nearby.items()
            if box in pair and any(g in pair for g in colliders)
        ]
        # Reproduce the old query's margin-only omission on the same geometry.
        legacy = SelfCollisionGuard(model, 0.005)
        legacy.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        legacy_contacts = [
            float(distance)
            for pair, distance in legacy._contacts(data.qpos).items()
            if box in pair and any(g in pair for g in colliders)
        ]
        row = {
            "case": name,
            "lift_extension_m": lift,
            "arm_path_fraction": fraction,
            "box": box_name,
            "source_mesh": model.mesh(model.geom_dataid[source]).name,
            "collision_geoms": [model.geom(g).name or f"geom_{g}" for g in colliders],
            "collider_distances_m": distances,
            "minimum_distance_m": min(distances),
            "source_surface": surface_crossing(model, data, source, box)
            if model.geom_type[box] == mujoco.mjtGeom.mjGEOM_BOX
            else None,
            "guard_margin_m": guard.margin,
            "guard_contacts_m": contacts,
            "legacy_midphase_query_contacts_m": legacy_contacts,
            "qpos": data.qpos.tolist(),
        }
        if lift is None:
            zero = model.numeric("neck_zero_deg").data
            row["api_pan_tilt_deg"] = [
                float(zero[0] - np.rad2deg(data.qpos[qadr(model, "head_pan_joint")])),
                float(zero[1] + np.rad2deg(data.qpos[qadr(model, "head_tilt_joint")])),
            ]
        cases.append(row)
        if render:
            render_pair(model, data, source, colliders, box, output, name)

    # Check all colliding samples in the identified paths/grid, rather than only
    # their deepest pose. Report the worst contact per sampled configuration.
    sweeps = []
    for label, configurations in (
        ("arms-lowered", [(0.0, float(t)) for t in np.linspace(0, 1, 101)]),
        ("arms-100mm", [(0.1, float(t)) for t in np.linspace(0, 1, 101)]),
        (
            "neck",
            [
                (float(p), float(t))
                for p in np.linspace(*model.joint("head_pan_joint").range, 25)
                for t in np.linspace(*model.joint("head_tilt_joint").range, 19)
            ],
        ),
    ):
        rows = []
        data = reset(model)
        for first, second in configurations:
            if label == "neck":
                data.qpos[qadr(model, "head_pan_joint")] = first
                data.qpos[qadr(model, "head_tilt_joint")] = second
                mujoco.mj_forward(model, data)
            else:
                set_arm_pose(model, data, first, second)
            contacts = [
                c
                for c in data.contact
                if c.dist < -0.0001 and model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]
            ]
            if not contacts:
                continue
            contact = min(contacts, key=lambda c: c.dist)
            pair = [int(contact.geom1), int(contact.geom2)]
            meshes = [g for g in pair if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            boxes = [g for g in pair if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            crossing = None
            if len(meshes) == len(boxes) == 1:
                source = meshes[0]
                body_name = model.body(model.geom_bodyid[source]).name
                if body_name in ("left_gripper_base", "right_gripper_base"):
                    source = housing_geoms(model, body_name.split("_")[0], 2)[0]
                crossing = surface_crossing(model, data, source, boxes[0])
            rows.append(
                {
                    "configuration": [first, second],
                    "penetration_m": float(-contact.dist),
                    "geoms": [model.geom(g).name or f"geom_{g}" for g in pair],
                    "source_surface": crossing,
                }
            )
        sweeps.append(
            {
                "name": label,
                "sampled": len(configurations),
                "colliding": len(rows),
                "source_surface_crossings": sum(
                    bool(r["source_surface"] and r["source_surface"]["crossing_triangles"])
                    for r in rows
                ),
                "poses": rows,
            }
        )
    endpoint = cases[1]
    # Keep the original 2.9mm BVH-margin regression as an explicitly synthetic
    # fixture. Moving the production frame forward correctly clears this pose.
    fixture = copy.copy(model)
    fixture_rail = fixture.geom("upper_rail_x_1_collision").id
    fixture.geom_pos[fixture_rail, :2] = [0, 0.2059]
    fd = reset(fixture)
    set_arm_pose(fixture, fd, 0.1, 1.0)
    fixture_parts = housing_geoms(fixture, "left", 3)
    fixture_contacts = [
        float(distance)
        for pair, distance in SelfCollisionGuard(fixture, 0.005)._contacts(fd.qpos).items()
        if fixture_rail in pair and any(g in pair for g in fixture_parts)
    ]
    checks = [
        {
            "name": "former hull bridge removed while retaining separated component surfaces",
            "passed": len(endpoint["collision_geoms"]) == 5
            and endpoint["minimum_distance_m"] > 0
            and endpoint["source_surface"]["crossing_triangles"] == 0,
        },
        {
            "name": "synthetic positive gripper/rail gap below 5 mm is detected by the query guard",
            "passed": bool(fixture_contacts)
            and all(0 < distance < 0.005 for distance in fixture_contacts),
        },
        {
            "name": "reviewed cases are clear or have source-supported intersections",
            "passed": all(
                case["minimum_distance_m"] >= 0
                or (
                    case["source_surface"] is not None
                    and case["source_surface"]["crossing_triangles"] > 0
                )
                for case in cases
            ),
        },
        {
            "name": "all retained worst contacts in sampled configurations cross source surfaces",
            "passed": all(row["colliding"] == row["source_surface_crossings"] for row in sweeps),
        },
        {
            "name": "query override leaves physics midphase and margins unchanged",
            "passed": not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE))
            and not np.any(model.geom_margin),
        },
    ]
    return {
        "method": __doc__,
        "hardware_clearance_verified": False,
        "cases": cases,
        "synthetic_margin_fixture": {
            "production_geometry_changed": False,
            "rail_chassis_xy_m": [0, 0.2059],
            "guard_contacts_m": fixture_contacts,
        },
        "checks": checks,
        "review_checks_passed": all(c["passed"] for c in checks),
        "sweeps": sweeps,
        "query_midphase_disabled": bool(
            guard.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        ),
        "physics_midphase_disabled": bool(
            model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "scene-sophia.mjcf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--expect-reviewed-geometry",
        action="store_true",
        help="Fail if the corrected shape and clearance invariants do not hold",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    report = review(model, args.output, args.render)
    sources = [
        Path(__file__),
        args.model,
        HERE / "sophia.mjcf",
        HERE / "assets_nero/gripper_base.stl",
        HERE / "assets_nero/head/topbase2.stl",
        ROOT / "robot/sim_collision.py",
        HERE / "sophia_collision_geometry.py",
    ]
    report["source_sha256"] = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (args.output / "shape-review.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        print(
            row["case"],
            "distance_mm",
            round(row["minimum_distance_m"] * 1000, 3),
            "CAD crossings",
            row["source_surface"]["crossing_triangles"]
            if row["source_surface"] is not None
            else "mesh/mesh clearance",
            flush=True,
        )
    for row in report["sweeps"]:
        print(
            row["name"],
            "colliding",
            row["colliding"],
            "CAD supported",
            row["source_surface_crossings"],
            flush=True,
        )
    if args.expect_reviewed_geometry and not report["review_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
