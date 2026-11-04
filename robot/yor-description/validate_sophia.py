"""Deterministic physical checks and repeatable visual review for Sophia.

    MUJOCO_GL=egl python robot/yor-description/validate_sophia.py --render

Checks model consistency, resting contact, lift coupling, gripper travel and
collision response. Passing does not establish sim-to-real dynamic calibration.
"""

import argparse
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sophia_sources import vendor_inertials

HERE = Path(__file__).resolve().parent


def qadr(model, name):
    return model.jnt_qposadr[model.joint(name).id]


def reset(model, pose="home"):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(pose).id)
    mujoco.mj_forward(model, data)
    return data


def step(model, data, seconds):
    mujoco.mj_step(model, data, nstep=round(seconds/model.opt.timestep))


def copy_state(source, data, target, dest):
    for j in range(source.njnt):
        name = source.joint(j).name
        try:
            to = target.joint(name).id
        except KeyError:
            continue
        size = 7 if source.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        dest.qpos[target.jnt_qposadr[to]:target.jnt_qposadr[to]+size] = data.qpos[source.jnt_qposadr[j]:source.jnt_qposadr[j]+size]
    for a in range(source.nu):
        try:
            target_actuator = target.actuator(source.actuator(a).name).id
        except KeyError:
            continue
        dest.ctrl[target_actuator] = data.ctrl[a]
    mujoco.mj_forward(target, dest)


def jaw_gap(model, data, side):
    """Inside gap between the actual collision-mesh surfaces along palm Y."""
    palm = model.body(f"{side}_gripper_base").id
    direction = data.xmat[palm].reshape(3, 3)[:, 1]
    extents = []
    for finger in "ab":
        geom = model.geom(f"{side}_finger_{finger}_collision").id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        projected = (world-data.xpos[palm]) @ direction
        extents.append((projected.min(), projected.max()))
    extents.sort()
    return float(extents[1][0]-extents[0][1])


def box_z_bounds(model, data, name):
    geom = model.geom(name).id
    half = abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom]
    return data.geom_xpos[geom, 2] + np.array([-half, half])


def box_world_bounds(model, data, name):
    geom = model.geom(name).id
    half = abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
    return np.array([data.geom_xpos[geom] - half, data.geom_xpos[geom] + half])


def validate(model):
    checks = []
    config = json.loads((HERE / "sophia.json").read_text())
    measured = config["measured"]
    travel = measured["shoulder_height_raised_m"] - measured["shoulder_height_lowered_m"]

    def check(name, ok, **details):
        checks.append({"name": name, "passed": bool(ok), **details})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {details}", flush=True)

    original = mujoco.MjModel.from_xml_path(str(HERE / "scene-full-agx.mjcf"))
    expected_total = float(original.body_mass.sum())
    vendor_errors = []
    for suffix, source in vendor_inertials().items():
        for side in ("left", "right"):
            name = f"{side}_{suffix}"
            b = model.body(name).id
            expected_total += source["mass"]-float(original.body(name).mass[0])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[b])
            rotation = rotation.reshape(3, 3)
            tensor = rotation @ np.diag(model.body_inertia[b]) @ rotation.T
            vendor_errors.append(max(abs(float(model.body_mass[b])-source["mass"]),
                                     float(np.max(abs(model.body_ipos[b]-source["pos"]))),
                                     float(np.max(abs(tensor-source["matrix"])))))
    check("pinned vendor inertias and expected total mass", max(vendor_errors) < 1e-9
          and abs(model.body_mass.sum()-expected_total) < 1e-8,
          mass_kg=float(model.body_mass.sum()), expected_mass_kg=expected_total,
          max_vendor_inertial_error=max(vendor_errors))
    data = reset(model, "lowered")
    frame_expected = np.array([measured["base_frame_depth_m"], measured["base_frame_width_m"],
                               measured["base_frame_height_m"]])
    frame_details = {}
    frame_ok = True
    for label, group in (("visual", 2), ("collision", 3)):
        bounds = [box_world_bounds(model, data, model.geom(g).name) for g in range(model.ngeom)
                  if model.geom_group[g] == group and (model.geom(g).name or "").startswith(
                      ("lower_rail_", "upper_rail_", "corner_post_"))]
        low, high = np.min(bounds, axis=(0, 1)), np.max(bounds, axis=(0, 1))
        frame_details[label] = {"size_xyz_m": (high-low).tolist(),
                                "floor_to_bottom_m": float(low[2]), "floor_to_top_m": float(high[2])}
        frame_ok &= np.allclose(high-low, frame_expected, atol=1e-8, rtol=0)
        frame_ok &= abs(low[2]-measured["base_frame_floor_clearance_m"]) < 1e-8
    check("measured base frame envelope and floor clearance", frame_ok, **frame_details,
          excludes_wheel_module_and_accessory_overhangs=True)
    sections = {}
    section_ok = True
    for name, prefix in (("fixed_column", "pillar_lower"), ("upper_column", "pillar_upper")):
        expected = [measured[f"{prefix}_depth_m"], measured[f"{prefix}_width_m"]]
        for suffix in ("", "_collision"):
            low, high = box_world_bounds(model, data, name+suffix)
            sections[name+suffix] = (high-low)[:2].tolist()
            section_ok &= np.allclose((high-low)[:2], expected, atol=1e-8, rtol=0)
    check("measured rectangular lower and upper pillar sections", section_ok,
          section_xy_m=sections, middle_section_still_estimated=True)
    # Internal arm geometry is unchanged; the measured flange mounts are new.
    data = reset(model, "reference")
    old = reset(original)
    copy_state(model, data, original, old)
    errors = []
    for side in ("left", "right"):
        positions = []
        for m, d in ((model, data), (original, old)):
            base = m.body(f"{side}_arm_base_link").id
            positions.append(d.xmat[base].reshape(3, 3).T @ (d.site_xpos[m.site(f"{side}_arm_ee").id]-d.xpos[base]))
        errors.append(float(np.linalg.norm(positions[0]-positions[1])))
    check("original arm kinematics relative to each flange preserved", max(errors) < 1e-7, position_errors_m=errors)
    ik = mujoco.MjModel.from_xml_path(str(HERE / "sophia-ik.mjcf"))
    ik_errors, rotation_errors = [], []
    for pose in ("lowered", "home", "reference", "raised"):
        data = reset(model, pose)
        ik_data = mujoco.MjData(ik)
        copy_state(model, data, ik, ik_data)
        for side in ("left", "right"):
            a, b = model.site(f"{side}_arm_ee").id, ik.site(f"{side}_arm_ee").id
            ik_errors.append(float(np.linalg.norm(data.site_xpos[a]-ik_data.site_xpos[b])))
            rotation_errors.append(float(np.max(abs(data.site_xmat[a]-ik_data.site_xmat[b]))))
    check("Sophia physics and companion IK mounts agree", max(ik_errors) < 1e-7 and max(rotation_errors) < 1e-7,
          max_position_error_m=max(ik_errors), max_rotation_matrix_error=max(rotation_errors))
    data = reset(model)
    spacing = float(np.linalg.norm(data.xpos[model.body("left_arm_base_link").id]-data.xpos[model.body("right_arm_base_link").id]))
    check("measured arm flange spacing", abs(spacing-measured["arm_flange_spacing_m"]) < 1e-7, spacing_m=spacing)
    neck = json.loads((HERE.parent / "neck" / "neck_calibration.json").read_text())
    zero = model.numeric("neck_zero_deg").data
    home = model.numeric("neck_home_deg").data
    limits, expected = [], []
    for i, (name, sign) in enumerate((("pan", -1), ("tilt", 1))):
        limits.append(sorted(zero[i] + sign*np.rad2deg(model.joint(f"head_{name}_joint").range)))
        motor = neck["motors"][name]
        resolution = 4096 if neck["model"].startswith(("sts", "sm")) else 1024
        half = (motor["range_max"]-motor["range_min"])/2*360/(resolution-1)
        expected.append([-half, half])
    check("neck limits and home match recorded calibration",
          np.allclose(limits, expected, atol=1e-6, rtol=0) and
          np.allclose(home, [neck["home_deg"][name] for name in ("pan", "tilt")]),
          api_limits_deg=np.asarray(limits).tolist(), home_deg=home.tolist())
    source = ast.parse((HERE.parent / "arm" / "can_gripper.py").read_text())
    stroke = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GRIPPER_STROKE_M" for t in node.targets))
    check("protocol stroke retained separately from physical jaw gap",
          np.allclose(model.numeric("gripper_protocol_stroke_m").data, stroke), protocol_stroke_m=stroke)
    for name in ("closed", "open"):
        data = reset(model)
        expected_gap = measured[f"gripper_gap_{name}_m"]
        for side in ("left", "right"):
            for f in "ab":
                joint = model.joint(f"{side}_gripper_finger_{f}")
                data.qpos[qadr(model, joint.name)] = joint.range[0 if name == "closed" else 1]
        mujoco.mj_forward(model, data)
        gaps = [jaw_gap(model, data, side) for side in ("left", "right")]
        check(f"measured {name} gap at collision surfaces", max(abs(g-expected_gap) for g in gaps) < .0001,
              measured_gap_m=expected_gap, mesh_gaps_m=gaps)
    for pose in ("home", "reference"):
        data = reset(model, pose)
        check(f"{pose}: no initial penetration", not any(c.dist < -1e-4 for c in data.contact), contacts=int(data.ncon))
        xy0 = data.qpos[:2].copy()
        step(model, data, 5)
        contacts = {model.body(model.geom_bodyid[c.geom2]).name for c in data.contact
                    if model.geom(c.geom1).name == "floor"}
        wheels = {"MK4_Colson_Hub_4_X_1_5" + suffix for suffix in ("", "_2", "_3", "_4")}
        drift = float(np.linalg.norm(data.qpos[:2]-xy0))
        check(f"{pose}: rests on four wheels", contacts == wheels and drift < .02 and not data.warning.number.any(),
              ground_bodies=sorted(contacts), drift_m=drift, warnings=data.warning.number.tolist())

    for pose in ("lowered", "raised"):
        data = reset(model, pose)
        step(model, data, 5)
        # Tape endpoints and build_sophia anchor the mounting-flange centres.
        # link1 is 138mm along the arm axis; its height changes with the reviewed
        # effective mount rotation and is not the flange's measurement point.
        heights = [float(data.xpos[model.body(f"{side}_arm_base_link").id, 2]) for side in ("left", "right")]
        target = measured[f"shoulder_height_{pose}_m"]
        check(f"{pose}: settled shoulder heights match tape", max(abs(h-target) for h in heights) < .003 and not data.warning.number.any(),
              measured_height_m=target, settled_heights_m=heights)

    overlaps, plate_gaps, plate_support, collision_match = [], [], [], []
    data = reset(model)
    for extension in np.linspace(0, travel, 9):
        data.qpos[qadr(model, "lift_joint")] = extension
        data.qpos[qadr(model, "sophia_middle_slide")] = extension/2
        mujoco.mj_forward(model, data)
        stages = [box_z_bounds(model, data, name) for name in ("fixed_column", "middle_column", "upper_column")]
        overlaps.extend(float(min(a[1], b[1])-max(a[0], b[0])) for a, b in zip(stages, stages[1:]))
        for suffix in ("", "_collision"):
            column = box_world_bounds(model, data, f"upper_column{suffix}")
            plate = box_world_bounds(model, data, f"shoulder_bridge{suffix}")
            plate_gaps.append(float(plate[0, 2]-column[1, 2]))
            plate_support.append(bool(np.all(column[0, :2] >= plate[0, :2]-1e-8)
                                      and np.all(column[1, :2] <= plate[1, :2]+1e-8)))
        collision_match.append(np.allclose(box_world_bounds(model, data, "upper_column"),
                                           box_world_bounds(model, data, "upper_column_collision"), atol=1e-9))
    check("estimated column stages stay overlapped over measured travel", min(overlaps) > .02,
          min_overlap_m=min(overlaps), physical_stage_lengths_still_unmeasured=True)
    check("upper pillar stays attached to shoulder plate through full lift travel",
          max(abs(gap) for gap in plate_gaps) < 1e-7 and all(plate_support) and all(collision_match)
          and model.geom_bodyid[model.geom("upper_column").id] == model.geom_bodyid[model.geom("shoulder_bridge").id],
          max_attachment_gap_m=max(abs(gap) for gap in plate_gaps),
          visual_and_collision_envelopes_match=bool(all(collision_match)), lift_positions_checked=9)

    data = reset(model)
    lift = qadr(model, "lift_joint")
    middle = qadr(model, "sophia_middle_slide")
    act = model.actuator("Lift").id
    check("lift joint travel uses tape endpoints", np.allclose(model.joint("lift_joint").range, [0, travel]),
          modeled_travel_m=float(model.joint("lift_joint").range[1]),
          encoder_reading_m=config["unresolved_readings"]["lift_encoder_travel_m"])
    max_coupling = 0.
    errors = []
    for target in (0., float(model.joint("lift_joint").range[1]), .25):
        start = data.ctrl[act]
        for value in np.linspace(start, target, 600):
            data.ctrl[act] = value
            mujoco.mj_step(model, data)
            max_coupling = max(max_coupling, abs(data.qpos[middle] - .5*data.qpos[lift]))
        step(model, data, 1)
        errors.append(float(abs(data.qpos[lift]-target)))
    check("lift full stroke and two-stage coupling", max(errors) < .005 and max_coupling < .002 and not data.warning.number.any(),
          height_errors_m=errors, max_stage_coupling_error_m=float(max_coupling))

    data = reset(model)
    for target, label in ((measured["gripper_gap_closed_m"]/2, "closed"), (measured["gripper_gap_open_m"]/2, "open")):
        for side in ("left", "right"):
            for f in "ab": data.ctrl[model.actuator(f"{side}_gripper_finger_{f}_pos").id] = target
        step(model, data, 2)
        values = {f"{side}_{f}": float(data.qpos[qadr(model, f"{side}_gripper_finger_{f}")]) for side in ("left", "right") for f in "ab"}
        check(f"jaws {label}", max(abs(v-target) for v in values.values()) < .002, finger_positions_m=values)

    # Insert a small object between each pair of jaws and verify holding under
    # gravity. Loading first in zero gravity prevents it falling before closure.
    for side in ("left", "right"):
        data = reset(model)
        root = ET.parse(HERE / "sophia.mjcf").getroot()
        root.find("compiler").set("meshdir", str(HERE / "assets_nero"))
        root.remove(root.find("keyframe"))
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .05")
        position = data.site_xpos[model.site(f"{side}_gripper_center").id]
        quat = data.xquat[model.body(f"{side}_gripper_base").id]
        probe = ET.SubElement(world, "body", name="grasp_probe", pos=" ".join(map(str, position)), quat=" ".join(map(str, quat)))
        ET.SubElement(probe, "freejoint", name="probe_free")
        ET.SubElement(probe, "geom", name="probe", type="box", size=".008 .015 .012", mass=".025", friction="1 .005 .0001")
        test_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        test_data = mujoco.MjData(test_model)
        copy_state(model, data, test_model, test_data)
        test_model.opt.gravity[:] = 0
        for f in "ab": test_data.ctrl[test_model.actuator(f"{side}_gripper_finger_{f}_pos").id] = 0
        step(test_model, test_data, 1)
        test_model.opt.gravity[:] = [0, 0, -9.81]
        step(test_model, test_data, 2)
        object_pos = test_data.xpos[test_model.body("grasp_probe").id]
        center = test_data.site_xpos[test_model.site(f"{side}_gripper_center").id]
        error = float(np.linalg.norm(object_pos-center))
        probe_geom = test_model.geom("probe").id
        touching = {test_model.body(test_model.geom_bodyid[c.geom2 if c.geom1 == probe_geom else c.geom1]).name
                    for c in test_data.contact if probe_geom in (c.geom1, c.geom2)}
        fingers = {f"{side}_gripper_finger_{f}" for f in "ab"}
        check(f"{side}: hold 25 g object with both fingers", error < .025 and fingers <= touching and not test_data.warning.number.any(),
              object_to_grasp_center_m=error, contacts=sorted(touching))
    return checks


def render(model, output):
    from PIL import Image

    data = reset(model, "reference")
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=1200, width=900) as renderer:
        for name, azimuth in [("front", 180), ("side", 90), ("rear", 0), ("perspective", 140)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, -.20955, .86]
            camera.distance, camera.azimuth, camera.elevation = 2.65, azimuth, -12
            renderer.update_scene(data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"sophia-reference-{name}.png")
        option.geomgroup[2], option.geomgroup[3] = 0, 1
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(output / "sophia-collisions.png")
        option.geomgroup[2], option.geomgroup[3] = 1, 0
        for pose in ("lowered", "raised"):
            renderer.update_scene(reset(model, pose), camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"sophia-{pose}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "output" / "asset-review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
    checks = validate(model)
    if args.render: render(model, args.output)
    report = {"mujoco_version": mujoco.__version__, "checks": checks,
              "configuration": json.loads((HERE / "sophia.json").read_text()),
              "passed": all(c["passed"] for c in checks)}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
