"""Build Sophia from the supplied YOR CAD and the four 2026-10-29 reference photos.

Run with numpy and mujoco installed. Geometry is in metres; no image generation or
mesh scaling is used. See sophia.json for provenance and uncertain dimensions.
The source YOR models remain available for comparison and hardware IK.
"""

import ast
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets_nero"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def arm_position_gains(simulation, side):
    """Use identified side/joint gains when supplied; retain scalar defaults."""
    overrides = simulation.get("arm_position_gains", {}).get(side, {})
    result = {}
    for name in ("kp", "kv"):
        values = np.asarray(overrides.get(name, simulation[f"arm_position_{name}"]), dtype=float)
        if values.ndim == 0:
            values = np.full(7, float(values))
        if values.shape != (7,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{side} arm {name} must contain seven positive finite gains")
        result[name] = values
    return result


def element(parent, tag, **attrs):
    if "name" in attrs:
        # Float-valued dimensions appear in some generated labels. Keep names
        # valid as USD identifiers as well as MuJoCo names.
        attrs["name"] = re.sub(r"[^A-Za-z0-9_]", "_", attrs["name"])
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def write_stl(path, triangles):
    """Binary STL per DAE material; preserve source triangle coordinates exactly."""
    triangles = np.asarray(triangles, dtype="<f4")
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(lengths[:, None], 1e-30)
    records = np.zeros(len(triangles), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    records["n"], records["v"] = normal, triangles
    with path.open("wb") as f:
        f.write(b"Sophia: original DAE material subset".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        f.write(records.tobytes())


def arm_materials(asset):
    """Read all COLLADA geometries, including link5's separate secondary mesh."""
    out = ASSETS / "sophia"
    out.mkdir(exist_ok=True)
    parts = {}
    for path in sorted((ASSETS / "dae").glob("*.dae")):
        root = ET.parse(path)
        effects = {
            e.get("id"): e.find(".//c:diffuse/c:color", NS).text
            for e in root.findall(".//c:effect", NS)
        }
        colors = {
            e.get("id"): effects[e.find("c:instance_effect", NS).get("url")[1:]]
            for e in root.findall(".//c:material", NS)
        }
        parts[path.stem] = []
        for node in root.findall(".//c:visual_scene//c:node", NS):
            matrix = node.find("c:matrix", NS)
            transform = np.eye(4) if matrix is None else np.fromstring(matrix.text, sep=" ").reshape(4, 4)
            for instance in node.findall("c:instance_geometry", NS):
                mesh = root.find(f'.//c:geometry[@id="{instance.get("url")[1:]}"]/c:mesh', NS)
                sources = {}
                for source in mesh.findall("c:source", NS):
                    stride = int(source.find("c:technique_common/c:accessor", NS).get("stride"))
                    sources[source.get("id")] = np.fromstring(source.find("c:float_array", NS).text, sep=" ").reshape(-1, stride)
                vertices = {v.get("id"): v.find('c:input[@semantic="POSITION"]', NS).get("source")[1:]
                            for v in mesh.findall("c:vertices", NS)}
                bindings = {b.get("symbol"): b.get("target")[1:]
                            for b in instance.findall(".//c:instance_material", NS)}
                for triangles in mesh.findall("c:triangles", NS):
                    inputs = triangles.findall("c:input", NS)
                    stride = max(int(i.get("offset")) for i in inputs) + 1
                    vi = next(i for i in inputs if i.get("semantic") == "VERTEX")
                    positions = sources[vertices[vi.get("source")[1:]]][:, :3]
                    indices = np.fromstring(triangles.find("c:p", NS).text, sep=" ", dtype=int).reshape(-1, stride)[:, int(vi.get("offset"))]
                    xyz = positions[indices].reshape(-1, 3, 3) @ transform[:3, :3].T + transform[:3, 3]
                    name = f"sophia_{path.stem}_{len(parts[path.stem])}"
                    write_stl(out / f"{name}.stl", xyz)
                    # Blender's COLLADA diffuse values are linear; MuJoCo's
                    # displayed RGB colors need the corresponding sRGB values.
                    rgba = np.fromstring(colors[bindings[triangles.get("material")]], sep=" ")
                    rgba[:3] = np.where(rgba[:3] <= .0031308, 12.92*rgba[:3], 1.055*rgba[:3]**(1/2.4)-.055)
                    element(asset, "material", name=name, rgba=numbers(rgba), specular="0.25", shininess="0.3")
                    # Logos/panels are open surfaces; their volume is not an inertia source.
                    element(asset, "mesh", name=name, file=f"sophia/{name}.stl", inertia="shell")
                    parts[path.stem].append(name)
    return parts


def visual(parent, name, shape, material, **attrs):
    return element(parent, "geom", name=name, type=shape, material=material,
                   group="2", contype="0", conaffinity="0", density="0", **attrs)


def solid(parent, name, shape, material, **attrs):
    visual(parent, name, shape, material, **attrs)
    return collision(parent, name, shape, **attrs)


def collision(parent, name, shape, **attrs):
    return element(parent, "geom", name=f"{name}_collision", type=shape,
                   group="3", contype="2", conaffinity="3", density="0",
                   friction="0.8 0.005 0.0001", **attrs)


def box(parent, name, center, half, material, collision=True):
    return (solid if collision else visual)(parent, name, "box", material,
                                           pos=numbers(center), size=numbers(half))


def cable(parent, name, points, radius=0.003, material="sophia_black"):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        visual(parent, f"{name}_{i}", "capsule", material,
               fromto=numbers([*a, *b]), size=str(radius))


def remove_geoms(body):
    for g in body.findall("geom"):
        body.remove(g)


def grounded_base_height(model):
    """Place the original tyre meshes on z=0 with an exactly upright chassis."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[:7] = [0, 0, 0, np.sqrt(.5), 0, np.sqrt(.5), 0]
    mujoco.mj_forward(model, data)
    bottoms = []
    for suffix in ("", "_2", "_3", "_4"):
        g = model.geom(f"MK4_Colson_Hub_4_X_1_5{suffix}_collision").id
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        bottoms.append(vertices[:, 2].min())
    return float(-min(bottoms))


def apply_mount_dimensions(root, measured, base_height, travel):
    """Shared rigid transforms for the physics model and its fixed-base IK model."""
    base = root.find('.//body[@name="base"]')
    base.set("pos", numbers([0, 0, base_height]))
    base.set("euler", numbers([0, np.pi/2, 0]))
    lift = root.find('.//body[@name="Lift_Middle"]')
    pos = np.fromstring(lift.get("pos"), sep=" ")
    # The chassis frame has -X up; keep the original fore/aft mount offset.
    shoulder_local_z = float(root.find('.//body[@name="left_arm_base_link"]').get("pos").split()[2])
    pos[0] = -(measured["shoulder_height_lowered_m"] - base_height - shoulder_local_z)
    lift.set("pos", numbers(pos))
    lift.find("joint").set("range", numbers([0, travel]))
    for side, sign in (("left", -1), ("right", 1)):
        arm = root.find(f'.//body[@name="{side}_arm_base_link"]')
        pos = np.fromstring(arm.get("pos"), sep=" ")
        pos[1] = sign * measured["arm_flange_spacing_m"] / 2
        arm.set("pos", numbers(pos))


def build(configuration=None, output_dir=None):
    from sophia_collision_geometry import apply_gripper_housing_collision
    from sophia_sources import (
        apply_confirmed_head_mount,
        apply_dataset_head_alignment,
        apply_head_calibration,
        apply_vendor_inertials,
        apply_wrist_cameras,
        spinal_neck_geometry,
        neck_reference,
        apply_spinal_assembly,
        apply_arm_reference,
        arm_reference_profile,
        apply_base_alignment,
    )

    cfg = json.loads((HERE / "sophia.json").read_text()) if configuration is None else configuration
    destination = HERE if output_dir is None else Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cad, photo, measured = cfg["cad"], cfg["photo_estimates"], cfg["measured"]
    travel = measured["shoulder_height_raised_m"] - measured["shoulder_height_lowered_m"]
    if travel <= 0 or not np.isfinite(list(measured[k] for k in measured if k.endswith("_m"))).all():
        raise ValueError("Measured dimensions must be finite and raised height must exceed lowered height")
    if not 0 <= photo["reference_lift_extension_m"] <= travel:
        raise ValueError("reference_lift_extension_m must lie within the lift travel")
    if not 0 <= cfg["simulation"]["home_lift_extension_m"] <= travel:
        raise ValueError("home_lift_extension_m must lie within the lift travel")
    dimensions = [measured[k] for k in (
        "base_frame_width_m", "base_frame_depth_m", "base_frame_height_m",
        "base_frame_floor_clearance_m", "pillar_lower_width_m", "pillar_lower_depth_m",
        "pillar_upper_width_m", "pillar_upper_depth_m")]
    if min(dimensions) <= 0:
        raise ValueError("Measured base and pillar dimensions must be positive")
    root = ET.parse(HERE / "yor-full-agx.mjcf").getroot()
    old = mujoco.MjModel.from_xml_path(str(HERE / "yor-full-agx.mjcf"))
    base_height = grounded_base_height(old)
    apply_mount_dimensions(root, measured, base_height, travel)
    apply_arm_reference(root, cfg)
    apply_vendor_inertials(root)
    apply_head_calibration(root)
    root.set("model", "sophia")
    root.insert(0, ET.Comment(" GENERATED by build_sophia.py. Edit sophia.json / generator. Photo estimates are not calibrated hardware dimensions. "))
    # Read hardware configuration without importing drivers or opening devices.
    gripper_source = ast.parse((HERE.parent / "arm" / "can_gripper.py").read_text())
    protocol_stroke = next(ast.literal_eval(node.value) for node in gripper_source.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GRIPPER_STROKE_M" for t in node.targets))
    jaw_closed, jaw_open = measured["gripper_gap_closed_m"], measured["gripper_gap_open_m"]
    if not 0 <= jaw_closed < jaw_open:
        raise ValueError("Measured jaw gaps must be nonnegative, with open > closed")
    root.find('.//default[@class="agx_gripper"]/joint').set("range", numbers([jaw_closed/2, jaw_open/2]))
    neck_calib = json.loads((HERE.parent / "neck" / "neck_calibration.json").read_text())
    resolution = 4096 if neck_calib["model"].startswith(("sts", "sm")) else 1024
    neck_home, neck_zero = neck_reference(cfg, neck_calib)
    # Hardware angles are centered on calibrated travel, not on raw tick 2048.
    # Geometric neutral is independently configurable; Home remains hardware data.
    custom = element(root, "custom")
    element(custom, "text", name="ik_model", data="sophia-ik.mjcf")
    element(custom, "text", name="head_camera_profile", data="zed.rectified")
    element(custom, "numeric", name="gripper_protocol_stroke_m", data=numbers([protocol_stroke]))
    element(custom, "numeric", name="gripper_physical_gaps_m", data=numbers([jaw_closed, jaw_open]))
    element(custom, "numeric", name="neck_zero_deg", data=numbers(neck_zero))
    element(custom, "numeric", name="neck_home_deg", data=numbers(neck_home))
    neck_rate = np.asarray(cfg["simulation"].get("neck_rate_limit_rad_s", [1.0, 1.0]))
    if neck_rate.shape != (2,) or not np.isfinite(neck_rate).all() or np.any(neck_rate <= 0):
        raise ValueError("neck_rate_limit_rad_s must contain two positive finite rates")
    element(custom, "numeric", name="neck_rate_limit_rad_s", data=numbers(neck_rate))
    for name in ("lift_speed_mps", "gripper_speed_strokes_s", "gripper_opening_speed_strokes_s"):
        rates = np.asarray(cfg["simulation"][name], dtype=float)
        if rates.shape != (2,) or not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError(f"{name} must contain two positive finite rates")
        element(custom, "numeric", name=name, data=numbers(rates))
    for side in ("left", "right"):
        gains = arm_position_gains(cfg["simulation"], side)
        for i in range(1, 8):
            actuator = root.find(f'.//position[@name="{side}_arm_joint{i}_pos"]')
            actuator.set("kp", str(gains["kp"][i-1]))
            actuator.set("kv", str(gains["kv"][i-1]))
        arm_root = root.find(f'.//body[@name="{side}_arm_base_link"]')
        for part in arm_root.iter("body"):
            part.set("gravcomp", str(cfg["simulation"]["arm_gravity_compensation"]))
    element(custom, "numeric", name="self_collision_margin_m", data=numbers([cfg["simulation"]["self_collision_margin_m"]]))
    element(custom, "numeric", name="reference_clearance_lift_m", data=numbers([cfg["simulation"]["reference_clearance_lift_m"]]))
    for name, zero, sign in zip(("pan", "tilt"), neck_zero, (-1, 1)):
        motor = neck_calib["motors"][name]
        half = (motor["range_max"] - motor["range_min"]) / 2 * 360 / (resolution - 1)
        limits = sorted(np.deg2rad(sign * (np.array([-half, half]) - zero)))
        root.find(f'.//joint[@name="head_{name}_joint"]').set("range", numbers(limits))
    asset = root.find("asset")
    materials = {
        "black": ("0.075 0.08 0.085 1", "0.15"),
        "rubber": ("0.018 0.02 0.023 1", "0.05"),
        "silver": ("0.65 0.67 0.69 1", "0.45"),
        "yellow": ("0.95 0.68 0.015 1", "0.25"),
        "blue": ("0.015 0.45 0.7 1", "0.2"),
        "purple": ("0.27 0.10 0.44 1", "0.08"),
        "red": ("0.7 0.012 0.02 1", "0.3"),
        "board": ("0.02 0.16 0.11 1", "0.1"),
        "lens": ("0.055 0.065 0.08 1", "0.7"),
        "white": ("0.82 0.82 0.8 1", "0.1"),
    }
    for name, (rgba, specular) in materials.items():
        element(asset, "material", name=f"sophia_{name}", rgba=rgba, specular=specular, shininess="0.3")
    parts = arm_materials(asset)
    # Rendering meshes carry zero mass. Explicit source CAD inertials govern dynamics.
    for geom in root.findall(".//worldbody//geom"):
        if geom.get("class") == "visual" and geom.get("mesh") in parts:
            geom.set("material", "sophia_black")
        if geom.get("class") in ("collision", "head_collision", "agx_gripper_collision"):
            geom.set("contype", "2")
            geom.set("conaffinity", "3")
            if geom.get("mesh"):
                geom.set("type", "mesh")
        if geom.get("class") == "wheel":
            geom.set("contype", "2")
            geom.set("conaffinity", "3")
    for body in root.findall(".//worldbody//body"):
        for geom in list(body.findall("geom")):
            mesh = geom.get("mesh")
            if geom.get("class") == "visual" and mesh in parts:
                body.remove(geom)
                for name in parts[mesh]:
                    visual(body, f'{body.get("name")}_{name}', "mesh", name, mesh=name)

    body = lambda name: root.find(f'.//body[@name="{name}"]')
    # CAD uses an x-down chassis frame. This child provides x-forward, y-left, z-up.
    base = body("base_profile_short")
    remove_geoms(base)
    chassis = element(base, "body", name="sophia_chassis", pos=f'0 {cad["base_center_y_m"]} 0',
                      quat="0.707106781 0 -0.707106781 0")
    z0 = base_height
    x, y = measured["base_frame_depth_m"] / 2, measured["base_frame_width_m"] / 2
    # Tape dimensions are the frame's outside surfaces, excluding wheel/module
    # overhangs. The rail cross-section is still an explicit photo estimate.
    rail_half = photo["base_rail_cross_section_m"] / 2
    bottom = measured["base_frame_floor_clearance_m"] + rail_half - z0
    top = measured["base_frame_floor_clearance_m"] + measured["base_frame_height_m"] - rail_half - z0
    for h, label, material in [(bottom, "lower", "silver"), (top, "upper", "yellow")]:
        for sign in (-1, 1):
            box(chassis, f"{label}_rail_x_{sign}", [0, sign * (y - rail_half), h], [x, rail_half, rail_half], f"sophia_{material}")
            box(chassis, f"{label}_rail_y_{sign}", [sign * (x - rail_half), 0, h], [rail_half, y - 2*rail_half, rail_half], f"sophia_{material}")
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(chassis, f"corner_post_{sx}_{sy}", [sx * (x - .02), sy * (y - .02), (top + bottom) / 2],
                [.01, .01, (top - bottom) / 2], "sophia_silver")
            box(chassis, f"motor_plate_{sx}_{sy}", [sx * .105, sy * .1556, bottom - .012], [.055, .056, .005], "sophia_black")
            solid(chassis, f"drive_motor_{sx}_{sy}", "cylinder", "sophia_black",
                  pos=numbers([sx * .105, sy * .1556, (bottom + top) / 2]), size="0.023 0.06")
            for ox in (-.035, .035):
                visual(chassis, f"plate_bolt_{sx}_{sy}_{ox}", "cylinder", "sophia_silver",
                       pos=numbers([sx * .105 + ox, sy * .1556 + .043, bottom - .006]), size="0.003 0.002")
    # Open electronics bay, blue e-stop plate and visible controllers from the photos.
    box(chassis, "battery", [-.055, 0, bottom + .065], [.055, .074, .055], "sophia_black")
    box(chassis, "battery_top", [-.055, 0, bottom + .122], [.052, .071, .003], "sophia_red", False)
    box(chassis, "estop_plate", [.04, .135, top + .012], [.09, .075, .003], "sophia_blue")
    solid(chassis, "estop", "cylinder", "sophia_red", pos=numbers([.04, .135, top + .027]), size=".014 .011")
    for i in range(4):
        cy = -.135 + i * .055
        box(chassis, f"controller_{i}", [.105, cy, bottom + .055], [.032, .022, .01], "sophia_board", False)
        box(chassis, f"heatsink_{i}", [.105, cy, bottom + .07], [.02, .015, .006], "sophia_silver", False)
        cable(chassis, f"power_wire_{i}", [[-.055, .03, bottom+.13], [0, cy, bottom+.15], [.10, cy, bottom+.08]], .0025, "sophia_red")
    for i in range(5):
        phase = i * .7
        points = [[.06 + .04*np.cos(t), -.04 + .10*np.sin(t), top-.025+.006*np.sin(2*t)]
                  for t in np.linspace(phase, phase + 5.5, 18)]
        cable(chassis, f"bay_cable_{i}", points, .002)
    # Exterior stage geometry follows the photographed three-section column.
    stage_length = photo["column_stage_length_m"]
    fixed_top = .57
    fixed_bottom = .105
    box(chassis, "column_crossmember", [0, 0, fixed_bottom-z0], [x-.02, .018, .012], "sophia_silver")
    lower_half = np.array([measured["pillar_lower_depth_m"], measured["pillar_lower_width_m"]]) / 2
    upper_half = np.array([measured["pillar_upper_depth_m"], measured["pillar_upper_width_m"]]) / 2
    # The middle section has not been measured. Interpolate its exterior width
    # rather than presenting the previous square CAD proxy as measured geometry.
    middle_half = (lower_half + upper_half) / 2
    box(chassis, "fixed_column", [0, 0, (fixed_bottom + fixed_top)/2-z0],
        [*lower_half, (fixed_top-fixed_bottom)/2], "sophia_black")
    visual(chassis, "antenna_body", "cylinder", "sophia_white",
           pos=numbers([.083, 0, .39-z0]), size=".014 .075")
    cable(chassis, "antenna_bar", [[.083, -.14, .48-z0], [.083, .14, .48-z0]], .0035, "sophia_white")
    box(chassis, "antenna_strap", [.084, 0, .41-z0], [.018, .019, .012], "sophia_purple", False)
    box(chassis, "antenna_mount_support", [.045, 0, .41-z0],
        [.025, .015, .006], "sophia_purple", False)
    lift = body("Lift_Middle")
    lift_inertial = lift.find("inertial")
    lift.remove(lift_inertial)
    remove_geoms(lift)
    lift.find("joint").set("range", numbers([0, travel]))
    middle = element(base, "body", name="sophia_middle_stage", pos=lift.get("pos"), quat=lift.get("quat"))
    middle.append(lift_inertial)
    element(middle, "joint", name="sophia_middle_slide", type="slide", axis="0 0 1",
            range=numbers([0, travel/2]), damping="5", armature="0.01")
    box(middle, "middle_column", [0, 0, -.10-stage_length/2], [*middle_half, stage_length/2], "sophia_black")
    torso = body("Lift_Top")
    remove_geoms(torso)
    # A bent aluminium shoulder bracket replaces the unrelated triangular CAD truss.
    flange_half = measured["arm_flange_spacing_m"] / 2
    plate_front = measured["arm_plate_front_edge_forward_m"]
    box(torso, "shoulder_bridge", [0, 0, .105],
        [plate_front, flange_half + .02232, .004], "sophia_silver")
    # The real pillar is bolted directly to the underside of the shoulder plate.
    # Anchor its top to that face, retaining the lower endpoint so full-stroke
    # telescoping overlap is unchanged. The visual and collider use one envelope.
    bridge = torso.find('geom[@name="shoulder_bridge"]')
    upper_top = float(bridge.get("pos").split()[2]) - float(bridge.get("size").split()[2])
    upper_bottom = -stage_length
    box(torso, "upper_column", [0, 0, (upper_bottom + upper_top)/2],
        [*upper_half, (upper_top-upper_bottom)/2], "sophia_black")
    for s in (-1, 1):
        box(torso, f"shoulder_flange_{s}", [0, s*(flange_half-.002), .058], [.055, .003, .047], "sophia_silver")
    # The metal plate front edge is confirmed. Bound the unmeasured black mount
    # to that envelope; rear symmetry, black plate width/thickness stay estimates.
    box(torso, "neck_mount_plate", [0, 0, .114],
        [plate_front, .061, .005], "sophia_black")
    box(torso, "computer_mount", [.078, 0, -.075], [.013, .074, .09], "sophia_purple")
    box(torso, "computer_mount_support", [.045, 0, -.075],
        [.025, .05, .006], "sophia_purple", False)
    box(torso, "computer", [.109, 0, -.075], [.025, .063, .075], "sophia_black")
    for i in range(12):
        box(torso, f"computer_fin_{i}", [.136, -.055+i*.01, -.075], [.004, .002, .064], "sophia_silver", False)
    for s in (-1, 1):
        cable(torso, f"shoulder_cable_{s}", [[.075, s*.04, -.05], [.085, s*.07, .13], [.02, s*.16, .15], [0, s*.21, .06]], .004)
    # The cable carrier is visual only; each visible section follows its own stage.
    for parent, start, end in [(chassis, .02, .44), (middle, -.48, -.12), (torso, -.32, .02)]:
        for i, z in enumerate(np.arange(start, end, .015)):
            box(parent, f'{parent.get("name")}_chain_{i}', [.084, 0, z], [.009, .018, .005], "sophia_black", False)
    # Existing head CAD contains the pan/tilt brackets. Replace its solid lower
    # neck with the open lattice tower visible in all four photos.
    neck = body("head_neck")
    neck.set("pos", "0 0 0.12")
    for g in list(neck.findall("geom")):
        if g.get("mesh") == "head_topbase1_cut":
            neck.remove(g)
    spinal = spinal_neck_geometry(cfg)
    tower = spinal["neck_extents_m"][2] if spinal else photo["neck_tower_height_m"]
    half_xy = np.array(spinal["neck_extents_m"][:2])/2 if spinal else [.043, .042]
    cap_positions = (.004, tower-.004) if spinal else (0, tower)
    for h in cap_positions:
        box(neck, f"tower_plate_{h}", [0, 0, h], [*half_xy, .004], "sophia_black")
    for xx in (-.03, .03):
        for yy in (-.03, .03):
            box(neck, f"tower_post_{xx}_{yy}", [xx, yy, tower/2], [.004, .004, tower/2], "sophia_black")
    for i in range(4):
        low, high = i*tower/4, (i+1)*tower/4
        for s in (-1, 1):
            for axis in (0, 1):
                a, b = [-.03, s*.03, low], [.03, s*.03, high]
                if i % 2: a[0], b[0] = b[0], a[0]
                if axis: a[0], a[1], b[0], b[1] = a[1], a[0], b[1], b[0]
                cable(neck, f"tower_brace_{i}_{s}_{axis}", [a, b], .0035)
    if spinal:
        # STL has disconnected surface patches, so it is a visual asset only.
        # Keep the post/plate collision proxies; do not convex-hull all holes shut.
        for g in list(neck.findall("geom")):
            if g.get("name", "").startswith("tower_") and g.get("group") != "3":
                neck.remove(g)
        element(root.find("asset"), "mesh", name="spinal_neck_lattice",
                file=spinal["asset"], inertia="shell")
        visual(neck, "spinal_neck_lattice", "mesh", "sophia_black", mesh="spinal_neck_lattice")
    # Recenter the complete source upper assembly, not just its pan pivot.
    # topbase2 is expressed in the old neck frame and must follow that shift.
    source_pan = np.fromstring(body("head_pan").get("pos"), sep=" ")
    shift = np.array([0., 0., tower]) - source_pan
    for g in neck.findall("geom"):
        if g.get("mesh") == "head_topbase2":
            g.set("pos", numbers(np.fromstring(g.get("pos", "0 0 0"), sep=" ") + shift))
            if g.get("class") == "head_collision":
                g.set("name", "neck_fixed_bracket_collision")
    # The retained neck mass includes the old lower tower too; its inertia is
    # still a source-model approximation, not a measured replacement assembly.
    body("head_pan").set("pos", f"0 0 {tower}")
    box(neck, "pan_servo_housing", [0, 0, tower+.015], [.026, .024, .015], "sophia_black")
    pan = body("head_pan")
    box(pan, "pan_bracket_base", [0, 0, .03], [.029, .025, .007], "sophia_black")
    box(pan, "pan_bracket_spine", [.027, 0, .06], [.005, .025, .03], "sophia_black")
    for g in root.findall('.//geom[@class="head_visual"]'):
        g.set("rgba", "0.075 0.08 0.085 1")
    tilt = body("head_tilt")
    tilt.find('geom[@name="head_zed2"]').set("size", "0.01595 0.07255 0.0149")
    # Match the rounded visual housing with a box and two cylinders. The old
    # single box filled the rounded corners; preserve the full camera envelope.
    for g in list(tilt.findall('geom[@class="head_collision"]')):
        if g.get("type") == "box":
            tilt.remove(g)
    collision(tilt, "zed_housing", "box", pos="-0.0276949 -0.00258203 0.0272061",
              size="0.01595 0.07255 0.0149")
    for sy in (-.07513203, .06996797):
        visual(tilt, f"zed_endcap_{sy}", "cylinder", "sophia_black",
               pos=f"-0.0276949 {sy} 0.0272061", size="0.0149 0.01595", quat="0.70710678 0 0.70710678 0")
        collision(tilt, f"zed_endcap_{sy}", "cylinder",
                  pos=f"-0.0276949 {sy} 0.0272061", size="0.0149 0.01595", quat="0.70710678 0 0.70710678 0")
    for sy in (-.062582, .057418):
        visual(tilt, f"zed_lens_{sy}", "cylinder", "sophia_lens",
               pos=f"-0.0439 {sy} 0.0272061", size="0.011 0.001", quat="0.70710678 0 0.70710678 0")
    for side in ("left", "right"):
        shoulder = body(f"{side}_arm_link1")
        visual(shoulder, f"{side}_shoulder_cap", "cylinder", "sophia_silver", pos="0 0 .0305", size=".033 .001")
        for g in body(f"{side}_gripper_base").findall("geom"):
            if g.get("class") == "agx_gripper_visual": g.set("material", "sophia_silver")
        wrist = body(f"{side}_arm_link5")
        cable(wrist, f"{side}_wrist_cable", [[.036, -.015, -.15], [.047, -.015, -.09], [.04, -.015, -.02]], .002)
        for letter in "ab":
            finger = body(f"{side}_gripper_finger_{letter}")
            for g in list(finger.findall("geom")):
                if g.get("class") == "agx_gripper_collision": finger.remove(g)
                else: g.set("material", "sophia_black")
            element(finger, "geom", name=f"{side}_finger_{letter}_collision", type="mesh",
                    mesh=f"agx_finger_{letter}", group="3", contype="2", conaffinity="3",
                    density="0", friction="2 0.005 0.0001", condim="4")
    for g in root.findall('.//geom[@class="visual"]'):
        if "MK4_Colson" in g.get("mesh", ""): g.set("material", "sophia_rubber")
        if g.get("mesh", "").startswith("Part_2"): g.set("material", "sophia_black")
    apply_wrist_cameras(root, cfg)
    apply_dataset_head_alignment(root, cfg)
    apply_spinal_assembly(root, cfg)
    apply_confirmed_head_mount(root, cfg)
    apply_base_alignment(root, cfg, base_height)
    if measured.get("neck_mount_coaxial_with_pillar", False):
        centre = np.fromstring(torso.find('geom[@name="upper_column"]').get("pos"), sep=" ")[:2]
        for mounted in (body("head_neck"), torso.find('geom[@name="neck_mount_plate"]')):
            if not np.allclose(np.fromstring(mounted.get("pos"), sep=" ")[:2], centre, atol=1e-9):
                raise ValueError("Neck mount and pillar must share the user-confirmed centreline")
    apply_gripper_housing_collision(root, ASSETS)
    equality = element(root, "equality")
    element(equality, "joint", name="lift_stage_coupling", joint1="sophia_middle_slide",
            joint2="lift_joint", polycoef="0 0.5 0 0 0", solref="0.01 1")
    for side in ("left", "right"):
        element(equality, "joint", name=f"{side}_jaw_coupling", joint1=f"{side}_gripper_finger_a",
                joint2=f"{side}_gripper_finger_b", polycoef="0 1 0 0 0")
    contact = root.find("contact")
    # Plate proxies fill the real tyre cutouts. Exclude only the wheel's own
    # chassis attachment; arms and external objects can still hit the wheels.
    for suffix in ("", "_2", "_3", "_4"):
        for chassis_body in ("base_profile_short", "sophia_chassis"):
            element(contact, "exclude", body1=chassis_body, body2=f"MK4_Colson_Hub_4_X_1_5{suffix}")
    # The forked wrist meshes enclose the next two joints. A convex hull fills
    # that empty space; these are connected wrist parts, not external obstacles.
    for side in ("left", "right"):
        for part in ("link7", "end_effector"):
            element(contact, "exclude", body1=f"{side}_arm_link5", body2=f"{side}_arm_{part}")
    for a, b in [("base_profile_short", "Lift_Top"), ("base_profile_short", "Lift_Middle"),
                 ("base_profile_short", "sophia_middle_stage"), ("sophia_chassis", "sophia_middle_stage"),
                 ("sophia_chassis", "Lift_Top"), ("sophia_middle_stage", "Lift_Top"),
                 ("sophia_middle_stage", "Lift_Middle")]:
        element(contact, "exclude", body1=a, body2=b)
    # Name-based keyframes avoid positional errors when adding the middle-stage DOF.
    root.remove(root.find("keyframe"))
    root.find("compiler").set("meshdir", str(ASSETS))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    keys = element(root, "keyframe")
    for pose in ("home", "reference", "lowered", "raised"):
        q = model.qpos0.copy()
        ctrl = old.key("home").ctrl.copy()
        for j in range(old.njnt):
            target = model.joint(old.joint(j).name).id
            n = 7 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            q[model.jnt_qposadr[target]:model.jnt_qposadr[target]+n] = old.key("home").qpos[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
        q[:7] = [0, 0, base_height, np.sqrt(.5), 0, np.sqrt(.5), 0]
        extension = {"home": cfg["simulation"]["home_lift_extension_m"],
                     "reference": photo["reference_lift_extension_m"],
                     "lowered": 0., "raised": travel}[pose]
        q[model.jnt_qposadr[model.joint("lift_joint").id]] = extension
        q[model.jnt_qposadr[model.joint("sophia_middle_slide").id]] = extension/2
        ctrl[model.actuator("Lift").id] = extension
        for name, home, zero, sign in zip(("pan", "tilt"), neck_home, neck_zero, (-1, 1)):
            angle = np.deg2rad(sign * (home - zero))
            q[model.jnt_qposadr[model.joint(f"head_{name}_joint").id]] = angle
            ctrl[model.actuator(f"head_{name}_pos").id] = angle
        for side, sign in [("left", 1), ("right", -1)]:
            if pose == "reference":
                angles = [sign*np.pi/2, np.pi/2, -sign*np.pi/2, 0, 0, 0, 0]
                reference_profile = arm_reference_profile(cfg)
                if side == "right" and reference_profile is not None:
                    angles = reference_profile["right_reference_q"]
                for i, angle in enumerate(angles, 1):
                    q[model.jnt_qposadr[model.joint(f"{side}_arm_joint{i}").id]] = angle
                    ctrl[model.actuator(f"{side}_arm_joint{i}_pos").id] = angle
            for letter in "ab":
                name = f"{side}_gripper_finger_{letter}"
                q[model.jnt_qposadr[model.joint(name).id]] = .025
                ctrl[model.actuator(f"{name}_pos").id] = .025
        element(keys, "key", name=pose, qpos=numbers(q), ctrl=numbers(ctrl))
    root.find("compiler").set("meshdir", "assets_nero" if destination == HERE else str(ASSETS))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination / "sophia.mjcf", encoding="unicode")
    ik = ET.parse(HERE / "nero-welded-base-and-lift.mjcf").getroot()
    ik.set("model", "sophia-ik")
    ik.insert(0, ET.Comment(" GENERATED by build_sophia.py. Same measured mounts as Sophia physics, with lift pinned to zero by SingleArmIK. "))
    apply_mount_dimensions(ik, measured, base_height, travel)
    apply_arm_reference(ik, cfg)
    apply_vendor_inertials(ik)
    ET.indent(ik, space="  ")
    if destination != HERE:
        ik.find("compiler").set("meshdir", str(ASSETS))
    ET.ElementTree(ik).write(destination / "sophia-ik.mjcf", encoding="unicode")
    scene = ET.Element("mujoco", model="scene-sophia")
    element(scene, "include", file="sophia.mjcf")
    element(scene, "statistic", center="0 -0.21 0.85", extent="1.85", meansize="0.04")
    vis = element(scene, "visual")
    element(vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0.15 0.15 0.15")
    element(vis, "global", azimuth="135", elevation="-15", offwidth="1600", offheight="1600")
    assets = element(scene, "asset")
    element(assets, "texture", type="skybox", builtin="gradient", rgb1="0.78 0.81 0.85",
            rgb2="0.92 0.93 0.94", width="512", height="3072")
    element(assets, "material", name="sophia_floor", rgba="0.55 0.56 0.57 1", reflectance="0")
    world = element(scene, "worldbody")
    element(world, "light", pos="1 0 3", dir="-0.3 0 -1", directional="true", diffuse="0.6 0.6 0.6")
    element(world, "geom", name="floor", type="plane", size="0 0 .05", material="sophia_floor", condim="4")
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(destination / "scene-sophia.mjcf", encoding="unicode")
    print("Wrote sophia.mjcf, scene-sophia.mjcf, sophia-ik.mjcf and DAE material meshes in assets_nero/sophia/")


if __name__ == "__main__":
    build()
