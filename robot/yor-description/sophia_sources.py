"""Pinned, validated calibration and vendor data used by the Sophia generator."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/agx_arm_urdf"
CAMERA_SOURCE = HERE / "calibration/camera_intrinsics_2026-10-30.json"
MOTION_SOURCE = HERE / "calibration/motion_summary_2026-10-30.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def neck_reference(configuration, calibration):
    """Keep the measured hardware parking pose separate from geometric neutral."""
    home = np.array([calibration["home_deg"][k] for k in ("pan", "tilt")], dtype=float)
    profile = configuration.get("neck_reference", {})
    zero = np.array(profile.get("geometric_zero_hardware_deg", home), dtype=float)
    if zero.shape != (2,) or not np.isfinite(zero).all():
        raise ValueError("neck_reference requires two finite hardware angles")
    resolution = 4096 if calibration["model"].startswith(("sts", "sm")) else 1024
    for i, name in enumerate(("pan", "tilt")):
        motor = calibration["motors"][name]
        half = (motor["range_max"] - motor["range_min"]) * 180 / (resolution - 1)
        if abs(zero[i]) > half or abs(home[i]) > half:
            raise ValueError("Neck Home and geometric zero must lie within hardware travel")
    return home, zero


def arm_reference_profile(configuration):
    profile = configuration.get("arm_reference", {})
    if not profile.get("enabled", False):
        return None
    path = HERE / profile["source"]
    if sha256(path) != profile["sha256"]:
        raise ValueError("Arm-reference calibration changed")
    data = json.loads(path.read_text())
    if (not configuration.get("spinal_assembly", {}).get("enabled", False)
            or configuration["spinal_assembly"]["sha256"] != data["requires_spinal_assembly_sha256"]):
        raise ValueError("Arm reference requires the reviewed full spinal CAD geometry")
    if not np.allclose(configuration["neck_reference"]["geometric_zero_hardware_deg"],
                       data["geometric_zero_hardware_deg"], atol=1e-9):
        raise ValueError("Arm and neck frame calibrations must be applied together")
    return data


def apply_arm_reference(root, configuration):
    """Effective reference rotation; preserves flange centres and servo angles."""
    data = arm_reference_profile(configuration)
    if data is None:
        return
    import mujoco

    v = np.deg2rad(data["right_mount_rotvec_parent_deg"])
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) > np.deg2rad(15):
        raise ValueError("Invalid effective arm reference rotation")
    angle = np.linalg.norm(v)
    delta = np.r_[np.cos(angle/2), v/max(angle, 1e-30)*np.sin(angle/2)]
    body = root.find(".//body[@name='right_arm_base_link']")
    old = np.empty(4)
    if body.get("euler") is not None:
        mujoco.mju_euler2Quat(old, np.fromstring(body.get("euler"), sep=" "), "xyz")
    else:
        old[:] = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
    result = np.empty(4)
    mujoco.mju_mulQuat(result, delta, old)
    body.set("quat", " ".join(map(str, result)))
    body.attrib.pop("euler", None)


def apply_spinal_assembly(root, configuration):
    """Replace the legacy upper neck with the supplied articulated CAD parts."""
    profile = configuration.get("spinal_assembly", {})
    if not profile.get("enabled", False):
        return
    path = HERE / profile["source"]
    if sha256(path) != profile["sha256"]:
        raise ValueError("Spinal assembly provenance changed")
    if configuration.get("head_alignment", {}).get("enabled", False):
        raise ValueError("Legacy head alignment cannot be combined with the spinal CAD assembly")
    import mujoco

    source = json.loads(path.read_text())
    bodies = {name: root.find(f".//body[@name='{name}']")
              for name in ("head_neck", "head_pan", "head_tilt")}
    pan, tilt = bodies["head_pan"], bodies["head_tilt"]
    original_pan = np.fromstring(pan.get("pos"), sep=" ")
    old_cameras = [tilt.find(f"camera[@name='head_zed_{eye}']") for eye in ("left", "right")]
    old_midpoint = np.mean([np.fromstring(c.get("pos"), sep=" ") for c in old_cameras], axis=0)
    pan.set("pos", " ".join(map(str, source["pan_position_neck_m"])))
    tilt.set("pos", " ".join(map(str, source["tilt_position_pan_m"])))
    tilt.find("joint").set("axis", " ".join(map(str, source["tilt_axis_direction_local"])))
    # Keep source mass assumptions, with COMs in their new body frames.
    inertial = pan.find("inertial")
    com = np.fromstring(inertial.get("pos"), sep=" ") + original_pan - source["pan_position_neck_m"]
    inertial.set("pos", " ".join(map(str, com)))
    inertial = tilt.find("inertial")
    com = np.fromstring(inertial.get("pos"), sep=" ") + source["camera_midpoint_tilt_m"] - old_midpoint
    inertial.set("pos", " ".join(map(str, com)))
    for body in bodies.values():
        for geom in list(body.findall("geom")):
            name = geom.get("name", "")
            if (geom.get("class") in ("head_visual", "head_collision")
                    or name.startswith(("pan_servo_", "pan_bracket_", "head_zed", "zed_"))):
                body.remove(geom)
    for part in source["parts"]:
        asset = HERE / "assets_nero" / part["asset"]
        if sha256(asset) != part["sha256"]:
            raise ValueError(f"Spinal CAD part changed: {part['name']}")
        name = "spinal_" + part["name"]
        ET.SubElement(root.find("asset"), "mesh", name=name, file=part["asset"], inertia="shell")
        camera = part["name"] == "zed_camera"
        for collision in (False, True):
            geom_name = ("zed_housing_collision" if collision else "head_zed2") if camera else (
                name + ("_collision" if collision else ""))
            attrs = dict(name=geom_name, type="mesh", mesh=name, density="0",
                         group="3" if collision else "2", contype="2" if collision else "0",
                         conaffinity="3" if collision else "0")
            if not collision:
                attrs["material"] = "sophia_black"
            ET.SubElement(bodies[part["body"]], "geom", **attrs)
    midpoint = np.array(source["camera_midpoint_tilt_m"])
    rotation = np.array(source["camera_rotation_local"])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    baseline = np.linalg.norm(np.fromstring(old_cameras[1].get("pos"), sep=" ")
                              - np.fromstring(old_cameras[0].get("pos"), sep=" "))
    for i, camera in enumerate(old_cameras):
        camera.set("pos", " ".join(map(str, midpoint + (i-.5)*baseline*rotation[:, 0])))
        camera.set("quat", " ".join(map(str, quat)))
        camera.attrib.pop("xyaxes", None)
    for site in tilt.findall("site"):
        if site.get("name", "").startswith("head_zed"):
            site.set("pos", " ".join(map(str, midpoint)))


def spinal_neck_geometry(configuration):
    """Load the reviewed static-neck mesh and dimensions, without runtime CAD tools."""
    profile = configuration.get("spinal_neck_geometry")
    if not profile or not profile.get("enabled", False):
        return None
    path = HERE / profile["source"]
    if sha256(path) != profile["sha256"]:
        raise ValueError("Spinal neck provenance changed; regenerate and review")
    data = json.loads(path.read_text())
    asset = HERE / "assets_nero" / data["asset"]
    if sha256(asset) != data["asset_sha256"]:
        raise ValueError("Spinal neck mesh differs from reviewed extraction")
    size = np.array(data["neck_extents_m"], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.min(size) <= 0:
        raise ValueError("Invalid spinal neck dimensions")
    return data


def vendor_inertials():
    """Only transfer inertials whose mesh coordinates match the local body frame."""
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    for relative, digest in provenance["files_sha256"].items():
        if sha256(VENDOR / relative) != digest:
            raise ValueError(f"Pinned vendor source changed: {relative}")
    for local, evidence in provenance["identical_local_meshes"].items():
        if sha256(HERE / "assets_nero" / local) != evidence["sha256"]:
            raise ValueError(f"Local mesh no longer matches vendor inertial frame: {local}")
    arm = ET.parse(VENDOR / "nero/urdf/nero_description.urdf").getroot()
    gripper = ET.parse(VENDOR / "nero/urdf/nero_with_gripper_description.xacro").getroot()
    links = [
        (f"arm_{name}", arm.find(f"link[@name='{name}']"))
        for name in ("base_link", *[f"link{i}" for i in range(1, 8)])
    ]
    links += [
        (f"gripper_finger_{letter}", gripper.find(f"link[@name='gripper_link{i}']"))
        for i, letter in enumerate("ab", 1)
    ]
    result = {}
    for local, link in links:
        collision_origin = link.find("collision/origin")
        if not all(
            np.allclose(np.fromstring(collision_origin.get(key), sep=" "), 0)
            for key in ("xyz", "rpy")
        ):
            raise ValueError(f"Mesh requires a frame conversion: {local}")
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        if not np.allclose(np.fromstring(origin.get("rpy"), sep=" "), 0):
            raise ValueError(f"Inertia requires a frame conversion: {local}")
        tensor = inertial.find("inertia")
        matrix = np.array(
            [[float(tensor.get(f"i{''.join(sorted((a, b)))}")) for b in "xyz"] for a in "xyz"]
        )
        mass = float(inertial.find("mass").get("value"))
        principal = np.linalg.eigvalsh(matrix)
        if (
            mass <= 0
            or not np.isfinite(matrix).all()
            or principal[0] <= 0
            or principal[-1] > sum(principal[:2])
        ):
            raise ValueError(f"Invalid vendor inertial data: {local}")
        result[local] = {
            "mass": mass,
            "pos": np.fromstring(origin.get("xyz"), sep=" "),
            "matrix": matrix,
            "fullinertia": np.array(
                [float(tensor.get(key)) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            ),
        }
    return result


def apply_vendor_inertials(root):
    for suffix, source in vendor_inertials().items():
        for side in ("left", "right"):
            body = root.find(f".//body[@name='{side}_{suffix}']")
            if body is None and suffix.startswith("gripper_"):
                continue  # The fixed-base IK model has no stock gripper fingers.
            if body is None:
                raise ValueError(f"Missing arm body: {side}_{suffix}")
            inertial = body.find("inertial")
            inertial.attrib.clear()
            inertial.set("mass", str(source["mass"]))
            for field in ("pos", "fullinertia"):
                inertial.set(field, " ".join(f"{v:.14g}" for v in source[field]))


def head_calibration():
    source = json.loads(CAMERA_SOURCE.read_text())["zed"]["rectified"]
    if not np.allclose(source["R_left_right"], np.eye(3)):
        raise ValueError("Head stream requires rectified parallel cameras")
    if not np.allclose(source["T_left_right_m"], [source["baseline_m"], 0, 0]):
        raise ValueError("Unexpected rectified stereo baseline")
    for side in ("left", "right"):
        data = source[side]
        values = [data[k] for k in ("width", "height", "fx", "fy", "cx", "cy")]
        if (
            not np.isfinite(values).all()
            or min(values[:4]) <= 0
            or not np.allclose(data["distortion"], 0)
        ):
            raise ValueError("Invalid rectified intrinsics")
    return source


def apply_head_calibration(root):
    source = head_calibration()
    cameras = [root.find(f".//camera[@name='head_zed_{side}']") for side in ("left", "right")]
    positions = np.array([np.fromstring(camera.get("pos"), sep=" ") for camera in cameras])
    midpoint = positions.mean(axis=0)
    direction = positions[1] - positions[0]
    direction /= np.linalg.norm(direction)
    for side, sign, camera in zip(("left", "right"), (-1, 1), cameras, strict=True):
        data = source[side]
        camera.attrib.pop("fovy", None)
        camera.set("sensorsize", "1 1")
        camera.set("resolution", f"{data['width']} {data['height']}")
        camera.set("focalpixel", f"{data['fx']:.12g} {data['fy']:.12g}")
        # OpenCV pixel (0,0) is centered at (0,0). MuJoCo's frustum offset
        # has the opposite sign on both image axes.
        camera.set(
            "principalpixel",
            f"{(data['width'] - 1) / 2 - data['cx']:.12g} "
            f"{(data['height'] - 1) / 2 - data['cy']:.12g}",
        )
        camera.set("ipd", "0")
        camera.set(
            "pos",
            " ".join(f"{v:.12g}" for v in midpoint + sign * source["baseline_m"] / 2 * direction),
        )


def apply_dataset_head_alignment(root, configuration):
    """Apply a provisional image-fitted rigid neck assembly, preserving encoders.

    Moving only the optical cameras can put them inside the unchanged housing.
    This correction carries visuals, collisions and inertials with their bodies.
    Hardware servo Home, travel, joint axes and lens intrinsics are unchanged.
    """
    profile = configuration.get("head_alignment")
    if not profile or not profile.get("enabled", False):
        return
    path = HERE / profile["source"]
    if sha256(path) != profile["sha256"]:
        raise ValueError("Dataset head alignment provenance changed")
    fit = json.loads(path.read_text())
    offset = np.array(fit["head_neck_parent_offset_m"], dtype=float)
    quat = np.array(fit["head_tilt_fixed_rotation_quat_wxyz"], dtype=float)
    if (configuration["measured"].get("neck_mount_coaxial_with_pillar", False)
            and offset.shape == (3,) and not np.allclose(offset[:2], 0, atol=1e-9)):
        raise ValueError("Head fit violates user-confirmed concentric neck/pillar mounting")
    if (offset.shape != (3,) or quat.shape != (4,) or not np.isfinite(offset).all()
            or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1)
            or np.max(np.abs(offset)) > .15):
        raise ValueError("Invalid dataset head alignment")
    neck = root.find(".//body[@name='head_neck']")
    tilt = root.find(".//body[@name='head_tilt']")
    if not np.allclose(np.fromstring(neck.get("pos"), sep=" "), fit["reference_head_neck_pos_m"]):
        raise ValueError("Head alignment requires refitting after nominal mount changes")
    if tilt.get("quat") is not None or tilt.get("euler") is not None:
        raise ValueError("Head alignment expects the original unrotated tilt body")
    position = np.array(fit["reference_head_neck_pos_m"]) + offset
    neck.set("pos", " ".join(f"{v:.12g}" for v in position))
    tilt.set("quat", " ".join(f"{v:.12g}" for v in quat))


def apply_confirmed_head_mount(root, configuration):
    """Place the stereo midpoint on the user-confirmed neutral lens plane.

    Lift_Top local -X is forward. Keep the neck axis fixed and move the complete
    ZED housing, lenses, collision shapes and optical frames together. The neck
    retains its articulated kinematics, so its projection changes during tilt.
    """
    import mujoco

    measured = configuration["measured"]
    position, rotation = np.zeros(3), np.eye(3)
    for name in ("head_neck", "head_pan", "head_tilt"):
        body = root.find(f".//body[@name='{name}']")
        if body.get("euler") is not None:
            raise ValueError("Confirmed camera mount requires explicit body quaternions")
        position += rotation @ np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        local_rotation = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation, np.fromstring(body.get("quat", "1 0 0 0"), sep=" "))
        rotation = rotation @ local_rotation.reshape(3, 3)
    tilt = root.find(".//body[@name='head_tilt']")
    cameras = [tilt.find(f"camera[@name='head_zed_{side}']") for side in ("left", "right")]
    midpoint = np.mean([np.fromstring(cam.get("pos"), sep=" ") for cam in cameras], axis=0)
    current = position + rotation @ midpoint
    target = np.array([-measured["head_lens_midpoint_forward_m"],
                       measured["head_lens_midpoint_lateral_m"], current[2]])
    shift = rotation.T @ (target - current)
    for node in tilt:
        name = node.get("name", "")
        if name.startswith(("head_zed", "zed_")):
            pos = np.fromstring(node.get("pos", "0 0 0"), sep=" ") + shift
            node.set("pos", " ".join(f"{v:.12g}" for v in pos))


def apply_wrist_cameras(root, configuration):
    """Measured lenses with explicitly accepted photo-estimated mounts."""
    profiles = json.loads(CAMERA_SOURCE.read_text())
    mount = configuration["wrist_cameras"]
    angle = np.deg2rad(mount["toe_in_deg"])
    roll = np.deg2rad(mount.get("mount_roll_deg", 0))
    rotation = np.array([[np.cos(roll), -np.sin(roll), 0],
                         [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    # The upper-side installation rotates both image axes around the approach
    # axis. Toe-in still points toward the grasp centre, with image-up above it.
    right = rotation @ [0, 1, 0]
    up = rotation @ [np.cos(angle), 0, np.sin(angle)]
    axes = np.concatenate([right, up])
    forward = -np.cross(right, up)
    position = np.asarray(mount["position_m"], dtype=float)
    for side in ("left", "right"):
        profile = profiles[f"{side}_gripper"]
        parent = root.find(f'.//body[@name="{side}_gripper_base"]')
        ET.SubElement(parent, "camera", {
            "name": f"{side}_wrist", "mode": "fixed", "ipd": "0",
            "pos": " ".join(map(str, mount["position_m"])),
            "xyaxes": " ".join(map(str, axes)), "sensorsize": "1 1",
            "resolution": f"{profile['width']} {profile['height']}",
            "focalpixel": f"{profile['fx']} {profile['fy']}",
            "principalpixel": f"{(profile['width']-1)/2-profile['cx']} "
                              f"{(profile['height']-1)/2-profile['cy']}",
        })
        ET.SubElement(parent, "geom", {
            "name": f"{side}_wrist_sensor", "type": "box",
            "pos": " ".join(map(str, position - .005 * forward)),
            "xyaxes": " ".join(map(str, axes)),
            "size": ".012 .017 .002", "group": "2", "density": "0",
            "contype": "0", "conaffinity": "0", "material": "sophia_board",
        })
        ET.SubElement(parent, "geom", {
            "name": f"{side}_wrist_lens", "type": "cylinder",
            "pos": " ".join(map(str, mount["position_m"])),
            "xyaxes": " ".join(map(str, axes)),
            "size": ".004 .001", "group": "2", "density": "0",
            "contype": "0", "conaffinity": "0", "material": "sophia_lens",
        })


def apply_base_alignment(root, configuration, base_height):
    """Apply the reviewed base fit without moving wheels, pillar, arms or head."""
    profile = configuration.get('base_alignment', {})
    if not profile.get('enabled', False):
        return
    path = HERE / profile['source']
    if sha256(path) != profile['sha256']:
        raise ValueError('Base calibration changed')
    data = json.loads(path.read_text())
    for dependency in ('arm_reference', 'spinal_assembly'):
        if (not configuration.get(dependency, {}).get('enabled', False) or
                configuration[dependency]['sha256'] != data['requires_' + dependency + '_sha256']):
            raise ValueError('Base alignment requires the reviewed head and arm frames')
    for name, value in data['measured_dimensions_m'].items():
        if not np.isclose(configuration['measured'][name], value, atol=1e-9, rtol=0):
            raise ValueError('Base calibration dimensions no longer match')
    shift = np.asarray(data['frame_translation_chassis_m'], dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all() or shift[2] != 0 or np.linalg.norm(shift) > .3:
        raise ValueError('Invalid base alignment translation')
    chassis = root.find('.//body[@name="sophia_chassis"]')
    def numbers(v):
        return ' '.join(f'{x:.12g}' for x in v)
    def set_position(name, value):
        found = [g for g in chassis.findall('geom') if g.get('name') in (name, name + '_collision')]
        if not found:
            raise ValueError('Missing base geometry: ' + name)
        for g in found:
            g.set('pos', numbers(value))
    for g in chassis.findall('geom'):
        name = g.get('name', '')
        if name.startswith(('lower_rail_', 'upper_rail_', 'corner_post_', 'battery', 'bay_cable_')):
            if g.get('fromto'):
                value = np.fromstring(g.get('fromto'), sep=' ').reshape(2, 3) + shift
                g.set('fromto', numbers(value.ravel()))
            else:
                g.set('pos', numbers(np.fromstring(g.get('pos'), sep=' ') + shift))
    # Old repeated controller blocks and sideways antenna were photo proxies.
    # Replace them with observed components. Hidden dimensions remain priors.
    for g in list(chassis.findall('geom')):
        if g.get('name', '').startswith(('controller_', 'heatsink_', 'power_wire_', 'antenna_bar_')):
            chassis.remove(g)
    a = data['accessories']
    # Chassis uses upright world axes; its origin is the grounded source base.
    z0 = float(base_height)
    xy = a['estop_plate_center_chassis_xy_m']
    set_position('estop_plate', [*xy, a['estop_plate_world_z_m'] - z0])
    set_position('estop', [*xy, a['estop_plate_world_z_m'] + .015 - z0])
    for name in ('antenna_body', 'antenna_strap', 'antenna_mount_support'):
        g = chassis.find(f'geom[@name="{name}"]')
        pos = np.fromstring(g.get('pos'), sep=' ')
        pos[1] = a['antenna_body_chassis_y_m']
        set_position(name, pos)
    for index, rod in enumerate(a['antenna_rods_world_chassis_xy_m']):
        points = np.array(rod)
        points[:, 2] -= z0
        ET.SubElement(chassis, 'geom', name=f'antenna_rod_{index}', type='capsule',
                      fromto=numbers(points.ravel()), size='.0035', material='sophia_white',
                      group='2', contype='0', conaffinity='0', density='0')
    for name, material in [('board', 'sophia_board'), ('finned_module', 'sophia_silver')]:
        pos = np.array(a[name + '_center_chassis_xy_world_z_m'])
        pos[2] -= z0
        ET.SubElement(chassis, 'geom', name='bay_' + name, type='box',
                      pos=numbers(pos), size=numbers(a[name + '_half_size_m']),
                      material=material, group='2', contype='0', conaffinity='0', density='0')
