"""Preserve disconnected CAD solids as separate MuJoCo collision meshes.

One convex hull around the complete gripper housing bridges the empty space
between its motor and wide top plate. Partition by exact shared STL vertices;
keep every original triangle and coordinate, without rescaling or decimation.
"""

import copy
import struct
import xml.etree.ElementTree as ET

import numpy as np


def read_binary_stl(path):
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"Truncated STL: {path}")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL with exact record length: {path}")
    records = np.frombuffer(
        payload,
        offset=84,
        count=count,
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    triangles = records["vertices"].copy()
    if not np.isfinite(triangles).all():
        raise ValueError(f"Nonfinite STL coordinates: {path}")
    return triangles


def connected_triangle_parts(triangles):
    """Deterministic partition; exact coordinate equality preserves CAD seams."""
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    parents = np.arange(len(vertices))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    faces = indices.reshape(-1, 3)
    for a, b, c in faces:
        parents[find(b)] = find(a)
        parents[find(c)] = find(a)
    labels = np.array([find(face[0]) for face in faces])
    parts = [triangles[labels == label] for label in np.unique(labels)]
    return sorted(parts, key=lambda part: tuple(part.min(axis=(0, 1))))


def write_collision_stl(path, triangles):
    records = np.zeros(
        len(triangles),
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    records["vertices"] = triangles
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    records["normal"] = normals
    with path.open("wb") as stream:
        stream.write(
            b"Sophia collision: unchanged CAD connected component".ljust(80, b"\0")
        )
        stream.write(struct.pack("<I", len(triangles)))
        stream.write(records.tobytes())


def apply_gripper_housing_collision(root, assets):
    parts = connected_triangle_parts(read_binary_stl(assets / "gripper_base.stl"))
    directory = assets / "sophia_collision"
    directory.mkdir(exist_ok=True)
    for i, triangles in enumerate(parts):
        name = f"sophia_gripper_housing_{i}"
        file = directory / f"{name}.stl"
        write_collision_stl(file, triangles)
        ET.SubElement(
            root.find("asset"), "mesh", name=name, file=str(file.relative_to(assets))
        )
    for side in ("left", "right"):
        body = root.find(f'.//body[@name="{side}_gripper_base"]')
        original = body.find('geom[@class="agx_gripper_collision"]')
        if original is None or original.get("mesh") != "agx_gripper_base":
            raise ValueError(f"Unexpected {side} gripper housing source")
        body.remove(original)
        for i in range(len(parts)):
            geom = copy.deepcopy(original)
            geom.set("name", f"{side}_gripper_housing_{i}_collision")
            geom.set("mesh", f"sophia_gripper_housing_{i}")
            geom.set("density", "0")
            body.append(geom)
