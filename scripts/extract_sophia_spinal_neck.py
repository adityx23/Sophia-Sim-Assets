"""Extract the reviewed static lattice from the supplied assembled spinal STL.

The reviewed triangle block contains its disconnected surface patches too.
Selection is valid only for the pinned source hash; a different export requires
review. Preserve triangles and dimensions, using only a rigid frame change and
the millimetre-to-metre unit conversion checked against the 120 mm lens baseline.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CAD Parts/SPINAL DESIGN.stl"
DESCRIPTION = ROOT / "robot/yor-description"
EXPECTED_SHA256 = "abce9493faff32b848605c3744614f8044bdbf7c3f8314841db210e57dccf48d"


def main():
    payload = SOURCE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Spinal STL changed; review the part selection before extracting")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + count * 50:
        raise ValueError("Invalid binary STL record count")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("a", "<u2")])
    records = np.frombuffer(payload, offset=84, dtype=dtype)
    start, stop = 76639, 89622
    triangles = records["vertices"][start:stop].astype(float)
    bounds = np.array([triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))])
    extents = bounds[1] - bounds[0]
    np.testing.assert_allclose(extents, [90, 75, 108.95000194], atol=1e-5)
    origin = np.r_[bounds[:, :2].mean(axis=0), bounds[0, 2]]
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    transformed = ((triangles - origin) @ rotation.T * 0.001).astype("<f4")
    normal = np.cross(transformed[:, 1] - transformed[:, 0], transformed[:, 2] - transformed[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
    result = np.zeros(len(transformed), dtype=dtype)
    result["vertices"], result["normal"] = transformed, normal
    directory = DESCRIPTION / "assets_nero/sophia"
    directory.mkdir(exist_ok=True)
    asset = directory / "spinal_neck_lattice.stl"
    asset.write_bytes(
        b"Sophia supplied spinal STL: static neck lattice, SI units".ljust(80, b"\0")
        + struct.pack("<I", len(result))
        + result.tobytes()
    )
    provenance = dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=digest,
        source_triangles=count,
        selected_triangle_slice_start_inclusive=start,
        selected_triangle_slice_stop_exclusive=stop,
        retained_triangles=len(result),
        selection=(
            "Reviewed contiguous static lattice block, including disconnected surface patches"
        ),
        source_bounds_mm=bounds.tolist(),
        origin_in_source_mm=origin.tolist(),
        rotation_source_to_neck=rotation.tolist(),
        units_scale=0.001,
        neck_extents_m=[
            float(extents[1] / 1000),
            float(extents[0] / 1000),
            float(extents[2] / 1000),
        ],
        asset=str(asset.relative_to(DESCRIPTION / "assets_nero")),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        physical_alignment=(
            "XY bounding-box centre on confirmed pillar centreline; base at neck Z=0"
        ),
        limitations=[
            "STL is a tessellated export, not exact BRep or joint/mate definitions.",
            "Bounds establish the static lattice envelope, not the pan/tilt encoder zero.",
            "Physics retains explicit primitive collision proxies "
            "and approximate source inertials.",
        ],
    )
    path = DESCRIPTION / "calibration/spinal_neck_geometry_2026-11-04.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            dict(asset=str(asset), extents_m=provenance["neck_extents_m"], triangles=len(result)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
