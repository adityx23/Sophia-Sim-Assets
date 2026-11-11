"""Export synchronized Sophia head and wrist RGB/depth views plus calibration."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.sim_camera import SophiaCameraRenderer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pose", choices=("home", "reference", "lowered", "raised"), default="home"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pinhole",
        dest="raw",
        action="store_false",
        help="Export flat rectified views (default; retained for compatibility)",
    )
    mode.add_argument(
        "--raw-distorted",
        dest="raw",
        action="store_true",
        help="Apply measured wrist lens distortion, including invalid edge masks",
    )
    parser.set_defaults(raw=False)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(ROOT / "robot/yor-description/scene-sophia.mjcf"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(args.pose).id)
    mujoco.mj_forward(m, d)
    camera = SophiaCameraRenderer(m, args.width, args.height)
    metadata = {}
    try:
        for name in camera.names:
            rgb, depth, valid, meta = camera.render(d, name, raw=args.raw)
            Image.fromarray(rgb).save(args.output / f"{name}.png")
            np.savez_compressed(args.output / f"{name}-depth.npz", depth_m=depth, valid=valid)
            metadata[name] = meta
    finally:
        camera.close()
    (args.output / "cameras.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote four synchronized views, depth maps, masks and metadata to {args.output}")


if __name__ == "__main__":
    main()
