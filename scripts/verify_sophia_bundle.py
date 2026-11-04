"""Verify every bundled checksum, portability and deterministic regeneration.

Run with the pinned Sophia environment. Extracts only into a temporary directory;
never imports hardware interfaces or changes the working simulation assets.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import mujoco
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("MANIFEST.json"))
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        if set(names) != set(manifest["files"]) | {"MANIFEST.json"}:
            raise ValueError("Manifest/file inventory mismatch")
        for name, entry in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid archive path")
            data = archive.read(name)
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError("Checksum mismatch: " + name)
        with tempfile.TemporaryDirectory(prefix="sophia-release-verify-") as tmp:
            root = Path(tmp)
            archive.extractall(root)
            desc = root / "robot/yor-description"
            files = ["sophia.mjcf", "sophia-ik.mjcf", "scene-sophia.mjcf"]
            before = {name: digest((desc / name).read_bytes()) for name in files}
            model = mujoco.MjModel.from_xml_path(str(root / manifest["entry_model"]))
            ik = mujoco.MjModel.from_xml_path(str(desc / "sophia-ik.mjcf"))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError("Nonfinite loaded state")
            env = dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1")
            completed = subprocess.run(
                [sys.executable, "robot/yor-description/build_sophia.py"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            changed = [name for name in files if digest((desc / name).read_bytes()) != before[name]]
            if changed:
                raise ValueError("Archive rebuild differs: " + ", ".join(changed))
            result = dict(
                passed=True,
                archive=str(args.archive.resolve()),
                archive_sha256=digest(args.archive.read_bytes()),
                release=manifest.get("release"),
                checked_files=len(manifest["files"]),
                portable_model_load=True,
                portable_ik_load=True,
                deterministic_model_rebuild=True,
                model_nq=model.nq,
                ik_nq=ik.nq,
                model_mass_kg=float(model.body_mass.sum()),
                rebuild_stdout=completed.stdout,
                hardware_accessed=False,
            )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
