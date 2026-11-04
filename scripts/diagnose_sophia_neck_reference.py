"""Compare servo reference and optical pitch hypotheses without editing assets.

Uses reviewed fingertip midpoints and measured joint states. Fits only episode-0
training labels; the other splits are development validation, not final tests.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.sim_camera import camera_intrinsics  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402
from scripts.fit_sophia_head_camera import project  # noqa: E402
from scripts.render_continuous_teleop import measured_pose  # noqa: E402

OUT = ROOT / "output/asset-review/pi-neck-reference-audit"


def main():
    sim = YORMujoco(ROOT / "robot/yor-description/scene-sophia.mjcf", headless=True)
    m, d = sim.model, sim.data
    cam = m.camera("head_zed_left").id
    k = camera_intrinsics(m, "head_zed_left", 1920, 1080)
    samples = []
    cache = ROOT / "output/asset-review/head-camera-fit"
    try:
        for episode in (0, 1):
            prefix = "episode-1-" if episode else ""
            geometry = json.loads((cache / f"{prefix}kinematics.json").read_text())
            labels = json.loads((cache / f"{prefix}annotations.json").read_text())["frames"]
            with np.load(
                ROOT
                / f"output/asset-review/hf-sanity-check/native-replay/episode_{episode:06d}.npz"
            ) as archive:
                a = {name: archive[name] for name in archive.files}
            for side in ("left", "right"):
                a[side + "_gripper"] = a[side + "_gripper"].reshape(-1)
            for row in geometry["records"]:
                label = labels.get(str(row["second"]))
                if not label:
                    continue
                measured_pose(sim, d, a, row["measured_index"])
                for side in ("left", "right"):
                    names = [side + "_" + letter for letter in "ab"]
                    if not all(name in label["pixels"] for name in names):
                        continue
                    points = []
                    for letter in "ab":
                        b = m.body(f"{side}_gripper_finger_{letter}").id
                        points.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ [0, 0, 0.0035])
                    split = (
                        "episode_1_validation"
                        if episode
                        else "train"
                        if label["split"] == "train"
                        else "episode_0_holdout"
                    )
                    samples.append(
                        dict(
                            episode=episode,
                            second=row["second"],
                            qpos=d.qpos.copy(),
                            points=np.array(points),
                            observed=np.mean([label["pixels"][n] for n in names], axis=0),
                            split=split,
                        )
                    )
        observed = np.array([s["observed"] for s in samples])
        groups = {
            name: np.array([s["split"] == name for s in samples])
            for name in ("train", "episode_0_holdout", "episode_1_validation")
        }

        # Parameters are additive *geometric* tilt, pan and optical pitch degrees.
        # A geometric tilt rotates the entire assembly about the actual joint.
        def predict(x, tilt_sign=1):
            optical = Rotation.from_rotvec([-np.deg2rad(x[2]), 0, 0]).as_matrix()
            predictions = []
            for s in samples:
                d.qpos[:] = s["qpos"]
                d.qpos[sim._tilt_adr] = tilt_sign * d.qpos[sim._tilt_adr] + np.deg2rad(x[0])
                d.qpos[sim._pan_adr] += np.deg2rad(x[1])
                mujoco.mj_forward(m, d)
                predictions.append(
                    project(
                        s["points"], d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3) @ optical, k
                    ).mean(0)
                )
            return np.array(predictions)

        definitions = {
            "current_home_as_zero": ([], np.zeros(3), 1),
            "hardware_zero_as_geometric_zero": ([], [-sim._neck_zero_deg[1], 0, 0], 1),
            "home_offset_added_twice": ([], [sim._neck_zero_deg[1], 0, 0], 1),
            "reversed_tilt_sign": ([], np.zeros(3), -1),
            "fit_geometric_tilt": ([0], np.zeros(3), 1),
            "fit_geometric_pan_tilt": ([0, 1], np.zeros(3), 1),
            "fit_optical_pitch": ([2], np.zeros(3), 1),
            "fit_joint_and_optical_pitch": ([0, 2], np.zeros(3), 1),
        }
        hypotheses = {}
        for name, (active, neutral, sign) in definitions.items():

            def expand(v, neutral=neutral, active=active):
                x = np.array(neutral, dtype=float)
                x[active] = v
                return x

            if active:
                result = least_squares(
                    lambda v, expand=expand, sign=sign: (
                        predict(expand(v), sign)[groups["train"]] - observed[groups["train"]]
                    ).ravel(),
                    np.zeros(len(active)),
                    bounds=(-25, 25),
                    loss="soft_l1",
                    f_scale=6,
                )
                x = expand(result.x)
                singular = np.linalg.svd(result.jac, compute_uv=False).tolist()
            else:
                x, singular = np.array(neutral), []
            predictions = predict(x, sign)
            scores = {}
            for split, mask in groups.items():
                error = predictions[mask] - observed[mask]
                scores[split] = dict(
                    n=int(mask.sum()),
                    rmse_px=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    axis_rmse_px=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                )
            hypotheses[name] = dict(
                geometric_tilt_delta_deg=float(x[0]),
                geometric_pan_delta_deg=float(x[1]),
                optical_pitch_down_deg=float(x[2]),
                scores=scores,
                jacobian_singular_values=singular,
            )
        report = dict(
            model_sha256=hashlib.sha256(
                (ROOT / "robot/yor-description/sophia.mjcf").read_bytes()
            ).hexdigest(),
            hardware_home_deg=sim._neck_home_deg.tolist(),
            current_geometric_zero_hardware_deg=sim._neck_zero_deg.tolist(),
            observations=len(samples),
            hypotheses=hypotheses,
            changes_applied=False,
            caveats=[
                "Sparse image landmarks do not uniquely identify physical mounting parameters.",
                "Episode 1 is previously inspected development validation.",
                "All measured arm poses, mount positions and intrinsics held fixed.",
                "Optical-only rotation is an explanatory ablation, "
                "not a physical mount correction.",
            ],
        )
        OUT.mkdir(parents=True, exist_ok=True)
        delta = hypotheses["fit_geometric_tilt"]["geometric_tilt_delta_deg"]
        report["candidate_geometric_zero_hardware_deg"] = [
            float(sim._neck_zero_deg[0]),
            float(sim._neck_zero_deg[1] - delta),
        ]
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        panels = []
        with mujoco.Renderer(m, height=360, width=640) as renderer:
            for second in (5, 32, 48):
                sample = next(s for s in samples if s["episode"] == 0 and s["second"] == second)
                real = cv2.imread(str(cache / f"real-{second:02d}s.jpg"))
                views = [cv2.cvtColor(cv2.resize(real, (640, 360)), cv2.COLOR_BGR2RGB)]
                for correction in (0, delta):
                    d.qpos[:] = sample["qpos"]
                    d.qpos[sim._tilt_adr] += np.deg2rad(correction)
                    mujoco.mj_forward(m, d)
                    renderer.update_scene(d, camera="head_zed_left", scene_option=option)
                    views.append(renderer.render().copy())
                panel = np.full((390, 1920, 3), 22, np.uint8)
                labels = ["REAL ZED", "CURRENT SIM", f"SIM neck reference +{delta:.2f} deg"]
                for j, (view, label) in enumerate(zip(views, labels, strict=True)):
                    panel[30:, j * 640 : (j + 1) * 640] = view
                    cv2.putText(
                        panel,
                        f"{second}s | {label}",
                        (j * 640 + 8, 22),
                        0,
                        0.55,
                        (240, 240, 240),
                        1,
                    )
                panels.append(panel)
        cv2.imwrite(
            str(OUT / "reference-ablation.jpg"), cv2.cvtColor(np.vstack(panels), cv2.COLOR_RGB2BGR)
        )
        (OUT / "reference-hypotheses.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(hypotheses, indent=2))
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
