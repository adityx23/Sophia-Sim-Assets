"""Exercise the policy action API using a privileged, simulation-only IK oracle.

This validates task reachability and physical grasp/release, not learned-policy
performance or hardware transfer. Does not connect to a robot.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.sophia_env import OBJECT_Z, SophiaPickPlaceEnv, solve_grasp_ik


class TaskOracle:
    def __init__(self, env, frame_callback=None):
        self.env = env
        self.info = None
        self.done = False
        self.returns = 0.0
        self.stages = []
        self.frame_callback = frame_callback

    def step(self, action):
        if self.done:
            return
        _, reward, terminated, truncated, self.info = self.env.step(action)
        self.returns += reward
        self.done = terminated or truncated
        if self.frame_callback:
            self.frame_callback(self.env)
        if self.done and not self.info["is_success"]:
            raise RuntimeError(f"Oracle episode failed: {self.info}")

    def hold(self, opening, seconds):
        for _ in range(round(seconds * self.env.contract.control_hz)):
            self.step(np.r_[np.zeros(7), opening])

    def move(self, target, opening, stage):
        self.stages.append(stage)
        target = np.asarray(target, dtype=float)
        corrected = target.copy()
        for _ in range(5):
            q = solve_grasp_ik(self.env.sim, corrected)
            for _ in range(240):
                if self.done:
                    return
                delta = (q - self.env.adapter.target) / self.env.contract.delta_rad
                self.step(np.r_[np.clip(delta, -1, 1), opening])
                if np.max(abs(q - self.env.adapter.target)) < 1e-5:
                    break
            self.hold(opening, 0.45)
            error = target - self.env.data.site_xpos[self.env.grasp_site]
            if np.linalg.norm(error) < 0.003 or self.done:
                return
            corrected += error * min(1.0, 0.015 / max(np.linalg.norm(error), 1e-9))
        if np.linalg.norm(error) > 0.008:
            raise RuntimeError(f"Unable to reach {stage}: residual {error}")

    def run(self):
        e = self.env
        pickup = e.data.xpos[e.object_id].copy()
        self.move(pickup + [0, 0, 0.1], 1.0, "approach")
        self.move(pickup, 1.0, "lower")
        self.stages.append("grasp")
        self.hold(0.0, 0.6)
        if e._task_state()["finger_contacts"] != 2:
            raise RuntimeError("Oracle failed to establish bilateral finger contact")
        self.move(pickup + [0, 0, 0.1], 0.0, "lift")
        if not e.lifted:
            raise RuntimeError("Object was not physically lifted")
        destination = np.r_[e.goal_xy, OBJECT_Z]
        self.move(destination + [0, 0, 0.1], 0.0, "transfer")
        offset = e.data.site_xpos[e.grasp_site] - e.data.xpos[e.object_id]
        self.move(destination + offset + [0, 0, 0.003], 0.0, "place")
        self.stages.append("release")
        self.hold(1.0, 0.5)
        self.move(destination + offset + [0, 0, 0.12], 1.0, "withdraw")
        self.hold(1.0, 0.5)
        if not self.done or not self.info["is_success"]:
            raise RuntimeError(f"Oracle did not complete the task: {self.info}")
        return {"return": self.returns, "stages": self.stages, **self.info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/asset-review/sophia-training"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record", action="store_true", help="Save the first oracle rollout as MP4"
    )
    parser.add_argument("--rgb-observations", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with SophiaPickPlaceEnv(
        observation_mode="rgb" if args.rgb_observations else "proprio"
    ) as env:
        (args.output / "policy-contract.json").write_text(
            json.dumps(env.contract.describe(), indent=2) + "\n"
        )
        for episode in range(args.episodes):
            seed = args.seed + episode
            env.reset(seed=seed)
            writer = None
            if args.record and episode == 0:
                import cv2

                writer = cv2.VideoWriter(
                    str(args.output / "pick-place-oracle.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (640, 360),
                )
                if not writer.isOpened():
                    raise RuntimeError("Unable to open video encoder")

            def record(e, writer=writer):
                if writer:
                    writer.write(cv2.cvtColor(e.render(), cv2.COLOR_RGB2BGR))

            try:
                result = TaskOracle(env, record).run()
                results.append({"seed": seed, **result})
                print(f"Seed {seed}: success in {result['steps']} steps", flush=True)
            finally:
                if writer:
                    writer.release()
    sources = [
        "robot/sophia_policy.py",
        "robot/sophia_env.py",
        "robot/yor-description/sophia.mjcf",
        "robot/yor-description/sophia.json",
        "scripts/run_sophia_pick_place.py",
    ]
    report = {
        "controller": "privileged simulation-only IK oracle through the policy action API",
        "learned_policy": False,
        "hardware_validated": False,
        "episodes": results,
        "all_passed": all(r["is_success"] for r in results),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
        },
    }
    (args.output / "task-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
