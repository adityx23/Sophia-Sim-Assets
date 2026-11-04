"""Fit effective arm gains and opening rate to existing native robot episodes.

No hardware access or source-parameter mutation. Recorded command snapshots are
not exact dispatch timestamps. Keep masses, friction, force limits and joint
offsets fixed; report sensitivity to the assumed command interpolation duration.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay_sophia_dataset_arm import fixture, replay

from robot.yor_mujoco import _SimArm

DESCRIPTION = ROOT / "robot/yor-description"
TRAIN = (0, 2, 4)
VALIDATION = tuple(i for i in range(18) if i not in TRAIN)
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Experiment:
    def __init__(self, root):
        self.model = fixture()
        self.data = mujoco.MjData(self.model)
        self.arm = _SimArm("right", self.model, str(DESCRIPTION / "sophia-ik.mjcf"))
        self.episodes, self.controls = {}, {}
        for index in (*TRAIN, *VALIDATION):
            path = root / f"episode_{index:06d}.npz"
            with np.load(path, allow_pickle=False) as source:
                record = {key: source[key].copy() for key in source.files}
            t = record["time_s"]
            q, commands = record["right_joints"], record["right_joints_cmd"]
            g, gc = record["right_gripper"].ravel(), record["right_gripper_cmd"].ravel()
            if (
                np.any(np.diff(t) <= 0)
                or t[0] != 0
                or not np.isfinite(np.c_[t, q, commands, g, gc]).all()
            ):
                raise ValueError(f"Invalid native sequence: {path}")
            opened = (g > 0.95) & (gc >= 0.99)
            last_closed = np.maximum.accumulate(np.where(~opened, t, -10.0))
            cmd_v, enc_v = record["cmd_vel"], record["enc_vel"]
            stationary = (
                (abs(cmd_v[:, :2]).max(axis=1) < 0.02)
                & (abs(cmd_v[:, 2]) < 0.05)
                & (abs(enc_v[:, :2]).max(axis=1) < 0.03)
                & (abs(enc_v[:, 2]) < 0.08)
            )
            record["fit_mask"] = (
                (t >= 1)
                & opened
                & (t - last_closed > 0.3)
                & stationary
                & record["right_target_active"]
            )
            record["opening_mask"] = (t >= 1) & (gc >= 0.99) & (g > 0.05) & (g < 0.95)
            record["path"] = path
            if record["fit_mask"].sum() < 30:
                raise ValueError(f"Insufficient selected motion: {path}")
            # Match repeated MuJoCo timestep addition, including command arrival
            # at nearly coincident sample/physics times.
            grid, now = [], 0.0
            while now < t[-1]:
                grid.append(now)
                now += self.model.opt.timestep
            record["step_times"] = np.array(grid)
            record["indices"] = np.searchsorted(t, grid, side="right") - 1
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.arm.qpos_adr] = q[0]
            self.data.ctrl[self.arm.act_ids] = q[0]
            for act, qopen, qclosed in self.arm.gripper_acts:
                adr = self.model.jnt_qposadr[self.model.actuator_trnid[act, 0]]
                self.data.qpos[adr] = qclosed + g[0] * (qopen - qclosed)
                self.data.ctrl[act] = self.data.qpos[adr]
            mujoco.mj_forward(self.model, self.data)
            initial = np.empty(mujoco.mj_stateSize(self.model, STATE))
            mujoco.mj_getState(self.model, self.data, initial, STATE)
            record["initial_state"] = initial
            self.episodes[index] = record

    def command_array(self, index, duration, rate):
        key = index, duration
        r = self.episodes[index]
        if key not in self.controls:
            arm = self.arm
            arm.q_cmd = r["right_joints"][0].copy()
            arm.q_goal = arm.q_cmd.copy()
            arm.q_vel = np.zeros(7)
            arm.gripper_cmd = arm.gripper_goal = float(r["right_gripper"][0, 0])
            controls = np.empty((len(r["indices"]), self.model.nu))
            last = None
            for step, sample in enumerate(r["indices"]):
                command = r["right_joints_cmd"][sample]
                if last is None or not np.array_equal(command, last):
                    arm.set_goal(command, duration)
                    last = command
                arm.tick(self.data, self.model.opt.timestep)
                controls[step] = self.data.ctrl
            self.controls[key] = controls
        controls = self.controls[key].copy()
        opened = float(r["right_gripper"][0, 0])
        step = rate * self.model.opt.timestep
        # Opening command is independent of gains/arm dynamics.
        goals = r["right_gripper_cmd"].ravel()
        trajectory = np.empty(len(r["indices"]))
        for i, sample in enumerate(r["indices"]):
            goal = min(1.0, max(0.0, float(goals[sample])))
            opened += min(step, max(-step, goal - opened))
            trajectory[i] = opened
        for act, qopen, qclosed in self.arm.gripper_acts:
            controls[:, act] = qclosed + trajectory * (qopen - qclosed)
        return controls

    def predict(self, index, kp, kv, duration=0.1, rate=5.0):
        r = self.episodes[index]
        acts = self.arm.act_ids
        self.model.actuator_gainprm[acts, 0] = kp
        self.model.actuator_biasprm[acts, 1] = -kp
        self.model.actuator_biasprm[acts, 2] = -kv
        state, _ = rollout.rollout(
            self.model,
            self.data,
            r["initial_state"],
            control=self.command_array(index, duration, rate)[None],
        )
        states = np.vstack([r["initial_state"], state[0]])
        if not np.isfinite(states).all() or np.any(np.diff(states[:, 0]) <= 0):
            raise RuntimeError("Nonfinite or nonadvancing rollout")
        addresses = np.r_[self.arm.qpos_adr, self.arm.finger_adr] + 1
        samples = np.column_stack(
            [
                np.interp(r["time_s"], states[:, 0], states[:, address])
                for address in addresses
            ]
        )
        normalized_gripper = np.clip(
            (states[:, 1 + self.arm.finger_adr] - self.arm.finger_closed)
            / (self.arm.finger_open - self.arm.finger_closed),
            0,
            1,
        )
        samples[:, 7] = np.interp(r["time_s"], states[:, 0], normalized_gripper)
        return samples

    def residual(self, log_parameters, duration):
        kp, kv = np.exp(log_parameters)
        parts = []
        for index in TRAIN:
            r = self.episodes[index]
            error = (self.predict(index, kp, kv, duration)[:, :7] - r["right_joints"])[
                r["fit_mask"]
            ]
            # Equal episode weighting; all seven joints have equal radian weight.
            parts.append(error.ravel() / np.sqrt(error.size))
        return np.concatenate(parts)

    def fit_gains(self, duration):
        results = []
        for start in ([500, 20], [200, 10], [1000, 40]):
            fit = least_squares(
                self.residual,
                np.log(start),
                args=(duration,),
                bounds=(np.log([100, 2]), np.log([1500, 80])),
                diff_step=0.002,
                max_nfev=45,
                ftol=1e-7,
                xtol=1e-6,
                gtol=1e-8,
            )
            results.append(fit)
        best = min(results, key=lambda fit: np.sum(fit.fun**2))
        singular = np.linalg.svd(best.jac, compute_uv=False)
        kp, kv = np.exp(best.x)
        result = {
            "kp": float(kp),
            "kv": float(kv),
            "interpolation_duration_s": duration,
            "training_equal_episode_rmse_rad": float(
                np.sqrt(
                    np.mean(
                        [
                            np.mean(
                                (
                                    self.predict(i, kp, kv, duration)[:, :7]
                                    - self.episodes[i]["right_joints"]
                                )[self.episodes[i]["fit_mask"]]
                                ** 2
                            )
                            for i in TRAIN
                        ]
                    )
                )
            ),
            "local_log_parameter_jacobian_singular_values": singular.tolist(),
            "optimizer_success": bool(best.success),
            "message": str(best.message),
            "active_bounds": best.active_mask.tolist(),
            "starts": [
                {
                    "kp": float(np.exp(f.x[0])),
                    "kv": float(np.exp(f.x[1])),
                    "squared_residual": float(f.fun @ f.fun),
                    "nfev": int(f.nfev),
                }
                for f in results
            ],
        }
        print(json.dumps({"gain_fit": result}), flush=True)
        return result

    def score(self, indices, kp, kv, duration, rate, output=None):
        rows, errors, opening_errors = [], [], []
        for index in indices:
            r = self.episodes[index]
            prediction = self.predict(index, kp, kv, duration, rate)
            error = prediction[:, :7] - r["right_joints"]
            selected = error[r["fit_mask"]]
            opening = (prediction[:, 7] - r["right_gripper"].ravel())[r["opening_mask"]]
            errors.append(selected)
            opening_errors.append(opening)
            rows.append(
                {
                    "episode": index,
                    "selected_samples": len(selected),
                    "selected_rmse_rad": float(np.sqrt(np.mean(selected**2))),
                    "selected_joint_rmse_rad": np.sqrt(
                        np.mean(selected**2, axis=0)
                    ).tolist(),
                    "all_post_initial_second_joint_rmse_rad": np.sqrt(
                        np.mean(error[r["time_s"] >= 1] ** 2, axis=0)
                    ).tolist(),
                    "moving_opening_samples": len(opening),
                    "moving_opening_rmse": float(np.sqrt(np.mean(opening**2)))
                    if len(opening)
                    else None,
                }
            )
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / f"episode_{index:06d}.npz",
                    time_s=r["time_s"],
                    measured_joints=r["right_joints"],
                    simulated_joints=prediction[:, :7],
                    measured_gripper=r["right_gripper"].ravel(),
                    simulated_gripper=prediction[:, 7],
                    arm_fit_mask=r["fit_mask"],
                    opening_fit_mask=r["opening_mask"],
                )
        error, opening = np.concatenate(errors), np.concatenate(opening_errors)
        return {
            "episodes": rows,
            "samples": len(error),
            "pooled_rmse_rad": float(np.sqrt(np.mean(error**2))),
            "pooled_joint_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "equal_episode_rmse_rad": float(
                np.sqrt(np.mean([x["selected_rmse_rad"] ** 2 for x in rows]))
            ),
            "moving_opening_samples": len(opening),
            "moving_opening_rmse": float(np.sqrt(np.mean(opening**2))),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "output/asset-review/hf-sanity-check/native-replay",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = {
        "training_episodes": TRAIN,
        "validation_episodes": VALIDATION,
        "primary_interpolation_duration_s": 0.1,
        "timing_sensitivity_durations_s": [1 / 30, 2 / 15],
        "arm_parameter_bounds": {"kp": [100, 1500], "kv": [2, 80]},
        "opening_rate_bounds_strokes_s": [1, 15],
        "baseline": {"kp": 500, "kv": 20, "opening_rate": 5},
        "selection": "Optimize only episodes 0,2,4. Primary duration fixed at previous 0.1 s assumption; alternative durations diagnose reconstruction uncertainty, not select new runtime timing.",
        "arm_candidate_acceptance": "At least 10% lower pooled selected validation RMSE; no individual joint RMSE worsens by >10%; improve at least 12/15 validation episodes; optimizer converges away from bounds. Integration checks required before any adoption.",
        "opening_candidate_acceptance": "At least 5% lower pooled moving-opening validation RMSE. Sparse transitions limit interpretation; closed/loaded response is not identified.",
        "held_fixed": [
            "Vendor masses and inertia",
            "Joint friction and armature",
            "Force limits",
            "Gravity compensation",
            "Joint offsets",
            "Geometry",
            "Timestamp alignment",
        ],
        "validation_note": "Episodes excluded from this optimization, but previously inspected in earlier release comparisons; not a new blind hardware experiment.",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    experiment = Experiment(args.root)
    print("Prepared native episodes and preregistered fit/validation split", flush=True)

    # Independently compare C++ batched rollouts to the earlier Python replay.
    old = args.output / "runtime-reference/episode_000000.npz"
    replay(args.root / "episode_000000.npz", old.parent, fixture(), grip_rate=5.0)
    reference = np.load(old)
    predicted = experiment.predict(0, 500, 20)
    parity = float(np.max(abs(predicted[:, :7] - reference["simulation_joints"])))
    gripper_parity = float(
        np.max(abs(predicted[:, 7] - reference["simulation_gripper"]))
    )
    if parity > 1e-8 or gripper_parity > 1e-8:
        raise RuntimeError(
            f"Fast replay differs from reference: {parity}, {gripper_parity}"
        )
    print(
        f"Runtime replay parity passed: {parity:g} rad, {gripper_parity:g} opening",
        flush=True,
    )
    primary = experiment.fit_gains(0.1)
    sensitivity = [
        experiment.fit_gains(x) for x in plan["timing_sensitivity_durations_s"]
    ]

    def opening_loss(rate):
        error = []
        for index in TRAIN:
            r = experiment.episodes[index]
            p = experiment.predict(index, 500, 20, rate=rate)
            error.extend((p[:, 7] - r["right_gripper"].ravel())[r["opening_mask"]])
        return float(np.mean(np.square(error)))

    rate_grid = np.linspace(1, 15, 29)
    losses = np.array([opening_loss(float(rate)) for rate in rate_grid])
    best = int(np.argmin(losses))
    interval = (
        float(rate_grid[max(0, best - 1)]),
        float(rate_grid[min(len(rate_grid) - 1, best + 1)]),
    )
    rate_fit = minimize_scalar(
        opening_loss, bounds=interval, method="bounded", options={"xatol": 0.001}
    )
    rate = float(rate_fit.x)
    print(f"Training-selected opening rate: {rate:g} strokes/s", flush=True)
    baseline = experiment.score(
        VALIDATION, 500, 20, 0.1, 5, args.output / "validation-baseline"
    )
    candidate = experiment.score(
        VALIDATION,
        primary["kp"],
        primary["kv"],
        0.1,
        rate,
        args.output / "validation-candidate",
    )
    arm_ratio = candidate["pooled_rmse_rad"] / baseline["pooled_rmse_rad"]
    joint_ratio = (
        np.array(candidate["pooled_joint_rmse_rad"]) / baseline["pooled_joint_rmse_rad"]
    )
    improvements = [
        b["selected_rmse_rad"] < a["selected_rmse_rad"]
        for a, b in zip(baseline["episodes"], candidate["episodes"], strict=True)
    ]
    rate_ratio = candidate["moving_opening_rmse"] / baseline["moving_opening_rmse"]
    # Bootstrap complete validation episodes, never individual correlated rows.
    rng = np.random.default_rng(3110)
    a = np.array(
        [
            r["selected_rmse_rad"] ** 2 * r["selected_samples"]
            for r in baseline["episodes"]
        ]
    )
    b = np.array(
        [
            r["selected_rmse_rad"] ** 2 * r["selected_samples"]
            for r in candidate["episodes"]
        ]
    )
    samples = rng.integers(0, len(a), size=(2000, len(a)))
    ratios = np.sqrt(b[samples].sum(axis=1) / a[samples].sum(axis=1))
    sources = [
        Path(__file__),
        ROOT / "scripts/replay_sophia_dataset_arm.py",
        ROOT / "robot/yor_mujoco.py",
        DESCRIPTION / "sophia.mjcf",
        DESCRIPTION / "sophia.json",
        old,
        *[r["path"] for r in experiment.episodes.values()],
    ]
    report = {
        "plan": plan,
        "primary_gain_fit": primary,
        "timing_sensitivity": sensitivity,
        "gripper_rate_fit": {
            "rate_strokes_s": rate,
            "optimizer_success": bool(rate_fit.success),
            "training_rmse": float(np.sqrt(rate_fit.fun)),
            "grid": [
                {"rate": float(r), "training_rmse": float(np.sqrt(e))}
                for r, e in zip(rate_grid, losses, strict=True)
            ],
        },
        "baseline_validation": baseline,
        "candidate_validation": candidate,
        "arm_validation_rmse_ratio": arm_ratio,
        "arm_per_joint_validation_rmse_ratio": joint_ratio.tolist(),
        "validation_episodes_improved": sum(improvements),
        "episode_bootstrap_rmse_ratio_95pct": np.percentile(
            ratios, [2.5, 97.5]
        ).tolist(),
        "gripper_validation_rmse_ratio": rate_ratio,
        "arm_candidate_passes_statistical_gate": bool(
            arm_ratio <= 0.9
            and np.all(joint_ratio <= 1.1)
            and sum(improvements) >= 12
            and primary["optimizer_success"]
            and not any(primary["active_bounds"])
        ),
        "gripper_candidate_passes_statistical_gate": bool(
            rate_ratio <= 0.95 and rate_fit.success
        ),
        "reference_replay_parity": {
            "max_joint_difference_rad": parity,
            "max_normalized_gripper_difference": gripper_parity,
        },
        "mask": "After 1 s, active right target, measured opening >0.95 and command >=0.99 for >0.3 s; low commanded and encoder-derived base velocity. This reduces but cannot prove absence of external contacts.",
        "fit_sample_counts": {
            str(i): int(r["fit_mask"].sum()) for i, r in experiment.episodes.items()
        },
        "limits": [
            "Effective closed-loop response, conditional on fixed vendor dynamics and input reconstruction; not unique physical gains.",
            "No force, payload mass, independent end-effector pose or independent base motion is observed in these telemetry arrays.",
            "Fixed-base unloaded fixture; selected intervals can still contain unobserved contacts.",
            "Command snapshots at about 30 Hz do not establish exact receipt time or interpolation semantics.",
            "Only three training episodes; bootstrap reflects this validation set, not all possible tasks.",
            "Opening-only identification does not validate left-side or object-constrained closing response.",
        ],
        "simulation_parameters_applied": False,
        "hardware_configuration_modified": False,
        "elapsed_s": time.monotonic() - started,
        "source_sha256": {
            str(p.resolve().relative_to(ROOT)): digest(p) for p in sources
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "arm_validation_rmse_ratio",
                    "validation_episodes_improved",
                    "gripper_validation_rmse_ratio",
                    "arm_candidate_passes_statistical_gate",
                    "gripper_candidate_passes_statistical_gate",
                    "elapsed_s",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
