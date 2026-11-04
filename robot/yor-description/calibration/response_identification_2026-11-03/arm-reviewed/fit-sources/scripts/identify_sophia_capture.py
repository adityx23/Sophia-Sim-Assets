"""Bounded subsystem identification on the 2026-11-02 real robot capture.

Source semantics: CAN gripper feedback is measured; null commands mean disabled.
No mass/contact fitting, no robot commands, and no automatic parameter adoption.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from replay_sophia_dataset_arm import fixture
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import _SimArm

HERE = ROOT / "robot/yor-description"
CAPTURE = HERE / "calibration/teleop_2026-11-02"
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
ARM_TRAIN = tuple(range(5))
ARM_VALIDATION = tuple(range(5, 15))
NECK_TRAIN = tuple(range(16, 21))
NECK_VALIDATION = tuple(range(21, 75))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(index, root=CAPTURE / "native-replay"):
    with np.load(root / f"episode_{index:06d}.npz", allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def stationary(record):
    c, v = record["cmd_vel"], record["enc_vel"]
    return (
        (abs(c[:, :2]).max(axis=1) < 0.02)
        & (abs(c[:, 2]) < 0.05)
        & (abs(v[:, :2]).max(axis=1) < 0.03)
        & (abs(v[:, 2]) < 0.08)
    )


def grid(record, dt):
    times = np.arange(int(np.ceil(record["time_s"][-1] / dt))) * dt
    return times, np.searchsorted(record["time_s"], times, side="right") - 1


def initial_state(model, data):
    mujoco.mj_forward(model, data)
    initial = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, initial, STATE)
    return initial


def simulate(model, initial, controls, times, addresses, finger_limits=None):
    states, _ = rollout.rollout(
        model, mujoco.MjData(model), initial, control=controls[None]
    )
    states = np.vstack([initial, states[0]])
    if not np.isfinite(states).all() or np.any(np.diff(states[:, 0]) <= 0):
        raise RuntimeError("Invalid or nonadvancing simulation rollout")
    samples = []
    for index, address in enumerate(addresses):
        values = states[:, 1 + address]
        if finger_limits is not None and index == len(addresses) - 1:
            closed, opened = finger_limits
            values = np.clip((values - closed) / (opened - closed), 0, 1)
        samples.append(np.interp(times, states[:, 0], values))
    return np.column_stack(samples)


class ArmExperiment:
    def __init__(self, side, indices):
        self.side = side
        self.model = fixture(side=side)
        self.arm = _SimArm(side, self.model, str(HERE / "sophia-ik.mjcf"))
        profile = json.loads((HERE / "sophia.json").read_text())["simulation"]
        side_index = ("left", "right").index(side)
        self.arm.gripper_speed = profile["gripper_speed_strokes_s"][side_index]
        self.arm.gripper_opening_speed = profile["gripper_opening_speed_strokes_s"][
            side_index
        ]
        self.records, self.cache = {}, {}
        for index in indices:
            r = load_episode(index)
            t, q = r["time_s"], r[f"{side}_joints"]
            commands = r[f"{side}_joints_cmd"]
            g, gc = r[f"{side}_gripper"].ravel(), r[f"{side}_gripper_cmd"].ravel()
            if not np.isfinite(np.c_[t, q, commands, g, gc]).all():
                raise ValueError(
                    "Disabled or missing commands: do not forward-fill for fitting"
                )
            opened = (g > 0.95) & (gc >= 0.99)
            last_closed = np.maximum.accumulate(np.where(~opened, t, -10.0))
            r["mask"] = (
                (t >= 1)
                & opened
                & (t - last_closed > 0.3)
                & stationary(r)
                & r[f"{side}_target_active"]
                & np.any(q != 0, axis=1)
            )
            r["opening_mask"] = (t >= 1) & (gc >= 0.99) & (g > 0.05) & (g < 0.95)
            r["grid"], r["indices"] = grid(r, self.model.opt.timestep)
            d = mujoco.MjData(self.model)
            d.qpos[self.arm.qpos_adr] = q[0]
            d.ctrl[self.arm.act_ids] = q[0]
            for act, opened, closed in self.arm.gripper_acts:
                adr = self.model.jnt_qposadr[self.model.actuator_trnid[act, 0]]
                d.qpos[adr] = d.ctrl[act] = closed + g[0] * (opened - closed)
            r["initial"] = initial_state(self.model, d)
            self.records[index] = r

    def controls(self, index, duration, opening_rate):
        key = index, duration
        r, arm, dt = self.records[index], self.arm, self.model.opt.timestep
        if key not in self.cache:
            d = mujoco.MjData(self.model)
            arm.q_cmd = arm.q_goal = r[f"{self.side}_joints"][0].copy()
            arm.q_vel = np.zeros(7)
            arm.gripper_cmd = arm.gripper_goal = float(r[f"{self.side}_gripper"][0, 0])
            controls = np.empty((len(r["indices"]), self.model.nu))
            last = None
            for step, sample in enumerate(r["indices"]):
                command = r[f"{self.side}_joints_cmd"][sample]
                if last is None or not np.array_equal(last, command):
                    arm.set_goal(command, duration)
                    last = command
                arm.tick(d, dt)
                controls[step] = d.ctrl
            self.cache[key] = controls
        controls = self.cache[key].copy()
        g = float(r[f"{self.side}_gripper"][0, 0])
        goals = r[f"{self.side}_gripper_cmd"].ravel()[r["indices"]]
        values = np.empty(len(goals))
        for step, goal in enumerate(goals):
            limit = (opening_rate if goal > g else arm.gripper_speed) * dt
            g += min(limit, max(-limit, goal - g))
            values[step] = g
        for act, opened, closed in arm.gripper_acts:
            controls[:, act] = closed + values * (opened - closed)
        return controls

    def predict(self, index, kp=500.0, kv=20.0, duration=2 / 30, rate=None):
        r, arm, model = self.records[index], self.arm, self.model
        kp, kv = np.asarray(kp, dtype=float), np.asarray(kv, dtype=float)
        rate = arm.gripper_opening_speed if rate is None else rate
        model.actuator_gainprm[arm.act_ids, 0] = kp
        model.actuator_biasprm[arm.act_ids, 1] = -kp
        model.actuator_biasprm[arm.act_ids, 2] = -kv
        result = simulate(
            model,
            r["initial"],
            self.controls(index, duration, rate),
            r["time_s"],
            [*arm.qpos_adr, arm.finger_adr],
            finger_limits=(arm.finger_closed, arm.finger_open),
        )
        return result

    def residual(self, parameters, indices, duration):
        kp, kv = np.exp(parameters)
        errors = []
        for index in indices:
            r = self.records[index]
            if r["mask"].sum() < 15:
                continue
            error = (
                self.predict(index, kp, kv, duration)[:, :7] - r[f"{self.side}_joints"]
            )[r["mask"]]
            errors.append(error.ravel() / np.sqrt(error.size))
        return np.concatenate(errors)

    def fit(self, duration):
        fits = [
            least_squares(
                self.residual,
                np.log(start),
                args=(ARM_TRAIN, duration),
                bounds=(np.log([100, 2]), np.log([1500, 80])),
                diff_step=0.002,
                max_nfev=35,
                ftol=1e-7,
                xtol=1e-6,
                gtol=1e-8,
            )
            for start in ([500, 20], [200, 10], [1000, 40])
        ]
        best = min(fits, key=lambda f: f.fun @ f.fun)
        return {
            "kp": float(np.exp(best.x[0])),
            "kv": float(np.exp(best.x[1])),
            "duration_s": duration,
            "success": bool(best.success),
            "active_bounds": best.active_mask.tolist(),
            "starts": [
                {
                    "kp": float(np.exp(f.x[0])),
                    "kv": float(np.exp(f.x[1])),
                    "loss": float(f.fun @ f.fun),
                    "nfev": f.nfev,
                }
                for f in fits
            ],
        }

    def score(self, indices, parameters, output=None):
        rows, errors = [], []
        for index in indices:
            r = self.records[index]
            prediction = self.predict(index, **parameters)
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / f"episode_{index:06d}.npz",
                    time_s=r["time_s"],
                    measured=r[f"{self.side}_joints"],
                    simulated=prediction[:, :7],
                    commands=r[f"{self.side}_joints_cmd"],
                    fit_mask=r["mask"],
                    gripper_measured=r[f"{self.side}_gripper"],
                    gripper_simulated=prediction[:, 7],
                    opening_mask=r["opening_mask"],
                )
            if r["mask"].sum() < 15:
                continue
            error = (prediction[:, :7] - r[f"{self.side}_joints"])[r["mask"]]
            errors.append(error)
            rows.append(
                {
                    "episode": index,
                    "samples": len(error),
                    "rmse_rad": float(np.sqrt(np.mean(error**2))),
                    "joint_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                }
            )
        all_errors = np.vstack(errors)
        return {
            "episodes": rows,
            "samples": len(all_errors),
            "rmse_rad": float(np.sqrt(np.mean(all_errors**2))),
            "joint_rmse_rad": np.sqrt(np.mean(all_errors**2, axis=0)).tolist(),
        }


def neck_fixture():
    original = ET.parse(HERE / "sophia.mjcf").getroot()
    full = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
    data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, data, full.key("home").id)
    mujoco.mj_forward(full, data)
    root = ET.Element("mujoco", model="Sophia neck response fixture")
    for tag in ("compiler", "option", "default", "asset"):
        root.append(copy.deepcopy(original.find(tag)))
    root.find("compiler").set("meshdir", str(HERE / "assets_nero"))
    body = copy.deepcopy(original.find('.//body[@name="head_pan"]'))
    body.set("childclass", "head")
    body.set("pos", " ".join(map(str, data.xpos[full.body("head_pan").id])))
    body.set("quat", " ".join(map(str, data.xquat[full.body("head_pan").id])))
    for geom in body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    ET.SubElement(root, "worldbody").append(body)
    acts = ET.SubElement(root, "actuator")
    for act in original.find("actuator"):
        if act.get("name", "").startswith("head_"):
            acts.append(copy.deepcopy(act))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class NeckExperiment:
    def __init__(self, indices, source=CAPTURE / "native-replay"):
        self.model = neck_fixture()
        home = json.loads((ROOT / "robot/neck/neck_calibration.json").read_text())[
            "home_deg"
        ]
        self.zero = np.array([home["pan"], home["tilt"]])
        self.records = {}
        for index in indices:
            r = load_episode(index, source)
            r["q"] = np.array([-1, 1]) * np.deg2rad(r["neck_deg"] - self.zero)
            r["goals"] = np.array([-1, 1]) * np.deg2rad(r["neck_cmd_deg"] - self.zero)
            r["grid"], r["indices"] = grid(r, self.model.opt.timestep)
            r["mask"] = (r["time_s"] >= 1) & stationary(r)
            d = mujoco.MjData(self.model)
            d.qpos[:] = d.ctrl[:] = r["q"][0]
            r["initial"] = initial_state(self.model, d)
            self.records[index] = r

    def predict(self, index, rates):
        r, dt = self.records[index], self.model.opt.timestep
        goals = r["goals"][r["indices"]]
        controls = np.empty_like(goals)
        q = r["q"][0].copy()
        step = np.asarray(rates) * dt
        for i, goal in enumerate(goals):
            q += np.minimum(step, np.maximum(-step, goal - q))
            controls[i] = q
        q = simulate(self.model, r["initial"], controls, r["time_s"], [0, 1])
        return self.zero + np.array([-1, 1]) * np.rad2deg(q)

    def score(self, indices, rates, output=None):
        errors, rows = [], []
        for index in indices:
            r = self.records[index]
            prediction = self.predict(index, rates)
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / f"episode_{index:06d}.npz",
                    time_s=r["time_s"],
                    measured=r["neck_deg"],
                    commands=r["neck_cmd_deg"],
                    simulated=prediction,
                    fit_mask=r["mask"],
                )
            if r["mask"].sum() < 15:
                continue
            error = (prediction - r["neck_deg"])[r["mask"]]
            errors.append(error)
            rows.append(
                {
                    "episode": index,
                    "samples": len(error),
                    "rmse_deg": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                }
            )
        errors = np.vstack(errors)
        return {
            "episodes": rows,
            "samples": len(errors),
            "rmse_deg": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsystem", choices=("arm", "neck"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "subsystem": args.subsystem,
        "arm_train": ARM_TRAIN,
        "arm_validation": ARM_VALIDATION,
        "neck_train": NECK_TRAIN,
        "neck_validation": NECK_VALIDATION,
        "arm_preview_s": 2 / 30,
        "preview_basis": "Deployed WebXR bridge default; exact launch override/receipt timestamps not recorded.",
        "gain_bounds": [[100, 2], [1500, 80]],
        "neck_rate_bounds": [0.3, 3.5],
        "gate": "At least 10% pooled improvement, no joint/axis >10% worse, 75% episodes improve, no optimizer bound; independent old-data check before adoption.",
        "mask": "After initialized first second, base stationary by command/encoders; arm additionally engaged and open-command/open-feedback settled >0.3s. Grasp/contact absence is not proven.",
        "fixed": [
            "masses",
            "inertias",
            "friction",
            "force limits",
            "joint offsets",
            "clock alignment",
        ],
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                HERE / "sophia.mjcf",
                HERE / "sophia.json",
                ROOT / "robot/yor_mujoco.py",
                CAPTURE / "MANIFEST.json",
            )
        },
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    if args.subsystem == "arm":
        experiment = ArmExperiment("left", (*ARM_TRAIN, *ARM_VALIDATION))
        fit = experiment.fit(2 / 30)
        print(json.dumps({"fit": fit}), flush=True)
        baseline = experiment.score(ARM_VALIDATION, {}, args.output / "baseline")
        candidate = experiment.score(
            ARM_VALIDATION,
            {"kp": fit["kp"], "kv": fit["kv"]},
            args.output / "candidate",
        )
        ratios = np.asarray(candidate["joint_rmse_rad"]) / baseline["joint_rmse_rad"]
        improved = sum(
            c["rmse_rad"] < b["rmse_rad"]
            for b, c in zip(baseline["episodes"], candidate["episodes"], strict=True)
        )
        passed = (
            candidate["rmse_rad"] <= 0.9 * baseline["rmse_rad"]
            and np.all(ratios <= 1.1)
            and improved >= 0.75 * len(baseline["episodes"])
            and fit["success"]
            and not any(fit["active_bounds"])
        )
        sensitivity = []
        for duration in (0.1, 2 / 15):
            sensitivity.append(experiment.fit(duration))

        # Opening-only candidates are reported separately; object-constrained closing is excluded.
        def opening_loss(rate, indices):
            errors = []
            for index in indices:
                r = experiment.records[index]
                if r["opening_mask"].any():
                    p = experiment.predict(index, rate=rate)[:, 7]
                    errors.extend((p - r["left_gripper"].ravel())[r["opening_mask"]])
            return float(np.mean(np.square(errors))), len(errors)

        grid_rates = np.linspace(1, 20, 20)
        losses = [opening_loss(rate, ARM_TRAIN)[0] for rate in grid_rates]
        best = int(np.argmin(losses))
        grip = minimize_scalar(
            lambda rate: opening_loss(rate, ARM_TRAIN)[0],
            method="bounded",
            bounds=(
                grid_rates[max(0, best - 1)],
                grid_rates[min(len(grid_rates) - 1, best + 1)],
            ),
        )
        gripper = {
            "candidate_rate": float(grip.x),
            "training_samples": opening_loss(grip.x, ARM_TRAIN)[1],
            "baseline_validation": opening_loss(5.0, ARM_VALIDATION),
            "candidate_validation": opening_loss(grip.x, ARM_VALIDATION),
            "adopted": False,
            "reason": "Sparse moving-opening samples; examine transition/episode support before any adoption.",
        }
        result = {
            "fit": fit,
            "baseline_validation": baseline,
            "candidate_validation": candidate,
            "joint_ratios": ratios.tolist(),
            "episodes_improved": int(improved),
            "candidate_passes_gate": bool(passed),
            "timing_sensitivity": sensitivity,
            "left_gripper": gripper,
        }
    else:
        experiment = NeckExperiment((*NECK_TRAIN, *NECK_VALIDATION))

        def residual(log_rates):
            rates = np.exp(log_rates)
            errors = []
            for index in NECK_TRAIN:
                r = experiment.records[index]
                if r["mask"].sum() < 15:
                    continue
                error = (experiment.predict(index, rates) - r["neck_deg"])[r["mask"]]
                errors.append(error.ravel() / np.sqrt(error.size))
            return np.concatenate(errors)

        fits = [
            least_squares(
                residual,
                np.log(start),
                bounds=(np.log([0.3, 0.3]), np.log([3.5, 3.5])),
                diff_step=0.01,
                max_nfev=35,
                ftol=1e-6,
                xtol=1e-5,
                gtol=1e-7,
            )
            for start in ([1, 2], [2, 3])
        ]
        best = min(fits, key=lambda f: f.fun @ f.fun)
        rates = np.exp(best.x)
        print(json.dumps({"neck_rates": rates.tolist()}), flush=True)
        baseline = experiment.score(NECK_VALIDATION, [1, 2], args.output / "baseline")
        candidate = experiment.score(NECK_VALIDATION, rates, args.output / "candidate")
        ratio = np.asarray(candidate["rmse_deg"]) / baseline["rmse_deg"]
        overall = float(
            np.linalg.norm(candidate["rmse_deg"]) / np.linalg.norm(baseline["rmse_deg"])
        )
        improved = sum(
            np.linalg.norm(c["rmse_deg"]) < np.linalg.norm(b["rmse_deg"])
            for b, c in zip(baseline["episodes"], candidate["episodes"], strict=True)
        )
        result = {
            "candidate_rates_rad_s": rates.tolist(),
            "baseline_validation": baseline,
            "candidate_validation": candidate,
            "axis_ratios": ratio.tolist(),
            "overall_ratio": overall,
            "episodes_improved": int(improved),
            "active_bounds": best.active_mask.tolist(),
            "candidate_passes_gate": bool(
                overall <= 0.9
                and np.all(ratio <= 1.1)
                and best.success
                and not any(best.active_mask)
                and improved >= 0.75 * len(baseline["episodes"])
            ),
        }
    result.update(
        elapsed_s=time.monotonic() - started, applied=False, hardware_calibrated=False
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "baseline_validation",
                    "candidate_validation",
                    "timing_sensitivity",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
