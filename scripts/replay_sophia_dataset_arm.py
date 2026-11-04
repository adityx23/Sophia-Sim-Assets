"""Benchmark the unloaded right arm against native recorded targets/feedback.

The fixed mount follows Sophia's CAD orientation. Scene contacts/base movement
are excluded to isolate actuator response; recorded grasped objects are unknown.
Target timestamps are snapshots, not exact controller receipt times. Assume the
driver's 0.1 s interpolation duration and exclude the first initialized second.
"""

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "robot/yor-description"
sys.path.insert(0, str(ROOT))
from robot.yor_mujoco import _SimArm


def fixture(kp=None, kv=None, gravcomp=None, side="right"):
    if side not in ("left", "right"):
        raise ValueError("Arm side must be left or right")
    original = ET.parse(HERE / "sophia.mjcf").getroot()
    full = mujoco.MjModel.from_xml_path(str(HERE / "scene-sophia.mjcf"))
    data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, data, full.key("home").id)
    mujoco.mj_forward(full, data)
    root = ET.Element("mujoco", model=f"Sophia {side} arm response fixture")
    for tag in ("compiler", "option", "default", "asset"):
        root.append(copy.deepcopy(original.find(tag)))
    root.find("compiler").set("meshdir", str(HERE / "assets_nero"))
    world = ET.SubElement(root, "worldbody")
    arm = copy.deepcopy(original.find(f'.//body[@name="{side}_arm_base_link"]'))
    body = full.body(f"{side}_arm_base_link").id
    arm.set("pos", " ".join(map(str, data.xpos[body])))
    for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        arm.attrib.pop(key, None)
    arm.set("quat", " ".join(map(str, data.xquat[body])))
    world.append(arm)
    for geom in arm.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    if gravcomp is not None:
        for body in arm.iter("body"):
            body.set("gravcomp", str(gravcomp))
    acts = ET.SubElement(root, "actuator")
    for actuator in original.find("actuator"):
        if actuator.get("name", "").startswith(f"{side}_"):
            actuator = copy.deepcopy(actuator)
            if "arm_joint" in actuator.get("name"):
                if kp is not None:
                    actuator.set("kp", str(kp))
                if kv is not None:
                    actuator.set("kv", str(kv))
            acts.append(actuator)
    eq = ET.SubElement(root, "equality")
    eq.append(copy.deepcopy(original.find(f'.//joint[@name="{side}_jaw_coupling"]')))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def replay(path, output, model, grip_rate=None, closing_rate=None):
    r = np.load(path, allow_pickle=False)
    t, measured, commands = r["time_s"], r["right_joints"], r["right_joints_cmd"]
    grip, grip_cmd = r["right_gripper"].ravel(), r["right_gripper_cmd"].ravel()
    if (
        np.any(np.diff(t) <= 0)
        or not np.isfinite(np.c_[t, measured, commands, grip, grip_cmd]).all()
    ):
        raise ValueError(f"Invalid native timestamps/signals in {path}")
    data = mujoco.MjData(model)
    arm = _SimArm("right", model, str(HERE / "sophia-ik.mjcf"))
    # Explicit temporary benchmark override; no source or hardware mutation.
    profile = json.loads((HERE / "sophia.json").read_text())["simulation"]
    arm.gripper_speed = profile["gripper_speed_strokes_s"][1] if grip_rate is None else grip_rate
    arm.gripper_opening_speed = (
        profile.get("gripper_opening_speed_strokes_s", profile["gripper_speed_strokes_s"])[1]
        if grip_rate is None else grip_rate
    )
    if closing_rate is not None:
        arm.gripper_speed = closing_rate
    arm.q_cmd = arm.q_goal = measured[0].copy()
    data.qpos[arm.qpos_adr] = measured[0]
    data.ctrl[arm.act_ids] = measured[0]
    for act, opened, closed in arm.gripper_acts:
        data.qpos[model.jnt_qposadr[model.actuator_trnid[act, 0]]] = closed + grip[
            0
        ] * (opened - closed)
        data.ctrl[act] = closed + grip[0] * (opened - closed)
    arm.gripper_cmd = arm.gripper_goal = float(grip[0])
    mujoco.mj_forward(model, data)
    sample = np.empty((len(t), 8))
    sample[0] = np.r_[measured[0], grip[0]]
    index, target = 0, 1
    last_command = None
    previous_time, previous = 0.0, sample[0].copy()
    while data.time < t[-1]:
        while index + 1 < len(t) and t[index + 1] <= data.time:
            index += 1
        if last_command is None or not np.array_equal(last_command, commands[index]):
            arm.set_goal(commands[index], 0.1)
            last_command = commands[index].copy()
        arm.gripper_goal = float(np.clip(grip_cmd[index], 0, 1))
        arm.tick(data, model.opt.timestep)
        mujoco.mj_step(model, data)
        if data.warning.number.any() or not np.isfinite(data.qpos).all():
            raise RuntimeError("Unstable arm replay")
        state = np.r_[data.qpos[arm.qpos_adr], arm.gripper_position(data)]
        while target < len(t) and t[target] <= data.time:
            alpha = (t[target] - previous_time) / (data.time - previous_time)
            sample[target] = previous + alpha * (state - previous)
            target += 1
        previous_time, previous = float(data.time), state
    valid = t >= 1
    error = sample[valid, :7] - measured[valid]
    # Opening is not constrained by an unknown held object. Closing errors are
    # retained separately and never treated as an identified free-space response.
    opening = valid & (grip_cmd >= 0.99)
    moving_opening = opening & (grip > 0.05) & (grip < 0.95)
    gm = moving_opening if moving_opening.any() else opening
    ge = sample[gm, 7] - grip[gm]
    row = {
        "episode": path.stem,
        "samples": int(valid.sum()),
        "arm_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "arm_bias_rad": np.mean(error, axis=0).tolist(),
        "arm_p95_abs_rad": np.percentile(abs(error), 95, axis=0).tolist(),
        "gripper_opening_motion_samples": int(gm.sum()),
        "gripper_opening_rmse_normalized": float(np.sqrt(np.mean(ge**2))),
        "gripper_all_rmse_normalized": float(
            np.sqrt(np.mean((sample[valid, 7] - grip[valid]) ** 2))
        ),
        "warnings": data.warning.number.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / path.name,
        time_s=t,
        hardware_joints=measured,
        command_joints=commands,
        simulation_joints=sample[:, :7],
        hardware_gripper=grip,
        command_gripper=grip_cmd,
        simulation_gripper=sample[:, 7],
        gripper_fit_mask=gm,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kv", type=float)
    parser.add_argument("--gravcomp", type=float, choices=(0.0, 1.0))
    parser.add_argument("--gripper-rate", type=float)
    parser.add_argument("--closing-rate", type=float)
    args = parser.parse_args()
    for rate in (args.gripper_rate, args.closing_rate):
        if rate is not None and (rate <= 0 or not np.isfinite(rate)):
            raise ValueError("Gripper rates must be finite and positive")
    model = fixture(args.kp, args.kv, args.gravcomp)
    paths = sorted((args.root / "native-replay").glob("*.npz"))
    if args.episodes is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in args.episodes]
    if not paths:
        raise ValueError("No episodes")
    rows = []
    for p in paths:
        row = replay(p, args.output, model, args.gripper_rate, args.closing_rate)
        rows.append(row)
        print(json.dumps(row), flush=True)
    weights = np.array([r["samples"] for r in rows])
    pooled = np.sqrt(
        np.average(
            np.array([r["arm_rmse_rad"] for r in rows]) ** 2, weights=weights, axis=0
        )
    )
    report = {
        "episodes": rows,
        "pooled_arm_rmse_rad": pooled.tolist(),
        "method": __doc__,
        "temporary_parameters": {
            "kp": args.kp,
            "kv": args.kv,
            "gravcomp": args.gravcomp,
            "gripper_rate_strokes_s": args.gripper_rate,
            "closing_rate_strokes_s": args.closing_rate,
        },
        "hardware_calibrated": False,
        "source_sha256": {
            str(p.resolve().relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__),
                HERE / "sophia.mjcf",
                ROOT / "robot/yor_mujoco.py",
                *paths,
            ]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Pooled arm RMSE:", pooled.tolist(), flush=True)


if __name__ == "__main__":
    main()
