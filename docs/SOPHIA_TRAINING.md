# Sophia policy interface and tabletop training environment

For unloaded base-and-arm trajectory comparison before policy deployment, see
[SOPHIA_REPLAY.md](SOPHIA_REPLAY.md). That workstation pipeline is separate from
the single-arm policy action contract and does not modify the joystick controller.

Phases 1 and 2 provide a shared single-arm policy interface and a Gymnasium
pick-and-place task. These are offline software foundations. No robot process
is launched or configured, and no trained checkpoint or hardware transfer claim
is included. Phase 3 calibration and phase 4 policy training remain separate.

## Install and run

From the YOR directory, use the existing Python 3.11 environment:

```bash
uv pip install --python .venv-sophia/bin/python --require-hashes -r requirements-sophia-training.txt
MUJOCO_GL=egl .venv-sophia/bin/python scripts/check_sophia.py --training --jobs 3 --output output/asset-review/sophia-training-2026-11-03/acceptance
```

The optional training lock adds Gymnasium and its dependencies while constraining
every existing simulation dependency to `requirements-sophia.txt`. The original
simulation-only installation and acceptance command still work without Gymnasium.
The full training suite includes the original simulation checks, interface and
environment regressions, and three randomized task executions.

```python
import gymnasium as gym
import numpy as np

env = gym.make("robot.sophia_env:SophiaPickPlace-v0", render_mode="rgb_array")
try:
    observation, info = env.reset(seed=7)
    # Hold the initial arm target and keep the gripper open.
    action = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    preview = env.render()  # spectator view; not a policy observation
finally:
    env.close()
```

`MUJOCO_GL=egl` must be set before importing MuJoCo for headless RGB rendering.
The environment uses CPU physics with no real-time sleeps or physics worker.
Independent environments have independent controllers and temporary scenes;
Gymnasium vector environments can own separate instances. GPU batching and
training-throughput optimization are not part of this milestone.

## Policy contract: sophia-single-arm-v1

`robot/sophia_policy.py` contains the simulator-independent conversion code. The
initial task uses the left arm, whose response was fitted to the recorded robot
data. The adapter also supports a separately fingerprinted right-arm contract.

| Field | Definition |
|---|---|
| Action | Float array of length 8: seven normalized joint increments, followed by absolute gripper opening |
| Joint order | `left_arm_joint1` through `left_arm_joint7` |
| Joint increments | Each value in `[-1, 1]`, multiplied by 0.02 rad at the default 30 Hz / 0.6 rad/s setting |
| Increment reference | Previous dispatched joint target, initialized from measured joints at reset; not Cartesian coordinates and not a fresh measured-pose delta every step |
| Absolute target limits | Clipped to the model's joint limits; clipping reported in `info` |
| Gripper | `0 = closed`, `1 = open`, matching `CanGripper`; never convert normalized commands to the 101.6 mm visual gap before dispatch |
| Dispatch | Existing `set_left_joint_target(joint_target, gripper_target, preview_time=1/30)` RPC signature |
| Physics timing | Accumulated 30 Hz deadlines over 5 ms physics ticks; timing error stays below one physics tick instead of accumulating |
| Other subsystems | Base velocity zero, lift held at nominal 0.3 m extension, neck and right-arm targets held constant; no policy actions for these subsystems yet |

Both backends use the same `prepare()` conversion. `dispatch()` calls a supplied,
already-connected client; constructing an adapter never opens a connection or
homes the robot. A simulator `False` result rejects the action without advancing
the stored target. A hardware RPC `None` result is recorded as `sent_unconfirmed`,
not proof that the robot executed the command. Exceptions propagate without
committing a new target. The hardware scheduler, watchdog, operator interlock and
actual execution checks belong to deployment integration, not this adapter.

Save `adapter.contract.describe()` and `adapter.contract.fingerprint` with any
future checkpoint. Side, joint limits, rate, scaling, image shape and conventions
are included in the fingerprint. Reject a checkpoint/adapter mismatch rather
than interpreting its actions under different settings.

The legacy policy packet in the main README has different delta semantics.
Existing absolute-joint or measured-relative delta packets are not v1 actions.
The legacy gripper documentation has been corrected to the driver's actual
closed/open convention. Hardware scripts and driver behavior are unchanged.

## Observations and camera handling

| Observation | Shape / type | Contents |
|---|---|---|
| `proprio` | `(15,)`, float32 | Measured joint angles in radians [0:7], measured normalized gripper opening [7], previous dispatched joint targets [8:15] |
| `gripper_valid` | `(1,)`, int8 | Whether the gripper observation is measured and valid |
| `head_rgb` | `(90,160,3)`, uint8 | Rectified head ZED left view, RGB |
| `wrist_rgb` | `(90,160,3)`, uint8 | Left wrist view with the supplied lens distortion, RGB |

`SophiaPolicyAdapter.observe(packet, images)` consumes the existing telemetry
keys `left_joints` and `left_gripper`, and image keys `head_zed_left` and
`left_wrist`. Image acquisition and synchronization remain the caller's job;
the adapter does not treat separately received frames as synchronized evidence.
It uses the same aspect-preserving resize and centered black padding for real
frames and simulation renders. Pass `color_order="BGR"` explicitly for OpenCV
frames. Both selected frames must be present when image observations are requested.

Missing gripper feedback becomes a zero placeholder with validity 0; a commanded
position is never substituted as a measurement. Joint feedback must be finite.
The observation regression tests include rows from the archived real teleoperation
capture. Use genuinely measured feedback from that stream or an equivalent robot
publisher; some legacy getters may fall back to their last command.

The actor receives no exact object pose, contacts, world camera transform, perfect
base localization or reward state. Privileged quantities are confined to reward
calculation and `info["privileged_metrics"]`. Do not feed that dictionary into a
deployed policy. The `observation_mode="proprio"` option is for fast diagnostics;
it cannot observe randomized object/target positions and is not a substitute for
the visual policy interface. Camera extrinsics remain the documented estimates;
this work does not establish a new optical calibration.

## Task, resets and scoring

The left arm moves a free 25 g rectangular object from the table onto a green
painted target. Object/goal XY positions are independently randomized by up to
20 mm around the nominal setup. The robot keeps its floating base and ordinary
lift/arm physics; these parts are held by controller targets, not welded in place.
The free object is never attached to the gripper or repositioned during a rollout.

`reset(seed=...)` restores physical and controller state, opens the grippers,
places the left arm in a reachable pregrasp posture, points the head toward the
table, samples the task and settles for 0.5 simulated seconds. Episode time then
starts at zero. Same seed and actions reproduce the same rollout. Teleportation
is limited to simulation reset; a real deployment will need an explicit setup
procedure. Optional `object_xy` and `goal_xy` reset overrides are checked for
workspace bounds and at least 100 mm pickup-to-goal separation.

Success requires all of the following:

- A previous lift of at least 45 mm with bilateral finger contact and no table support.
- Object center within 35 mm of the goal, resting on the table near its expected height.
- Object speed below 20 mm/s, gripper opening above 0.8, no finger contact, and at least 70 mm gripper clearance.
- Those placement conditions sustained for 0.3 seconds.

Pushing the object onto the goal, hovering above it or retaining it in the gripper
does not count. Command rejection, self-collision guard intervention, object drop,
robot/table penetration above 2 mm, base displacement above 25 mm or numerical
instability ends the episode as a failure. The default 900-step time limit is a
truncation, separate from task termination. Call reset after either outcome.

The dense reward is the change in a bounded reach/grasp/lift/placement potential,
minus 0.001 per step, plus 10 on success or minus 1 on failure. Task poses and
contacts are available to the training reward but not the actor. Reward design
and policy learning performance still require evaluation in the training phase.

## Inspect the task without training a policy

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/run_sophia_pick_place.py --episodes 3 --rgb-observations --record --output output/asset-review/sophia-training-2026-11-03/demo
```

This controller is a privileged simulation-only IK oracle. It uses true simulated
grasp/object positions to choose joint targets, then goes through the same bounded
eight-value action interface as a policy. It demonstrates task reachability,
controller integration and frictional object transport. It is not a trained policy
and cannot be deployed unchanged. Outputs include `policy-contract.json`, a
source-hashed `task-validation.json` and, with `--record`, `pick-place-oracle.mp4`.

The Gymnasium API follows the upstream [environment contract](https://gymnasium.farama.org/api/env/).
Its checker, real-telemetry interface tests, deterministic-reset tests, RGB checks,
failure/timeout tests and independent randomized oracle episodes are part of
`check_sophia.py --training`.
