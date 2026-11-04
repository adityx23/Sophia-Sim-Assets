# Sophia workstation replay and comparison pipeline

For live simulation teleoperation with the physical robot following, see
[Sophia live follow](SOPHIA_LIVE_FOLLOW.md). This document describes the separate
fixed-trajectory replay and comparison tools.

This pipeline generates a shared command file, executes it in local MuJoCo,
records incoming telemetry through a passive ZMQ subscriber, and compares an
imported recording against the simulation. There is no hardware-command mode,
no Pi deployment, and no modification to `robot/teleop/joystick.py`.

The initial sequence is object-free: hold, drive forward with smooth ramps, stop,
then execute an unloaded left-arm reach/lower/close/lift/transfer/lower/open/retract
motion and return to the starting posture. Lift, neck and right-arm targets stay
fixed. This measures unloaded motion and timing, not grasp force or contact physics.

## Setup and complete local check

From the YOR directory:

```bash
uv pip install --python .venv-sophia/bin/python --require-hashes -r requirements-sophia-replay.txt
MUJOCO_GL=egl .venv-sophia/bin/python scripts/check_sophia_replay.py --video --output output/asset-review/my-replay-check
```

Use a new output directory for each run. The optional replay lock adds Matplotlib
for standalone plots while keeping the existing simulation/training dependencies
pinned. FFmpeg/ffprobe are optional system tools used to produce and verify H.264
video; the simulator also writes an OpenCV MP4 when `--video` is selected.

The completed local check is under
`output/asset-review/sophia-replay-2026-11-03-final/`:

| File | Purpose |
|---|---|
| `trajectory.json` | Versioned command sequence, initial conditions and content hash |
| `simulation/recording.json` | Actual simulated commands, dispatch times, measured states and source metadata |
| `simulation/physics-states.npz` | Recorded MuJoCo states for inspection; no state forcing during replay |
| `replay-preview.mp4` | Approximately 33-second unloaded replay video |
| `self-check/report.html` | Same-run comparison: software check, not real-robot evidence |
| `injected-error-check/report.html` | Synthetic 0.02 rad joint bias and 20 mm lateral drift recovered by the comparator |
| `pipeline-status.json` | Dependency/source hashes, test results and protected hardware-file hashes |

The reference has 990 commands at 30 Hz and a 32.97-second command horizon. Its
forward velocity command integrates to 0.25 m; actual simulated displacement is
recorded separately (about 0.230 m in the validated run). The robot is driven
through ordinary swerve and joint controllers, so command integration is not
silently substituted for physical travel.

## Run individual stages

```bash
# Generate only; never issues hardware commands.
.venv-sophia/bin/python scripts/sophia_replay.py generate --output output/asset-review/new-reference/trajectory.json

# Local physics replay, optionally with a video.
MUJOCO_GL=egl .venv-sophia/bin/python scripts/sophia_replay.py simulate --trajectory output/asset-review/new-reference/trajectory.json --output output/asset-review/new-reference/simulation --video
```

The generator uses the current model to solve arm waypoints once. It exports
absolute joint targets and base velocity commands. Replay does not run Cartesian
IK, track objects or correct the base to force a desired displacement. The same
command file can therefore be consumed by an independently provided robot-side
runner later; this workstation implementation does not provide that runner.

## Trajectory contract

Schema: `sophia.replay.trajectory.v1`. Its SHA256 is over canonical JSON excluding
the `sha256` field. Changing any command or initial condition requires a new hash.
The simulator also checks that the referenced model/configuration hashes match.

| Field | Meaning |
|---|---|
| `frames[].seq` / `time_s` | Contiguous sequence starting at zero; elapsed planned time on the 30 Hz grid |
| `frames[].base_velocity` | `[vx, vy, yaw_rate]`, body frame, x forward/y left/yaw counterclockwise, m/s and rad/s |
| `frames[].left_joints` | Seven absolute joint targets in radians, joint1 through joint7 |
| `frames[].left_gripper` | Absolute normalized opening: 0 closed, 1 open |
| `frames[].preview_time_s` | Arm interpolation duration, 1/30 second |
| `frames[].stage` | Human-readable segment label |
| `initial` | Required arm/gripper targets, physical lift extension plus its assumed encoder mapping, and neck configuration |

The initial version limits forward commands to 0.05 m/s, commanded base
acceleration to 0.1 m/s², arm target velocity to 0.5 rad/s and target acceleration
to 1.5 rad/s². These are chosen bounds for this sequence, not newly measured
hardware limits. Base commands begin/end at zero, and the arm returns to its
initial target with the gripper open. Simulator initialization sets the pose and
settles for one second before starting the replay clock; that is not a robot
homing procedure. A later physical run needs a separately established start state.

## Passive recording

The subscriber requires an explicit endpoint. For example, to capture a stream
forwarded to local port 5558:

```bash
.venv-sophia/bin/python scripts/sophia_replay.py listen --endpoint tcp://127.0.0.1:5558 --duration 120 --output output/asset-review/replay-capture
```

This command only subscribes. It does not start motion, initialize the robot,
open an RPC client or publish commands. No live capture was started by the
workstation implementation; the transport tests use an isolated loopback publisher.
Stop a capture with Ctrl-C, SIGTERM or a `STOP` file inside its output directory.
The listener updates `status.json` and preserves exact message bytes in
`packets.jsonl` alongside parsed payloads and receiver timestamps. Zero packets is
reported as no measurement. PUB/SUB capture can lose packets; keep any source-side
recording as the authoritative copy when available.

It accepts these two-part `[topic, JSON object]` messages:

| Topic | Payload |
|---|---|
| `sophia/replay/meta` | Recording metadata below, including `run_id` |
| `sophia/replay/command` | One trajectory frame plus `run_id`, actual `dispatch_time_s` and `status` |
| `sophia/replay/state` | One measured sample below plus `run_id` |
| `sophia/replay/event` | `run_id`, `event`, `time_s`; completion also supplies `sample_count` |
| `teleop/episode`, `episode/signal`, `teleop/episode_result` | Existing stream preserved for observation-only import |

`status` is `accepted` or `sent_unconfirmed`; neither is evidence of measured
execution without feedback. Events include `start`, `complete`, `manual_takeover`,
`aborted` and `fault`. An interruption remains recorded even if a later completion
event appears. Commands and states have separate contiguous sequence counters.

## Recording and clock contract

Schema: `sophia.replay.recording.v1`. The generated
`simulation/recording.json` is a complete concrete example of the file structure.
The canonical importer assembles the same structure from topic messages.

Required metadata: `schema`, `run_id`, `trajectory_sha256`, `source` (`robot`,
`simulation` or `synthetic`), `clock="seconds_from_run_start"`, `pose_frame`,
`pose_source`, and measured `initial_state`. The importer adds `complete`,
`end_reason`, `commands`, `samples` and `events` from the capture.

`initial_state` includes `left_joints`, `right_joints`, `left_gripper`,
`lift_extension_m`, `neck_deg` (calibrated hardware API convention), and
`steer_rad`. Comparison checks the two runs' initial measurements within 0.05 rad
for arm/steering angles, 0.1 normalized opening, 5 mm lift extension and 2 degrees
neck angle. These are initial engineering setup tolerances, not accuracy claims.
Missing initial-state measurements prevent an unqualified matched-run result.

Each sample has:

- `seq`, `time_s`: measurement sequence and elapsed **source** snapshot time.
- `left_joints`: seven measured radians, or null if unavailable.
- `left_gripper`: measured normalized opening, or null; no commanded-position fallback.
- `base_pose`: base-center `[x,y,yaw]` in a fixed local world frame, or null.
- `base_pose_time_s`: the pose sensor's own sample time expressed in the same run clock.
- Optional `base_velocity_body`, `steer_rad`, `drive_m_s`: wheel-derived body velocity and four module measurements in FL/FR/RR/RL order.

The pose frame identifier is `local_world_base_center_xyyaw_m_rad`. Real pose
sources may be `zed_base_pose` or `external_base_tracking`. A raw ZED camera pose
must first be transformed to the base center with consistent axes; it must not
be relabeled as base pose. Encoder-integrated odometry is not independent physical
travel evidence and is excluded from the base-position comparison. Wheel-derived
velocities remain useful and are plotted separately.

Use one run start and a common source time domain. If the pose source uses another
clock, establish and record its clock conversion before supplying these samples.
Receiver arrival time is preserved for diagnostics but never used to fabricate
sensor timestamps or command dispatch times. Repeated pose timestamps with the
same value are stale samples, not fresh observations. Conflicting reused timestamps
are rejected. Comparison does not estimate away delay by shifting or warping time.

## Import and compare

```bash
.venv-sophia/bin/python scripts/sophia_replay.py import --packets output/asset-review/replay-capture/packets.jsonl --run-id YOUR_RUN_ID --trajectory output/asset-review/new-reference/trajectory.json --output output/asset-review/robot-replay.json

.venv-sophia/bin/python scripts/sophia_replay.py compare --trajectory output/asset-review/new-reference/trajectory.json --reference output/asset-review/new-reference/simulation/recording.json --observed output/asset-review/robot-replay.json --output output/asset-review/robot-replay-comparison
```

The comparator verifies trajectory identity, dispatch values/order, initial state,
completion/intervention events, sample gaps and coverage. It reports joint/gripper
errors, independent base displacement and heading errors when available, wheel
response and dispatch timing. Outputs are `comparison.json`, `aligned-samples.csv`,
`comparison.png` and `report.html`.

Only initial translation and heading are removed from base poses. No fitted scale,
endpoint correction, per-joint time shift, long-gap interpolation or extrapolation
is applied. At least 95% coverage is required for a qualified data comparison;
this is a data-quality gate, not an accuracy pass threshold. The report always
retains `hardware_validated=false`: a dataset comparison alone does not establish
grasp performance, safe deployment or policy transfer.

For an existing `frames.jsonl` / `.jsonl.gz` file, `import-teleop` accepts
`--frames`, `--episode-id`, an explicit `--start-timestamp` in the publisher's
Unix clock, `--trajectory` and `--output`. It preserves the observed joint/gripper
feedback and wheel-derived body velocity. It labels the association as supplied
by the caller, leaves independent pose unavailable and does not invent missing
dispatch records or completion events. These results stay diagnostic-only.

The present deliverable is the complete workstation pipeline and its simulation
reference. Real response comparison remains pending a robot recording; no changes
to the joystick controller are included or proposed here.
