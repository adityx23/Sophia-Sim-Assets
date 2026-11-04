# Live simulation teleoperation with Sophia following

The existing WebXR page and `WholeBodyTeleop` controller drive the updated Sophia
MuJoCo model. A `LiveLink` adapter replaces the bridge's ordinary robot link for
this session. It solves arm IK once in simulation, bounds the resulting joint
targets, checks the simulator's collision guard, and sends the accepted targets
over SSH to a small endpoint on the Pi. The endpoint calls the existing local
YOR RPC service. Both arms, both AgileX grippers, base, neck and lift are supported.

```text
Quest -> existing VR controller -> MuJoCo IK / command limits / collision checks
                                    |                         |
                              simulated response        accepted commands
                                                              |
                                                       authenticated SSH
                                                              |
                                                 Pi endpoint -> yor.py -> robot
                                                              |
                                           actual dispatch + measured feedback
                                                              |
                                              paired workstation + Pi recording
```

`robot/teleop/joystick.py` and the Pi's existing hardware drivers are unchanged.
This is a live command-following experiment, not deployment of a trained policy.
Simulation collision clearance alone does not establish real-world clearance.

## Installed state and launch

The Pi endpoint is installed in `/home/yor/sophia-live-2026-11-03-r4`, with a verified
source/contract manifest. It uses `/home/yor/miniforge3/envs/yor-nero/bin/python`
(observed NumPy 1.26.4 and commlink 0.4.0). The workstation uses the separately
locked `requirements-sophia-live.txt`. No physical motions were executed during setup.

During setup the Pi was already running `robot/yor.py` and
`robot/whole_body_teleop.py`. **Before a live session, turn existing teleop off,
then stop its `whole_body_teleop.py` process in its terminal. Keep `yor.py` running.**
The new endpoint refuses motion mode while a known competing teleop writer is
running. Do not run `joystick.py` alongside this session: the new endpoint reads
the joystick directly as a stop input, without editing that file.

Start from YOR on this workstation:

```bash
bash scripts/run_sophia_live.sh --ip 192.168.123.89
```

The wrapper uses headless rendering with EGL by default, so it works from a
terminal without a desktop display. Stereo headset video remains enabled. If
your shell already sets `MUJOCO_GL` to another backend, explicitly use
`MUJOCO_GL=egl bash scripts/run_sophia_live.sh --ip 192.168.123.89`.

The launcher prints the workstation's headset URLs. On the Quest, first open
`https://WORKSTATION_IP:8444/` and accept its local video certificate, then open
`https://WORKSTATION_IP:8443/?tilt=0` and enter VR. `/monitor` shows the real feedback,
simulated feedback, and follow status. A second controller connection is rejected;
video-only viewers can still connect.

### Workcell and looking down

The default scene contains only an empty floor. The existing mobile-manipulation
demo also supplies two tables, two free parcels and floor markings. Build just
that environment, without starting its scripted task:

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/render_sophia_demo.py \
  --scene-only --output output/sophia-vr-workcell
```

After stopping the current launcher, select it for live following:

```bash
bash scripts/run_sophia_live.sh --ip 192.168.123.89 \
  --model output/sophia-vr-workcell/demo-scene.mjcf
```

For simulation only, use the existing launcher instead; it does not connect to
the Pi or move the physical robot:

```bash
MUJOCO_GL=egl .venv-sophia/bin/python robot/sim_vr_teleop.py --headless \
  --ip 192.168.123.89 --model output/sophia-vr-workcell/demo-scene.mjcf
```

Run only one launcher at a time because they share headset/video ports. The
workcell is virtual furniture, not a reconstruction of the robot's physical room.
Live following still moves the real robot when enabled.

The headset shows the simulated ZED stereo cameras, with their calibrated image
field of view (about 53 degrees vertically); it is not a full 360-degree avatar
view. In the recorded `20261103-180857` arm/lift pose, the arms enter the image at
roughly 50 degrees down and the shoulder plate is visible around 65 degrees.
They are outside the camera image at the roughly 28-degree tilt reached there.

Live following now captures your comfortable headset pitch on each enable,
just as it captures yaw. Neutral pitch now maps to the calibrated yor.py neck
home, with zero additional viewing offset. `--view-down-deg 30` optionally
selects the previous downward working view; the default is 0. These are
operator view preferences, not changes to the physical mount or calibration.

Open `https://192.168.123.89:8443/?tilt=0` to clear the old URL/stored view offset
for the session. The earlier `?tilt=-25` advice would add another 25 degrees to
this new neutral view. While teleop is OFF, the right stick still adjusts that
additional offset (left = further down). The working view takes effect gradually
on enable and commands both simulated and physical necks within the existing
angle/rate limits. Startup still uses measured robot pose; Home returns the
neck to its calibrated straight-ahead pose. Nothing moves automatically at
startup. Camera intrinsics and mounting geometry are unchanged.

The simulator initializes **once** from measured robot joints, gripper openings,
neck angles and homed lift reading. It does not repeatedly force its state to
match the robot. The endpoint never initializes drivers or invokes blocking
hardware homing routines. Operator-requested Home is streamed as ordinary bounded
targets, as described below. Stopping
the previous VR bridge disables the grippers; on your explicit A+X enable, the
live endpoint can reactivate them only if measured and requested openings are
already at least 0.99. This avoids using AgileX's enable routine to unexpectedly
open closed jaws. Missing gripper feedback, disabled closed grippers, an unhomed
lift or a start mismatch prevents following. Establish those states with the
existing controls before handing over.

| Input | Effect |
|---|---|
| A + X, tap | Enable/disable following; starts OFF |
| Hold either grip | Drive that arm from the simulated VR motion |
| Each trigger while clutched | Operate that AgileX gripper |
| B + Y | Toggle base mode; left stick translates, right stick turns |
| Head movement | Command neck pan/tilt while enabled |
| A + Y / X + B, hold | Lift down / up; release stops |
| Pi gamepad Back, or movement of a translation/rotation stick | Abort the live session; no automatic resumption |
| Ctrl-C in workstation terminal | Close session and request stops |
| A + X, hold 1.5 seconds | Return both arms and neck to their existing home targets and open both grippers; ends OFF |

The gamepad mapping uses the same Linux/Xbox Back index (6) as `joystick.py`,
and axes 0, 1, 3, 4 above 25% travel. Its disconnect ends the session. Validate
the stop input locally before movement if using a different gamepad mapping.
During Home, the base is stopped and the homed lift returns to yor.py's
`LIFT_HOME_M = 0.648` encoder-coordinate park height using up/down/stop commands. The return uses the same
simulation collision checks, acknowledgement pacing, tracking checks and speed
limits as following; it never resets joint coordinates or re-homes the lift
encoder. Completion requires measured arm/neck/gripper agreement. A+X tap during
Home cancels it; so do stale input and the local joystick stop. Home times out
after 60 seconds instead of running indefinitely.

Stationary lift holds preserve their commanded height through OFF, pause and
Home transitions. The forwarded lift target is the commanded setpoint converted
to encoder coordinates; the measured simulated height is logged separately.
This fixes the `Lift start/tracking mismatch` in `20261103-184446`: repeatedly
replacing the hold with gravity-sagged measurements had accumulated a false
height change. A regression starts from that session's recorded pose, performs
30 stop/re-enable cycles in the furnished workcell, completes Home and enables
following again through the real endpoint code with a synthetic robot.

A stale headset/control sample pauses motion. Once input recovers, turn teleop
OFF and then ON to re-enable; fresh frames alone do not resume motion. No process
restart is needed for that input pause. Pi faults, collision rejection and
tracking failures remain latched and require a session restart.
Expiry of an older computed target is handled separately: if the current
headset input is fresh and the Pi has stopped, the old target is discarded and
a new stationary handshake uses measured feedback. An old expiry flag cannot
mark a newer valid headset sample stale. Actual input loss still requires the
operator OFF/ON transition.

Commands that advance an active session carry `requires_active`. If the transport
has sent OFF while the producer was computing such a command, that command is
discarded rather than treated as a new enable. The producer then builds a new
stationary handshake from measured feedback. Before native-profile enable, arm
and neck readings must stay within 0.02 rad / 0.5 degrees of a reference sample
for 100 ms, allowing deceleration after OFF. The existing 2-degree neck start
alignment check is retained. This corrects the race recorded at wire sequences
541–542 in session `20261103-190633`; no Pi endpoint update is required.

The first enable packet is stationary. The robot checks arm agreement within
0.05 rad, gripper opening within 0.1, neck within 2 degrees and lift within 10 mm
of encoder coordinates before accepting motion. These are initial engineering
bounds, not calibrated accuracy specifications. The default `yor` command profile
uses Pi-verified yor.py/driver configuration: joint preview 1.0 s, joint velocity
3 rad/s and acceleration 15 rad/s², base limits [1, 1, 1.57] in m/s, m/s, rad/s,
base acceleration [1, 1, 1.57], and neck velocity 2000 servo steps/s (175.78125°/s)
with servo acceleration 80. Grippers receive normalized position directly, as
in yor.py, without an added stroke-rate cap. Base limits are per axis; the
joystick’s separate 0.5 m/s preset is not yor.py’s 1 m/s default. The Pi driver
performs joint interpolation. The active adapter does not impose the former
packet-rate faults or fixed 0.25-rad/15-degree target-to-feedback limits, which
conflicted with a one-second interpolation horizon. Start alignment, absolute
bounds, collision checks, ownership checks and stop watchdogs remain active. Lift uses the
existing nonblocking up/down/stop API and the recorded travel bounds.

## Recording and calibration export

Every session is recorded automatically, regardless of the VR episode buttons:

- Workstation: `output/live-sophia/UTC_SESSION/metadata.json` and `events.jsonl`.
- Pi: `/home/yor/YOR/teleop_data/sim_follow/UTC_SESSION/` with the same file names.
- Workstation on exit: `analysis/report.json` and `analysis/identification-input.npz`.

The logs keep raw VR-derived commands, accepted simulator joint targets, source
sequences, model/source/contract hashes, actual per-method Pi dispatch times and
return statuses, both arms' measured and commanded states, measured gripper
feedback, neck, raw lift encoder/reference, and wheel-derived body velocity.
The Pi writes command intent before dispatch and then records feedback. A `None`
RPC return is `sent_unconfirmed`, not proof of motor execution. Partial dispatches
and stop attempts are retained when an error occurs.

Pi and workstation clocks are separate. Logs preserve source timestamps and SSH
round-trip intervals; sequence numbers associate commands and replies. The export
does not silently shift time, invent feedback or treat a network reply as a
simultaneous measurement. Controller snapshot timestamps do not establish the age
of every individual CAN sample; missing/stale flags from the drivers are retained.

To re-run the audit manually:

```bash
.venv-sophia/bin/python scripts/analyze_sophia_live.py \
  --session output/live-sophia/UTC_SESSION --output output/asset-review/live-audit
```

The NPZ contains separate actual joint dispatch times, robot timestamps, sim sample
times, commands and measured responses. Use complete sessions as fitting/held-out
units for actuator delay and response identification. The audit reports robot
joint tracking error against dispatched targets; it does not call the robot and
simulation samples time-aligned ground truth. It never updates physics parameters
during control. Independent base pose, load/contact forces and accurate camera
mount transforms are not supplied by this stream; unloaded motion cannot identify
all masses, friction coefficients or grasp performance.

## Stops and transport behavior

SSH uses existing key authentication and host-key verification, without storing
passwords or exposing a new command port. There is one outstanding request, a
latest-frame buffer and no command backlog. Each request must answer the Pi's
one-use challenge within 250 ms. Source frames older than 200 ms are not forwarded.
The Pi checks a 250 ms receive lease, limits, measured tracking error and evidence
of another writer. Its loop checks local joystick stop between dispatch calls.

Freshness uses the actual headset receipt timestamp, including time spent
computing a target. Repeated bridge ticks cannot make an old input fresh. Input
pauses are logged with timing details. Bridge/HUD callbacks read cached simulator
snapshots rather than waiting on physics or inverse kinematics. Stereo OpenGL
rendering runs in a separate process fed by a queue holding at most one state;
GPU stalls cannot hold the control interpreter. Startup waits for the first
published frame. The model, calibrated stereo projection and stream format are
the same as before this process separation.
The launcher also uses one BLAS/OpenMP math worker for the small IK and collision
calculations; this machine otherwise selected 32 BLAS workers. Verification uses
the same threading configuration. Motor settings follow the yor.py profile described above. The source deadline
is now 200 ms: furnished-scene load testing reproduced a 126 ms processing stall
that exceeded the previous 120 ms cutoff. The Pi's independent 250 ms lease is
unchanged, and expired targets are still discarded.

The r4 endpoint uses the `yor` profile and delegates motion-rate handling to
the Pi drivers. The old commissioning profile and its jitter-pacing checks remain
available for regression tests only. Restart the launcher to load both the
updated workstation and the versioned r4 endpoint; there is no hot replacement
of a running motion endpoint.

The workstation also waits for acknowledgement of each distinct simulation
target before advancing another control step. Repeated frames keep the same
source sequence and target while waiting; an acknowledgement for an older target
cannot release a newer one. The latest headset intent is used after acceptance,
without a catch-up burst. Disable, stale-input and fault handling take precedence
over this wait. High network latency therefore reduces motion speed. This fixes
the further rate fault in `20261103-180001`, where 100–130 ms transport delays
previously let several sim steps accumulate before the next Pi dispatch.
The Pi directory is r4; restart the workstation launcher to load this sender
fix. `acknowledged_source_seq` in workstation replies identifies the originating
sim target accepted by the endpoint, not a measured hardware pose achievement.

On ordinary timeout/EOF/stop, base zero and lift stop are attempted first, followed
by holds from fresh measured arm/neck positions. Gripper torque remains unchanged.
Local RPC calls have 60 ms send/receive timeouts and failures are logged. These
are software stops: a stuck hardware RPC server, Pi failure or `kill -9` can
prevent stop delivery, especially for the lift, which requires an explicit stop.
The endpoint does not replace the physical emergency stop or add a hardware lift
watchdog. Process checks and command-readback checks also cannot provide exclusive
ownership against arbitrary clients that ignore this workflow.

## Verification and package provenance

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/check_sophia_live.py \
  --output output/asset-review/new-live-check
# Optional real-Pi read-only readiness check; never supplies --allow-motion:
MUJOCO_GL=egl .venv-sophia/bin/python scripts/check_sophia_live.py \
  --output output/asset-review/new-live-readiness --pi-host yor@192.168.123.55
```

The live tests use real MuJoCo, the shared VR controller, the actual endpoint
process and local RPC against an explicitly synthetic test endpoint. Test logs
cannot be mistaken for hardware calibration data. The readiness check only polls
real feedback. Session `20261103-174948` contains initial physical following
and the neck timing fault. The combined dispatch-pacing and sender-acknowledgement
fixes and a real operator's stop test still require a supervised session; offline
tests do not establish those results. Regression tests include synthetic
100–130 ms outbound delays followed by fast packets, with real MuJoCo, transport
and endpoint code connected to a fake robot.

The `vr-teleop` checkout is based on Pi revision
`626351530db2c5726a78dee3f719bc3773f5e7bf`; its original bridge matches the archived
Pi source byte-for-byte. The ZMQ simulated-camera source/config from simulation
revision `dbbf864f3b4facc4101d82fcd84dde13132a38c5` is restored. The only bridge
integration changes are an injectable link factory and an input-freshness flag;
the web server also rejects simultaneous control connections.
The uncommitted submodule changes must accompany the revision when transferring
this workspace; a gitlink alone does not contain them.

Build a fresh Pi bundle after changing endpoint sources or the contract:

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/live_sophia_teleop.py prepare \
  --output output/asset-review/new-pi-bundle.zip
```

Deploy to a new directory and update the launcher's `--remote-dir`; endpoint and
workstation contract hashes must agree. No model fitting results are automatically
deployed to the robot.
