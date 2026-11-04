# Sophia offline audit, calibration preparation and standalone setup

The finalized asset baseline is **Sophia sim v1.0.0 (2026-11-04)**. See
[the release record](SOPHIA_SIM_ASSETS.md#finalized-baseline-v100--november-4-2026)
for the versioned archive, checksum, validation scope and current model entry
point. This release closes the offline asset preparation with accepted
assumptions; further hardware accuracy and policy-transfer validation are a
separate evaluation phase. The workflow history below remains available.


The workstation-only [trajectory replay pipeline](SOPHIA_REPLAY.md) provides an
object-free reference, MuJoCo execution, passive recording/import and comparison
reports. Its dedicated check is `scripts/check_sophia_replay.py`; joystick and
hardware-control sources remain unchanged.

The phase 1/2 policy interface and training environment use an optional Gymnasium
lock and acceptance gates. See [SOPHIA_TRAINING.md](SOPHIA_TRAINING.md) for setup,
the exact action/observation contract and task validation. They do not connect to
hardware or establish learned-policy transfer.

Sophia's functional simulation release passes its acceptance suite using the
supplied measurements, CAD, photos, camera calibration and recorded robot data,
plus explicitly accepted engineering assumptions. Collision-shape corrections
and the legacy/current mount-frame review are complete. The user has now expanded
the objective to real-world accuracy, offered precise mount CAD and confirmed
that a short teleoperation recording is possible. The functional release is the
baseline; physical fidelity remains to be established using the workflow below.
`sophia.json` records the current defaults under `release`. The scripts never
apply calibration fits to the robot or silently overwrite measurement files.

## Standalone setup

The standalone dependency lock is `requirements-sophia.txt`, generated from
`requirements-sophia.in`. It contains 21 pinned packages with download hashes,
including OpenCV for stereo video. It excludes the hardware/ROS environment
freeze in the repository's general `requirements.txt`.

Tested on Linux x86-64 with CPython 3.11, MuJoCo 3.17.0 and EGL. Create an isolated
environment with no inherited system packages, from the repository/bundle root:

```bash
python3.11 -m venv .venv-sophia
.venv-sophia/bin/python -m pip install --require-hashes -r requirements-sophia.txt
MUJOCO_GL=egl .venv-sophia/bin/python scripts/check_sophia.py --render
```

With `uv`, the equivalent install is `uv pip sync requirements-sophia.txt --python
.venv-sophia/bin/python --require-hashes`. The current checkout also has a prepared
`.venv-sophia` environment. Rendering needs a working system EGL/OpenGL driver;
the Python lock does not supply the operating-system graphics driver.

Launch with a viewer and stereo video:

```bash
.venv-sophia/bin/python robot/yor_mujoco.py --model robot/yor-description/scene-sophia.mjcf
```

Use `--headless --no-video` for controller-only runs. The native stereo test uses
an in-process ZMQ connection and a temporary red target to verify rendering,
JPEG encoding/decoding, RGB/BGR ordering, metadata and left/right disparity.
It does not require a camera or network connection. Collision proxies are hidden
from the video renderer.

The acceptance runner checks exact package versions, deterministic regeneration,
asset integrity, functional asset validation, calibration math, stereo transport,
collision rejection, Sophia/legacy motion and extended simulation behavior. It
only runs an explicit simulation test list; it never discovers hardware tests.
The suite also checks the hardware-summary comparisons and a two-parcel mobile
manipulation task using free objects and gripper contacts. Logs and
`acceptance.json` go to `output/asset-review/acceptance/`. The report keeps
`simulation_checks_passed` separate from `hardware_calibrated` and includes
accepted assumptions. The latter cannot become true from these tests alone.
The current suite also verifies the four camera projections, depth, rigid mounts,
assumed lift/gripper mappings and consistency between runtime rates and collision
prediction. It checks exact housing-triangle preservation, detected sub-margin
clearance and CAD-surface support for the retained collision samples.
Latest results: `output/asset-review/dataset-system-identification/acceptance/acceptance.json`.

The mobile task closes the loop on simulated gripper positions to correct the
position servos' load-dependent reach error. Corrections use the same guarded
joint commands, with no gain changes or forces applied directly to the parcels.
The report records reach residuals, finger contacts, carry offsets, final table
support, and robot/table penetration. This demonstrates a functioning simulated
task; it does not measure the real robot's tracking or grasp accuracy.

## Applied calibration and vendor corrections — 2026-10-30

The head cameras now use the supplied Orin **rectified ZED** profiles at
1920×1080: focal lengths 1072.864624 px, principal point (983.690857, 526.938049)
px and a 119.991852 mm stereo baseline. The original 70° vertical-FOV assumption
is replaced with the calibrated pinhole projection. Raw lens distortion is not
applied to an already rectified image.

MuJoCo stores principal-point offsets relative to the image center with the
opposite sign. The generator converts the pixel-center convention explicitly;
rendered targets are checked against independent pinhole projections after JPEG
transport. The publisher reports per-eye intrinsics matching its actual output.
For resized images, `u_out + 0.5 = sx * (u_in + 0.5)` (likewise for v). Non-16:9
outputs resize the full calibrated frame rather than cropping it silently.
Legacy models retain their FOV-based projection.

The untouched source is
`robot/yor-description/calibration/camera_intrinsics_2026-10-30.json`. It also
contains raw ZED and both gripper-camera distortion/intrinsic profiles. The two
wrist cameras now use those measured lens profiles with accepted photo-based
mounts: `(42, 0, 52)` mm in each gripper palm frame and 10° toe-in toward the
grasp center. Image-right follows palm +Y; optical-forward follows the fingers.
Mount uncertainty is recorded as ±15 mm / ±10°, not as measured calibration.
Head mounts retain their source CAD transforms. Each recording's `meta/cameras.json` takes priority
for that recording because ZED calibration can vary slightly between opens.

Export all four synchronized views, optical-Z depth maps, validity masks and
per-camera K/distortion/world/parent transforms:

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/render_sophia_cameras.py --output output/sophia-cameras
MUJOCO_GL=egl .venv-sophia/bin/python scripts/render_sophia_cameras.py --pinhole --output output/sophia-cameras-pinhole
```

`SophiaCameraRenderer` in `robot/sim_camera.py` is also usable by simulation
clients under their physics/render lock. It corrects OpenGL's square-pixel
frustum to each supplied K and applies measured wrist distortion for raw output.
The supplied wrist distortion polynomials do not invert reliably at every
corner: invalid raw pixels are black, with NaN depth and an explicit false mask
(approximately 1.5% left / 9.1% right at 640×360). Pinhole output has complete
coverage. Direct MuJoCo camera rendering is ideal pinhole; use this helper for
exact lens projection. The existing ZED stereo publisher remains available.

## Gripper and inertia audit findings

```bash
.venv-sophia/bin/python robot/yor-description/audit_sophia.py
```

The source investigation found the official
[AgileX arm description repository](https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd00db7e2628563d2edc5540963045de).
The URDFs, MIT license, commit and file/mesh checksums are pinned under
`robot/yor-description/vendor/agx_arm_urdf/`. Only bodies with verified identical
mesh coordinates receive vendor inertials. Joint limits, effort limits, joint
transforms and controller gains retain their existing robot-specific settings.

- All eight arm meshes (base plus seven links) and both finger meshes match the
  vendor STL files **byte-for-byte**. Both arms and fingers now use vendor masses,
  centers of mass and full inertia tensors, including off-diagonal terms.
- Both `arm_link7` masses are corrected from **0.000001 kg (1 mg)** to the
  vendor's **0.001 kg (1 g)**. The existing inertia now has a consistent mass;
  its largest radius of gyration is approximately 14.14 mm.
- Each arm base plus links now totals **4.5812127 kg**. The separate adapter is
  **0.05 kg**, the gripper assembly is **0.5 kg**, and total modeled robot mass is
  **42.0053061 kg**. These are model totals, not a weigh-in of the installed robot.
- The original left/right arm-only URDFs remain legacy references for other
  variants. Sophia explicitly supersedes their inertials with the pinned vendor
  source; the audit retains their differences as historical evidence.
- The installed grippers are user-confirmed stock AgileX Nero. The finger source
  is resolved; the current vendor **housing** differs from the supplied housing.
  Housing geometry/inertia and the custom adapter are retained as accepted source
  approximations. The configured palm remains `(0, 0.012, 0)` m and the grasp
  site `(0, 0.132, 0)` m in each end-effector frame.
- The unused, unitless `gripper.STL` remains an optional identification task. It
  is not required to use the current model and has not replaced an active mesh.
- Chassis/lift/accessory inertials and friction retain source defaults. These
  support the tested simulated tasks; payload, force and tipping predictions are
  approximate. Hardware validation is optional future work.

## Hardware response comparison

```bash
.venv-sophia/bin/python robot/yor-description/compare_sophia_hardware.py
```

The supplied Pi summary is preserved under
`robot/yor-description/calibration/motion_summary_2026-10-30.json`. The comparison
reconstructs **eight neck ramps** from reported endpoints/durations and exercises
both lift directions. It does not present these synthetic command ramps as raw
hardware replay or fit unsupported dynamics parameters.

The larger neck sweeps produce approximately 57.3–57.4°/s in simulation, compared
with reported hardware rates around 57.4–58.9°/s. Other ramps differ; matching the
large sweeps alone does not validate response timing. The hardware lag estimator
is unspecified, so the sim's mid-ramp error-equivalent lag is labeled separately.

The release assumes an affine lift mapping: **0.6726 encoder m spans 0.5588
physical m**, anchored at the summary's 0.6744 m upper hard stop. The inferred
lower encoder coordinate is 0.0018 m and scale is 0.830805828. Multiplying the
summary's 0.02649 / 0.0308 encoder m/s produces **0.022008 / 0.025589 physical
m/s up/down**, replacing the old 0.05 m/s placeholder. Runtime commands and
collision prediction share these rates. `robot/sophia_profile.py` provides
explicit conversions; the simulator API still takes physical extension in m.
No hardware encoder configuration changes. This is an accepted reconciliation
assumption, not an independently fitted calibration.

Normalized gripper opening maps linearly to the measured 0–101.6 mm gap. Right
opening now uses 6 strokes/s after the identification pass below; both closing
directions and left opening retain 5 strokes/s as assumptions. The base retains
source CAD geometry, friction and four functioning simulated wheel encoders.
The physical motor-3 reconstruction issue is outside nominal asset behavior.

## Blank measurement templates

```bash
.venv-sophia/bin/python robot/sophia_calibration.py templates --output output/asset-review/measurements
```

This creates CSV headers and an unfilled camera JSON template, with no invented
measurements. It refuses to replace existing files. Use a new directory per
measurement session and retain robot configuration, date, units and instrument
uncertainty alongside the data.

| File | Contents |
|---|---|
| `lift.csv` | `sample_id, split, encoder_value, shoulder_height_m` |
| `gripper.csv` | `sample_id, split, side, raw_width_um, jaw_gap_m` |
| `points.csv` | Paired `source_x_m/y_m/z_m` and `target_x_m/y_m/z_m`, plus sample ID and split |
| `poses.csv` | Sample ID, side, physical lift extension, seven joint angles in radians, measured XYZ and quaternion `qw,qx,qy,qz` |
| `trajectory.csv` | Monotonic `time_s` and named scalar signals; add joint/base columns as needed |
| `gripper_can.csv` | `time_s, side, arbitration_id, data_hex` for received frames |
| `camera.json` | Image dimensions, intrinsic matrix, OpenCV distortion, parent frame and optical-frame mounting transform |

For fits, `split` is exactly `train` or `validation`. Use at least three distinct
training positions and additional held-out measurements; three training values
alone cannot validate a mapping. Left and right grippers are fitted separately.
Capture repeats and both motion directions to expose hysteresis. Unit tests use
explicitly synthetic fixtures and are not included as measurement templates.

## Pillar-to-shoulder attachment correction — 2026-11-03

The user identified a visible gap between the lift pillar and the metal plate
carrying the arms. The reference photos show that connection is rigid. The
previous upper pillar ended at local z=0, while the plate underside is z=0.101 m.
The generator now derives the upper endpoint from the plate's lower face and
retains the pillar's lower endpoint at -0.430 m. Its visual and collision shapes
therefore span a 0.531 m assembly envelope. This is a modeled extent constrained
by the attachment, not an additional tape measurement.

The validator checks the attachment, plate coverage and matching collider at nine
lift heights, including both travel limits. The previous 101 mm gap is zero to
numerical precision; minimum telescoping overlap remains 50.6 mm. Arm mounts,
measured shoulder heights, mass/inertias and fitted response gains are unchanged.
Evidence: `robot/yor-description/calibration/pillar_attachment_2026-11-03.json`.
Integration checks: `output/asset-review/pillar-connection-fix/acceptance/acceptance.json`.
Updated bundle: `output/asset-review/sophia-sim-2026-11-03-pillar-fixed.zip`.

## Physics response adapted from the live capture — 2026-11-03

Sophia now uses separate effective position-control gains for each **left-arm
joint**, fitted to the new real-robot recording. The scalar kp500/kv20 defaults
remain for the right arm. `sophia.json` stores the left override under
`simulation.arm_position_gains.left`; the builder validates and applies all
seven kp/kv pairs. Masses, inertias, friction, force limits, mounts, gripper rates,
neck rates and lift settings retain their prior values.

| Comparison | Previous RMSE | Adapted RMSE | Change |
|---|---:|---:|---:|
| Development: 10 episodes, 3,220 selected samples | 0.009473 rad | 0.006213 rad | 34.4% lower |
| Final check: 2 intervals of later episode 15, 804 selected samples | 0.006133 rad | 0.003411 rad | 44.4% lower |

Every joint improves in both comparisons. Training uses cycle 51; cycles 52/53
are development checks. The final check uses previously unscored intervals from
the later episode 15. It is one later episode, not an independent day or a
payload/contact validation. The final-interval selection uses only engagement,
command availability and base stationarity; it does not select by fit error.
Scoring excludes the first initialized second, closed/grasp intervals and moving
base intervals. Open grippers reduce but do not prove absence of external contact.

A shared left-arm gain fit was rejected because joints 1/2 worsened. That finding
motivated the regularized per-joint model, so the previously used development
episodes are not described as a fresh final test. The neck candidate was also
rejected: pan hit its bound and pooled validation error increased. A left-opening
candidate had only 12 training and 11 validation transition samples; it remains
unapplied. The new recording does not identify right-arm gains, lift speed,
contact friction, grip force or physical masses.

Read-only inspection of the Pi's actual WebXR bridge and driver resolved the
missing provenance flag: non-null gripper positions are measured CAN feedback;
null commands mean **disabled/limp**, not an instruction to hold the previous
command. Raw imported flags/NaNs remain intact. The matching source files predate
the capture; their hashes and copies are retained. They are not a capture-time
process attestation. The bridge defaults to 2/30 s arm interpolation, while exact
launch overrides and dispatch times were not logged. The fitted gains are
effective simulator parameters conditional on that timing and the retained
dynamics, not uniquely identified motor constants.

Evidence and decisions: `robot/yor-description/calibration/response_identification_2026-11-03.json`.
Full plans, source snapshots and per-episode predictions:
`robot/yor-description/calibration/response_identification_2026-11-03/`.
The C++ rollout was checked against independent Python stepping of `_SimArm`
for both sides. The isolated neck fixture preserves the full model's neck mass
matrix and gravity terms. Compiled-model comparison verifies that only the left
actuator gains changed.

```bash
.venv-sophia/bin/python scripts/identify_sophia_capture.py --subsystem arm --output output/asset-review/new-arm-fit
.venv-sophia/bin/python scripts/identify_sophia_capture.py --subsystem neck --output output/asset-review/new-neck-fit
.venv-sophia/bin/python scripts/identify_sophia_joint_gains.py --output output/asset-review/new-joint-fit
```

These analysis tools retain the original kp500/kv20 comparison baseline and
write candidates; they do not apply changes themselves. Each requires a new
output directory. Full integration checks:
`output/asset-review/live-capture-identification/acceptance/acceptance.json`.
Updated bundle: `output/asset-review/sophia-sim-2026-11-03-response-fit.zip`.

## Imported real-robot capture — 2026-11-02

The user-requested listener has stopped cleanly. Its final capture is included in
the simulation assets at `robot/yor-description/calibration/teleop_2026-11-02/`:
**75 complete episodes, 115,791 native telemetry frames, 3,892.84 seconds of
recorded episode time**, with no detected sequence loss, duplicates, malformed
packets or nonincreasing source timestamps. Episode completeness refers to the
recording, not successful task completion or validated physical accuracy.

Publisher labels identify 16 cube manipulation/handoff episodes and 59 navigation
episodes. These labels are retained for selecting relevant comparison intervals;
they are not an independent assessment of task outcomes.

The directory contains losslessly compressed original packet/frame logs,
receiver timestamps, publisher task/cycle/outcome messages, the original sim
baseline, and 75 `native-replay/episode_*.npz` files. `summary.json` maps each NPZ
to its source episode ID and reports signal ranges/coverage. `MANIFEST.json`
hashes imported files and the uncompressed source files. Load arrays with
`numpy.load(path, allow_pickle=False)`; no hardware connection is required.

Import another **stopped** session into a fresh directory with:

```bash
.venv-sophia/bin/python scripts/import_sophia_capture.py --source output/live-sophia/SESSION --output robot/yor-description/calibration/NEW_CAPTURE
```

The importer cross-checks every data packet against the decoded log, recomputes
the listener's quality results, and rejects incomplete or changing sources.
Missing numeric values stay NaN with explicit `*_valid` masks; timestamps are
not resampled. Gripper provenance is encoded as -1 unknown, 0 command fallback,
or 1 measured. All provenance flags are unknown in this capture. Left/right
gripper commands are null in 84,108 / 84,394 frames respectively; source
command-hold semantics must be confirmed before reconstructing those inputs.
The older arm replay expects finite commands and cannot consume these files
directly without resolving that issue.

Both arms' joint positions and targets, base command/encoder velocity, and neck
state/targets are present. Lift height remains constant at publisher coordinate
0.6738 and lift command is zero, so this capture adds no lift speed/stroke
identification. No camera frames, independent base poses, raw force, payload mass,
or exact command dispatch/interpolation timings were received. Existing videos
remain separate sources that can be matched by episode ID if available.

This import adds evidence and reproducible inputs. **No gains, friction, masses,
mounts or other simulation parameters were fitted or changed from these runs.**
The subsequent November 3 analysis above resolves command semantics and adopts
validated left-arm effective gains. The imported capture is
included automatically by `scripts/package_sophia.py`.
Updated standalone bundle:
`output/asset-review/sophia-sim-2026-11-02-teleop-capture.zip`.

## Live episode listener — 2026-11-02

Listen to the existing Pi ZMQ publisher without initializing the robot or making
RPC/motion calls:

```bash
.venv-sophia/bin/python scripts/listen_sophia_episodes.py --endpoint tcp://192.168.123.55:5558 --output output/live-sophia/session-001 --duration 7200
```

The output directory must be new. The subscriber waits through disconnections
and exits after the duration limit, Ctrl-C/SIGTERM, or creation of a `STOP` file
inside the session directory. `status.json` updates every five seconds and
distinguishes TCP connection, received packets, and per-episode data quality.
`summary.json` is written on exit. `packets.jsonl` retains exact multipart wire
bytes as base64 plus local wall/monotonic receive times; `frames.jsonl` retains
decoded episode objects for later replay. Source timestamps and measurement
flags are preserved. Model/config/controller snapshots and hashes are recorded
under `sim-baseline/` and `metadata.json` before capture.

Subscriptions are `teleop/episode`, `episode/signal`, and `teleop/episode_result`.
The wire format is two parts: UTF-8 topic, then a JSON object. Existing teleop
publishes motion frames only while an episode is recording; its start requires
a task prompt. A second script can subscribe to the existing publisher; it must
not attempt to bind another publisher to the same occupied address/port.

Quality checks count sequence gaps, duplicates, out-of-order frames, missing
initial/trailing sequences when the publisher's final frame count is available,
nonincreasing source timestamps, and invalid/missing signals. A complete sequence
does not guarantee valid feedback: gripper command fallback is explicitly flagged.
Keep the Pi's local episode recording; ZMQ PUB/SUB is not a lossless archive.
Source and receiver clocks must be synchronized before interpreting their
difference as latency. Prefer source timestamps for deterministic offline replay.

The next comparison replays **recorded commands** from the measured initial
state, then checks independent measured responses. Setting simulation positions
to each measured frame would hide response error. The current right-arm response
benchmark can be used after validating/converting a run to its native NPZ schema;
this listener itself does not yet replay or tune the simulation automatically.
Fit on one set of runs and validate on separate runs before adopting changes.

Useful first recordings: stationary unloaded left/right arm motion, repeated
gripper opening/closing without an object, then a pick-and-place task. Label the
task and payload when known, and retain matching camera recordings by episode ID.
Joint targets/feedback support response fitting; camera observations are needed
for independent alignment, object contact and base slip checks. The local
publisher does not currently include lift, joint velocity, torque or exact
command dispatch/interpolation timing. The extra Pi logger can add those fields
with units and source timestamps when available; this listener preserves them
without asserting unsupported units or fabricating missing measurements.

## Passive getter and CAN capture tools

The state recorder calls only `get_collection_state` on an already running
server; it does not initialize, enable or command the robot. It preserves getter
responses, client request start/end timestamps, a source label and a complete/
incomplete footer. RPC waits are bounded by `--timeout-s` (default 5 seconds);
timeouts mark the capture incomplete and close only the local client.
A normalized gripper value that falls back to a command is
not promoted to measured jaw width.

```bash
.venv-sophia/bin/python robot/sophia_calibration.py record-state --host ROBOT_HOST --source-label real-robot-session-001 --duration 20 --output session-state.jsonl
```

The Linux SocketCAN recorder receives only gripper feedback ID `0x2A8` from an
existing interface. It neither configures the interface nor sends CAN frames.
Specify the actual interface and side; the script does not infer robot wiring.

```bash
.venv-sophia/bin/python robot/sophia_calibration.py record-gripper --interface YOUR_INTERFACE --side left --duration 20 --output left-feedback.csv
.venv-sophia/bin/python robot/sophia_calibration.py decode-gripper left-feedback.csv --output left-widths.csv
```

Decoding preserves signed raw width, overtravel, raw effort, status and payload;
it filters angle-mode frames instead of treating them as widths. Raw effort is
not asserted to be a calibrated force. Pair stationary raw widths with measured
jaw gaps in `gripper.csv`. CAN timestamps are relative to that capture's start;
separate recordings require explicit synchronization. No physical CAN interface
or robot server was accessed during offline preparation; recorder tests use
synthetic inputs.

## Fit mappings and compare measurements

Examples below assume populated templates. Output files must be new paths.

```bash
.venv-sophia/bin/python robot/sophia_calibration.py fit-lift lift.csv --encoder-unit motor_turns --output lift-fit.json
.venv-sophia/bin/python robot/sophia_calibration.py fit-gripper gripper.csv --output gripper-fit.json
.venv-sophia/bin/python robot/sophia_calibration.py fit-transform points.csv --output mounting-fit.json
.venv-sophia/bin/python robot/sophia_calibration.py compare-poses poses.csv --site gripper_center --output pose-errors.json
.venv-sophia/bin/python robot/sophia_calibration.py compare-series real.csv sim.csv --signals lift_m pan_deg --max-gap-s 0.1 --output motion-errors.json
.venv-sophia/bin/python robot/sophia_calibration.py check-camera camera.json --output camera-check.json
```

- **Lift fit:** fits `shoulder_height_m = slope * encoder_value + intercept`.
  Specify the unit actually recorded; it may be raw turns, counts or an existing
  reported distance. Establish the lowered-height zero before converting this to
  extension. This optional fitting tool can supersede the release's assumed
  **558.8 / 672.6** scale if independent data later becomes available; it does
  not automatically rescale hardware.
- **Gripper fit:** fits raw width directly to measured gap, including inverted
  coordinates and overtravel. Review residuals for nonlinearity. The **70 mm
  protocol coordinate** is not substituted for the **101.6 mm physical opening**.
- **Frame fit:** estimates a rigid transform from metric 3D point correspondences
  using `p_target = T_target_source @ p_source`. Collinear/coincident point sets
  are rejected. It does not infer units or scale.
- **Pose comparison:** uses the stationary `chassis_ground` frame: X forward,
  Y left, Z up, origin on the floor under the average steering-axis XY position.
  Specify `arm_ee` or `gripper_center` and measure the corresponding frame.
  Joint angles are radians, XYZ and physical lift extension are metres, and
  quaternions are normalized WXYZ. It evaluates kinematics without gravity
  settling. Do not feed lift-independent hardware API poses as floor-referenced
  measurements without converting frames.
- **Motion comparison:** reports per-signal RMSE, mean absolute and maximum
  errors. It rejects duplicate/nonmonotonic times, nonfinite values, excessive
  interpolation gaps and extrapolation. Signals must share units and frames;
  unwrap angular signals consistently. Any known clock offset is supplied as
  `--candidate-time-offset-s` and recorded; delay is not automatically optimized
  away to improve agreement.
- **Camera check:** validates data structure and a proper rigid transform, not
  optical accuracy. `T_parent_camera_optical` maps optical coordinates (+X right,
  +Y down, +Z forward) into `head_tilt` or `chassis_ground`. A chassis-relative
  transform must include `calibration_pose` with `lift_m`, `pan_deg`, `tilt_deg`
  and applies only at that pose. MuJoCo camera axes require conversion with
  `diag(1, -1, -1, 1)` before using an optical transform as a camera mount.

Fits report training and validation errors separately, include source hashes,
and remain proposals for review. Empty data, missing columns, duplicate sample
IDs and degenerate fits fail explicitly. No fit changes `sophia.json`, MJCF,
hardware calibration, CAN mappings or robot limits.

## Portable bundle

```bash
.venv-sophia/bin/python scripts/package_sophia.py --output output/asset-review/sophia-sim.zip
```

The deterministic ZIP contains models, required source meshes, the generator,
audit/calibration tools, controller, tests, docs, license and dependency lock.
`MANIFEST.json` records every bundled file's size and SHA256. It excludes Python
environments, physical-driver libraries, raw recordings and reference photos.
The supplied calibration JSON sources and pinned vendor references are included.
Extract into a new directory, install the lock, and run `scripts/check_sophia.py`
from that root. Reports and logs are regenerated locally, not presented as fresh
hardware validation in the bundle.

## Remaining work — offline model consistency review

The confirmed base is now **431.8 mm left-to-right × 190.5 mm front-to-back ×
203.2 mm bottom-to-top**, with **63.5 mm clearance** below it. Its top is therefore
266.7 mm above the floor. Lower and upper pillar sections are **88.9 × 50.8 mm**
and **76.2 × 50.8 mm**, respectively, with the larger dimension left-to-right.
These measurements are applied to visuals and collisions and are no longer
missing inputs. The middle section is provisionally **82.55 × 50.8 mm**; rail
cross-sections, stage lengths, overlaps and motion ratio remain estimates.

The user confirms that the physical gripper and wrist-camera mounts have not
changed. Hardware continuity is established, and the model-frame reconciliation
below is complete. The retained wrist transforms remain accepted photo estimates.

| Completed area | Current evidence and implementation |
|---|---|
| Lift mapping and speed | Pi log provides 26.49 mm/s up and 30.8 mm/s down in encoder coordinates. The accepted linear scale 558.8/672.6 gives 22.008/25.589 physical mm/s. Mapping, endpoint conversion, runtime rates and collision prediction are implemented and tested. |
| Base and pillar dimensions | User measurements are applied to both visual and collision geometry. |
| Gripper endpoints and response | Physical gap 0–101.6 mm; explicit linear normalized mapping; 70 mm protocol coordinate retained separately. Dataset identification updates right opening to 6 strokes/s; left opening and both closing directions retain 5. Aggregate opening error improves on 15 separate episodes, with variation across episodes. |
| Arm response and inertials | All 18 available native episodes benchmarked; retained arm gains and verified vendor arm/finger inertials. |
| Physical mount continuity | User confirms unchanged gripper and wrist-camera mounts. No new measurement request. |
| Model-frame reconciliation | Stock source and Sophia gripper transforms agree; full Piper/iPhone camera chains composed and identified as a different assembly. Current photo-estimated wrist transforms retained. |
| Camera lenses and rendering | Supplied head/wrist intrinsics, RGB/depth, distortion handling, masks and transforms implemented; projection and rigid-mount tests pass. |
| Collision shapes | Whole-housing convex-hull bridging removed using five source-connected meshes per gripper; every source triangle retained. Identified residual intersections checked against original CAD surfaces. |
| Clearance guard | Separate query model now detects the reproduced 2.9 mm positive gap inside its 5 mm margin. Live physics options, collision masks and exclusions are unchanged. |
| Functional integration | Current acceptance includes direction-specific opening/closing prediction, mount-frame review, collision geometry, cameras and the free-object mobile task; results are in `dataset-system-identification/acceptance/acceptance.json`. |

| Category | Remaining work for the expanded real-world accuracy objective |
|---|---|
| CAD / simple reference information | Obtain the offered assembled mount CAD; provide ordinary-scale object masses or labeled masses with uncertainty. No robot disassembly or individual-link weighing is proposed. |
| On-robot observations | Defer a new session until existing data has been used. Collect only observations still missing for the intended fidelity claim; a short session remains available if needed. |
| Offline work | Existing-data response identification completed below. Import offered mount CAD; investigate video-derived object/base motion and remaining mass/contact uncertainties; extend only identifiable fits and validate predictions separately. |

The 5 mm guard is a simulation clearance requirement, not a measured bound on
geometry uncertainty. Completing the source review does not establish measured
hardware clearances or extrinsics.

The legacy wrist offset `(0, -36, 100)` mm belongs to a camera body nested under
`left_arm_iphone_body`, whose own translation/rotation must also be applied.
Its composed position is approximately `(28.5, 15.0, 117)` mm in that old
gripper frame. The current photo estimate is `(42, 0, 52)` mm in the current
gripper frame. The completed review finds that the old camera is attached to a
different gripper/phone assembly on a six-joint Piper arm. Its declared stock
housing mesh is unused in that assembly, so matching mesh filenames do not
establish an equivalent camera frame. The old camera offset is not adopted. The local
`yor-full.mjcf` contains head cameras and neck offsets; the Pi excerpt describing
their absence refers to a different model state. Existing assets and source
comparison are the inputs for this review.

Accepted approximations remain: 20 mm rails/posts, interpolated middle pillar,
430 mm middle section, upper envelope extended to the shoulder plate and
half-speed middle stage; source assembly masses and
contact friction; symmetric left-side/closing response; linear lift/jaw mapping;
and nominal four-wheel base feedback. These are documented defaults, not
additional measurement tasks. Friction and gripping force concern contact
dynamics separately from collision-shape placement. The physical motor-3 fault
is outside nominal simulation asset scope. `hardware_calibrated=false` is
retained. The earlier accepted-assumption release remains complete as a
functional baseline. Its defaults now need the real-world fidelity review below;
the 16 simulation acceptance gates alone do not establish physical accuracy.

## Real-world fidelity using CAD and a short recording — 2026-10-31

The user can supply precise mount CAD and can record a short teleoperation
session if needed, but the existing dataset takes priority. The aim is to fit the parameters that affect the intended tasks and
validate predictions on independent real examples. Exact identification of every
individual mass, friction coefficient and motor parameter is unnecessary, and
some parameter combinations cannot be separated with the available sensors.
This follows the recorded-input/output approach in
[MIT's system-identification notes](https://underactuated.csail.mit.edu/sysid.html)
and the [MuJoCo identification toolbox](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md).
The locked local MuJoCo environment includes `mujoco.sysid`; its availability is
not evidence that any new physical parameters have already been fitted.

| Area | Practical method | What it can establish |
|---|---|---|
| Gripper / camera mounts | Import an assembled STEP or equivalent CAD export, preserving units and flange references. Locate the camera optical frame from its camera model/drawing. If unavailable, use multiple views of a known-size printed target with recorded arm poses. | CAD assembly offsets and an independently checked camera transform. Mount CAD alone does not identify the optical center or installation tolerances. |
| Mass / center of mass / inertia | Keep verified vendor arm/finger properties. Combine part mass specifications with actual CAD solids and material density; model batteries/electronics as separate component masses at their assembly positions. | Physically informed mass estimates without disassembly. Do not infer solid mass from outer collision hull volume. Printed infill, hollow tubes and missing internal parts need explicit uncertainty. |
| Arm and neck response | Replay recorded commands; fit a small bounded set of delay, rate, damping and effective actuator-response parameters, retaining vendor inertias unless load data contradicts them. | Predictive tracking in the tested pose/speed/load range. Do not vary all masses and gains freely to hide timing errors. Existing right-arm and neck fits are useful baselines. |
| Grasp behavior | Record repeated low-height grasps/holds/transfers with two known-mass objects and the normal effort setting. Record actual width and raw feedback. | Effective holding/slip behavior and bounds on contact parameters for the tested object surfaces. Successful holds provide lower bounds; they do not identify a slip threshold. No deliberate drops are needed. |
| Base motion | Record short translations, rotations and stops while the head/lift are stationary; compare wheel signals against stereo or fixed-reference visual motion. | Velocity response, stopping behavior and slip bounds. Existing encoder-derived body velocity is not independent ground-truth motion. Keep the known motor-3 reconstruction issue separate from floor friction. |
| Validation | Reserve whole repeats and a different motion/object case before fitting; compare physical motion and grasp outcomes with replay predictions. | Task-specific accuracy and operating limits, not merely a successful simulation demo. |

The existing recordings contain commands and joint positions, but no raw joint
torques, raw gripper effort/width, known object masses or independent base poses.
Their end-effector poses are forward-kinematics outputs. Those fields cannot
independently validate kinematic geometry. Right-side motion is represented;
left-side command response is not. The Pi lift log already supplies directional
speed and onset/coast observations, so basic lift speed need not be remeasured.
Physical travel continues to use the accepted encoder scale until independently
checked; further onset/coast fitting must distinguish acquisition latency.

For two opposing approximately horizontal jaw contacts, a static vertical hold
constrains the product of friction and normal force (approximately `2*mu*N >= m*g`).
With no independent force observation, a stronger motor and a lower friction
coefficient can predict the same hold. Retain plausible force/friction ranges,
and validate the combined behavior. The CAN effort field remains raw until its
units and relationship to jaw force are established; a driver command in Nm is
not a measured jaw force in N. If separate force identification becomes necessary,
an ordinary pull gauge/scale is a possible additional observation, not a current
requirement. Uncertainty sweeps supplement, rather than replace, real validation.

### Suggested first session

Budget about 10–15 minutes including object changes. Use normal operator-controlled
motions and operating effort limits. No automatic motion script is proposed.

1. Start the usual synchronized robot/video recording. Save per-session camera
   calibration, source timestamps and a session note with object IDs, masses,
   mass source (scale or label), surface materials and normal effort settings.
2. With the base stationary, record both arms moving through several ordinary
   working poses at slow and normal speed. Include pauses, reversals and unloaded
   gripper opening/closing. Repeat part of the sequence for validation.
3. For each of two light representative objects, close, lift a few centimeters
   above the table, hold, transfer and place. Repeat on both arms. Record natural
   slip, deformation and failure outcomes as well as successful grasps. Keep
   several complete repeats out of fitting.
4. With head/lift stationary, record a few short straight/lateral base moves,
   rotations and stops in view of stationary features. Video must provide motion
   independent of wheel odometry. Timestamp alignment and visual tracking quality
   must be checked before interpreting an apparent slip or lag.
5. If the new CAD cannot establish the optical frame, include a separate known-size
   target segment with diverse wrist orientations and stationary captures.
   [OpenCV hand-eye calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
   combines target observations and robot poses. The printed target's physical
   scale must be known; pauses reduce camera/robot synchronization sensitivity.

### Recording readiness

The existing `record-state` getter and episode recorder already retain joint
positions, commands, normalized gripper states and neck/base information. Normal
video recording supplies views, episode IDs and timing metadata. The separate
`record-gripper` tool passively preserves raw `0x2A8` bytes, but uses its own
relative clock, so it needs an explicit shared-time alignment before fitting.
The current driver discards the feedback effort bytes in its normal state path.

`ArmNode.get_joint_state()` exposes position, velocity and torque feedback, while
`get_collection_state()` does not currently include the latter two. Prefer the
raw state with an explicit validity flag: some convenience getters return zeros
when hardware reads fail, which must not become apparent measurements. Record
available torque feedback with its source and unit semantics, and test it against
known-load changes before using it for inertial fitting. Logger channel/timestamp
preparation is offline work pending before the session; no logger or robot code
has been deployed for this fidelity phase.

The mass-source inventory and first bounded response-identification pass now
exist. Further offline work concerns CAD, video-derived independent observations,
contact/mass uncertainty and validation of supported additional fits.
Accuracy thresholds should be tied to the intended objects and clearances;
millimeter-level or payload-limit claims require corresponding independent
evidence. Current scope/status: `output/asset-review/real-world-fidelity/remaining-work.json`.

## Mount-frame reconciliation — 2026-10-31

`scripts/review_sophia_mounts.py` compares the compiled transforms in
`robot.mjcf`, `arm-upright.mjcf`, `yor-full.mjcf`, `yor-full-agx.mjcf` and
`sophia.mjcf`. It composes the entire legacy camera hierarchy, compares mesh
identities and checks both current mounts over 20 arm/lift configurations.

| Source | Finding and adopted decision |
|---|---|
| Stock AgileX source → Sophia | Link-7-to-palm transforms agree to floating-point precision. Retain the source adapter and palm placement. |
| Legacy Piper/iPhone assembly | Uses a different 37,342-triangle `gripper_body` and a six-joint arm. Its identical stock `gripper_base` mesh is declared but unused. Retain the composed camera transform as reference, not as an installed Nero camera calibration. |
| Custom gripper camera-mount STLs | Belong to the articulated custom gripper in `yor-full.mjcf`; that model defines no wrist optical cameras. No supported stock-camera transform follows from those meshes alone. |
| Current wrist cameras | Retain `(42, 0, 52)` mm in each stock palm frame, with 10° toe-in and the supplied per-camera lens calibration. Photos and dataset views support the arrangement; metric extrinsics remain estimates. |

For both sides, the adopted palm origin is `(0, 45, -23.5)` mm relative to
`arm_link7`: the end-effector offset `(0, 33, -23.5)` plus the palm's 12 mm
offset along end-effector +Y. Palm +Z points along link-7 +Y; fingers open along
approximately opposite palm ±Y axes. The grasp site is `(0, 165, -23.5)` mm
relative to link 7, or `(0, 132, 0)` mm relative to the end-effector body.

The legacy camera's fully composed optical position is approximately
`(28.500, 14.971, 117.000)` mm in its **legacy** gripper frame, not the raw
`(0, -36, 100)` mm camera-body translation. Its optical orientation is also
composed and recorded in the report. No rigid transform between the different
installed gripper assemblies is established by the available files.

The five source/frame checks pass, with maximum matrix errors below `5e-16`
for stock-to-Sophia agreement and pose-invariant mounting. This establishes
model consistency. `make_agx_variant.py` explicitly labels its installed mount
offset/orientation as estimated, so inheritance cannot turn it into a measurement.
No supported numeric mounting correction was found. The existing placement is
the final accepted engineering default, and source reconciliation is closed.

```bash
.venv-sophia/bin/python scripts/review_sophia_mounts.py --output output/asset-review/mount-review
```

Portable evidence: `robot/yor-description/calibration/mount_frame_review_2026-10-31.json`.
Current acceptance: `output/asset-review/mount-review/acceptance/acceptance.json`.
Current work status: `output/asset-review/mount-review/remaining-work.json`.
Updated bundle: `output/asset-review/sophia-sim-2026-10-31-mount-review.zip`.

## Collision-shape review and fixes — 2026-10-31

The original gripper housing STL combines separate motor/cap/plate surfaces.
One convex collider around the complete assembly filled their intervening empty
space, consistent with [MuJoCo's documented convex mesh collision behavior](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).
`sophia_collision_geometry.py` partitions its **12,988 triangles into five exact
connected components** (7,566 / 1,544 / 1,930 / 532 / 1,416 triangles). Each
component receives its own collider on both grippers. Original coordinates,
visuals, mount transforms, body inertias, force limits and masks are preserved.

| Identified case | Review evidence | Result |
|---|---|---|
| Fully lowered arm transition | 57 source-CAD triangles cross the corner-post box, inset by 0.1 mm. | Approximately 2.4 mm intersection retained; path rejected. |
| Hanging endpoint at 100 mm lift | No source-CAD surface crosses the upper rail. The old 18.8 mm penetration came from the whole-assembly hull. | Corrected to **2.904 mm positive separation**; still below the 5 mm clearance margin. |
| Approach to that endpoint | 388 source-CAD triangles cross the rail at the retained worst transition pose. | Approximately 7.3 mm maximum intersection remains; the approach path is rejected. |
| Extreme neck pose | 255 source-CAD triangles cross the camera housing box. | Approximately 11.3 mm intersection retained; pose rejected. |

Across the sampled paths, fully lowered collisions remain **3/101**, 100 mm-lift
collisions fall from **22/101 to 14/101**, and neck collisions remain **11/475**.
Every retained worst contact in those colliding configurations has an original
CAD-surface crossing. Eight false colliding samples were removed. This supports
the geometry decisions in the retained model; it is not an independent physical
clearance measurement or a global proof for unsampled poses.

The same review exposed a query-margin defect: on MuJoCo 3.17, the mesh BVH
midphase culled the 2.904 mm-separated gripper/rail pair even after query margins
were inflated to 5 mm. The separate guard model now disables that midphase and
detects the pair. Live physics retains its original options and zero margins.
No collision pairs were disabled and the 5 mm requirement was not reduced.

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/review_sophia_collisions.py --render --expect-reviewed-geometry --output output/collision-shape-review
```

The review writes original-CAD and actual-convex-hull views, triangle crossing
witnesses, per-component distances, margin-query comparisons and all colliding
sample configurations. Before/after evidence is under
`output/asset-review/collision-review/`; portable provenance is in
`robot/yor-description/calibration/collision_shape_review_2026-10-31.json`.

The collision-corrected snapshot passed **15/15 acceptance gates**, including **18 collision
regression tests** and the two-parcel mobile manipulation task. All **32 generated
files** reproduce exactly. Its historical bundle is
`output/asset-review/sophia-sim-2026-10-31-collision-review.zip`; its work-status
snapshot is `output/asset-review/collision-review/remaining-work.json`. The mount
review above supersedes that snapshot's remaining-work item.

## Dataset system identification — 2026-10-31

`scripts/identify_sophia_response.py` performs bounded nonlinear least squares
on native recorded targets/feedback. It uses runtime-equivalent control sequences
and MuJoCo rollouts, checked against the earlier Python replay to below `1e-8`.
Episodes **0, 2, 4** select parameters; the other **15** evaluate the selected
candidate. These episodes were excluded from this optimizer, although inspected
in earlier release comparisons; this is not a new blind hardware experiment.

For arm fitting, select active-target samples after the first second with the
gripper measured >95% open, commanded open for at least 0.3 s, and low commanded
and encoder-derived base velocity. This reduces unknown payload/contact effects
without proving their absence. The validation mask contains **3,009 samples**.
Masses, inertias, friction, force limits, encoder offsets and timestamp alignment
stay fixed. Three optimizer starts check convergence of shared effective kp/kv.

| Result | Evidence and decision |
|---|---|
| Shared arm gains: candidate kp 275.25 / kv 10.64 | Selected validation RMSE improves 0.005571 → 0.004487 rad, about 19.5%, and aggregate error improves in all 15 episodes. Joint 1 worsens 0.002684 → 0.005691 rad, violating the preregistered per-joint regression limit. **Reject**; keep kp 500 / kv 20. |
| Right gripper opening: fitted 6.038, rounded to 6 strokes/s | With retained arm gains and closing rate 5, validation moving-opening RMSE improves **0.107495 → 0.082016 normalized strokes**, about **23.7%**, across 99 samples. **Apply right opening only**; left opening and both closing directions remain at 5. |
| Timing sensitivity | Assumed command interpolation 33.3/100/133.3 ms yields similar training error but fitted damping about 29.14/10.64/2.00. These are effective response parameters conditional on input reconstruction, not uniquely identified motor constants. |

The adopted gripper rate improves 9 validation episodes and worsens 6. An
episode-level bootstrap gives an RMSE-ratio interval about **0.604–1.025**;
it includes no improvement. The change improves this dataset's aggregate error
and passes the predefined point-estimate gate, but the sparse transitions do not
establish uniform or statistically conclusive improvement on future recordings.
Intermediate gap geometry, grip force and object-constrained closing are not
identified by this fit. Collision trajectory prediction uses the same
direction-specific rate as runtime, and legacy models retain their prior rates.

```bash
.venv-sophia/bin/python scripts/identify_sophia_response.py --output output/asset-review/new-identification-run
```

Use a fresh output directory. Native inputs default to the previously downloaded
`output/asset-review/hf-sanity-check/native-replay/`; the dataset is not included
in the portable asset ZIP. `plan.json` records the fit split, bounds and adoption
criteria before fitting. The summary retains source hashes, masks, all validation
episodes, optimizer results and timing sensitivity. Pre-adoption sources are
archived under `dataset-system-identification/fit-sources/`. The independent
Python replay of the actual adopted configuration is in `adopted-validation/`.
Portable evidence: `calibration/response_sysid_2026-10-31.json`.
Integration results: `output/asset-review/dataset-system-identification/acceptance/acceptance.json`.
Updated bundle: `output/asset-review/sophia-sim-2026-10-31-dataset-sysid.zip`.

The dataset remains useful for further visual object/base-motion reconstruction.
Those observations need scale, optical frames and timestamp validation; the
existing FK end-effector fields and wheel-derived body velocity are not independent
pose measurements. Training policies on the dataset does not resolve parameter
ambiguity or create the missing physical observations.

## Initial right-arm and gripper response — 2026-10-30

`scripts/replay_sophia_dataset_arm.py` uses native timestamps, holds each recorded
target until it changes, assumes the driver's 0.1 s interpolation and excludes
the initialized first second. The arm is mounted at its source orientation with
scene contacts/base acceleration excluded. This isolates unloaded response;
unknown carried objects and exact command receipt timing limit the comparison.

Episodes 0, 2 and 4 select candidates; the other 15 episodes are validation data.
The retained arm model (kp 500, kv 20, no explicit gravity compensation) has
validation per-joint RMSE **0.00240, 0.00190, 0.00289, 0.01248, 0.00258, 0.00227,
0.00818 rad**. Tested gravity-compensation/higher-gain candidates performed worse.
Gripper rates 2, 5, 10 and 20 strokes/s were compared on training data; 5 was best.
Validation moving-opening RMSE decreases **0.42472 → 0.10750 normalized strokes**.
The validation mask contains 99 samples from short opening transitions, not 99
independent trials. Closing against an unknown held object is excluded from the
fit. Both closing and left-side rates use the stated symmetry assumption.

Evidence, source hashes and limitations are packaged in
`robot/yor-description/calibration/response_release_2026-10-30.json`.

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/replay_sophia_dataset_arm.py --root output/asset-review/hf-sanity-check --output output/arm-replay
```

## Additional evidence from the robot dataset — 2026-10-30

The [Pi Atlas 1080p dataset](https://huggingface.co/datasets/adityx23/Pi-Atlas_Sanity_Check_Dataset_1080p/tree/ed7c9f2f1a78ceec46543229aacb0387120ea8f7)
provides native robot telemetry from October 26. The user confirmed that the
grippers, camera mounts, lift and shoulder brackets match the October 29 reference
configuration. The inspected revision is
`ed7c9f2f1a78ceec46543229aacb0387120ea8f7`; its metadata lists 23 episodes, but
only episodes 0–17 have files. All 18 available telemetry files were inspected:
31,804 native samples spanning approximately 17.7 minutes. Downloaded source files
are verified against the revision's Git/LFS hashes.

This dataset supplies right-arm joint commands/feedback, right-gripper normalized
commands/feedback and neck commands/angles for comparison.
Base commands and encoder-derived body velocity are available for diagnostics.
Another generic recording is unnecessary for obtaining those same signals.

The following source limitations are handled by the release assumptions above:

- Lift commands are zero throughout each episode; encoder height is fixed at
  0.59153 m in two episodes and 0.65579 m in the others. These are reported encoder
  coordinates, not measured floor-to-shoulder heights or calibrated sim extension.
- The left arm is never actively targeted and the left gripper is never commanded.
- All recorded base/camera pose fields are null. There are no raw per-wheel
  signals, so encoder-derived body velocity cannot establish independent ground
  motion or repair motor 3's feedback.
- Gripper values are normalized; raw width/effort, measured jaw gaps, object
  masses, joint torques and physical mounting measurements are absent.
- End-effector poses in the supplied driver are forward-kinematics outputs.
  They do not replace external pose measurements for validating mount geometry.
- IMU masks are entirely false. The converted 30 Hz data can repeat source frames
  and accepts camera/robot matches up to 100 ms away. Use native Pi timestamps
  for robot response comparisons and verify image timing for visual calibration.
- Stereo/depth and wrist video can support further visual alignment work; no
  per-recording intrinsic files or camera-to-robot transforms are included. A
  sampled wrist video shows task motion, not an established calibration target.

The analysis and native replay exports are under
`output/asset-review/hf-sanity-check/`. These commands use the downloaded source
files; they make no network connection and do not change model parameters:

```bash
# Analysis interpreter needs numpy and pyarrow; pyarrow is not a sim dependency.
python3 scripts/audit_sophia_dataset.py --root output/asset-review/hf-sanity-check
MUJOCO_GL=egl .venv-sophia/bin/python scripts/replay_sophia_dataset_neck.py --root output/asset-review/hf-sanity-check
```

`dataset-audit.json` records coverage, missingness, timestamps and source hashes.
`neck-replay.json` preserves the original model comparison; new default runs write
`neck-replay-current.json`, using native timestamps and no fitted clock shift. It is a subsystem
benchmark: the base, arms and lift stay at simulation home because the physical
lift mapping was unresolved during that neck-only benchmark. Individual output traces are retained for inspection.
The replay exposed a concrete rate mismatch: the old 1 rad/s tilt command limit
could not follow recorded tilts reaching approximately 100–109 degrees/s. A
2 rad/s candidate was selected using episode 0 and evaluated separately on
episodes 1, 3, 8 and 17. Their pooled tilt RMSE improved from **1.207 to 0.441°**.
Two slow episodes increased slightly (about 0.014° and 0.001° RMSE); the faster
episodes improved substantially. Worst tilt error across the five compared
episodes decreased from approximately **15.9° to 4.94°**.

Sophia now embeds `neck_rate_limit_rad_s = [1, 2]`, in pan/tilt order. Both normal
control and collision-path prediction use these per-axis rates. Legacy models,
pan speed, joint limits, actuator gains and hardware settings retain their prior
values. This is an empirical simulation parameter within the recorded motion
envelope, not a certified hardware speed limit or complete neck calibration.
The evidence and limitations are pinned in
`robot/yor-description/calibration/neck_response_2026-10-26.json`.
