# Sophia simulation assets

## Finalized baseline: v1.0.0 — November 4, 2026

The accepted Sophia assets are frozen as `sophia-sim-v1.0.0-2026-11-04`.
Use `robot/yor-description/scene-sophia.mjcf` as the simulation entry point and
`sophia-ik.mjcf` as its matching IK model. `sophia.json` records the release version,
configuration and accepted assumptions. The versioned ZIP, SHA256 checksum,
full acceptance report and archive verification are under
`output/releases/sophia-sim-v1.0.0-2026-11-04/`. The archive includes the models,
source meshes, generator, pinned calibration profiles, dependency locks, runtime,
training/replay tools, documentation and acceptance evidence, with a per-file
checksum manifest. No robot-side installation is part of this release.

This baseline includes the supplied neck CAD, accepted effective arm/neck
reference alignment, measured chassis/pillar dimensions, stereo-corrected base
placement and user-confirmed wrist cameras above the grippers. Subsequent model
or calibration changes should be released under a new version; retain this
archive for reproducible policy and replay comparisons. Record the release ID
and archive checksum alongside each training run and robot comparison.

The offline geometry/software work is finalized with the documented assumptions.
It is a stable basis for training and staged transfer evaluation. Payload/contact
physics, broad-workspace accuracy and real policy transfer retain their stated
validation limits; the release does not turn estimated parameters into measured
hardware calibration. Historical recording height alignment remains opt-in and
file-hash-bound. The following dated sections preserve the calibration history.


For policy training, the shared single-arm interface and repeatable Gymnasium
pick-and-place task are documented in [SOPHIA_TRAINING.md](SOPHIA_TRAINING.md).
This completes the offline software foundation for phases 1 and 2; learned-policy
training and real-robot transfer validation remain outstanding.

`robot/yor-description/scene-sophia.mjcf` is the Sophia configuration built from
the supplied YOR CAD and the four `PXL_20261029_*.RAW-01.jpg` photographs. The
original `scene-full*.mjcf` models and hardware IK model remain available.
Confirmed physical measurements in `sophia.json` override the old base envelope,
pillar cross-sections, mounting heights, flange spacing, provisional lift travel
and protocol-based jaw opening.

The functional simulation release uses user-authorized assumptions. Collision
shape fixes and offline mount-frame reconciliation are complete, as recorded
in [the work table](SOPHIA_OFFLINE_TOOLS.md#remaining-work--offline-model-consistency-review).
The user has expanded the objective to real-world accuracy and can provide
precise mount CAD. A new recording is now captured and packaged: 75 episodes,
115,791 telemetry frames (about 65 minutes), with source logs, sim-baseline
snapshots, episode arrays and integrity hashes under
`calibration/teleop_2026-11-02/`. Subsequent identification applies per-joint
left-arm position-control gains: development error decreases 34.4%, and a final
check on two intervals of a later episode decreases 44.4%. Every joint improves
in both checks. See `calibration/response_identification_2026-11-03.json` for the
scope, source semantics and rejected candidates. The
[fidelity workflow](SOPHIA_OFFLINE_TOOLS.md#real-world-fidelity-using-cad-and-a-short-recording--2026-10-31)
tracks the remaining parameter identification and real-data validation. See
`sophia.json`'s `release` section for the current defaults. Simulation acceptance results are in
`output/asset-review/pillar-connection-fix/acceptance/acceptance.json`.

## Wrist cameras above grippers — November 4

Both wrist camera assemblies now sit on the user-confirmed upper side of the
stock AgileX grippers. The palm-relative lens offset is `[-42, 0, 52]` mm, replacing
`[42, 0, 52]` mm, with a 180-degree mount roll around the finger approach axis.
Both optical frames rotate with the mount: image-up is upright in the recorded
task pose and the retained 10-degree toe-in points toward the grasp centre. The
visible boards and lenses use the same configured mounting transform; their
positions are no longer separately hard-coded in the builder. In the reviewed
episode-0 frame, the lenses are approximately 29 mm (left) and 38 mm (right)
above their palm origins, rather than 52/45 mm below them. These world heights
vary with wrist pose; the mount is rigid relative to each palm.

Upper-versus-lower placement is user confirmed. Exact offset magnitudes, board
sizes and toe-in remain estimates with the existing stated uncertainty. Measured
camera intrinsics and distortion, head cameras, arm kinematics, body inertias,
actuator settings, base placement and collision shapes are unchanged. Eleven
camera/release/head-alignment regression tests pass. Before/after views, raw
wrist images, checks and a regenerated episode-0 real/sim comparison are in
`output/asset-review/wrist-camera-above/`. The comparison video is
`episode-0-real-vs-sim-zed-cameras-above.mp4` in that directory; earlier base
review artifacts describe the model before this wrist-camera correction.

## Base stereo alignment — November 4

The five-step base review is complete and enabled locally. The previously accepted
head/arm calibration was held fixed throughout. The yellow electronics frame was
about 163 mm too far back and 45 mm too far to the robot's right in the model.
Its corrected chassis translation is `[0.162881, 0.044997, 0]` metres, with X
forward and Y left. The frame retains its confirmed 431.8 mm width, 190.5 mm depth,
203.2 mm height and 63.5 mm floor clearance. Wheel/module locations, the pillar,
shoulder mounts, head CAD, camera intrinsics, actuator settings and assembly
masses are unchanged.

| Step | Result and evidence |
| --- | --- |
| 1. Lock head/arms | Compiled body positions/rotations, joint axes/ranges, cameras, masses/inertias and actuator gains match the accepted model exactly. |
| 2. Reconstruct landmarks | Rectified stereo and yellow-rail line intersections yield 6 training corners from 3 frames in episodes 0/3, and 9 held-out corners from 5 frames in episodes 0/1/2/3. Occluded corners, weak stereo matches and neck speeds above 5 deg/s are excluded. Pillar placement stays fixed because these views do not establish an independent pillar transform. |
| 3. Fit chassis-frame placement | Only frame XY and two recording-specific vertical alignment terms are fitted; no dimension, camera, arm or frame-tilt fitting. Held-out projection RMSE falls from 174.6 px to 16.7 px using geometry alone. |
| 4. Correct visible accessories and historical height | Antennas extend forward instead of sideways; blue e-stop plate moves to the right side; repeated placeholder controllers are replaced by an approximate observed board/finned module. Historical replay alignment reduces held-out error to 8.7 px / 11.7 mm stereo RMSE. |
| 5. Validate and render | 5,119 discrete collision cases: 1,404 named-pose/neck-grid cases and 3,715 trajectory samples across episodes 0/1/2/3/8/17. No new/deeper penetrations over 1 mm. Arm-to-head-camera relative positions stay unchanged within numerical precision. All 36 relevant regression tests pass. |

The height terms are **historical recording alignments**, not new lift encoder
calibration: -36.131 mm for the first static recording setting (episodes 0/1),
and -99.144 mm for the second (episodes 2–17). Native recording file hashes are
required before either can be applied; the later episodes inherit their matching
static setting, with base-landmark validation only in episodes 0–3. Other
recordings and live commands retain the existing lift mapping. The two static
settings cannot identify lift gain, speed, or the cause of the historical offset.
Hardware Home and robot scripts are unchanged. The default `measured_pose` API
also retains zero alignment unless a caller explicitly opts in.

The profile is pinned at
`robot/yor-description/calibration/base_alignment_2026-11-04.json` and enabled by
`sophia.json["base_alignment"]`. `robot/sophia_recording_alignment.py` verifies
recording/profile hashes and calibration dependencies. Both visible geometry and
corresponding frame/plate collision shapes move together. The existing source
assembly inertia is retained; this visual fit does not identify mass distribution.
The proximity-query tests use explicit synthetic rail fixtures to retain their
known 2.9 mm gap and closed-to-open collision cases after the real frame moves;
production geometry and guard margins are not changed to satisfy tests.

Reproduce the offline fit and acceptance from the cached source videos/native
recordings and frozen baseline in `output/asset-review/base-calibration/`:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv-sophia/bin/python scripts/calibrate_sophia_base.py
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv-sophia/bin/python scripts/fit_sophia_base.py
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv-sophia/bin/python scripts/check_sophia_base.py --promote --video
```

The fitter stages a model without changing defaults; the checker promotes only
if its evidence, settings and acceptance gates still match. Reports include
`stereo-base-points.json`, `candidate-profile.json`, `acceptance.json`,
`final-regressions.log`, `completion.json` and `video.json`. The new 55.4-second,
30 fps, 1920×640 comparison is
`output/asset-review/base-calibration/episode-0-real-vs-sim-zed-base-corrected.mp4`.
Useful visible-base checkpoints are 3 seconds and 31–32 seconds. This video
replays measured poses; it does not test physics tracking or recreate the real
table, room or manipulated object.

Residual limits: held-out stereo errors reach about 16 mm, including a small
rail-plane roll/edge discrepancy that was not fitted away. Electronics positions
are approximate to about 20 mm, with hidden faces, rail sections and thicknesses
still estimated. Broad-workspace camera accuracy, dynamics, contact/friction,
payloads and real deployment validation remain separate work. Episodes 8/17
contribute collision/relative-pose checks, not base-landmark accuracy, because
no usable yellow-frame landmarks are visible there.

## Full neck CAD and frame calibration — November 4

The active default now uses the complete reviewed upper spinal assembly, not
only its lattice. `spinal_assembly_2026-11-04.json` records eight extracted part
blocks, hashes, coordinate transforms and cylindrical-surface axis recovery.
The pan axis is about 10.45 mm forward of the centred mount; the tilt axis is
about 41.05 mm forward. The prior model placed the tilt axis only 1 mm from
the centreline and translated the camera to obtain its overhang. This gave the
wrong camera trajectory during neck motion. CAD fixes the articulated path,
while the neck foot stays centred on the pillar and the neutral lens midpoint
retains its confirmed 50.8 mm overhang. The supplied stereo baseline and
intrinsics are unchanged. These CAD fit residuals describe the tessellation,
not independent real-world metrology.

The remaining large image error was isolated to the right arm. A training-only
effective right-arm reference rotation of about 5.8 degrees corrects it without
moving flange centres, changing link lengths or changing any measured joint
values. This can include joint-zero/mount-reference effects and must not be
interpreted as a measured bracket angle. The neck's fitted geometric neutral
is hardware pan 1.642963 degrees / tilt -16.258976 degrees; hardware Home remains
pan 2.02 / tilt -3.25. The local physics and IK models use the same calibrated
arm reference. The sim-only hanging reference is recomputed with IK to preserve
its previous end-effector pose; its old joint angles are still correctly blocked
near the chassis at low lift. Hardware Home joint targets remain unchanged.

The complete process was run and passed acceptance on 34 reviewed finger-pair
observations across episodes 0, 1, 3, 8 and 17. Only episode-0 training labels
enter fitting. Validation left-eye RMSE is 5.5–16.9 pixels and right-eye RMSE is
3.0–18.7 pixels; triangulated validation point errors are about 5–12 mm. An
independent approximate wrist-feature check improves from 102 to 29 pixels
(feature-centre annotation uncertainty about 20 pixels; not used in fitting).
The joint-reference and arm-frame corrections are saved together in
`image_frame_alignment_2026-11-04.json` with hashes and scope.

The corrected CAD bracket parts remove the legacy fixed-bracket camera overlap.
All acceptance gates pass across 936 calibrated named-pose combinations and
2,985 native trajectory samples: no new/deeper penetration over 1 mm. Baseline
full-CAD geometry has no sampled penetration over 1 mm either. Finger rays and
56 segmented stereo views per model show no neck/shoulder mount obstruction.
The source mass/inertia assumptions are retained, with COM frames updated;
this is geometric calibration, not new dynamic identification. Visual validation
covers roughly 30–46 degrees tilt and a limited manipulation workspace. Other
workspaces, payload/contact physics and independent hardware deployment remain
separate validation tasks. Recorded Pi EE fields derived from its legacy FK are
not independent pose ground truth; joint-based replay preserves measured angles.

Reproduce the isolated build, training-only fit, arm/wrist audit, collision and
visibility checks, and video with the cached data:

```bash
MUJOCO_GL=egl .venv-sophia/bin/python scripts/run_sophia_full_frame_calibration.py
```

Add `--promote` to enable a passing result locally, or `--no-video` to omit
rendering. Failed acceptance returns a nonzero status and leaves defaults alone.
The process never connects to the robot. Fresh STL extraction, when needed,
uses `scripts/extract_sophia_spinal_assembly.py`; the bundle can build from its
pinned generated meshes without the original CAD export. Inputs and outputs are
under `output/asset-review/full-neck-cad-calibration/`, including
`combined-fit-report.json`, `arm-frame-audit.json`, `acceptance.json`,
`canonical-verification.json`, and the final 55.4-second video
`episode-0-real-vs-sim-zed-calibrated.mp4`. The final default projection matches
the accepted candidate to numerical precision. No Pi/Orin controller files or
`joystick.py` were changed.

The earlier November 4 static-only update below is retained as an audit trail
and is superseded by the complete assembly and accepted frame calibration.
It uses the supplied `CAD Parts/SPINAL DESIGN.stl`.
The reviewed static lattice is extracted without shape scaling: its source
millimetres are converted to metres and its frame is rigidly aligned with the
confirmed pillar centreline. The 12,983 retained triangles include disconnected
surface patches. The lattice envelope is 75 mm fore/aft by 90 mm left/right by
108.950 mm high. This height replaces the 138.280 mm photo estimate for the upper
neck assembly location. The new visual mesh retains the CAD lattice openings;
post and end-plate collision proxies are resized to its envelope. Approximate
source inertials and the legacy pan/tilt bracket geometry remain uncalibrated.
The 50.8 mm neutral lens overhang, centred mounting, servo Home and recorded
angle conventions remain unchanged. Provenance is in
`calibration/spinal_neck_geometry_2026-11-04.json`; reproduction uses
`scripts/extract_sophia_spinal_neck.py` with the pinned source STL.

The recording-offset audit checked the collection schema's `vr-teleop` commit
`6a432bf` and every available native/converted pair across 18 dataset episodes.
The historical controller generates pan from relative headset yaw plus Home pan,
and tilt from signed headset pitch relative to the horizon plus Home tilt.
The published `neck_deg` is measured `Present_Position` in the calibrated servo
frame; `neck_cmd_deg` is the final calibrated hardware goal. Neither is a
per-episode delta. The native arrays match raw telemetry exactly, and converted
state/action neck fields match the referenced native rows within 0.000002 degrees
(float32 rounding). The episode codec's delta compression preserves the initial
absolute sample and cumulatively decodes the original values. No `view_pitch_deg`
setting exists in these Oct26 raw episodes; converted placeholders are masked
invalid. Do not add Home, headset recentering or view-pitch offsets again during
replay. These angles are relative to calibrated servo travel, not an independently
measured optical horizon. Historical per-episode servo calibration snapshots are
still unavailable. Audit evidence and source snapshots are under
`output/asset-review/spinal-cad-review/neck-offset-audit/`; reproduce with
`scripts/audit_sophia_neck_recording.py` in the analysis environment.

The STL correction alone reduces the reviewed episode-0 held-out landmark
projection RMSE from 229.32 to 214.47 pixels; the camera-to-arm mismatch remains.
A diagnostic optical pitch correction of 9.71 degrees downward reduces that
error to 59.42 pixels (59.49 pixels on previously inspected episode 1). This
diagnostic is not enabled in the assets: image fitting cannot yet separate
optical mounting pitch from the hardware-to-model angular reference. Results
are in `output/asset-review/spinal-cad-review/post-stl-diagnosis/`.
The updated 55.4-second comparison is
`output/real2sim-videos/pi-atlas-episode-0/episode-0-real-vs-sim-zed-stl-neck.mp4`.

Read-only Orin inspection on November 4 traced `~/yor_ws/start_collection.sh`
through `collect.py`, `recorders.py`, `zed_rgbd.py`, and the actual dataset
converter (`tools/pi_atlas_sanity.py` and `tools/pi_atlas.py`). Neither collector
nor converter adds a neck angle offset. ZED colour comes from the SDK's rectified
left/right views; the inspected capture path has no application pitch rotation,
crop or flip. The archived headset client vertically centres display planes,
which does not transform recorded pixels. Episode 0's original camera metadata
reports focal length 1072.393799 pixels versus the later calibration's
1072.864624 pixels (0.044% difference). All 399 exported episode records mark IMU
absent. The unresolved angular reference is still the simulation's use of Home
as `neck_zero_deg`, rather than a separately established optical neutral pose.
Source snapshots, hashes and findings are in
`output/asset-review/orin-collection-offset-audit/`. Current Orin capture source
postdates Oct26; original episode metadata and the numeric recording audit
corroborate these findings but cannot reconstruct every historical SDK setting.

The Pi follow-up traced its active `robot/whole_body_teleop.py` through the bridge
and checked the installed LeRobot motor normalization source. Pi and local neck
calibration match exactly. Servo feedback is `(Present_Position - range_midpoint)
* 360/4095`; the servo homing offset has already been written to hardware.
`yor.py` loads Home through `Neck`, rather than defining an optical neutral pose.
The calibration utility saves Home from an operator-positioned neck, so it does
not independently establish the optical horizon.

`scripts/diagnose_sophia_neck_reference.py` compares actual articulated FK with
camera-only rotation on 25 reviewed fingertip-pair observations. Reversing tilt
sign gives over 4600 px held-out error; using hardware zero as geometric zero
still gives 145.6 px; applying Home twice gives 289.3 px. Fitting a full-assembly
geometric tilt offset of +9.159 degrees gives 63.9 px on episode-0 held-out
labels and 64.9 px on episode 1, versus 214.5 and 233.7 px currently. It would
correspond to hardware geometric-zero tilt -12.409 degrees while retaining
hardware Home -3.25 degrees. Optical-only pitch gives similar errors, so the
physical causes remain ambiguous. No candidate was enabled in runtime assets.
Evidence, source hashes and a three-frame visual comparison are in
`output/asset-review/pi-neck-reference-audit/`. Both robot processes and files
were left untouched.

The complete offline calibration process is now available as:

```bash
bash scripts/run_sophia_head_calibration.sh
```

It requires the cached stereo MP4s and native/converted telemetry for episodes
0, 1, 3, 8 and 17 under `output/asset-review/head-calibration-pipeline/data/`.
Their hashes are saved in `data/manifest.json`; episode 0's left video matches
the previously reviewed source byte for byte. `SOPHIA_DATA_PYTHON` selects a
Python with numpy/pyarrow; `SOPHIA_SIM_PYTHON` defaults to `.venv-sophia/bin/python`.
The launcher rebuilds, prepares timestamp alignment, fits, checks and renders.
It returns status 1 when acceptance fails, after writing the reports and video.
No stage connects to or commands the robot. Repeated checks can skip video with
`--no-video`. The separately run mounting/controller/video regression suite
passed all 28 tests.

Home and geometric neutral are separate generator inputs: `neck_reference` in
`sophia.json` defines geometric zero in hardware degrees. Hardware Home remains
in `robot/neck/neck_calibration.json`. Joint limits and every saved pose/controller
target are mapped consistently when zero changes. The current default zero is
unchanged pending acceptance. Candidate XMLs preserve Home and transform the
camera housing, lens visuals, collision shapes, sites and both optical cameras
together. Camera changes do not shift the neck/pillar centreline, static lattice,
neutral lens XY, stereo baseline or intrinsics. Tilt-body inertials remain approximate.

The November 4 run used 34 finger-pair observations: 11 episode-0 training labels
and 23 validation observations across five episodes. Stereo matching uses
epipolar, reverse-match and pair-consistency rejection. Nine episode-0 stereo
correspondences were additionally inspected by eye to resolve repeated-finger
ambiguity; displaced old matches at 50/52 s were not manually endorsed. Seven
training and eleven validation stereo pairs survived. Labels and review provenance
are in `calibration/head_camera_labels_2026-11-04.json`; new validation labels have
an estimated 14-pixel annotation uncertainty. Fitting uses training labels only.
These are development validations, not a final untouched test set.

Camera frames and native robot samples are matched on Orin `capture_ns/recv_ns`,
not by assuming Pi clock equality. Left/right captures must agree, stale samples
are masked, and downloaded native row timestamps must match the existing exports.
The selected non-boundary solution rotates the rigid camera assembly downward
9.664 degrees, with no joint-zero or positional change. Its actual compiled FK
agrees with the fitting projection to better than 1e-5 pixels. It improves each
validation episode by over 50%, but left-eye RMSE remains 37.4–74.4 pixels.
Validation stereo-point errors are about 26–43 mm, with only one reliable pair
in episode 8. It fails the fixed 25-pixel left-eye, 35-pixel right-eye and
30-mm stereo-point acceptance requirements. More flexible fits hit the allowed
mount adjustment bounds and were excluded from selection.

Collision comparison covered 936 pan/tilt/lift/arm-pose combinations and 2,985
native trajectory samples (about 10 Hz): no new penetration over 1 mm or existing
penetration deepened by over 1 mm. Both models already have 32 colliding sampled
poses, including fixed neck bracket versus ZED housing at extreme pan and
60–73-degree down tilt (examples reach about 11 mm). These have not been hidden
or removed by calibration. Finger sightline rays found no mount obstruction;
segmentation found zero mount pixels in 56 reviewed views per model. The visually
validated neck tilt range is only 29.8–46.1 degrees, so those observations do not
validate imagery at mechanical travel limits.

The candidate remains separate and **is not promoted**. Results are in
`output/asset-review/head-calibration-pipeline/fit-report.json` and
`acceptance.json`; the full 55.4-second measured-state comparison is
`episode-0-real-vs-sim-zed-calibration-candidate.mp4` in that same directory.
Its timestamps and measured poses are reconstructed for the new mapping rather
than reinterpreting old physics-replay qpos. Unresolved work is the relative
arm/camera geometry behind the residual image error, plus the extreme-angle
fixed-bracket collision geometry; the recording offset hypothesis is unsupported.

From the YOR directory, launch the robot with its existing controller:

```bash
python robot/yor_mujoco.py --model robot/yor-description/scene-sophia.mjcf --no-video
```

For VR, the launcher now forwards `--model` to the simulator:

```bash
python robot/sim_vr_teleop.py --model robot/yor-description/scene-sophia.mjcf
```

The usual VR prerequisites still apply, including the `vr-teleop` submodule.
For standalone stereo video, omit `--no-video` and install `opencv-python` as
described in the VR setup guide.
Neither command changes the physical robot's configuration. The original
`--gripper agx|custom` selections still work when `--model` is omitted.

## Appearance and geometry

- The upper lift pillar meets the underside of the metal shoulder plate directly.
  Its visual and collision envelopes share that attachment and move rigidly with
  the plate. This corrects the former 101 mm gap without moving the arm mounts.
- Nero arm visuals use **the original material assignments in the DAE assets**,
  including black housings, silver panels and modelled branding. The generator
  exports each material's triangles into an STL; it retains all geometries,
  including the second mesh in `link5.dae`. Original STL meshes remain the arm
  collision shapes. Materials are converted from COLLADA linear RGB to display
  RGB. Visual meshes use shell inertia because some logo surfaces have no volume;
  they have zero density and do not change the explicit body inertias.
- Yellow upper rails, aluminium lower rails/posts, the open electronics bay,
  blue e-stop panel, black column, shoulder brackets, lattice neck tower, camera
  lenses, rear computer, antenna and representative wiring follow the photos.
- The source arm joint axes, link lengths, limits, end-effector frames and
  actuator types are preserved; response gains follow the documented fits.
  Both grippers use verified stock AgileX finger meshes,
  with black fingers and silver housings. The installed housing revision and
  adapter transform retain the supplied CAD values as accepted approximations.

The measured inside jaw gap is **101.6 mm open and 0 mm closed**. The model uses
two symmetric 50.8 mm slides and an assumed linear normalized opening command.
Right-gripper opening uses 6 strokes/s after dataset identification and separate
episode evaluation; left opening and both closing directions retain 5 strokes/s.
Those unchanged directions remain assumptions; the right-opening improvement is
aggregate across the validation set rather than uniform across episodes.
The builder records
the nominal **70 mm protocol coordinate** from `robot/arm/can_gripper.py` separately
from the physical gap. It does not change the CAN driver or hardware calibration.
It also reads
`robot/neck/neck_calibration.json`: pan travel is about 251.077 degrees, tilt
travel 145.055 degrees, and saved home is pan 2.02 / tilt -3.25 degrees.
Like `robot/neck/neck.py`, these angles are centered on each calibrated tick
range and use 360/4095 degrees per tick for the STS3215. Raw tick 2048 is not
the API's zero. The simulator reports these calibrated angles and homes to the
saved values; legacy models retain their original zero-centered convention.
The user subsequently confirmed that the neck mount and pillar share the same
vertical centreline, with the arm mounting plate between them. This fixes the
neck mount's horizontal position at the pillar centre. The original default
model already has this arrangement. Earlier fitted candidates that translate
the entire neck sideways or forward are superseded; the generator rejects
activating those offsets. Their image improvements do not establish correct
physical mounting dimensions. Resolve viewpoint errors within this centred
assembly, including the camera-to-axis geometry and provisional plate envelope.

The user also confirmed that the camera lens plane overhangs that centreline
by 2 inches (50.8 mm), stays centred left-to-right, and aligns with the metal
arm plate's front edge. The generator now fixes the neutral stereo midpoint
at Lift_Top XY `[-0.0508, 0]`, moving the ZED housing, optical frames and collision
shapes together. The stereo baseline is unchanged. This is a neutral-geometry
reference assumption; the horizontal projection changes with pan/tilt, and the
optical angular zero remains provisional. The metal plate front extent is
50.8 mm; rear symmetry and bounding the black mounting plate to that envelope
are explicit assumptions. Neck/pillar coaxiality remains fixed. Twenty-five
mounting, collision and video regression checks pass after this update.

The November 3 optional dataset alignment previously estimated a replacement for the
provisional Home-to-horizon assumption. It is **disabled in default assets**:
the candidate fails the existing full-downward-neck clearance acceptance check
at API pan 2.02 / tilt 72.527 degrees. The collision guard remains active.
`calibration/head_alignment_2026-11-03.json` fits a fixed 9.87-degree downward tilt
alignment and a whole-neck mounting offset from episode 0's measured joints and
reviewed fingertip-pair midpoints, with additional rectified stereo constraints.
The rigid bodies carry camera housing, collision shapes and inertials together;
servo Home, joint travel and camera intrinsics retain their hardware values.
For the candidate at 1920x1080, withheld episode-0 landmark RMSE improves from
222.7 to 13.5 pixels.
Episode 1 was excluded from fitting: its eight reviewed pair midpoints improve
from 239.2 to 25.0 pixels RMSE. These are image-alignment results, not centimetre
accuracy measurements of physical mounts. Arm-visible training views cover only
about 36-46 degrees of tilt, and tilt/translation remain correlated. Yaw and roll
retain their nominal values. Other neck poses and individual finger silhouettes
remain unverified. Fit inputs, sensitivity estimates, before/after images and
the review video and opt-in candidate scene are in
`output/asset-review/head-camera-fit/`; reproduce with
`scripts/fit_sophia_head_camera.py --help`. Resolve the bracket/camera clearance
model before promoting this fit to the default assets. No hardware settings are modified.

The `reference` keyframe has hanging arms and 0.400 m of lift extension for visual
comparison. This is an estimated photo pose, not recovered joint encoder data.
The `home` keyframe uses the controller's original bent-arm posture and 0.300 m
extension, leaving lift travel in both directions. The simulator starts in
`home`; rendered reference views use `reference`.
The additional `lowered` and `raised` keyframes use the home arm posture at the
measured lift endpoints. The validator renders both using the same camera.

The builder also generates `sophia-ik.mjcf` with the same shoulder height and
flange spacing as the physics model. A model-embedded `ik_model` path makes the
simulator select it automatically. Original arm link geometry, joint axes and
joint limits are unchanged. Cartesian poses remain lift-independent, expressed
in the calibrated model's frame at zero lift; they are no longer numerically
identical to poses from the old hardware IK model. Hardware files are untouched.

## Dimensions and physical assumptions

Dimensions and provenance are recorded in
[`sophia.json`](../robot/yor-description/sophia.json).

| Quantity | Value | Basis |
|---|---:|---|
| Base frame front-to-back × side-to-side | 190.5 × 431.8 mm | Confirmed 7.5 × 17 inches; outside frame surfaces, excluding wheel/module and accessory overhangs |
| Base frame bottom-to-top height | 203.2 mm | Confirmed 8 inches; excludes ground clearance |
| Floor to frame bottom / top | 63.5 / 266.7 mm | Confirmed 2.5-inch clearance; top is clearance plus frame height, or 10.5 inches |
| Lower pillar left-to-right × front-to-back | 88.9 × 50.8 mm | Confirmed 3.5 × 2 inches |
| Upper pillar left-to-right × front-to-back | 76.2 × 50.8 mm | Confirmed 3 × 2 inches |
| Middle pillar left-to-right × front-to-back | 82.55 × 50.8 mm | Provisional interpolation; not measured |
| Wheel diameter | about 101.5 mm | Original wheel mesh |
| Shoulder joint height, lift lowered | 812.8 mm | Confirmed floor-to-shoulder measurement: 32 inches |
| Shoulder joint height, lift raised | 1371.6 mm | Confirmed floor-to-shoulder measurement: 54 inches |
| Nominal shoulder height in the reference pose | 1212.8 mm | Measured lowered height plus estimated 400 mm extension |
| Physical lift travel | 558.8 mm | Difference of confirmed shoulder heights: 22 inches |
| Encoder-reported travel | 672.6 mm | Used only in the accepted affine mapping to 558.8 mm physical travel |
| Arm mounting-flange separation | 241.3 mm | Confirmed 9.5 inches, modeled symmetrically about the column |
| Inside jaw gap, closed / open | 0 / 101.6 mm | Confirmed 0 / 4 inches, applied to both grippers |
| Rail/post cross-section | 20 × 20 mm | Photo estimate; measured frame envelope does not establish rail thickness |
| Middle column / upper assembly envelope length | 430 / 531 mm | Estimated middle; upper extends to the shoulder plate while retaining its lower endpoint |
| Total simulated robot mass | 42.0053061 kg | Matched vendor arm/finger inertials; other source bodies retained |

The frame and lower/upper pillar measurements update both visual and collision
geometry. Frame top is **266.7 mm above the floor**, not 203.2 mm. Wheel/module
positions and mounting-plate overhangs retain the source CAD coordinates; the
new frame envelope does not establish wheelbase or track. Accessory placements
remain estimates. Body masses and inertias are not inferred from these outside
dimensions.

The 558.8 mm physical height difference supersedes the previous provisional
416 mm simulation travel. The encoder's 672.6 mm differs by **113.8 mm**. The release
assumes a 558.8/672.6 affine scale, with the summary's 674.4 mm upper hard stop
implying a 1.8 mm lower encoder coordinate. This preserves the measured physical
travel while deriving 22.008 mm/s up and 25.589 mm/s down from the supplied motion
summary. `robot/sophia_profile.py` exposes the conversion separately from the
physical-extension simulator API. Hardware encoder calibration is untouched.

The chassis starts upright with its CAD tyre surfaces on the floor. Carriage
height is then set from the measured shoulder center, so initial floor clearance
does not introduce the old roughly 7 mm height error. Physics settling introduces
small contact/servo deflection: validation checks both shoulder centers against
the tape measurements with a 3 mm tolerance.

Sophia uses two moving stages: `lift_joint` is total carriage travel and
`sophia_middle_slide` follows half that distance. This **stage ratio and the
430 mm middle section length remain assumptions**. The upper envelope spans
531 mm to meet the shoulder plate; that length is derived from the modeled
attachment, not a new physical measurement. The original
middle-stage mass moves with that stage. The validator checks that the section
proxies retain overlap through the measured 558.8 mm travel; this does not
establish the real section dimensions. Absolute lift commands now default to
the loaded model's range, allowing the full stroke beyond the former 500 mm
API cap. An explicit caller-supplied limit is still honored.

The source geometry and adopted release assumptions are:

| Part | Established evidence | Adopted approximation |
|---|---|---|
| Lift | Confirmed end sections, shoulder heights, stroke and rigid plate attachment | Interpolated 430 mm middle section, 531 mm upper envelope reaching the plate, half-speed middle stage, explicit encoder mapping |
| Base | Confirmed outer dimensions and clearance; source swerve layout | 20 mm rails/posts and source wheel/module positions |
| Shoulders | Measured 241.3 mm spacing and floor heights | Retain source fore/aft and vertical flange offsets and bracket orientation |
| Grippers | Stock Nero identity, matching finger meshes and measured endpoint gaps | Supplied housing/adapter, 132 mm end-effector-to-grasp distance, linear gap mapping; fitted right opening at 6 strokes/s, left opening and both closing directions at 5 |
| Neck/cameras | Calibrated neck range/home and supplied lens intrinsics | Photo-estimated tower, CAD head mount and photo-estimated wrist mounts |
| Accessories | Placement visible in the photos | Visual dimensions/supports; mass included in source assembly inertials |

## Remaining work and accepted assumptions

Lift mapping and direction-specific speed are **implemented and tested**, using
the Pi log and accepted physical/encoder scale. Base/pillar measurements, gripper
endpoint gaps, right-arm/gripper response comparisons and calibrated camera lenses
are also complete. The user confirms that physical gripper and wrist-camera
mounts have not changed. Existing-asset frame comparison is complete: Sophia
preserves the stock AgileX mounting chain. Older Piper/iPhone and custom-gripper
camera mounts belong to different assemblies and provide no supported replacement
for the current photo-estimated wrist transforms.

| Category | Remaining work for real-world accuracy |
|---|---|
| CAD / simple reference information | Offered precise mount CAD and representative object masses from an ordinary scale or labels with uncertainty. |
| On-robot observations | Recording completed: 75 episodes are now bundled. Lift stayed stationary; raw force and independent camera/base observations were not in this stream. Use available data before requesting further observations. |
| Offline work | Left-arm response fit applied and command semantics resolved. Import offered CAD and investigate independent visual motion, timing uncertainty and remaining mass/contact uncertainty. Broader payload/contact accuracy remains unvalidated. |

The detailed evidence and adopted decisions are in the
[offline work table](SOPHIA_OFFLINE_TOOLS.md#remaining-work--offline-model-consistency-review).
Collision review is complete. Five exact connected-component housing colliders
replace the single hull that incorrectly bridged empty CAD space. The former
18.82 mm overlap at the 100 mm hanging endpoint becomes 2.904 mm separation.
This remains inside the 5 mm guard margin; intermediate motion still has a
CAD-supported intersection of about 7.3 mm. The fully lowered post interference
(2.4 mm) and extreme-neck interference (11.3 mm) also cross source CAD surfaces
and remain rejected. See the
[review evidence](SOPHIA_OFFLINE_TOOLS.md#collision-shape-review-and-fixes--2026-10-31).

Mass distribution, friction, grasp force, left gripper response and
unmeasured construction details remain accepted approximations. Left-arm
position-control gains now have limited real-trajectory validation. These are not
new physical-work requests. The current wrist extrinsics retain their
photo-estimate label: the completed source-frame review found no supported
replacement. Both current mounts remain rigid across 20 sampled arm/lift poses,
and the gripper transforms match their stock source. This closes the offline
review without claiming measured hardware extrinsics. That functional release
remains the baseline for the expanded real-world fidelity work above; its passed
simulation tests do not replace independent physical validation.

The subsequent [dataset system-identification pass](SOPHIA_OFFLINE_TOOLS.md#dataset-system-identification--2026-10-31)
fits on three episodes and evaluates on the other 15. Right opening is now
6 normalized strokes/s, reducing pooled moving-opening error by 23.7%; left
opening and closing rates remain 5. This improves 9 episodes and worsens 6, so
the aggregate improvement is provisional. A shared arm-gain candidate lowered
average error but doubled joint-1 error and was rejected. Physical masses,
friction and grip force have not been inferred from policy-training success.

## Source geometry notes

The chassis is rotated approximately 90 degrees about Y by the MJCF. Its 543 mm
mesh-X extent is therefore vertical and includes the fixed lift structure; it
is not a 543 mm ground footprint. Similarly, `Lift_Top.stl` includes mounting
structure: its full bounding-box height is not a bare telescoping tube length.
The 430 mm middle length and 531 mm upper envelope remain modeling estimates;
the upper endpoint is constrained to the shoulder plate's underside.

`assets_nero/gripper.STL` has raw bounds spanning approximately 55.33 × 44.13 ×
60.43 source units. STL has no unit metadata; these coordinates are consistent
with millimetres, unlike the metre-scale finger meshes currently mounted in the
model. No existing MJCF/URDF references this file. Its units, part identity and
mount transform must be established before substituting it for those fingers.

Gripper CAN feedback is available, but the driver inverts its width coordinate
when configured to do so, normalizes by 70 mm, and clamps the result to [0, 1].
It also commands endpoint overtravel. A normalized reported position is thus
not an independent physical jaw-gap measurement, and may hide overtravel.
The confirmed endpoint gaps are now known; corresponding raw feedback values
would establish the mapping between protocol coordinates and those physical gaps.
No hardware was connected or commanded while building these assets.

Most original robot collision geoms used `contype=0, conaffinity=1`, preventing
robot-to-robot contact. Sophia enables those contacts, uses mesh hulls for arms
and fingers, five source-connected hulls per gripper housing, and explicit
primitive collision shapes for the new structure. Every original housing
triangle is preserved; the partition removes only whole-assembly hull bridging.
Adjacent wrist fork parts, telescoping stages and chassis/tyre attachments have
targeted exclusions to avoid contacts inside overlapping CAD/proxy geometry.
The two fingers of each gripper are coupled. Visual wiring, labels, electronics
details and antenna parts do not add contact surfaces or inferred mass.

Arm and finger masses, centers of mass and full inertias now use the pinned
official AgileX URDFs, verified against identical local meshes. This resolves the
1 mg link7 error and conflicting arm inertial sources. Chassis, lift, housing,
accessory masses and friction remain source approximations. The arm gains are
retained after comparison with all 18 available native recordings. Geometry,
contact dynamics and camera mounts are adequate for the tested simulation tasks;
payload, tipping and force predictions remain approximate. Independent physical
calibration is optional future work rather than an unfinished asset requirement.

## Rebuild and inspect

The generator needs NumPy and MuJoCo. Rendering additionally needs Pillow and a
working OpenGL backend. Tested with MuJoCo 3.11 and 3.17; use a recent MuJoCo for
the visual mesh `inertia="shell"` attribute.

This checkout has an isolated `.venv-sophia` environment installed from the
hashed `requirements-sophia.txt` lock. Activate it with
`source .venv-sophia/bin/activate` before the commands below. See
[standalone setup and offline tools](SOPHIA_OFFLINE_TOOLS.md) for installation,
the asset/inertia audit, calibration preparation and portable packaging.
Run the complete offline suite with `python scripts/check_sophia.py --render`.

```bash
python robot/yor-description/build_sophia.py
MUJOCO_GL=egl python robot/yor-description/validate_sophia.py --render
python robot/test_yor_mujoco.py --model robot/yor-description/scene-sophia.mjcf
python robot/test_sophia_collision.py
MUJOCO_GL=egl python robot/yor-description/evaluate_sophia.py
```

Omit `MUJOCO_GL=egl` on systems using the normal windowing backend. The asset
validator writes front, side, rear and perspective views, lowered/raised views,
a collision view and `validation.json` into `output/asset-review/`. It verifies
unchanged arm kinematics relative to each flange, vendor inertias and expected
total mass, agreement with
the companion IK model, measured frame envelope/clearance and pillar sections,
measured flange spacing and settled shoulder heights,
wheel contact, stage overlap/coupling, actual jaw-mesh gaps, and a 25 g object
held by each gripper under gravity. These are functional
checks of the model, not measurements of the real robot.

## Extended evaluation results — 2026-10-29 baseline

The metrics below describe the original baseline. The 2026-10-30 vendor-inertia
and camera update is checked separately in
[`2026-10-30-update/acceptance.json`](../output/asset-review/2026-10-30-update/acceptance.json);
the measured-base/pillar update is recorded in
[`measured-base-pillar/acceptance.json`](../output/asset-review/measured-base-pillar/acceptance.json).
The latest asset validator includes 23 checks, including the measured visual and
collision envelopes. In the measured-frame baseline, the hanging-arm transition had
3/101 colliding samples at zero lift (maximum 2.41 mm penetration) and 22/101
at 100 mm lift (18.82 mm). Both paths remain rejected; the separately tested
300 and 558.8 mm paths remain clear. The neck sweep is unchanged. Refer to the
latest report for current hashes, logs and results; the following baseline
numbers are retained for comparison.
The October 30 release, including wrist cameras, lift-rate mapping and gripper
response, was checked in
[`release/acceptance/acceptance.json`](../output/asset-review/release/acceptance/acceptance.json).
The October 31 collision correction is checked in
[`collision-review/acceptance/acceptance.json`](../output/asset-review/collision-review/acceptance/acceptance.json).
Its 100 mm-lift path has 14/101 colliding samples with about 7.3 mm maximum
penetration. All remaining tested intersections have source-CAD surface support;
the approach remains blocked. At zero lift and in the neck sweep, counts are
unchanged. Historical numbers below describe their respective earlier models.

The model passes the tested operation checks, and the controller rejects the
identified self-colliding paths. This does **not** mean every combination of
joint angles is collision-free. Results on MuJoCo 3.17.0 at a 5 ms physics timestep:

| Suite | Result | Coverage |
|---|---:|---|
| `validate_sophia.py --render` | 21/21 passed | Measurements, matching IK, wheel contact, lift coupling, jaw travel, static 25 g grasps, review renders |
| `test_yor_mujoco.py --model robot/yor-description/scene-sophia.mjcf` | 15/15 passed | Cartesian arm commands, home, grippers, lift, neck and base through the simulator API |
| `evaluate_sophia.py` | 16/16 passed | Long holds, guarded configuration sweeps, simultaneous motion, driving at both lift endpoints, stereo RGB/depth rendering |
| `test_sophia_collision.py` | 16/16 passed | Alignment, clear/rejected commands, pending motion, runtime hold, atomic bimanual rejection, cancellation and bend–raise–unfold sequence |
| `test_yor_mujoco.py --gripper agx` | 15/15 passed | Regression of the original model with the shared controller |

The generator preserves the source CAD alignment between the fixed upper neck
bracket and pan pivot when recentering them on the tower. Camera collision
geometry matches the visual housing's central box and rounded endcaps. Its
collision masks remain enabled, and calibrated pan/tilt travel is unchanged.

The extended evaluator retains raw collision counts and screenshots for these
restricted configurations and requires the guard to reject them:

- **Neck interference:** 11 of 475 sampled pan/tilt combinations have more than
  0.1 mm robot-to-robot penetration. Worst overlap is 11.28 mm between the camera
  housing (`head_tilt`) and fixed upper neck bracket (`head_neck`), at API
  pan −125.54° / tilt +72.53°. All 11 colliding combinations are rejected. Clear
  extreme angles remain accessible; usable tilt depends on pan and the path.
- **Hanging-arm transition at zero lift:** 7 of 101 joint-space samples collide;
  worst overlap is 7.82 mm between a finger and the lower base rail.
- **Hanging-arm transition at 100 mm lift:** 27 of 101 samples collide; worst
  overlap is 13.98 mm between a gripper housing and an upper base rail. The same
  path has no detected penetration at the separately tested 300 and 558.8 mm
  lift extensions. Those samples do not establish a continuous safe workspace.

These are intersections of the current simulated collision shapes. Whether they
represent real interference or inaccurate proxy geometry remains unverified.
They are now blocked by the controller. The evaluator does not remove the raw
collisions from its report or claim that those poses became physically reachable.

### Guarded simulation commands

Sophia's model metadata opts into a **5 mm self-clearance margin**. Preflight
checks use the live physics frame, both arms, both jaw openings, lift height,
neck angles and pending goals. Independent joint arrival times are sampled with
at most 20 mrad or 2 mm per joint step, including intermediate configurations;
checking endpoints alone would miss the gripper passing through an upper rail
during lift descent. Link-length and geometry-radius bounds further refine
sampling so a long arm cannot sweep over a thin rail between coarse angular
samples. Configurations already inside the clearance margin can retreat while
increasing separation. The clearance margin applies only to a separate query
model, so physics contacts, friction and grasp behavior are unchanged.
The query model disables mesh BVH midphase culling so that positive mesh/box
gaps within the inflated margin are detected; the corrected 2.904 mm
gripper/rail case is a regression test. Live physics keeps its original midphase
setting and zero geom margins.

At every physics step, the guard checks the next commands and a 50 ms velocity
projection. If that check fails, it cancels pending manipulator motion and holds
position. This is a sampled simulation guard, not a continuous collision proof
or a hardware safety controller. It respects the model's existing exclusions,
ignores intentional contact between the two fingers of the same gripper, and
does not check world obstacles or certify payload/braking performance. Neck
commands were rate-limited to 1 rad/s in this baseline. The dataset-backed update
uses 1 rad/s pan and 2 rad/s tilt; see `SOPHIA_OFFLINE_TOOLS.md` for the recorded
trajectory comparisons and remaining calibration limits.

Arm, gripper and lift commands return `False` on rejection and leave previously
accepted goals intact. Bimanual Cartesian targets are committed together or
rejected together. Neck responses retain `pan`/`tilt` and add `accepted`.
`get_motion_status()` reports the last action and rejection, including the
offending geoms, measured clearance and required clearance. The same status is
included in `get_collection_state()["motion_status"]`. A blocking lift request
returns `False` if stopped, superseded or canceled by the runtime guard.

`move_to_reference_pose()` is an explicit simulation-only sequence: move both
arms to their bent home posture, raise the lift to at least **300 mm**, then
unfold into the reference hanging posture and open the jaws. Each leg passes
the guard; it returns `False` if a leg cannot be executed. Other commands do not
implicitly raise the lift or move the other arm. The 300 mm waypoint and 5 mm
margin are provisional simulation parameters in `sophia.json`, not measured
hardware limits. A test executes this sequence from the fully lowered lift.
Legacy models without the guard metadata keep their original unguarded behavior.

All four named poses held for 60 simulated seconds with less than 0.54 mm base
drift and no MuJoCo warnings. A 30-second combined arm/lift/neck/gripper trajectory
had at most 0.0088 rad arm tracking error, 4.54 mm lift tracking error, and no
robot-to-robot penetration. Forward/strafe motion at 0.2 m/s and yaw at 0.5 rad/s,
with stops, passed at both lift endpoints; maximum base tilt was under 0.13°.
These outcomes depend on the source mass, inertia, friction and actuator values;
they do not establish real payload or tipping performance.

Both stereo cameras render nonblank RGB and finite positive depth in home and
raised poses. The current model uses the measured 119.991852 mm baseline, parallel
rectified camera frames and the Orin intrinsics; see the offline guide for pixel conventions.
This is a renderer smoke test in the simple floor/sky scene. The separate
`test_sophia_video.py` checks the native stereo publisher through rendering,
JPEG encoding, in-process ZMQ transport and OpenCV decoding, including colour,
stereo disparity and metadata. New projection tests check the supplied intrinsics
at both native and resized aspect ratios. Camera mounting calibration, image
realism and network performance with a remote client remain unverified.

The reproducible extended evaluator writes its metrics, source hashes and
restricted-pose closeups into `output/asset-review/evals/` and exits with status 1 while
any tested condition fails:

- [Extended JSON results](../output/asset-review/evals/extended-evaluation.json)
- [Asset validation JSON](../output/asset-review/evals/validation.json)
- [Neck collision closeup](../output/asset-review/evals/neck-interference-collision.png)
- [Arm/base closeup at 100 mm lift](../output/asset-review/evals/arm-interference-lift-0.1000-visual.png)

The output folder is ignored by Git; rerun the commands to regenerate it. For
the asset-validation output in the table, use
`validate_sophia.py --render --output output/asset-review/evals`.

Edit the configuration or generator, then rebuild; do not hand-edit
`sophia.mjcf`, `scene-sophia.mjcf`, `sophia-ik.mjcf` or `assets_nero/sophia/*.stl`. The source DAE and
STL files are untouched. Reference photographs are read from the supplied folder
and are not copied into the repository.
