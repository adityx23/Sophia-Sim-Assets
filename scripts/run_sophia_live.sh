#!/usr/bin/env bash
# Starts disarmed; the operator enables live motion with A+X in the headset.
set -euo pipefail
sophia_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sophia_session="$(date -u +%Y%m%d-%H%M%S)"
cd "$sophia_repo"
# The headset uses offscreen stereo rendering; no desktop GLFW window is needed.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec .venv-sophia/bin/python scripts/live_sophia_teleop.py run \
  --headless \
  --host yor@192.168.123.55 \
  --remote-dir /home/yor/sophia-live-2026-11-03-r4 \
  --remote-python /home/yor/miniforge3/envs/yor-nero/bin/python \
  --remote-output "/home/yor/YOR/teleop_data/sim_follow/$sophia_session" \
  --output "output/live-sophia/$sophia_session" \
  --joystick-device /dev/input/js0 \
  --allow-motion "$@"
