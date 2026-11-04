#!/usr/bin/env bash
# Offline only. Requires the five cached episodes documented in SOPHIA_SIM_ASSETS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_PYTHON="${SOPHIA_DATA_PYTHON:-python}"
SIM_PYTHON="${SOPHIA_SIM_PYTHON:-.venv-sophia/bin/python}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
"$SIM_PYTHON" robot/yor-description/build_sophia.py
"$DATA_PYTHON" scripts/prepare_sophia_head_calibration.py
"$SIM_PYTHON" scripts/calibrate_sophia_head.py
"$SIM_PYTHON" scripts/check_sophia_head_calibration.py "$@"
