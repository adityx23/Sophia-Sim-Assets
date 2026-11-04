"""Trace recorded neck values through native and converted Pi-Atlas episodes.

Run in the analysis environment with numpy and pyarrow. This reads repository
history and dataset files only; it never connects to or commands robot hardware.
"""

import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/asset-review/hf-sanity-check"
OUT = ROOT / "output/asset-review/spinal-cad-review/neck-offset-audit"
REVISION = "6a432bf"


def historic(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "vr-teleop"), "show", f"{REVISION}:{path}"], text=True
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((DATA / "source/meta/info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    si = [state_names.index(f"neck_deg.{axis}") for axis in ("pan", "tilt")]
    ai = [action_names.index(f"neck_cmd_deg.{axis}") for axis in ("pan", "tilt")]
    view_index = state_names.index("view_pitch_deg")
    results = []
    for path in sorted((DATA / "native-replay").glob("episode_*.npz")):
        a = np.load(path)
        name = path.stem
        table = pq.read_table(DATA / f"source/data/chunk-000/{name}.parquet")
        native_table = pq.read_table(
            DATA / f"source/native/chunk-000/{name}/teleop.parquet", columns=["raw"]
        )
        raw = [json.loads(row) for row in native_table["raw"].to_pylist()]
        measured = np.array([row["neck_deg"] for row in raw])
        commands = np.array([row["neck_cmd_deg"] for row in raw])
        np.testing.assert_array_equal(measured, a["neck_deg"])
        np.testing.assert_array_equal(commands, a["neck_cmd_deg"])
        indices = np.array(table["source.robot_row_index"].to_pylist(), dtype=int)
        valid = (indices >= 0) & (indices < len(measured))
        indices = np.clip(indices, 0, len(measured) - 1)
        state = np.array(table["observation.state"].to_pylist())
        action = np.array(table["action"].to_pylist())
        smask = np.array(table["annotation.state_valid"].to_pylist())[:, si].all(1) & valid
        amask = np.array(table["annotation.action_valid"].to_pylist())[:, ai].all(1) & valid
        se = state[smask][:, si] - measured[indices[smask]]
        ae = action[amask][:, ai] - commands[indices[amask]]
        np.testing.assert_allclose(se, 0, atol=1e-5)
        np.testing.assert_allclose(ae, 0, atol=1e-5)
        ticks = measured * 4095 / 360
        results.append(
            dict(
                episode=int(name.split("_")[-1]),
                native_rows=len(measured),
                initial_measured_pan_tilt_deg=measured[0].tolist(),
                raw_to_native_export_max_error_deg=0,
                converted_state_matches_native_rows=int(smask.sum()),
                converted_state_max_error_deg=float(np.max(np.abs(se))),
                converted_action_matches_native_rows=int(amask.sum()),
                converted_action_max_error_deg=float(np.max(np.abs(ae))),
                measured_quantization_max_residual_ticks=float(np.max(abs(ticks - np.rint(ticks)))),
                raw_view_pitch_present_rows=sum("view_pitch_deg" in row for row in raw),
                converted_view_pitch_valid_rows=int(
                    np.array(table["annotation.state_valid"].to_pylist())[:, view_index].sum()
                ),
            )
        )
    for path in ("server/whole_body.py", "server/yor_bridge.py", "server/episode_store.py"):
        (OUT / (Path(path).stem + f"-{REVISION}.py.txt")).write_text(historic(path))
    codec = types.ModuleType("historical_episode_codec")
    exec(
        compile(historic("server/episode_store.py"), "historical_episode_codec", "exec"),
        codec.__dict__,
    )
    frames = [{"neck_deg": [2.285714, 35.692307]}, {"neck_deg": [2.373626, 35.78022]}]
    _, decoded = codec.decode(codec.encode({}, frames, ["neck_deg"]))
    np.testing.assert_allclose(decoded["neck_deg"], [v["neck_deg"] for v in frames], atol=0.0005)
    report = dict(
        history_revision=REVISION,
        episodes=results,
        findings={
            "neck_deg": (
                "Measured calibrated servo angles from Present_Position, not per-episode deltas"
            ),
            "neck_cmd_deg": (
                "Calibrated servo goals after headset sign, Home offset, clipping/rate limits"
            ),
            "historical_headset_pan": (
                "Relative yaw: sign * wrap(headset_yaw - enable_yaw) + home_pan"
            ),
            "historical_headset_tilt": (
                "sign * headset_pitch_from_horizon + home_tilt; "
                "no enable-pitch subtraction in this revision"
            ),
            "current_optional_relative_pitch": (
                "Newer relative-pitch controller subtracts enable pitch before computing "
                "a hardware goal; feedback fields remain calibrated servo angles"
            ),
            "view_pitch_deg": (
                "Separate operator setting in newer schema; absent in the reviewed Oct26 "
                "raw episodes, converted placeholder zeros are invalid-masked"
            ),
            "on_disk_delta_encoding": (
                "Compression stores deltas including the first absolute sample; "
                "decode cumulatively reconstructs the original angles"
            ),
            "normalization_reference": (
                "Calibrated servo travel midpoint, not gravity/horizon or episode start"
            ),
            "replay": (
                "Use measured neck_deg directly for pose comparison; "
                "do not add Home or headset offsets a second time"
            ),
        },
        codec_fixture_input=frames,
        codec_fixture_decoded=decoded["neck_deg"].tolist(),
        historical_calibration_snapshot_available=False,
        caveat=(
            "Current/source-revision code and stored numeric values agree; an unrecorded "
            "historical calibration change cannot be reconstructed from this alone."
        ),
        robot_or_dataset_modified=False,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                episodes=len(results),
                max_state_error_deg=max(v["converted_state_max_error_deg"] for v in results),
                max_action_error_deg=max(v["converted_action_max_error_deg"] for v in results),
                view_pitch_present_rows=sum(v["raw_view_pitch_present_rows"] for v in results),
                codec_first_decoded=decoded["neck_deg"][0].tolist(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
