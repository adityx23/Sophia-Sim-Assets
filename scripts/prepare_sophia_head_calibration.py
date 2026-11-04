"""Export capture-time alignment from Orin parquet (requires pyarrow)."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/asset-review/head-calibration-pipeline"


def main():
    report = {}
    for ep in (0, 1, 3, 8, 17):
        native = np.load(
            ROOT / f"output/asset-review/hf-sanity-check/native-replay/episode_{ep:06d}.npz"
        )
        table = pq.read_table(OUT / f"data/episode-{ep}-data.parquet")
        recv = native["recv_ns"]
        original = pq.read_table(OUT / f"data/episode-{ep}-native.parquet")
        np.testing.assert_array_equal(original["recv_ns"].to_numpy(), recv)
        captures = table["source.camera_capture_ns.zed_left"].to_numpy()
        hi = np.clip(np.searchsorted(recv, captures), 0, len(recv) - 1)
        lo = np.maximum(hi - 1, 0)
        indices = np.where(abs(recv[lo] - captures) < abs(recv[hi] - captures), lo, hi)
        legacy = np.clip(
            np.searchsorted(native["robot_timestamp"], captures / 1e9, side="right") - 1,
            0,
            len(recv) - 1,
        )
        valid = (
            table["annotation.camera_valid.zed_left"].to_numpy()
            & table["annotation.camera_valid.zed_right"].to_numpy()
            & (np.abs(recv[indices] - captures) <= 50e6)
        )
        right_captures = table["source.camera_capture_ns.zed_right"].to_numpy()
        valid &= captures == right_captures
        records = [
            dict(
                frame=i,
                index=int(j),
                legacy_index=int(legacy[i]),
                valid=bool(valid[i]),
                capture_ns=int(captures[i]),
                robot_gap_ms=float((recv[j] - captures[i]) / 1e6),
            )
            for i, j in enumerate(indices)
        ]
        (OUT / f"data/episode-{ep}-alignment.json").write_text(json.dumps(records) + "\n")
        report[ep] = dict(
            frames=len(records),
            valid=int(valid.sum()),
            legacy_row_difference_max=int(abs(indices[valid] - legacy[valid]).max()),
            clock_and_transport_median_ms=float(
                np.median(native["robot_timestamp"] * 1000 - recv / 1e6)
            ),
            matching="nearest native robot recv_ns to stereo image capture_ns on Orin clock",
        )
    (OUT / "alignment-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
