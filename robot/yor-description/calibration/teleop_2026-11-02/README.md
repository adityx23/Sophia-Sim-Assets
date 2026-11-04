# Sophia robot capture

This is calibration evidence, not an applied parameter fit. See summary.json for coverage, missing fields and limitations.

- source/ contains the stopped capture and its original sim-baseline snapshots. The two large JSONL files are compressed losslessly with gzip.
- native-replay/ contains one NPZ per episode in source order. Load with numpy.load(path, allow_pickle=False). Keep validity masks and NaNs.
- episode-events.jsonl retains task, cycle, outcome and episode boundary messages.
- MANIFEST.json hashes all imported files and the original uncompressed sources.

The older arm benchmark requires finite gripper commands; these nullable exports must not be passed to it without resolving command/feedback semantics first.
