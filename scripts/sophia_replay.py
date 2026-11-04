"""Workstation pipeline: generate / simulate / listen / import / compare.

No hardware command mode exists. The listener opens a SUB socket only.
"""

import argparse
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.sophia_replay import build_trajectory, load_trajectory, simulate  # noqa: E402


def new_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="write a versioned, object-free command trajectory")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("simulate", help="execute only in local MuJoCo")
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    p = sub.add_parser("listen", help="passively record an explicitly supplied ZMQ endpoint")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=120)
    p = sub.add_parser("import", help="import a canonical replay run from a passive capture")
    p.add_argument("--packets", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("import-teleop", help="legacy observation-only diagnostic import")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--start-timestamp", type=float, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare", help="write JSON, CSV, plots and an HTML report")
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--observed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        trajectory = build_trajectory()
        new_file(args.output, trajectory)
        print(
            f"{args.output}: {trajectory['duration_s']:.2f}s, "
            f"{len(trajectory['frames'])} commands; sha256={trajectory['sha256']}"
        )
    elif args.command == "simulate":
        result = simulate(load_trajectory(args.trajectory), args.output, video=args.video)
        print(
            f"Simulation complete: {len(result['samples'])} samples; "
            f"{args.output / 'recording.json'}"
        )
    elif args.command == "listen":
        from robot.sophia_replay_io import capture

        stopping = False

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(f"Passive capture only: {args.endpoint}", flush=True)
        result = capture(args.endpoint, args.output, args.duration, lambda: stopping)
        print(json.dumps(result, indent=2))
    elif args.command in ("import", "import-teleop"):
        from robot.sophia_replay_io import import_capture, import_teleop

        trajectory = load_trajectory(args.trajectory)
        if args.command == "import":
            result = import_capture(args.packets, args.run_id, trajectory)
        else:
            result = import_teleop(args.frames, args.episode_id, args.start_timestamp, trajectory)
        new_file(args.output, result)
        print(
            f"Imported {len(result['samples'])} samples, "
            f"complete={result['complete']}; {args.output}"
        )
    else:
        from robot.sophia_replay_compare import compare

        report = compare(
            load_trajectory(args.trajectory),
            json.loads(args.reference.read_text()),
            json.loads(args.observed.read_text()),
            args.output,
        )
        print(f"{report['status']}; {args.output / 'report.html'}")
        print("Quality issues:", report["quality_issues"])


if __name__ == "__main__":
    main()
