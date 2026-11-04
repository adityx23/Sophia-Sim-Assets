"""Regularized per-joint follow-up, with an untouched later-episode final check.

The shared-gain candidate's validation results motivated this model class.
Episodes 5..14 are therefore development checks, not a fresh final test.
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import identify_sophia_capture as capture
import numpy as np
from scipy.optimize import least_squares


def segment_record(record, begin, end):
    result = {key: value[begin:end].copy() for key, value in record.items()}
    result["time_s"] -= result["time_s"][0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "training_episodes": list(capture.ARM_TRAIN),
        "development_episodes": list(capture.ARM_VALIDATION),
        "final_test": "Episode 15; contiguous engaged/commanded, stationary-base intervals >=3s, initialized separately; exclude first second and closed/grasp intervals as before.",
        "parameters": "Seven left kp and seven left kv; right unchanged.",
        "bounds": {"kp": [100, 1500], "kv": [2, 80]},
        "log_prior": {"kp": 500, "kv": 20, "residual_weight_rad": 0.002},
        "interpolation_duration_s": 2 / 30,
        "gate": "Both development and final check: pooled RMSE <=0.9 baseline, every joint <=1.1 baseline, at least 75% intervals improve. Optimizer converges without an active bound. Effective gains only, not physical motor constants.",
        "source_sha256": {
            str(path.relative_to(capture.ROOT)): capture.digest(path)
            for path in (
                Path(__file__),
                Path(capture.__file__),
                capture.HERE / "sophia.json",
                capture.HERE / "sophia.mjcf",
                capture.CAPTURE / "MANIFEST.json",
            )
        },
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    experiment = capture.ArmExperiment(
        "left", (*capture.ARM_TRAIN, *capture.ARM_VALIDATION)
    )
    prior = np.log([500.0] * 7 + [20.0] * 7)

    def residual(parameters):
        kp, kv = np.exp(parameters[:7]), np.exp(parameters[7:])
        errors = []
        for index in capture.ARM_TRAIN:
            r = experiment.records[index]
            if r["mask"].sum() < 15:
                continue
            error = (experiment.predict(index, kp=kp, kv=kv)[:, :7] - r["left_joints"])[
                r["mask"]
            ]
            errors.append(error.ravel() / np.sqrt(error.size))
        errors.append(0.002 * (parameters - prior))
        return np.concatenate(errors)

    fits = []
    for start in (prior, np.log([300.0] * 7 + [20.0] * 7)):
        fit = least_squares(
            residual,
            start,
            bounds=(np.log([100.0] * 7 + [2.0] * 7), np.log([1500.0] * 7 + [80.0] * 7)),
            diff_step=0.002,
            max_nfev=50,
            ftol=1e-7,
            xtol=1e-6,
            gtol=1e-8,
        )
        fits.append(fit)
        print(
            json.dumps(
                {
                    "kp": np.exp(fit.x[:7]).tolist(),
                    "kv": np.exp(fit.x[7:]).tolist(),
                    "nfev": fit.nfev,
                    "loss": float(fit.fun @ fit.fun),
                }
            ),
            flush=True,
        )
    best = min(fits, key=lambda fit: fit.fun @ fit.fun)
    parameters = {"kp": np.exp(best.x[:7]).tolist(), "kv": np.exp(best.x[7:]).tolist()}
    scores = {}
    for label, exp, indices in [("development", experiment, capture.ARM_VALIDATION)]:
        scores[label] = {
            "baseline": exp.score(indices, {}, args.output / label / "baseline"),
            "candidate": exp.score(
                indices, parameters, args.output / label / "candidate"
            ),
        }
    # Only inspect final test predictions after training and model selection above.
    original = capture.load_episode(15)
    enabled = (
        original["left_target_active"]
        & np.isfinite(original["left_gripper_cmd"].ravel())
        & capture.stationary(original)
    )
    indices = np.flatnonzero(enabled)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    segments, ranges = {}, {}
    for group in groups:
        if (
            len(group)
            and original["time_s"][group[-1]] - original["time_s"][group[0]] >= 3
        ):
            key = 1500 + len(segments)
            segments[key] = segment_record(original, int(group[0]), int(group[-1]) + 1)
            ranges[key] = [int(group[0]), int(group[-1]) + 1]
    with patch.object(
        capture, "load_episode", side_effect=lambda index: segments[index]
    ):
        final = capture.ArmExperiment("left", tuple(segments))
    scores["final"] = {
        "baseline": final.score(tuple(segments), {}, args.output / "final/baseline"),
        "candidate": final.score(
            tuple(segments), parameters, args.output / "final/candidate"
        ),
    }
    passes = []
    for label, score in scores.items():
        baseline, candidate = score["baseline"], score["candidate"]
        ratios = np.asarray(candidate["joint_rmse_rad"]) / baseline["joint_rmse_rad"]
        improved = sum(
            c["rmse_rad"] < b["rmse_rad"]
            for b, c in zip(baseline["episodes"], candidate["episodes"], strict=True)
        )
        score.update(
            joint_ratios=ratios.tolist(),
            overall_ratio=candidate["rmse_rad"] / baseline["rmse_rad"],
            intervals_improved=int(improved),
        )
        passes.append(
            score["overall_ratio"] <= 0.9
            and np.all(ratios <= 1.1)
            and improved >= 0.75 * len(baseline["episodes"])
        )
    report = {
        "parameters": parameters,
        "scores": scores,
        "final_source_episode": 15,
        "final_segment_sample_ranges": ranges,
        "optimizer_success": bool(best.success),
        "active_bounds": best.active_mask.tolist(),
        "candidate_passes_gate": bool(
            all(passes) and best.success and not any(best.active_mask)
        ),
        "simulation_parameters_applied": False,
        "limits": [
            "Final test comprises intervals of one later episode, not independent days or hardware configurations.",
            "No torque or payload identification. Effective closed-loop gains depend on the retained inertias, friction and timing assumptions.",
            "The new model class used development results from the rejected shared-gain fit; only episode 15 was reserved for a fresh final test.",
        ],
        "elapsed_s": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "candidate_passes_gate": report["candidate_passes_gate"],
                "scores": {
                    key: {
                        k: v
                        for k, v in value.items()
                        if k not in ("baseline", "candidate")
                    }
                    for key, value in scores.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
