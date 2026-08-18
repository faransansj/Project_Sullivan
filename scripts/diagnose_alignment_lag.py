#!/usr/bin/env python3
"""Write a validation-only prediction/target lag sweep to JSON and CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.alignment_diagnostic import best_lag, lag_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="NPZ with predictions, targets, optional mask"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=["val", "validation", "test", "train"])
    parser.add_argument("--frame-rate", required=True, type=float)
    parser.add_argument("--metric", choices=["rmse", "dimension_mean_pcc"], default="rmse")
    parser.add_argument("--min-lag-ms", type=int, default=-300)
    parser.add_argument("--max-lag-ms", type=int, default=300)
    parser.add_argument("--step-ms", type=int, default=20)
    parser.add_argument("--config", help="Optional config path recorded in output")
    parser.add_argument("--checkpoint", help="Optional checkpoint path recorded in output")
    parser.add_argument("--split-manifest", help="Optional split manifest path recorded in output")
    args = parser.parse_args()

    data = np.load(args.input)
    results = lag_sweep(
        data["predictions"],
        data["targets"],
        frame_rate=args.frame_rate,
        split=args.split,
        mask=data["mask"] if "mask" in data else None,
        min_lag_ms=args.min_lag_ms,
        max_lag_ms=args.max_lag_ms,
        step_ms=args.step_ms,
        metric=args.metric,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "input": args.input,
        "frame_rate": args.frame_rate,
        "metric": args.metric,
        "best_validation_lag": best_lag(results),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split_manifest": args.split_manifest,
        "results": results,
    }
    (output_dir / "alignment_lag_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "alignment_lag_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
