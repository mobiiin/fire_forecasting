"""Evaluate persistence baselines for several candidate target channels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from scripts.evaluate_persistence_baseline import evaluate_persistence_for_channel


DEFAULT_CANDIDATE_CHANNELS = (50, 51, 52, 53, 54, 55)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate persistence baselines for candidate target channels."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATE_CHANNELS),
        help="Candidate channels to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/logs/persistence_candidate_targets.csv",
        help="CSV path for the persistence summary table.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for channel in args.channels:
        result = evaluate_persistence_for_channel(
            config_path=args.config,
            target_channel=int(channel),
            num_visualizations=0,
            compute_threshold_metrics=False,
            output_dir=None,
            verbose=False,
        )
        regression_metrics = result["regression_metrics"]
        distribution = result["target_distribution"]
        rows.append(
            {
                "channel": int(channel),
                "mae": float(regression_metrics["test_mae"]),
                "rmse": float(regression_metrics["test_rmse"]),
                "active_mae": float(regression_metrics["test_active_mae"]),
                "min": float(distribution["min"]),
                "max": float(distribution["max"]),
                "mean": float(distribution["mean"]),
                "std": float(distribution["std"]),
                "active_pixel_fraction": float(distribution["active_pixel_fraction"]),
            }
        )

    fieldnames = [
        "channel",
        "mae",
        "rmse",
        "active_mae",
        "min",
        "max",
        "mean",
        "std",
        "active_pixel_fraction",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("channel, mae, rmse, active_mae, min, max, mean, std, active_pixel_fraction")
    for row in rows:
        print(
            f"{row['channel']}, {row['mae']:.6f}, {row['rmse']:.6f}, {row['active_mae']:.6f}, "
            f"{row['min']:.6g}, {row['max']:.6g}, {row['mean']:.6g}, {row['std']:.6g}, "
            f"{row['active_pixel_fraction']:.6f}"
        )
    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
