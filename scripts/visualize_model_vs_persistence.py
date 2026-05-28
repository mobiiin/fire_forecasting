"""Visualize model forecasts against a persistence baseline on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.visualize_predictions import visualize_model_vs_persistence


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Visualize model predictions against a persistence baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of chronological test samples to compare and save.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/model_vs_persistence",
        help="Directory for saved comparison plots.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    visualize_model_vs_persistence(
        config_path=args.config,
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
