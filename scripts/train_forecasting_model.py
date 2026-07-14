"""Train the wildfire forecasting model selected by the config."""

from __future__ import annotations

import argparse

from src.training.train import train_model


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training."""

    parser = argparse.ArgumentParser(description="Train the wildfire forecasting architecture selected by the config.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--run_name", default=None, help="Optional explicit run name.")
    parser.add_argument("--output_root", default=None, help="Override training.output.root_dir.")
    parser.add_argument("--overwrite_run", action="store_true", help="Allow writing into an existing explicit run directory.")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    train_model(args.config, run_name=args.run_name, output_root=args.output_root, overwrite_run=args.overwrite_run)


if __name__ == "__main__":
    main()
