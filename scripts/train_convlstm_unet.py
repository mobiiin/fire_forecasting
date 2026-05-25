"""Train the ConvLSTM U-Net model."""

from __future__ import annotations

import argparse

from src.training.train import train_model


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training."""

    parser = argparse.ArgumentParser(description="Train the ConvLSTM U-Net wildfire model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
