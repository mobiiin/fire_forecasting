"""Evaluate a trained ConvLSTM U-Net checkpoint on the configured test split."""

from __future__ import annotations

import argparse

from src.training.train import evaluate_model_on_test_set


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for standalone test evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvLSTM U-Net checkpoint on the test split."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. If omitted, resolve from config.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Which config-derived checkpoint to use when --checkpoint is not provided.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    results = evaluate_model_on_test_set(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint_kind=args.checkpoint_kind,
    )

    print(f"checkpoint: {results['checkpoint_path']}")
    print(f"test samples: {results['num_test_samples']}")
    for metric_name, metric_value in results["test_results"].items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
