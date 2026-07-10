"""Evaluate an Earthformer-lite checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.models.evaluation import evaluate_checkpoint_on_split


def _config_override() -> dict:
	return {
		"model": {"architecture": "earthformer_lite", "name": "earthformer_lite"},
		"checkpoint": {
			"path": "./artifacts/checkpoints/earthformer_lite/latest_model.pt",
			"best_path": "./artifacts/checkpoints/earthformer_lite/best_model.pt",
		},
	}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate an Earthformer-lite checkpoint.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
	parser.add_argument("--checkpoint_kind", choices=("best", "latest"), default="best", help="Which configured checkpoint to use when --checkpoint is omitted.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Which split to evaluate.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	result = evaluate_checkpoint_on_split(
		config_path=args.config,
		split=args.split,
		checkpoint_path=args.checkpoint,
		checkpoint_kind=args.checkpoint_kind,
		config_override=_config_override(),
	)
	print(f"checkpoint: {result['checkpoint_path']}")
	print(f"split: {result['split']}")
	print(f"num_samples: {result['num_samples']}")
	print("aggregate metrics")
	for metric_name, metric_value in sorted(result["aggregate_results"].items()):
		print(f"{metric_name}: {metric_value:.6f}")
	print("per-fire metrics")
	for dataset_name in sorted(result["per_dataset_results"]):
		print(f"[{dataset_name}]")
		for key, value in result["per_dataset_results"][dataset_name].items():
			if key == "num_samples":
				print(f"{key}: {int(value)}")
			else:
				print(f"{key}: {value:.6f}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI safeguard
		print(f"error: {exc}", file=sys.stderr)
		sys.exit(1)
