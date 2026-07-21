"""Run the multitask persistence baseline on one dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.baselines import evaluate_baseline, predict_persistence_for_sample


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate the persistence baseline.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to evaluate.")
	parser.add_argument("--mode", choices=("patch", "full_domain_tiled"), default="patch", help="Evaluation data mode.")
	parser.add_argument("--num_samples", type=int, default=None, help="Optional sample cap for quick runs.")
	parser.add_argument("--output_csv", default=None, help="Optional CSV output path.")
	parser.add_argument("--save_predictions", action="store_true", help="Save per-sample prediction .npz files.")
	parser.add_argument("--save_visualizations", action="store_true", help="Save per-sample prediction visualizations.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	output_csv = args.output_csv
	if output_csv is None:
		output_csv = Path(f"artifacts/logs/persistence_baseline_{args.split}.csv")
	result = evaluate_baseline(
		config_path=args.config,
		split=args.split,
		method_name="persistence",
		predict_fn=predict_persistence_for_sample,
		mode=args.mode,
		num_samples=args.num_samples,
		output_csv=output_csv,
		save_predictions=bool(args.save_predictions),
		save_visualizations=bool(args.save_visualizations),
	)
	print(f"method: {result['method']}")
	print(f"split: {result['split']}")
	sequence = result.get("sequence", {})
	if sequence:
		print(f"input_sequence_length: {sequence.get('input_sequence_length')}")
		print(f"prediction_horizon: {sequence.get('prediction_horizon')}")
	print(f"num_samples: {result['num_samples']}")
	for metric_name, metric_value in sorted(result["aggregate_results"].items()):
		print(f"{metric_name}: {metric_value:.6f}")
	print(f"output_csv: {Path(output_csv).expanduser().resolve()}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI safeguard
		print(f"error: {exc}", file=sys.stderr)
		sys.exit(1)
