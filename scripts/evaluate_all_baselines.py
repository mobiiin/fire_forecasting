"""Evaluate all non-neural baselines across one or more splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

from src.baselines import (
	evaluate_baseline,
	predict_linear_extrapolation_for_sample,
	predict_persistence_for_sample,
)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate all non-neural wildfire baselines.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"], help="Splits to evaluate.")
	parser.add_argument("--mode", choices=("patch", "full_domain_tiled"), default="patch", help="Evaluation data mode.")
	parser.add_argument("--num_samples", type=int, default=None, help="Optional per-run sample cap for quick runs.")
	parser.add_argument("--output_csv", default="artifacts/logs/all_baselines.csv", help="Combined CSV output path.")
	return parser


def _write_rows(output_csv: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	fieldnames: list[str] = []
	seen = set()
	for row in rows:
		for key in row.keys():
			if key not in seen:
				seen.add(key)
				fieldnames.append(str(key))
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def main() -> None:
	args = build_argument_parser().parse_args()
	runs = (
		("persistence", predict_persistence_for_sample),
		("linear_extrapolation", predict_linear_extrapolation_for_sample),
	)
	all_rows: list[dict[str, Any]] = []
	for split in args.splits:
		for method_name, predict_fn in runs:
			result = evaluate_baseline(
				config_path=args.config,
				split=split,
				method_name=method_name,
				predict_fn=predict_fn,
				mode=args.mode,
				num_samples=args.num_samples,
				output_csv=None,
				save_predictions=False,
				save_visualizations=False,
			)
			all_rows.extend(result["rows"])
			aggregate = result["aggregate_results"]
			print(f"{method_name} | {split} | samples={result['num_samples']}")
			for metric_name, metric_value in sorted(aggregate.items()):
				print(f"  {metric_name}: {metric_value:.6f}")

	output_csv = Path(args.output_csv).expanduser().resolve()
	_write_rows(output_csv, all_rows)
	print(f"output_csv: {output_csv}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI safeguard
		print(f"error: {exc}", file=sys.stderr)
		sys.exit(1)
