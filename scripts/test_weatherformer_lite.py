"""Evaluate a WeatherFormer-lite checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

from src.models.evaluation import evaluate_checkpoint_on_split


def _config_override() -> dict:
	return {
		"model": {"architecture": "weatherformer_lite", "name": "weatherformer_lite"},
		"checkpoint": {
			"path": "./artifacts/checkpoints/weatherformer_lite/latest_model.pt",
			"best_path": "./artifacts/checkpoints/weatherformer_lite/best_model.pt",
		},
	}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	fieldnames: list[str] = []
	seen = set()
	for row in rows:
		for key in row.keys():
			if key not in seen:
				seen.add(key)
				fieldnames.append(str(key))
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate a WeatherFormer-lite checkpoint.")
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

	aggregate_csv = Path("artifacts/logs/weatherformer_lite_test_metrics.csv").resolve()
	per_fire_csv = Path("artifacts/logs/weatherformer_lite_test_metrics_by_fire.csv").resolve()
	aggregate_row = [{"split": result["split"], "num_samples": result["num_samples"], **result["aggregate_results"]}]
	per_fire_rows = [{"split": result["split"], "dataset_name": dataset_name, **metrics} for dataset_name, metrics in sorted(result["per_dataset_results"].items())]
	_write_csv(aggregate_csv, aggregate_row)
	_write_csv(per_fire_csv, per_fire_rows)
	print(f"aggregate_csv: {aggregate_csv}")
	print(f"per_fire_csv: {per_fire_csv}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI safeguard
		print(f"error: {exc}", file=sys.stderr)
		sys.exit(1)
