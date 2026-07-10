"""Inspect multi-fire split sizes and sliding-window patch counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data.discovery import discover_multiple_datasets
from src.data.splits import build_sliding_patch_refs_for_split, manual_fire_holdout_splits, multi_fire_chronological_splits


def _ensure_config_path(config: dict, config_path: str | Path) -> dict:
	resolved_path = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved_path)
	config["_config_path"] = str(resolved_path)
	return config


def _base_split_refs(config: dict, dataset_records):
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "multi_dataset_chronological":
		split_mode = "multi_fire_chronological"
	if split_mode == "manual_fire_holdout":
		manual = dict(config.get("manual_fire_split", {})) if isinstance(config.get("manual_fire_split"), dict) else {}
		return manual_fire_holdout_splits(
			dataset_records=dataset_records,
			train_fire_names=manual.get("train_fires", []),
			val_fire_names=manual.get("val_fires", []),
			test_fire_names=manual.get("test_fires", []),
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			config=config,
		)
	return multi_fire_chronological_splits(
		dataset_records=dataset_records,
		input_sequence_length=int(config["input_sequence_length"]),
		prediction_horizon=int(config["prediction_horizon"]),
		train_fraction=float(config.get("train_fraction", 0.7)),
		val_fraction=float(config.get("val_fraction", 0.15)),
		test_fraction=float(config.get("test_fraction", 0.15)),
	)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inspect multi-fire split sizes and patch counts.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config = _ensure_config_path(load_config(args.config), args.config)
	dataset_records = discover_multiple_datasets(config)
	sample_refs = _base_split_refs(config, dataset_records)
	for split in ("train", "val", "test"):
		print(f"\n[{split}]")
		_ = build_sliding_patch_refs_for_split(
			dataset_records=dataset_records,
			sample_refs=sample_refs[split],
			split=split,
			config=config,
		)


if __name__ == "__main__":
	main()
