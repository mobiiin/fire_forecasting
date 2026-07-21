"""Inspect multi-fire split sizes and sliding-window patch counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data.discovery import discover_multiple_datasets
from src.data.splits import build_sliding_patch_refs_for_split, manual_fire_holdout_splits, multi_fire_chronological_splits
from src.data.temporal_trim import max_valid_local_start, resolve_temporal_trim


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
	parser.add_argument("--index", default=None, help="Optional fire dataset index override.")
	return parser


def _print_trim_table(config: dict, dataset_records) -> None:
	t_in = int(config["input_sequence_length"])
	horizon = int(config["prediction_horizon"])
	print("\nTemporal trim summary")
	print("Split | Fire | Original frames | Trim start | Trim end | Trimmed frames | Valid samples")
	manual = dict(config.get("manual_fire_split", {})) if isinstance(config.get("manual_fire_split"), dict) else {}
	assignments = {}
	for split, key in (("train", "train_fires"), ("val", "val_fires"), ("test", "test_fires")):
		for name in manual.get(key, []):
			assignments[str(name)] = split
	for record in dataset_records:
		trim = resolve_temporal_trim(record)
		valid_samples = max(0, max_valid_local_start(record, t_in, horizon) + 1)
		split = assignments.get(str(record["dataset_name"]), "all")
		print(
			f"{split.upper():<5} | {str(record['dataset_name']):<24} | "
			f"{int(trim['original_num_frames']):<15} | {int(trim['trim_start_index']):<10} | "
			f"{int(trim['trim_end_index']):<8} | {int(trim['trimmed_num_frames']):<14} | {valid_samples}"
		)


def main() -> None:
	args = build_argument_parser().parse_args()
	config = _ensure_config_path(load_config(args.config), args.config)
	if args.index:
		config["fire_dataset_index_json"] = str(Path(args.index).expanduser().resolve())
	dataset_records = discover_multiple_datasets(config)
	_print_trim_table(config, dataset_records)
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
