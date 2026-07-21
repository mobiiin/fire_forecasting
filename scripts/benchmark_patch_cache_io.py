"""Benchmark patch-cache/DataLoader throughput for one split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.training.train import _as_batch, _ensure_config_path, _loader_summary


def _select_loader(loaders: tuple[Any, Any, Any], split: str):
	train_loader, val_loader, test_loader = loaders
	split_name = split.lower()
	if split_name == "train":
		return train_loader
	if split_name in {"val", "validation"}:
		return val_loader
	if split_name == "test":
		if test_loader is None:
			raise ValueError("No test DataLoader is configured.")
		return test_loader
	raise ValueError(f"Unsupported split: {split!r}")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark configured wildfire DataLoader/cache I/O.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
	parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to benchmark.")
	parser.add_argument("--num_batches", type=int, default=100, help="Maximum number of batches to iterate.")
	parser.add_argument("--warmup_batches", type=int, default=5, help="Warmup batches excluded from timing.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config = _ensure_config_path(load_config(args.config), args.config)
	loader = _select_loader(create_dataloaders(config), args.split)
	max_batches = max(1, int(args.num_batches))
	warmup_batches = max(0, int(args.warmup_batches))

	print(f"split: {args.split}")
	print(f"dataset_samples: {len(loader.dataset)}")
	print(f"dataloader: {_loader_summary(loader)}")
	dataset = getattr(loader, "dataset", None)
	print(f"dataset_type: {type(dataset).__name__}")
	if hasattr(dataset, "cache_dir"):
		print(f"cache_dir: {Path(dataset.cache_dir)}")
	if hasattr(dataset, "manifest"):
		print(f"cache_format: {getattr(dataset, 'manifest', {}).get('shard_format', 'unknown')}")
	if hasattr(dataset, "input_sequence_length"):
		print(f"input_sequence_length: {getattr(dataset, 'input_sequence_length')}")
	if hasattr(dataset, "prediction_horizon"):
		print(f"prediction_horizon: {getattr(dataset, 'prediction_horizon')}")

	measured_batches = 0
	measured_samples = 0
	total_data_wait = 0.0
	start = None
	for batch_index, batch in enumerate(loader, start=1):
		if batch_index > max_batches + warmup_batches:
			break
		x_batch, _y_batch = _as_batch(batch)
		batch_size = int(x_batch.shape[0])
		now = time.perf_counter()
		if batch_index <= warmup_batches:
			start = time.perf_counter()
			continue
		if start is None:
			start = now
		total_data_wait += now - start
		measured_batches += 1
		measured_samples += batch_size
		start = time.perf_counter()

	elapsed = max(total_data_wait, 1.0e-9)
	print(f"measured_batches: {measured_batches}")
	print(f"measured_samples: {measured_samples}")
	print(f"avg_data_wait_s: {elapsed / max(measured_batches, 1):.6f}")
	print(f"samples_per_second: {measured_samples / elapsed:.2f}")


if __name__ == "__main__":
	main()
