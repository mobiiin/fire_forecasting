"""Run lightweight reliability checks for the wildfire forecasting pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.data.splits import chronological_split_indices


def _resolve_path(base_path: Path, configured_path: str | Path) -> Path:
	"""Resolve a path relative to the config location."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (base_path.parent / path).resolve()


def _extract_numeric_suffix(name: str) -> int | None:
	digits = []
	for character in reversed(name):
		if character.isdigit():
			digits.append(character)
		else:
			break
	if not digits:
		return None
	return int("".join(reversed(digits)))


def _sort_chronologically(paths: list[Path]) -> list[Path]:
	numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in paths]
	if all(value is not None for value in numeric_suffixes):
		return [path for _, path in sorted(zip(numeric_suffixes, paths), key=lambda item: item[0])]
	return sorted(paths, key=lambda path: path.name)


def _check_chronological_split_no_leakage(config: dict, file_count: int) -> None:
	"""Ensure each split stays inside its temporal segment."""

	t_in = int(config["input_sequence_length"])
	horizon = int(config["prediction_horizon"])
	train_fraction = float(config.get("train_fraction", 0.7))
	val_fraction = float(config.get("val_fraction", 0.15))
	test_fraction = float(config.get("test_fraction", 0.15))
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	splits = chronological_split_indices(
		num_timesteps=file_count,
		input_sequence_length=t_in,
		prediction_horizon=horizon,
		train_fraction=train_fraction,
		val_fraction=val_fraction,
		test_fraction=test_fraction,
		split_mode=split_mode,
	)

	train_end = int(np.floor(file_count * train_fraction))
	val_end = train_end + int(np.floor(file_count * val_fraction))

	for index in splits["train"]:
		target_t = index + t_in - 1 + horizon
		if target_t >= train_end:
			raise AssertionError("Train split leakage detected into val/test segment.")
	for index in splits["val"]:
		target_t = index + t_in - 1 + horizon
		if split_mode != "train_val_external_test" and target_t >= val_end:
			raise AssertionError("Validation split leakage detected into test segment.")


def _check_normalization_training_only(config: dict, file_count: int) -> None:
	"""Verify normalization timestamps come only from train input windows."""

	t_in = int(config["input_sequence_length"])
	horizon = int(config["prediction_horizon"])
	train_fraction = float(config.get("train_fraction", 0.7))
	val_fraction = float(config.get("val_fraction", 0.15))
	test_fraction = float(config.get("test_fraction", 0.15))
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	splits = chronological_split_indices(
		num_timesteps=file_count,
		input_sequence_length=t_in,
		prediction_horizon=horizon,
		train_fraction=train_fraction,
		val_fraction=val_fraction,
		test_fraction=test_fraction,
		split_mode=split_mode,
	)

	train_end = int(np.floor(file_count * train_fraction))
	train_input_indices: set[int] = set()
	for start_index in splits["train"]:
		train_input_indices.update(range(start_index, start_index + t_in))
	if train_input_indices and max(train_input_indices) >= train_end:
		raise AssertionError("Normalization would include non-train timestamps.")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run lightweight smoke checks for the project.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration.")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	for key in ("data_dir", "file_pattern", "input_sequence_length", "prediction_horizon", "target_channel"):
		if key not in config:
			raise KeyError(f"Missing required config key: {key}")

	data_dir = _resolve_path(config_path, str(config["data_dir"]))
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
	files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' with pattern '{config['file_pattern']}'.")

	_check_chronological_split_no_leakage(config, len(files))
	_check_normalization_training_only(config, len(files))

	try:
		train_loader, val_loader, test_loader = create_dataloaders(config)
		batch = next(iter(train_loader))
		x_batch, y_batch = batch[:2]
		if x_batch.ndim != 5 or y_batch.ndim != 4:
			raise AssertionError("Unexpected dataloader tensor ranks.")
		print(f"DataLoader shape check passed: X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
		if test_loader is None:
			print(f"Split sample counts: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, external_test=not_configured")
		else:
			print(f"Split sample counts: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, external_test={len(test_loader.dataset)}")
	except ImportError:
		print("PyTorch unavailable: skipped dataloader/model smoke checks.")

	print("Smoke checks passed.")


if __name__ == "__main__":
	main()
