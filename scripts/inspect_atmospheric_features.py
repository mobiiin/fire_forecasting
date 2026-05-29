"""Inspect atmospheric engineered features for one wildfire dataset sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.dataset import (
	FireSequenceDataset,
	_resolve_atmospheric_features_config,
	_resolve_path,
	_sort_chronologically,
	build_atmospheric_features,
)


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the CLI parser."""

	parser = argparse.ArgumentParser(description="Inspect atmospheric engineered features for one dataset sample.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--sample_index", type=int, default=0, help="Dataset sample index to inspect.")
	return parser


def _sorted_files(config: dict, config_path: Path) -> list[Path]:
	data_dir = _resolve_path(config_path, config["data_dir"])
	files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{config['file_pattern']}'.")
	return files


def _stats(array: np.ndarray) -> dict[str, float]:
	values = np.asarray(array, dtype=np.float32)
	finite = values[np.isfinite(values)]
	if finite.size == 0:
		return {
			"min": float("nan"),
			"max": float("nan"),
			"mean": float("nan"),
			"std": float("nan"),
			"p50": float("nan"),
			"p90": float("nan"),
			"p99": float("nan"),
		}
	return {
		"min": float(finite.min()),
		"max": float(finite.max()),
		"mean": float(finite.mean()),
		"std": float(finite.std()),
		"p50": float(np.percentile(finite, 50)),
		"p90": float(np.percentile(finite, 90)),
		"p99": float(np.percentile(finite, 99)),
	}


def _print_stats(name: str, array: np.ndarray) -> None:
	stats = _stats(array)
	print(
		f"{name}: min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g} "
		f"std={stats['std']:.6g} p50={stats['p50']:.6g} p90={stats['p90']:.6g} p99={stats['p99']:.6g}"
	)


def main() -> None:
	args = build_argument_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)
	atmospheric = _resolve_atmospheric_features_config(config)
	if not atmospheric["enabled"]:
		raise ValueError("atmospheric_features.enabled is false; there are no atmospheric features to inspect.")

	files = _sorted_files(config, config_path)
	dataset = FireSequenceDataset(
		file_paths=files,
		sample_indices=None,
		input_sequence_length=int(config["input_sequence_length"]),
		prediction_horizon=int(config["prediction_horizon"]),
		target_channel=int(config.get("target_channel", 0)),
		input_channel_count=int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0))),
		input_channel_indices=config.get("input_channel_indices"),
		task_type=str(config.get("task_type", "regression")),
		fire_threshold=float(config.get("fire_threshold", 0.5)),
		use_patches=bool(config.get("use_patches_for_eval", False)),
		patch_size=int(config.get("patch_size", 64)),
		active_patch_probability=float(config.get("active_patch_probability", 0.7)),
		active_threshold=float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		normalization_stats=None,
		normalize_target=False,
		return_metadata=True,
		config=config,
	)

	_, _, metadata = dataset[int(args.sample_index)]
	current_frame = np.load(Path(metadata["current_file_path"]).expanduser().resolve(), mmap_mode="r", allow_pickle=False)
	current_frame = np.asarray(current_frame, dtype=np.float32)
	patch_top = metadata.get("patch_top")
	patch_left = metadata.get("patch_left")
	patch_size = metadata.get("patch_size")
	if patch_top is not None and patch_left is not None and patch_size is not None:
		patch_top = int(patch_top)
		patch_left = int(patch_left)
		patch_size = int(patch_size)
		current_frame = current_frame[patch_top : patch_top + patch_size, patch_left : patch_left + patch_size, :]

	num_levels = int(atmospheric["num_vertical_levels"])
	variables_per_level = int(atmospheric["variables_per_level"])
	low_level_indices = [int(index) for index in atmospheric["low_level_indices"]]
	atmospheric_features = build_atmospheric_features(current_frame, config)

	horizontal_count = num_levels if atmospheric["add_horizontal_wind_speed"] else 0
	low_level_count = 1 if atmospheric["add_low_level_mean_wind_speed"] else 0
	updraft_count = num_levels if atmospheric["add_updraft"] else 0
	offset = 0
	horizontal_slice = slice(offset, offset + horizontal_count)
	offset += horizontal_count
	low_level_slice = slice(offset, offset + low_level_count)
	offset += low_level_count
	updraft_slice = slice(offset, offset + updraft_count)

	def _channel(base_level: int, variable_offset: int) -> np.ndarray:
		return np.asarray(current_frame[:, :, base_level * variables_per_level + variable_offset], dtype=np.float32)

	panels: list[tuple[str, np.ndarray]] = [
		("U z0", _channel(0, 0)),
		("V z0", _channel(0, 1)),
		("W z0", _channel(0, 2)),
	]
	for z_level in range(min(3, num_levels)):
		if horizontal_count > z_level:
			panels.append((f"Horizontal wind speed z{z_level}", atmospheric_features[:, :, horizontal_slice.start + z_level]))
	if low_level_count == 1:
		panels.append((
			f"Low-level mean wind speed z{low_level_indices}",
			atmospheric_features[:, :, low_level_slice.start],
		))
	for z_level in range(min(3, num_levels)):
		if updraft_count > z_level:
			panels.append((f"Updraft z{z_level}", atmospheric_features[:, :, updraft_slice.start + z_level]))

	print(f"sample_index: {int(args.sample_index)}")
	print(f"current_file: {metadata['current_file_path']}")
	print(f"atmospheric engineered feature shape: {tuple(atmospheric_features.shape)}")
	for title, panel in panels:
		_print_stats(title, panel)

	output_dir = _resolve_path(config_path, "outputs/atmospheric_feature_inspection")
	output_dir.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 5, figsize=(24, 10), dpi=150, constrained_layout=True)
	for axis, (title, panel) in zip(axes.flatten(), panels):
		image = axis.imshow(np.asarray(panel, dtype=np.float32), origin="lower", cmap="viridis")
		axis.set_title(title)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	for axis in axes.flatten()[len(panels):]:
		axis.axis("off")
	output_path = output_dir / f"sample_{int(args.sample_index):05d}.png"
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	print(f"saved figure: {output_path}")


if __name__ == "__main__":
	main()
