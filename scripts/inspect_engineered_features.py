"""Inspect engineered features for one sample from the wildfire dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.dataset import FireSequenceDataset, _resolve_channel_layout, _resolve_engineered_features_config, _sort_chronologically


def _resolve_path(base_path: Path, configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (base_path.parent / path).resolve()


def _sorted_files(config: dict, config_path: Path) -> list[Path]:
	data_dir = _resolve_path(config_path, config["data_dir"])
	files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{config['file_pattern']}'.")
	return files


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inspect engineered features for one dataset sample.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--sample_index", type=int, default=0, help="Dataset sample index to inspect.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)
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

	x_tensor, y_tensor, metadata = dataset[int(args.sample_index)]
	x_array = x_tensor.detach().cpu().numpy()
	y_array = y_tensor.detach().cpu().numpy()
	last_timestep = x_array[-1]
	base_channels = int(dataset.base_input_channel_count)
	engineered_channels = int(dataset.engineered_channel_count)
	layout = _resolve_channel_layout(config)
	engineered = _resolve_engineered_features_config(config)

	print(f"X shape: {tuple(x_array.shape)}")
	print(f"y shape: {tuple(y_array.shape)}")
	print(f"base input channels: {base_channels}")
	print(f"engineered channels: {engineered_channels}")

	offset = base_channels
	channel_ranges: list[tuple[str, tuple[int, int]]] = []
	if engineered["add_flux_delta"]:
		channel_ranges.append(("flux_delta", (offset, offset + len(layout["flux_channels"]) - 1)))
		offset += len(layout["flux_channels"])
	if engineered["add_fuel_delta"]:
		channel_ranges.append(("fuel_delta", (offset, offset + len(layout["fuel_channels"]) - 1)))
		offset += len(layout["fuel_channels"])
	if engineered["add_step_consumed_fuel"]:
		channel_ranges.append(("step_consumed_fuel", (offset, offset + len(layout["fuel_channels"]) - 1)))
		offset += len(layout["fuel_channels"])
	if engineered["add_cumulative_consumed_fuel"]:
		channel_ranges.append(("cumulative_consumed_fuel", (offset, offset + len(layout["fuel_channels"]) - 1)))
		offset += len(layout["fuel_channels"])
	for name, (start, end) in channel_ranges:
		print(f"{name}: channels [{start}, {end}]")

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

	flux_delta_start = base_channels
	fuel_delta_start = flux_delta_start + len(layout["flux_channels"])
	step_consumed_start = fuel_delta_start + len(layout["fuel_channels"])
	cumulative_start = step_consumed_start + len(layout["fuel_channels"])

	panel_specs = [
		("Selected flux channel", current_frame[:, :, layout["flux_channels"][0]]),
		("Selected flux delta", last_timestep[flux_delta_start]),
		("Surface fuel", current_frame[:, :, layout["surface_fuel_channel"]]),
		("Canopy fuel", current_frame[:, :, layout["canopy_fuel_channel"]]),
		("Surface fuel delta", last_timestep[fuel_delta_start + 0]),
		("Canopy fuel delta", last_timestep[fuel_delta_start + 1]),
		("Surface step consumed fuel", last_timestep[step_consumed_start + 0]),
		("Canopy step consumed fuel", last_timestep[step_consumed_start + 1]),
		("Surface cumulative consumed fuel", last_timestep[cumulative_start + 0]),
		("Canopy cumulative consumed fuel", last_timestep[cumulative_start + 1]),
		("Target surface consumed fuel", y_array[0]),
		("Target canopy consumed fuel", y_array[1]),
		("Target mask", y_array[2]),
	]

	output_dir = _resolve_path(config_path, "outputs/engineered_feature_inspection")
	output_dir.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(5, 3, figsize=(18, 24), dpi=150, constrained_layout=True)
	for axis, (title, panel) in zip(axes.flatten(), panel_specs):
		image = axis.imshow(np.asarray(panel, dtype=np.float32), origin="lower", cmap="inferno")
		axis.set_title(title)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	for axis in axes.flatten()[len(panel_specs):]:
		axis.axis("off")
	output_path = output_dir / f"sample_{int(args.sample_index):05d}.png"
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	print(f"saved figure: {output_path}")


if __name__ == "__main__":
	main()
