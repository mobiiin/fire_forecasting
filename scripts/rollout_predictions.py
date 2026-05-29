"""Autoregressive rollout visualization for a one-step ConvLSTM U-Net.

This script rolls a single-step model forward over future timesteps. The model
predicts only the target fire-intensity channel, so non-target channels must be
filled either from the true future exogenous files (teacher_forced_exogenous)
or copied from the latest known frame (constant_exogenous).

Limitation:
    A one-step model that predicts only fire intensity cannot truly forecast all
    atmospheric, fuel, or flux variables unless those future exogenous channels
    are available or the model is trained to predict the full set of channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.data.dataset import FireSequenceDataset
from src.data.preprocessing import inverse_normalize_channel_map as inverse_normalize_scalar_channel_map
from src.data.preprocessing import load_normalization_stats, normalize_tensor
from src.data.splits import chronological_split_indices
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.visualization.animate import save_rollout_animation, save_side_by_side_maps, save_single_map


def _get_section(config: Mapping[str, Any], *names: str) -> dict[str, Any]:
	"""Return the first nested mapping found under any of the provided names."""

	for name in names:
		section = config.get(name)
		if isinstance(section, dict):
			return section
	return {}


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to a config file when available."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_path is None:
		return path.resolve()
	return (base_path.parent / path).resolve()


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
	"""Attach the config path so downstream helpers can resolve relative paths."""

	resolved_path = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved_path)
	config["_config_path"] = str(resolved_path)
	return config


def _extract_numeric_suffix(name: str) -> int | None:
	"""Extract a trailing numeric suffix from a filename stem if present."""

	digits = []
	for character in reversed(name):
		if character.isdigit():
			digits.append(character)
		else:
			break
	if not digits:
		return None
	return int("".join(reversed(digits)))



def _sort_chronologically(file_paths: list[Path]) -> list[Path]:
	"""Sort files by numeric suffix when available, otherwise lexicographically."""

	numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in file_paths]
	if all(value is not None for value in numeric_suffixes):
		return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
	return sorted(file_paths, key=lambda path: path.name)



def _discover_files(config: Mapping[str, Any]) -> list[Path]:
	"""Discover dataset files in chronological order."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	data_dir = _resolve_path(config_path, config["data_dir"])
	file_pattern = str(config["file_pattern"])
	files = _sort_chronologically(list(data_dir.glob(file_pattern)))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
	return files



def _build_test_dataset(config: Mapping[str, Any], normalization_stats) -> FireSequenceDataset:
	"""Build the configured test dataset and expose metadata for visualization."""

	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "train_val_external_test":
		test_data_dir = config.get("test_data_dir")
		if test_data_dir in (None, "", "null"):
			raise ValueError(
				"No external test_data_dir configured. This project now uses data_dir only for train/val. "
				"Set test_data_dir in the config to run rollout predictions on an external test dataset."
			)
		config_path_value = config.get("config_path", config.get("_config_path"))
		config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
		test_dir = _resolve_path(config_path, test_data_dir)
		external_file_pattern = str(config.get("external_test_file_pattern", config["file_pattern"]))
		files = _sort_chronologically(list(test_dir.glob(external_file_pattern)))
		if not files:
			raise FileNotFoundError(
				f"No external test files found in '{test_dir}' using pattern '{external_file_pattern}'."
			)
	else:
		files = _discover_files(config)
	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	input_channel_count = int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0)))
	if input_channel_count <= 0:
		raise KeyError("Config must define a positive input_channel_count or model.input_channels.")
	if split_mode == "train_val_external_test":
		max_start_index = len(files) - input_sequence_length - prediction_horizon
		sample_indices = [] if max_start_index < 0 else list(range(max_start_index + 1))
	else:
		split_indices = chronological_split_indices(
			num_timesteps=len(files),
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
			train_fraction=float(config.get("train_fraction", 0.7)),
			val_fraction=float(config.get("val_fraction", 0.15)),
			test_fraction=float(config.get("test_fraction", 0.15)),
			split_mode=split_mode,
		)
		sample_indices = split_indices["test"]
	return FireSequenceDataset(
		file_paths=files,
		sample_indices=sample_indices,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
		target_channel=int(config["target_channel"]),
		input_channel_count=input_channel_count,
		normalization_stats=normalization_stats,
		normalize_target=bool(_get_section(config, "normalization").get("normalize_target", False)),
		return_metadata=True,
	)



def _load_raw_frame(file_path: Path) -> np.ndarray:
	"""Load a raw numpy frame from disk and validate that it is 3D."""

	array = np.load(file_path, mmap_mode="r", allow_pickle=False)
	if array.ndim != 3:
		raise ValueError(f"Expected a 3D tensor in {file_path}, got shape {array.shape}.")
	return np.asarray(array, dtype=np.float32)



def _resolve_checkpoint_path(config: Mapping[str, Any]) -> Path:
	"""Resolve the best checkpoint path, falling back to the latest checkpoint when needed."""

	checkpoint_config = _get_section(config, "checkpoint")
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	latest_path, best_path = latest_and_best_checkpoint_paths(_resolve_path(config_path, checkpoint_path))
	selected = best_path if best_path.exists() else latest_path
	if not selected.exists():
		raise FileNotFoundError(
			"No checkpoint found for rollout. "
			f"Checked best='{best_path}' and latest='{latest_path}'."
		)
	return selected



def _build_model(config: Mapping[str, Any], input_channels: int):
	"""Instantiate the trained ConvLSTM U-Net architecture."""

	return build_model_from_config(config, input_channels=input_channels)



def _normalize_window(window: np.ndarray, normalization_stats) -> np.ndarray:
	"""Normalize a rollout window using the same statistics as training."""

	if normalization_stats is None:
		return window
	return normalize_tensor(window, normalization_stats["mean"], normalization_stats["std"]).astype(np.float32, copy=False)



def _window_to_model_input(window: np.ndarray, normalization_stats) -> np.ndarray:
	"""Convert a raw rollout window into a model-ready tensor shaped (1, T, C, H, W)."""

	if window.ndim != 4:
		raise ValueError(f"Expected a raw rollout window with shape (T, H, W, C), got {window.shape}.")
	normalized = _normalize_window(window, normalization_stats)
	normalized = np.transpose(normalized, (0, 3, 1, 2))
	normalized = np.ascontiguousarray(normalized, dtype=np.float32)
	model_input = np.expand_dims(normalized, axis=0)
	if model_input.ndim != 5:
		raise ValueError(f"Expected a 5D model input, got shape {model_input.shape}.")
	return model_input



def _update_window(
	current_window: np.ndarray,
	predicted_target: np.ndarray,
	next_true_frame: np.ndarray | None,
	rollout_mode: str,
	input_channel_count: int,
) -> np.ndarray:
	"""Advance the input window using only the model input channels.

	The model is trained on the first ``input_channel_count`` channels, while the
	target label stays separate and is not fed back into the input state.
	"""

	if current_window.ndim != 4:
		raise ValueError(f"Expected current_window shape (T, H, W, C), got {current_window.shape}.")
	if current_window.shape[-1] != input_channel_count:
		raise ValueError(
			f"Expected current_window to have {input_channel_count} channels, got {current_window.shape[-1]}."
		)
	if rollout_mode == "teacher_forced_exogenous":
		if next_true_frame is None:
			raise ValueError("teacher_forced_exogenous requires a future ground-truth frame.")
		new_frame = np.asarray(next_true_frame[:, :, :input_channel_count], dtype=np.float32).copy()
	elif rollout_mode == "constant_exogenous":
		new_frame = current_window[-1, :, :, :input_channel_count].copy()
	else:
		raise ValueError(f"Unsupported rollout_mode: {rollout_mode}")

	updated = np.concatenate([current_window[1:, :, :, :input_channel_count], new_frame[None, ...]], axis=0)
	if updated.shape != current_window.shape:
		raise ValueError("Updated rollout window changed shape unexpectedly.")
	return updated



def _sample_output_name(start_sample_index: int, future_step: int, future_index: int) -> str:
	"""Build an informative filename for rollout outputs."""

	return f"sample_{start_sample_index:05d}_step_{future_step:03d}_t{future_index:05d}.png"



def rollout_predictions(
	config_path: str | Path,
	start_index: int = 0,
	rollout_steps: int = 30,
	rollout_mode: str = "constant_exogenous",
) -> list[Path]:
	"""Run autoregressive rollout visualization from the configured test dataset."""

	if torch is None:
		raise ImportError("PyTorch is required to run rollout predictions.")

	config = _ensure_config_path(load_config(config_path), config_path)
	set_seed(int(config.get("seed", _get_section(config, "training").get("seed", 42))))
	logger = setup_logging(str(_get_section(config, "logging").get("level", "INFO")))

	normalization_config = _get_section(config, "normalization")
	normalization_path = normalization_config.get("path")
	if not normalization_path:
		raise KeyError("Rollout visualization requires normalization stats under config.normalization.path.")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path_obj = Path(config_path_value).expanduser().resolve() if config_path_value else None
	resolved_normalization_path = _resolve_path(config_path_obj, normalization_path)
	if not resolved_normalization_path.exists():
		raise FileNotFoundError(f"Normalization stats not found: {resolved_normalization_path}")
	normalization_stats = load_normalization_stats(resolved_normalization_path)

	test_dataset = _build_test_dataset(config, normalization_stats)
	if len(test_dataset) == 0:
		raise ValueError("Configured test dataset is empty; cannot run rollout visualization.")
	if start_index < 0 or start_index >= len(test_dataset):
		raise IndexError(f"start_index must be in [0, {len(test_dataset) - 1}], got {start_index}.")
	input_channel_count = int(getattr(test_dataset, "input_channel_count", test_dataset.num_channels))

	files = test_dataset.file_paths
	start_sample_index = int(test_dataset.sample_indices[start_index])
	input_sequence_length = int(test_dataset.input_sequence_length)
	target_channel = int(test_dataset.target_channel)
	rollout_steps = int(rollout_steps)
	if rollout_steps <= 0:
		raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")

	input_window_files = files[start_sample_index : start_sample_index + input_sequence_length]
	if len(input_window_files) != input_sequence_length:
		raise ValueError("Unable to assemble the initial rollout input window.")
	current_window = np.stack([
		_load_raw_frame(file_path)[:, :, :input_channel_count] for file_path in input_window_files
	], axis=0)
	if current_window.ndim != 4:
		raise ValueError(f"Expected rollout window shape (T, H, W, C), got {current_window.shape}.")
	if current_window.shape[-1] != input_channel_count:
		raise ValueError(f"Expected {input_channel_count} channels, got {current_window.shape[-1]}.")

	device_setting = str(config.get("device", _get_section(config, "training").get("device", "auto"))).lower()
	if device_setting == "auto":
		device_setting = "cuda" if torch.cuda.is_available() else "cpu"
	if device_setting == "cuda" and not torch.cuda.is_available():
		device_setting = "cpu"
	device = torch.device(device_setting)

	checkpoint_path = _resolve_checkpoint_path(config)
	logger.info("Loading checkpoint: %s", checkpoint_path)
	checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
	model = _build_model(config, input_channels=input_channel_count).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	visualization_config = _get_section(config, "visualization")	
	output_root = _resolve_path(config_path_obj, visualization_config.get("output_path", "./outputs/visualizations"))
	output_dir = output_root / "rollouts" / f"start_{start_index:05d}_{Path(files[start_sample_index]).stem}"
	output_dir.mkdir(parents=True, exist_ok=True)
	cmap = str(visualization_config.get("cmap", "inferno"))
	dpi = int(visualization_config.get("dpi", 150))
	fps = int(visualization_config.get("fps", 2))

	rollout_frames: list[np.ndarray] = []
	ground_truth_frames: list[np.ndarray | None] = []
	titles: list[str] = []
	saved_paths: list[Path] = []
	max_future_with_truth = len(files) - (start_sample_index + input_sequence_length)
	rollout_limit = rollout_steps if rollout_mode == "constant_exogenous" else min(rollout_steps, max_future_with_truth)
	if rollout_limit <= 0:
		raise ValueError("No future timesteps are available for the requested rollout.")

	logger.info(
		"Rolling from configured test sample %s (raw index %s) for %s steps in %s mode.",
		start_index,
		start_sample_index,
		rollout_limit,
		rollout_mode,
	)

	with torch.no_grad():
		for rollout_step in range(rollout_limit):
			absolute_target_index = start_sample_index + input_sequence_length + rollout_step
			model_input = _window_to_model_input(current_window, normalization_stats)
			model_input_tensor = torch.from_numpy(model_input).to(device)
			if model_input_tensor.ndim != 5:
				raise ValueError(f"Expected a 5D tensor for model input, got {tuple(model_input_tensor.shape)}.")

			prediction = model(model_input_tensor)
			if prediction.ndim != 4:
				raise ValueError(f"Model output must have shape (B, C, H, W), got {tuple(prediction.shape)}.")
			if prediction.shape[0] != 1:
				raise ValueError(f"Expected batch size 1 during rollout, got {prediction.shape[0]}.")
			if prediction.shape[1] < 1:
				raise ValueError("Model must predict at least one channel.")

			predicted_target = prediction[0, 0].detach().cpu().numpy()
			if bool(getattr(test_dataset, "normalize_target", False)):
				if test_dataset.target_mean is None or test_dataset.target_std is None:
					raise ValueError("Target normalization is enabled, but target stats are unavailable.")
				predicted_target = inverse_normalize_scalar_channel_map(
					predicted_target,
					test_dataset.target_mean,
					test_dataset.target_std,
				)
			rollout_frames.append(np.asarray(predicted_target, dtype=np.float32))

			next_true_frame = None
			ground_truth_map = None
			if absolute_target_index < len(files):
				next_true_frame = _load_raw_frame(files[absolute_target_index])
				ground_truth_map = np.asarray(next_true_frame[:, :, target_channel], dtype=np.float32)
				ground_truth_frames.append(ground_truth_map)
			else:
				ground_truth_frames.append(None)

			titles.append(f"Start {start_index:05d} | step {rollout_step + 1:03d} | target t{absolute_target_index:05d}")
			frame_output_name = _sample_output_name(start_index, rollout_step + 1, absolute_target_index)
			frame_output_path = output_dir / frame_output_name
			saved_paths.append(
				save_single_map(
					map_array=predicted_target,
					output_path=frame_output_path,
					title=titles[-1] + " | predicted fire intensity",
					cmap=cmap,
					dpi=dpi,
				)
			)

			if ground_truth_map is not None:
				comparison_output_path = output_dir / f"comparison_{frame_output_name}"
				saved_paths.append(
					save_side_by_side_maps(
						left_map=predicted_target,
						right_map=ground_truth_map,
						output_path=comparison_output_path,
						left_title="Predicted future fire intensity",
						right_title="Ground truth future fire intensity",
						title=titles[-1],
						cmap=cmap,
						dpi=dpi,
						right_available=True,
					)
				)

			future_true_frame = next_true_frame if rollout_mode == "teacher_forced_exogenous" else None
			current_window = _update_window(
				current_window=current_window,
				predicted_target=predicted_target,
				next_true_frame=future_true_frame,
				rollout_mode=rollout_mode,
				input_channel_count=input_channel_count,
			)

	animation_ground_truth = ground_truth_frames if any(frame is not None for frame in ground_truth_frames) else None
	animation_output_path = output_dir / "rollout_animation"
	saved_paths.append(
		save_rollout_animation(
			predicted_frames=rollout_frames,
			ground_truth_frames=animation_ground_truth,
			output_path=animation_output_path,
			titles=titles,
			cmap=cmap,
			fps=fps,
		)
	)

	logger.info("Saved %s rollout artifacts in %s", len(saved_paths), output_dir)
	return saved_paths



def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the command-line parser for rollout visualization."""

	parser = argparse.ArgumentParser(description="Run autoregressive ConvLSTM U-Net rollout visualizations.")
	parser.add_argument(
		"--config",
		default="configs/default.yaml",
		help="Path to the YAML configuration file.",
	)
	parser.add_argument("--start_index", type=int, default=0, help="Start index within the configured test dataset.")
	parser.add_argument("--rollout_steps", type=int, default=30, help="Number of autoregressive rollout steps to run.")
	parser.add_argument(
		"--rollout_mode",
		choices=("teacher_forced_exogenous", "constant_exogenous"),
		default="constant_exogenous",
		help="How to fill non-target channels for each autoregressive step.",
	)
	return parser



def main() -> None:
	"""CLI entry point."""

	args = build_argument_parser().parse_args()
	rollout_predictions(
		args.config,
		start_index=args.start_index,
		rollout_steps=args.rollout_steps,
		rollout_mode=args.rollout_mode,
	)


if __name__ == "__main__":
	main()
