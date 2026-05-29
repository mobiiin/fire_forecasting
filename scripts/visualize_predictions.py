"""Visualize wildfire predictions and multitask targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import DataLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	DataLoader = None

from scripts.evaluate_persistence_baseline import build_persistence_sample, discover_files, regression_metrics
from src.config import load_config
from src.data.dataset import FireSequenceDataset, _resolve_multitask_config, _sort_chronologically
from src.data.preprocessing import inverse_normalize_channel_map as inverse_normalize_scalar_channel_map
from src.data.preprocessing import load_normalization_stats
from src.data.splits import chronological_split_indices, chronological_train_val_split_indices
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.visualization.fuel_reconstruction import reconstruct_future_fuel_bed
from src.visualization.plot_maps import (
	plot_model_vs_persistence_grid,
	plot_multitask_prediction_grid,
	plot_prediction_grid,
)


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


def _discover_files(config: Mapping[str, Any]) -> list[Path]:
	"""Discover and chronologically sort dataset files."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	data_dir = _resolve_path(config_path, config["data_dir"])
	file_pattern = str(config["file_pattern"])
	files = _sort_chronologically(list(data_dir.glob(file_pattern)))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
	return files


def _build_dataset_for_split(config: Mapping[str, Any], normalization_stats, split: str) -> FireSequenceDataset:
	"""Build a dataset for validation or external test visualization."""

	split = str(split).lower()
	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	common_kwargs = {
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_channel": int(config.get("target_channel", 0)),
		"input_channel_count": int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0))),
		"input_channel_indices": config.get("input_channel_indices"),
		"task_type": str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))),
		"fire_threshold": float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
		"normalization_stats": normalization_stats,
		"normalize_target": bool(_get_section(config, "target_normalization").get("enabled", _get_section(config, "normalization").get("normalize_target", False))),
		"return_metadata": True,
		"config": config,
	}

	if split == "val":
		files = _discover_files(config)
		split_mode = str(config.get("split_mode", "train_val_test")).lower()
		if split_mode == "train_val_external_test":
			splits = chronological_train_val_split_indices(
				num_timesteps=len(files),
				input_sequence_length=input_sequence_length,
				prediction_horizon=prediction_horizon,
				train_fraction=float(config.get("train_fraction", 0.85)),
				val_fraction=float(config.get("val_fraction", 0.15)),
			)
		else:
			splits = chronological_split_indices(
				num_timesteps=len(files),
				input_sequence_length=input_sequence_length,
				prediction_horizon=prediction_horizon,
				train_fraction=float(config.get("train_fraction", 0.7)),
				val_fraction=float(config.get("val_fraction", 0.15)),
				test_fraction=float(config.get("test_fraction", 0.15)),
				split_mode=split_mode,
			)
		return FireSequenceDataset(file_paths=files, sample_indices=splits["val"], **common_kwargs)

	if split == "test":
		test_data_dir = config.get("test_data_dir")
		if test_data_dir in (None, "", "null"):
			raise ValueError(
				"No external test_data_dir configured. This project now uses data_dir only for train/val. "
				"Set test_data_dir in the config to visualize external test predictions."
			)
		config_path_value = config.get("config_path", config.get("_config_path"))
		config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
		test_data_dir_path = _resolve_path(config_path, test_data_dir)
		external_file_pattern = str(config.get("external_test_file_pattern", config["file_pattern"]))
		files = _sort_chronologically(list(test_data_dir_path.glob(external_file_pattern)))
		if not files:
			raise FileNotFoundError(
				f"No external test files found in '{test_data_dir_path}' using pattern '{external_file_pattern}'."
			)
		return FireSequenceDataset(file_paths=files, sample_indices=None, **common_kwargs)

	raise ValueError(f"split must be 'val' or 'test', got {split!r}.")


def _build_test_dataset(config: Mapping[str, Any], normalization_stats) -> FireSequenceDataset:
	"""Backward-compatible wrapper for external test visualization dataset."""

	return _build_dataset_for_split(config, normalization_stats, split="test")


def _build_test_loader(dataset):
	"""Create a chronological, sample-by-sample test DataLoader."""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to build the visualization DataLoader.")
	return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)


def _select_device(config: Mapping[str, Any]):
	"""Select the inference device from config, defaulting to CPU when unavailable."""

	device_setting = str(config.get("device", _get_section(config, "training").get("device", "auto"))).lower()
	if device_setting == "auto":
		device_setting = "cuda" if torch.cuda.is_available() else "cpu"
	if device_setting == "cuda" and not torch.cuda.is_available():
		device_setting = "cpu"
	return torch.device(device_setting)


def _extract_first_item(value):
	"""Normalize collated metadata values to their first scalar/string item."""

	if torch is not None and torch.is_tensor(value):
		return value.reshape(-1)[0].item() if value.numel() else None
	if isinstance(value, (list, tuple)):
		return value[0]
	return value


def _metadata_to_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
	"""Convert a collated metadata batch into a simple dictionary."""

	return {key: _extract_first_item(value) for key, value in metadata.items()}


def _build_checkpoint_path(config: Mapping[str, Any]) -> Path:
	"""Resolve the best checkpoint path and fall back to the latest checkpoint if needed."""

	checkpoint_config = _get_section(config, "checkpoint")
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	latest_path, best_path = latest_and_best_checkpoint_paths(_resolve_path(config_path, checkpoint_path))
	selected = best_path if best_path.exists() else latest_path
	if not selected.exists():
		raise FileNotFoundError(
			"No checkpoint found for visualization. "
			f"Checked best='{best_path}' and latest='{latest_path}'."
		)
	return selected


def _build_model(config: Mapping[str, Any], input_channels: int):
	"""Instantiate and load the trained ConvLSTM U-Net."""

	return build_model_from_config(config, input_channels=input_channels)


def _sample_output_name(metadata: Mapping[str, Any]) -> str:
	"""Build an informative filename from the sample metadata."""

	sample_index = metadata.get("sample_index")
	file_path = metadata.get("target_file_path")
	file_stem = Path(str(file_path)).stem if file_path else "sample"
	if sample_index is None:
		return f"{file_stem}.png"
	try:
		sample_index_int = int(sample_index)
	except (TypeError, ValueError):
		return f"{file_stem}.png"
	return f"sample_{sample_index_int:05d}_{file_stem}.png"


def _load_raw_frame(file_path: str | Path) -> np.ndarray:
	"""Load one raw dataset frame from disk."""

	array = np.load(Path(file_path).expanduser().resolve(), mmap_mode="r", allow_pickle=False)
	if array.ndim != 3:
		raise ValueError(f"Expected a 3D raw frame, got shape {array.shape}.")
	return np.asarray(array, dtype=np.float32)


def _crop_map_from_metadata(channel_map: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
	"""Crop a 2D map to the evaluated patch when metadata contains patch coordinates."""

	patch_top = metadata.get("patch_top")
	patch_left = metadata.get("patch_left")
	patch_size = metadata.get("patch_size")
	if patch_top is None or patch_left is None or patch_size is None:
		return np.asarray(channel_map, dtype=np.float32)
	patch_top = int(patch_top)
	patch_left = int(patch_left)
	patch_size = int(patch_size)
	return np.asarray(channel_map[patch_top : patch_top + patch_size, patch_left : patch_left + patch_size], dtype=np.float32)


def _prediction_to_single_map(predicted: torch.Tensor, test_dataset: FireSequenceDataset, task_type: str) -> np.ndarray:
	"""Convert a single-target model output tensor into a display-ready map."""

	if task_type == "segmentation":
		predicted_map = torch.sigmoid(predicted[0, 0]).detach().cpu().numpy()
	else:
		predicted_map = predicted[0, 0].detach().cpu().numpy()
	if bool(getattr(test_dataset, "normalize_target", False)) and test_dataset.target_mean is not None and test_dataset.target_std is not None and task_type == "regression":
		predicted_map = inverse_normalize_scalar_channel_map(predicted_map, test_dataset.target_mean, test_dataset.target_std)
	return np.asarray(predicted_map, dtype=np.float32)


def _build_multitask_visualization_maps(
	predicted: torch.Tensor,
	y_sample: torch.Tensor,
	metadata: Mapping[str, Any],
	config: Mapping[str, Any],
	test_dataset: FireSequenceDataset,
) -> dict[str, np.ndarray]:
	"""Build multitask fuel/mask maps for visualization."""

	multitask = _resolve_multitask_config(config)
	current_frame = _load_raw_frame(metadata["current_file_path"])
	current_surface_fuel = _crop_map_from_metadata(current_frame[:, :, int(multitask["surface_fuel_channel"])], metadata)
	current_canopy_fuel = _crop_map_from_metadata(current_frame[:, :, int(multitask["canopy_fuel_channel"])], metadata)

	true_surface_consumed = y_sample[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
	true_canopy_consumed = y_sample[0, 1].detach().cpu().numpy().astype(np.float32, copy=False)
	true_mask = y_sample[0, 2].detach().cpu().numpy().astype(np.float32, copy=False)
	pred_surface_consumed = predicted[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
	pred_canopy_consumed = predicted[0, 1].detach().cpu().numpy().astype(np.float32, copy=False)

	if bool(getattr(test_dataset, "normalize_target", False)) and test_dataset.target_mean is not None and test_dataset.target_std is not None:
		target_mean = np.asarray(test_dataset.target_mean, dtype=np.float32)
		target_std = np.asarray(test_dataset.target_std, dtype=np.float32)
		if target_mean.shape[0] >= 2 and target_std.shape[0] >= 2:
			true_surface_consumed = inverse_normalize_scalar_channel_map(true_surface_consumed, target_mean[0], target_std[0])
			true_canopy_consumed = inverse_normalize_scalar_channel_map(true_canopy_consumed, target_mean[1], target_std[1])
			pred_surface_consumed = inverse_normalize_scalar_channel_map(pred_surface_consumed, target_mean[0], target_std[0])
			pred_canopy_consumed = inverse_normalize_scalar_channel_map(pred_canopy_consumed, target_mean[1], target_std[1])

	true_future_surface_fuel, true_future_canopy_fuel = reconstruct_future_fuel_bed(
		current_surface_fuel=current_surface_fuel,
		current_canopy_fuel=current_canopy_fuel,
		pred_surface_consumed=true_surface_consumed,
		pred_canopy_consumed=true_canopy_consumed,
		clamp_nonnegative=True,
	)
	pred_future_surface_fuel, pred_future_canopy_fuel = reconstruct_future_fuel_bed(
		current_surface_fuel=current_surface_fuel,
		current_canopy_fuel=current_canopy_fuel,
		pred_surface_consumed=pred_surface_consumed,
		pred_canopy_consumed=pred_canopy_consumed,
		clamp_nonnegative=True,
	)
	pred_mask_probability = torch.sigmoid(predicted[0, 2]).detach().cpu().numpy().astype(np.float32, copy=False)
	pred_mask = (pred_mask_probability > 0.5).astype(np.float32, copy=False)

	return {
		"current_surface_fuel": current_surface_fuel,
		"true_future_surface_fuel": np.asarray(true_future_surface_fuel, dtype=np.float32),
		"pred_future_surface_fuel": np.asarray(pred_future_surface_fuel, dtype=np.float32),
		"current_canopy_fuel": current_canopy_fuel,
		"true_future_canopy_fuel": np.asarray(true_future_canopy_fuel, dtype=np.float32),
		"pred_future_canopy_fuel": np.asarray(pred_future_canopy_fuel, dtype=np.float32),
		"true_mask": true_mask,
		"pred_mask_probability": pred_mask_probability,
		"pred_mask": pred_mask,
	}


def visualize_predictions(config_path: str | Path, num_samples: int = 10, split: str = "val") -> list[Path]:
	"""Render chronological forecast visualizations for several test samples."""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to visualize predictions.")

	config = _ensure_config_path(load_config(config_path), config_path)
	set_seed(int(config.get("seed", _get_section(config, "training").get("seed", 42))))
	logger = setup_logging(str(_get_section(config, "logging").get("level", "INFO")))

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path_obj = Path(config_path_value).expanduser().resolve() if config_path_value else None
	normalization_path = _get_section(config, "normalization").get("path")
	normalization_stats = None
	if normalization_path:
		resolved_normalization_path = _resolve_path(config_path_obj, normalization_path)
		if resolved_normalization_path.exists():
			normalization_stats = load_normalization_stats(resolved_normalization_path)

	selected_split = str(split).lower()
	test_dataset = _build_dataset_for_split(config, normalization_stats, split=selected_split)
	if len(test_dataset) == 0:
		raise ValueError(f"Requested {selected_split} dataset is empty; cannot visualize predictions.")
	test_loader = _build_test_loader(test_dataset)
	task_type = str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))).lower()
	first_batch = next(iter(test_loader))
	x_batch, y_batch = first_batch[:2]
	if x_batch.ndim != 5 or y_batch.ndim != 4:
		raise ValueError(f"Unexpected visualization batch shapes: X={tuple(x_batch.shape)} y={tuple(y_batch.shape)}")

	input_channels = int(x_batch.shape[2])
	device = _select_device(config)
	checkpoint_path = _build_checkpoint_path(config)
	logger.info("Loading checkpoint: %s", checkpoint_path)
	checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
	model = _build_model(config, input_channels=input_channels).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	visualization_config = _get_section(config, "visualization")
	output_dir = _resolve_path(config_path_obj, visualization_config.get("output_path", "./outputs/visualizations_multitask"))
	output_dir.mkdir(parents=True, exist_ok=True)
	cmap = str(visualization_config.get("cmap", "inferno"))
	dpi = int(visualization_config.get("dpi", 150))

	saved_paths: list[Path] = []
	max_samples = min(int(num_samples), len(test_loader.dataset))
	with torch.no_grad():
		for sample_index, batch in enumerate(test_loader):
			if sample_index >= max_samples:
				break
			x_sample, y_sample, metadata = batch
			metadata_dict = _metadata_to_dict(metadata)
			x_sample = x_sample.to(device)
			predicted = model(x_sample)
			output_name = _sample_output_name(metadata_dict)
			output_path = output_dir / output_name
			if output_path.exists():
				output_path = output_dir / f"{output_path.stem}_{sample_index:05d}{output_path.suffix}"

			if task_type == "multitask":
				plot_inputs = _build_multitask_visualization_maps(predicted, y_sample, metadata_dict, config, test_dataset)
				saved_path = plot_multitask_prediction_grid(
					current_surface_fuel=plot_inputs["current_surface_fuel"],
					true_future_surface_fuel=plot_inputs["true_future_surface_fuel"],
					pred_future_surface_fuel=plot_inputs["pred_future_surface_fuel"],
					current_canopy_fuel=plot_inputs["current_canopy_fuel"],
					true_future_canopy_fuel=plot_inputs["true_future_canopy_fuel"],
					pred_future_canopy_fuel=plot_inputs["pred_future_canopy_fuel"],
					true_mask=plot_inputs["true_mask"],
					pred_mask_probability=plot_inputs["pred_mask_probability"],
					pred_mask=plot_inputs["pred_mask"],
					output_path=output_path,
					title=f"{selected_split.capitalize()} multitask forecast | sample {sample_index + 1}/{max_samples}",
					cmap=cmap,
					dpi=dpi,
				)
			else:
				current_map = x_sample[0, -1, -1].detach().cpu().numpy()
				ground_truth_map = y_sample[0, 0].detach().cpu().numpy()
				predicted_map = _prediction_to_single_map(predicted, test_dataset, task_type)
				if task_type == "regression" and bool(getattr(test_dataset, "normalize_target", False)) and test_dataset.target_mean is not None and test_dataset.target_std is not None:
					ground_truth_map = inverse_normalize_scalar_channel_map(
						ground_truth_map,
						test_dataset.target_mean,
						test_dataset.target_std,
					)
				saved_path = plot_prediction_grid(
					current_map=current_map,
					ground_truth_map=ground_truth_map,
					predicted_map=predicted_map,
					output_path=output_path,
					title=f"{selected_split.capitalize()} forecast | sample {sample_index + 1}/{max_samples}",
					threshold=0.5 if task_type == "segmentation" else float(config.get("fire_threshold", 0.5)),
					cmap=cmap,
					dpi=dpi,
					normalization_stats=normalization_stats,
					channel_index=int(x_sample.shape[2] - 1),
					draw_contours=True,
				)
			saved_paths.append(saved_path)
			logger.info("Saved visualization: %s", saved_path)

	return saved_paths


def visualize_model_vs_persistence(
	config_path: str | Path,
	num_samples: int = 20,
	output_dir: str | Path = "outputs/model_vs_persistence",
) -> dict[str, Any]:
	"""Save side-by-side model-vs-persistence comparison plots for regression targets."""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to visualize predictions.")
	config = _ensure_config_path(load_config(config_path), config_path)
	task_type = str(config.get("task_type", "regression")).lower()
	if task_type != "regression":
		raise ValueError("visualize_model_vs_persistence currently supports regression mode only.")

	set_seed(int(config.get("seed", _get_section(config, "training").get("seed", 42))))
	logger = setup_logging(str(_get_section(config, "logging").get("level", "INFO")))
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path_obj = Path(config_path_value).expanduser().resolve() if config_path_value else None

	normalization_path = _get_section(config, "normalization").get("path")
	normalization_stats = None
	if normalization_path:
		resolved_normalization_path = _resolve_path(config_path_obj, normalization_path)
		if resolved_normalization_path.exists():
			normalization_stats = load_normalization_stats(resolved_normalization_path)

	test_dataset = _build_test_dataset(config, normalization_stats)
	test_loader = _build_test_loader(test_dataset)
	input_channels = int(next(iter(test_loader))[0].shape[2])
	device = _select_device(config)
	checkpoint_path = _build_checkpoint_path(config)
	checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
	model = _build_model(config, input_channels=input_channels).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	files = discover_files(Path(config["config_path"]).expanduser().resolve(), config)
	resolved_output_dir = _resolve_path(config_path_obj, output_dir)
	resolved_output_dir.mkdir(parents=True, exist_ok=True)
	cmap = str(_get_section(config, "visualization").get("cmap", "inferno"))
	dpi = int(_get_section(config, "visualization").get("dpi", 150))
	eps = float(_get_section(config, "metrics").get("eps", _get_section(config, "training").get("eps", 1e-6)))
	active_threshold = float(config.get("active_threshold", config.get("fire_threshold", 0.0)))

	saved_paths: list[Path] = []
	persistence_predictions: list[np.ndarray] = []
	model_predictions: list[np.ndarray] = []
	ground_truth_targets: list[np.ndarray] = []
	max_samples = min(int(num_samples), len(test_loader.dataset))
	with torch.no_grad():
		for sample_position, batch in enumerate(test_loader):
			if sample_position >= max_samples:
				break
			x_sample, _, metadata = batch
			metadata_dict = _metadata_to_dict(metadata)
			predicted = model(x_sample.to(device))
			persistence_sample = build_persistence_sample(
				config=config,
				files=files,
				target_channel=int(config["target_channel"]),
				sample_start=int(metadata_dict["sample_index"]),
				patch_top=int(metadata_dict["patch_top"]) if metadata_dict.get("patch_top") is not None else None,
				patch_left=int(metadata_dict["patch_left"]) if metadata_dict.get("patch_left") is not None else None,
				patch_size=int(metadata_dict["patch_size"]) if metadata_dict.get("patch_size") is not None else None,
			)
			last_input_target_map = np.asarray(persistence_sample["current_map"], dtype=np.float32)
			ground_truth_map = np.asarray(persistence_sample["true_future_map"], dtype=np.float32)
			persistence_prediction = np.asarray(persistence_sample["persistence_prediction"], dtype=np.float32)
			model_prediction = _prediction_to_single_map(predicted, test_dataset, task_type)

			output_path = resolved_output_dir / _sample_output_name(metadata_dict)
			saved_path = plot_model_vs_persistence_grid(
				last_input_target_map=last_input_target_map,
				ground_truth_map=ground_truth_map,
				persistence_prediction_map=persistence_prediction,
				model_prediction_map=model_prediction,
				output_path=output_path,
				title=f"Model vs persistence | sample {sample_position + 1}/{max_samples}",
				cmap=cmap,
				dpi=dpi,
			)
			saved_paths.append(saved_path)
			persistence_predictions.append(persistence_prediction)
			model_predictions.append(model_prediction)
			ground_truth_targets.append(ground_truth_map)
			logger.info("Saved model-vs-persistence visualization: %s", saved_path)

	def _aggregate(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
		total_abs_error = 0.0
		total_squared_error = 0.0
		total_pixels = 0
		for prediction, target in zip(predictions, targets):
			error = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
			total_abs_error += float(np.abs(error).sum())
			total_squared_error += float(np.square(error).sum())
			total_pixels += int(error.size)
		return {
			"mae": total_abs_error / total_pixels,
			"rmse": float(np.sqrt(total_squared_error / total_pixels + eps)),
		}

	persistence_summary = _aggregate(persistence_predictions, ground_truth_targets)
	model_summary = _aggregate(model_predictions, ground_truth_targets)
	ratio = float(model_summary["mae"] / persistence_summary["mae"]) if persistence_summary["mae"] > 0.0 else float("inf")
	print(f"persistence MAE: {persistence_summary['mae']:.6f}")
	print(f"model MAE: {model_summary['mae']:.6f}")
	print(f"persistence RMSE: {persistence_summary['rmse']:.6f}")
	print(f"model RMSE: {model_summary['rmse']:.6f}")
	print(f"ratio: {ratio:.6f}" if np.isfinite(ratio) else "ratio: inf")

	return {
		"checkpoint_path": str(checkpoint_path),
		"output_dir": str(resolved_output_dir),
		"num_samples": max_samples,
		"saved_paths": [str(path) for path in saved_paths],
		"persistence_metrics": persistence_summary,
		"model_metrics": model_summary,
		"ratio": ratio,
	}


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the command-line argument parser."""

	parser = argparse.ArgumentParser(description="Visualize ConvLSTM U-Net wildfire predictions.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", choices=("val", "test"), default="val", help="Which split to visualize.")
	parser.add_argument(
		"--num_samples",
		type=int,
		default=10,
		help="Number of chronological validation or external test samples to visualize.",
	)
	return parser


def main() -> None:
	"""CLI entry point."""

	args = build_argument_parser().parse_args()
	visualize_predictions(args.config, num_samples=args.num_samples, split=args.split)


if __name__ == "__main__":
	main()
