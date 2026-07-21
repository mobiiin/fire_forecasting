"""Shared evaluation helpers for non-neural wildfire baselines."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional visualization
	plt = None

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import DataLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	DataLoader = None

from src.config import load_config
from src.data.cached_patch_dataset import CachedPatchDataset
from src.data.cache import get_patch_cache_dir, target_definition_version, temporal_target_offsets
from src.data.dataset import MultiFirePatchSequenceDataset, metadata_batch_to_list
from src.data.discovery import discover_multiple_datasets
from src.data.patching import resolve_patching_config, resolve_split_patch_mode
from src.data.preprocessing import load_normalization_stats
from src.data.splits import build_sliding_patch_refs_for_split, manual_fire_holdout_splits, multi_fire_chronological_splits
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.train import _coerce_loss_result, _denormalize_target_tensors_for_metrics

from src.baselines.common import ensure_geometry, ensure_initial_fuel, resolve_patch


BaselinePredictor = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, int] | None], np.ndarray]


def _sequence_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
	offsets = temporal_target_offsets(config)
	return {
		"input_sequence_length": int(config["input_sequence_length"]),
		"prediction_horizon": int(config["prediction_horizon"]),
		"target_offset_from_start": int(offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
	}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
	merged = dict(base)
	for key, value in override.items():
		current_value = merged.get(key)
		if isinstance(current_value, Mapping) and isinstance(value, Mapping):
			merged[key] = _deep_merge(current_value, value)
		else:
			merged[key] = value
	return merged


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def _resolve_path(config: Mapping[str, Any], configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path_value = config.get("config_path", config.get("_config_path"))
	if config_path_value:
		return (Path(config_path_value).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
	resolved_path = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved_path)
	config["_config_path"] = str(resolved_path)
	return config


def _resolve_normalization_stats(config: Mapping[str, Any]) -> Mapping[str, np.ndarray] | None:
	normalization_path = _get_section(config, "normalization").get("path")
	if not normalization_path:
		return None
	resolved = _resolve_path(config, normalization_path)
	if not resolved.exists():
		return None
	return load_normalization_stats(resolved)


def _build_split_refs(config: Mapping[str, Any], dataset_records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "multi_dataset_chronological":
		split_mode = "multi_fire_chronological"
	if split_mode == "manual_fire_holdout":
		manual = _get_section(config, "manual_fire_split")
		return manual_fire_holdout_splits(
			dataset_records=dataset_records,
			train_fire_names=manual.get("train_fires", []),
			val_fire_names=manual.get("val_fires", []),
			test_fire_names=manual.get("test_fires", []),
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			config=config,
		)
	if split_mode == "multi_fire_chronological":
		return multi_fire_chronological_splits(
			dataset_records=dataset_records,
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			train_fraction=float(config.get("train_fraction", 0.7)),
			val_fraction=float(config.get("val_fraction", 0.15)),
			test_fraction=float(config.get("test_fraction", 0.15)),
		)
	raise ValueError(
		"Baseline evaluation currently supports split_mode=manual_fire_holdout or multi_fire_chronological."
	)


def _cache_enabled_for_patch_mode(config: Mapping[str, Any]) -> bool:
	cache = _get_section(config, "cache")
	return bool(cache.get("enabled", False) and cache.get("use_precomputed_patches", False))


def _build_cached_split_loader(
	config: Mapping[str, Any],
	split: str,
	normalization_stats: Mapping[str, np.ndarray] | None,
):
	if DataLoader is None:
		raise ImportError("PyTorch is required to build cached baseline DataLoaders.")
	cache_dir = get_patch_cache_dir(config)
	dataset = CachedPatchDataset(
		cache_dir=cache_dir,
		split=split,
		config=config,
		normalization_stats=normalization_stats,
		return_metadata=True,
	)
	batch_size = int(_get_section(config, "training").get("batch_size", config.get("batch_size", 4)))
	num_workers = int(_get_section(config, "training").get("num_workers", config.get("num_workers", 0)))
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=False,
		drop_last=False,
	)


def _build_dynamic_split_loader(
	config: Mapping[str, Any],
	split: str,
	mode: str,
	dataset_records: Sequence[Mapping[str, Any]],
	normalization_stats: Mapping[str, np.ndarray] | None,
):
	if DataLoader is None:
		raise ImportError("PyTorch is required to build dynamic baseline DataLoaders.")
	sample_refs = _build_split_refs(config, dataset_records)
	if split not in sample_refs:
		raise KeyError(f"Unknown split {split!r}.")
	selected_refs = list(sample_refs[split])
	patching = resolve_patching_config(config)
	use_patches = False
	split_patch_mode = resolve_split_patch_mode(config, split, prefer_cache=False)
	if mode == "patch":
		if split_patch_mode == "sliding_window":
			selected_refs = build_sliding_patch_refs_for_split(dataset_records=dataset_records, sample_refs=selected_refs, split=split, config=config)
			use_patches = True
		elif split == "train":
			use_patches = bool(patching["enabled"])
	elif mode == "full_domain_tiled":
		if split_patch_mode == "sliding_window":
			selected_refs = build_sliding_patch_refs_for_split(dataset_records=dataset_records, sample_refs=selected_refs, split=split, config=config)
			use_patches = True
	else:
		raise ValueError(f"Unsupported baseline evaluation mode: {mode!r}.")

	dataset = MultiFirePatchSequenceDataset(
		dataset_records=dataset_records,
		sample_refs=selected_refs,
		input_sequence_length=int(config["input_sequence_length"]),
		prediction_horizon=int(config["prediction_horizon"]),
		target_channel=int(config.get("target_channel", 0)),
		input_channel_count=int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0))),
		input_channel_indices=config.get("input_channel_indices"),
		task_type=str(config.get("task_type", "multitask")),
		fire_threshold=float(config.get("fire_threshold", 0.5)),
		use_patches=use_patches,
		patch_size=int(patching["patch_height"]),
		active_patch_probability=float(patching["active_patch_probability"]),
		active_threshold=float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		normalization_stats=normalization_stats,
		normalize_target=bool(_get_section(config, "target_normalization").get("enabled", False)),
		return_metadata=True,
		config=config,
		split=split,
	)
	batch_size = int(_get_section(config, "training").get("batch_size", config.get("batch_size", 4)))
	num_workers = int(_get_section(config, "training").get("num_workers", config.get("num_workers", 0)))
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=False,
		drop_last=False,
	)


def _prepare_dataset_records(dataset_records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
	prepared: dict[str, dict[str, Any]] = {}
	for record in dataset_records:
		record_dict = dict(record)
		record_dict["initial_fuel"] = ensure_initial_fuel(record_dict, config)
		record_dict["geometry"] = ensure_geometry(record_dict, config)
		prepared[str(record_dict["dataset_name"])] = record_dict
	return prepared


def _normalize_prediction_for_loss(loader, prediction: torch.Tensor) -> torch.Tensor:
	dataset = getattr(loader, "dataset", None)
	normalize_target = bool(getattr(dataset, "normalize_target", False))
	target_mean = getattr(dataset, "target_mean", None)
	target_std = getattr(dataset, "target_std", None)
	task_type = str(getattr(dataset, "task_type", "regression")).lower()
	if not normalize_target or target_mean is None or target_std is None:
		return prediction
	prediction = prediction.clone()
	if task_type == "multitask":
		mean = torch.as_tensor(target_mean, dtype=prediction.dtype, device=prediction.device).reshape(1, -1, 1, 1)
		std = torch.clamp(torch.as_tensor(target_std, dtype=prediction.dtype, device=prediction.device).reshape(1, -1, 1, 1), min=1.0e-6)
		regression_channels = min(int(mean.shape[1]), 2)
		prediction[:, :regression_channels] = (prediction[:, :regression_channels] - mean[:, :regression_channels]) / std[:, :regression_channels]
		return prediction
	if task_type == "regression":
		mean_value = torch.as_tensor(float(target_mean), dtype=prediction.dtype, device=prediction.device)
		std_value = torch.clamp(torch.as_tensor(float(target_std), dtype=prediction.dtype, device=prediction.device), min=1.0e-6)
		return (prediction - mean_value) / std_value
	return prediction


def _resolve_dataset_record(
	prepared_records: Mapping[str, Mapping[str, Any]],
	metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
	dataset_name = str(metadata.get("dataset_name", metadata.get("fire_name", "")))
	if dataset_name in prepared_records:
		return prepared_records[dataset_name]
	raise KeyError(f"Could not resolve dataset record for metadata dataset_name={dataset_name!r}.")


def _save_prediction_artifact(output_root: Path, metadata: Mapping[str, Any], prediction: np.ndarray, target: np.ndarray) -> None:
	dataset_name = str(metadata.get("dataset_name", metadata.get("fire_name", "dataset")))
	sample_index = int(metadata.get("sample_index", -1))
	patch = resolve_patch(metadata=metadata)
	patch_suffix = ""
	if patch is not None:
		patch_suffix = f"_y{patch['y0']:03d}_x{patch['x0']:03d}"
	output_root.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output_root / f"{dataset_name}_sample{sample_index:05d}{patch_suffix}.npz",
		prediction=np.asarray(prediction, dtype=np.float32),
		target=np.asarray(target, dtype=np.float32),
	)


def _save_visualization(output_root: Path, metadata: Mapping[str, Any], prediction: np.ndarray, target: np.ndarray) -> None:
	if plt is None:  # pragma: no cover - optional dependency
		return
	dataset_name = str(metadata.get("dataset_name", metadata.get("fire_name", "dataset")))
	sample_index = int(metadata.get("sample_index", -1))
	patch = resolve_patch(metadata=metadata)
	patch_suffix = ""
	if patch is not None:
		patch_suffix = f"_y{patch['y0']:03d}_x{patch['x0']:03d}"
	channel_titles = ("surface", "canopy", "mask_logits", "energy")
	figure, axes = plt.subplots(2, min(4, prediction.shape[0]), figsize=(4 * min(4, prediction.shape[0]), 8))
	if min(4, prediction.shape[0]) == 1:
		axes = np.asarray(axes).reshape(2, 1)
	for channel_index in range(min(4, prediction.shape[0])):
		axes[0, channel_index].imshow(target[channel_index], cmap="inferno")
		axes[0, channel_index].set_title(f"true {channel_titles[channel_index]}")
		axes[1, channel_index].imshow(prediction[channel_index], cmap="inferno")
		axes[1, channel_index].set_title(f"pred {channel_titles[channel_index]}")
		axes[0, channel_index].axis("off")
		axes[1, channel_index].axis("off")
	figure.suptitle(f"{dataset_name} sample={sample_index}")
	output_root.mkdir(parents=True, exist_ok=True)
	figure.tight_layout()
	figure.savefig(output_root / f"{dataset_name}_sample{sample_index:05d}{patch_suffix}.png", dpi=150)
	plt.close(figure)


def _write_csv(output_csv: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def evaluate_baseline(
	config_path: str | Path,
	split: str,
	method_name: str,
	predict_fn: BaselinePredictor,
	mode: str = "patch",
	num_samples: int | None = None,
	max_batches: int | None = None,
	config_override: Mapping[str, Any] | None = None,
	output_csv: str | Path | None = None,
	save_predictions: bool = False,
	save_visualizations: bool = False,
) -> dict[str, Any]:
	if torch is None:
		raise ImportError("PyTorch is required to evaluate wildfire baselines.")

	config = _ensure_config_path(load_config(config_path), config_path)
	if isinstance(config_override, Mapping):
		config = _deep_merge(config, dict(config_override))
	config["return_metadata"] = True
	normalization_stats = _resolve_normalization_stats(config)
	dataset_records = discover_multiple_datasets(config)
	prepared_records = _prepare_dataset_records(dataset_records, config)
	if mode == "patch" and _cache_enabled_for_patch_mode(config):
		loader = _build_cached_split_loader(config=config, split=split, normalization_stats=normalization_stats)
	else:
		loader = _build_dynamic_split_loader(
			config=config,
			split=split,
			mode=mode,
			dataset_records=dataset_records,
			normalization_stats=normalization_stats,
		)

	criterion = get_loss_function(config)
	total_samples = 0
	total_loss = 0.0
	aggregate_metrics: dict[str, float] = defaultdict(float)
	aggregate_loss_components: dict[str, float] = defaultdict(float)
	per_dataset: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

	prediction_output_root = Path("artifacts/baselines") / method_name / split / "predictions"
	visualization_output_root = Path("artifacts/baselines") / method_name / split / "visualizations"

	if max_batches is not None and int(max_batches) <= 0:
		raise ValueError("max_batches must be positive when provided.")

	for batch_index, batch in enumerate(loader):
		if max_batches is not None and batch_index >= int(max_batches):
			break
		if not isinstance(batch, (tuple, list)) or len(batch) < 3:
			raise TypeError("Baseline evaluation requires batches with input tensors, target tensors, and metadata.")
		y_batch = batch[1]
		metadata_items = metadata_batch_to_list(batch[2], batch_size=int(y_batch.shape[0]))
		batch_limit = len(metadata_items)
		if num_samples is not None:
			remaining = int(num_samples) - total_samples
			if remaining <= 0:
				break
			batch_limit = min(batch_limit, remaining)

		for sample_index in range(batch_limit):
			metadata = metadata_items[sample_index]
			dataset_record = _resolve_dataset_record(prepared_records, metadata)
			patch = resolve_patch(metadata=metadata)
			prediction_np = predict_fn(
				dataset_record=dataset_record,
				sample_ref=metadata,
				config=config,
				patch=patch,
			)
			target_np = np.asarray(y_batch[sample_index].detach().cpu().numpy(), dtype=np.float32)
			prediction_tensor = torch.from_numpy(np.ascontiguousarray(prediction_np[None, ...], dtype=np.float32)).to(torch.float32)
			target_tensor = y_batch[sample_index : sample_index + 1].detach().to(torch.float32)
			loss_input = _normalize_prediction_for_loss(loader, prediction_tensor)
			loss_result = criterion(loss_input, target_tensor)
			loss, loss_components = _coerce_loss_result(loss_result)
			metric_pred, metric_true = _denormalize_target_tensors_for_metrics(loader, prediction_tensor.clone(), target_tensor.clone())
			sample_metrics = compute_metrics(metric_pred, metric_true, config)
			dataset_name = str(metadata.get("dataset_name", metadata.get("fire_name", "dataset")))

			total_samples += 1
			total_loss += float(loss.detach().item())
			per_dataset[dataset_name]["num_samples"] += 1.0
			per_dataset[dataset_name]["test_loss"] += float(loss.detach().item())
			for component_name, component_value in loss_components.items():
				aggregate_loss_components[component_name] += float(component_value)
				per_dataset[dataset_name][f"test_{component_name}"] += float(component_value)
			for metric_name, metric_value in sample_metrics.items():
				aggregate_metrics[metric_name] += float(metric_value)
				per_dataset[dataset_name][f"test_{metric_name}"] += float(metric_value)

			if save_predictions:
				_save_prediction_artifact(
					output_root=prediction_output_root,
					metadata=metadata,
					prediction=prediction_np,
					target=metric_true[0].detach().cpu().numpy(),
				)
			if save_visualizations:
				_save_visualization(
					output_root=visualization_output_root,
					metadata=metadata,
					prediction=prediction_np,
					target=metric_true[0].detach().cpu().numpy(),
				)

	if total_samples == 0:
		raise ValueError(f"Baseline evaluation for split={split!r} produced no samples.")

	aggregate_results: dict[str, float] = {"test_loss": total_loss / total_samples}
	for component_name, total_value in aggregate_loss_components.items():
		aggregate_results[f"test_{component_name}"] = total_value / total_samples
	for metric_name, total_value in aggregate_metrics.items():
		aggregate_results[f"test_{metric_name}"] = total_value / total_samples

	per_dataset_results: dict[str, dict[str, float]] = {}
	for dataset_name, totals in per_dataset.items():
		sample_count = max(int(totals.get("num_samples", 0.0)), 1)
		row = {"num_samples": float(sample_count)}
		for key, value in totals.items():
			if key == "num_samples":
				continue
			row[key] = float(value) / sample_count
		per_dataset_results[dataset_name] = row

	rows: list[dict[str, Any]] = []
	sequence_metadata = _sequence_metadata(config)
	rows.append(
		{
			"method": method_name,
			"split": split,
			"scope": "aggregate",
			"dataset_name": "",
			"num_samples": total_samples,
			**sequence_metadata,
			**aggregate_results,
		}
	)
	for dataset_name in sorted(per_dataset_results):
		rows.append(
			{
				"method": method_name,
				"split": split,
				"scope": "per_fire",
				"dataset_name": dataset_name,
				**sequence_metadata,
				**per_dataset_results[dataset_name],
			}
		)

	if output_csv is not None:
		_write_csv(Path(output_csv).expanduser().resolve(), rows)

	return {
		"method": method_name,
		"split": split,
		"num_samples": total_samples,
		"sequence": sequence_metadata,
		"aggregate_results": aggregate_results,
		"per_dataset_results": per_dataset_results,
		"rows": rows,
	}
