"""Shared checkpoint evaluation helpers for wildfire models."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping
import warnings

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.data.cache import target_definition_version, temporal_target_offsets
from src.data.dataset import create_dataloaders, metadata_batch_to_list
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, validate_checkpoint_model_compatibility
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.hardware import autocast_context, choose_amp_dtype
from src.training.input_normalization import (
	apply_input_normalization,
	build_input_normalizer_for_loader,
	compare_normalization_metadata,
	normalization_metadata_from_loader,
)
from src.training.train import (
	_coerce_loss_result,
	_denormalize_target_tensors_for_metrics,
	_ensure_config_path,
	_get_device,
	_infer_input_channels_from_loader,
	_resolve_training_paths,
)


def _select_loader(train_loader, val_loader, test_loader, split: str):
	split_name = str(split).lower()
	if split_name == "train":
		return train_loader
	if split_name == "val":
		return val_loader
	if split_name == "test":
		if test_loader is None:
			raise ValueError("No test loader is configured for split='test'.")
		return test_loader
	raise ValueError(f"split must be one of train, val, test. Got {split!r}.")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
	"""Recursively merge mapping overrides without dropping sibling config keys."""

	merged = dict(base)
	for key, value in override.items():
		current_value = merged.get(key)
		if isinstance(current_value, Mapping) and isinstance(value, Mapping):
			merged[key] = _deep_merge(current_value, value)
		else:
			merged[key] = value
	return merged


def _validate_checkpoint_architecture(checkpoint: Mapping[str, Any], expected_architecture: str | None, checkpoint_path: Path) -> None:
	"""Raise if checkpoint metadata names a different architecture than requested."""

	if expected_architecture in (None, ""):
		return
	checkpoint_architecture = checkpoint.get("architecture")
	if checkpoint_architecture in (None, ""):
		warnings.warn(
			f"Checkpoint {checkpoint_path} has no architecture metadata; continuing with requested architecture {expected_architecture!r}.",
			RuntimeWarning,
			stacklevel=2,
		)
		return
	if str(checkpoint_architecture).lower() != str(expected_architecture).lower():
		raise ValueError(
			"Checkpoint architecture mismatch: "
			f"checkpoint={checkpoint_architecture!r}, expected={expected_architecture!r}, path={checkpoint_path}"
		)


def _expected_sequence_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	offsets = temporal_target_offsets(config)
	return {
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_offset_from_start": int(offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
	}


def _validate_checkpoint_sequence(
	checkpoint: Mapping[str, Any],
	config: Mapping[str, Any],
	checkpoint_path: Path,
	allow_sequence_mismatch: bool = False,
) -> None:
	"""Raise if checkpoint temporal-target metadata does not match the evaluation config."""

	expected = _expected_sequence_metadata(config)
	mismatches: list[str] = []
	for key in ("input_sequence_length", "prediction_horizon", "target_offset_from_start", "target_offset_from_last_input"):
		if key not in checkpoint:
			mismatches.append(f"{key}: checkpoint=<missing>, expected={expected[key]!r}")
			continue
		try:
			checkpoint_value = int(checkpoint[key])
		except (TypeError, ValueError):
			mismatches.append(f"{key}: checkpoint={checkpoint.get(key)!r}, expected={expected[key]!r}")
			continue
		if checkpoint_value != int(expected[key]):
			mismatches.append(f"{key}: checkpoint={checkpoint_value!r}, expected={expected[key]!r}")
	target_key = "target_definition_version"
	checkpoint_target_definition = checkpoint.get(target_key)
	if checkpoint_target_definition in (None, ""):
		mismatches.append(f"{target_key}: checkpoint=<missing>, expected={expected[target_key]!r}")
	elif str(checkpoint_target_definition) != str(expected[target_key]):
		mismatches.append(f"{target_key}: checkpoint={checkpoint_target_definition!r}, expected={expected[target_key]!r}")

	if not mismatches:
		return
	message = (
		f"Checkpoint sequence metadata mismatch for {checkpoint_path}:\n"
		+ "\n".join(f"  - {item}" for item in mismatches)
	)
	if allow_sequence_mismatch:
		warnings.warn(message, RuntimeWarning, stacklevel=2)
		return
	raise ValueError(message + "\nPass --allow_sequence_mismatch only for intentional compatibility/debug runs.")


def evaluate_checkpoint_on_split(
	config_path: str | Path,
	split: str = "test",
	checkpoint_path: str | Path | None = None,
	checkpoint_kind: str = "best",
	config_override: Mapping[str, Any] | None = None,
	max_batches: int | None = None,
	expected_architecture: str | None = None,
	allow_sequence_mismatch: bool = False,
	allow_normalization_mismatch: bool = False,
) -> dict[str, Any]:
	"""Evaluate one checkpoint on the requested split using existing metrics/losses."""

	if torch is None:
		raise ImportError("PyTorch is required to evaluate wildfire models.")

	config = _ensure_config_path(load_config(config_path), config_path)
	if isinstance(config_override, Mapping):
		config = _deep_merge(config, dict(config_override))
	config["return_metadata"] = True
	if max_batches is not None and int(max_batches) <= 0:
		raise ValueError("max_batches must be positive when provided.")

	train_loader, val_loader, test_loader = create_dataloaders(config)
	selected_loader = _select_loader(train_loader, val_loader, test_loader, split)
	if len(selected_loader.dataset) == 0:
		raise ValueError(f"Selected split {split!r} is empty.")

	input_channels = _infer_input_channels_from_loader(train_loader)
	device = _get_device(config)
	model = build_model_from_config(config, input_channels=input_channels).to(device)
	criterion = get_loss_function(config)
	input_normalizer = build_input_normalizer_for_loader(selected_loader, device, input_channels, config)
	normalization_metadata = normalization_metadata_from_loader(selected_loader, config, input_channels)

	if checkpoint_path is None:
		latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
		resolved_checkpoint_path = best_checkpoint_path if str(checkpoint_kind).lower() == "best" else latest_checkpoint_path
	else:
		resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
	if not resolved_checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

	checkpoint = load_checkpoint(resolved_checkpoint_path, map_location=device)
	_validate_checkpoint_architecture(checkpoint, expected_architecture, resolved_checkpoint_path)
	_validate_checkpoint_sequence(checkpoint, config, resolved_checkpoint_path, allow_sequence_mismatch=allow_sequence_mismatch)
	normalization_mismatches = compare_normalization_metadata(checkpoint.get("normalization"), normalization_metadata)
	if normalization_mismatches:
		message = (
			f"Checkpoint normalization metadata mismatch for {resolved_checkpoint_path}:\n"
			+ "\n".join(f"  - {item}" for item in normalization_mismatches)
		)
		if allow_normalization_mismatch:
			warnings.warn(message, RuntimeWarning, stacklevel=2)
		else:
			raise ValueError(message + "\nPass --allow_normalization_mismatch only for intentional compatibility/debug runs.")
	validate_checkpoint_model_compatibility(model, checkpoint, resolved_checkpoint_path)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	amp_dtype = choose_amp_dtype(config, device)

	aggregate_loss_total = 0.0
	aggregate_metric_totals: dict[str, float] = defaultdict(float)
	aggregate_metric_counts: dict[str, int] = defaultdict(int)
	aggregate_loss_component_totals: dict[str, float] = defaultdict(float)
	aggregate_samples = 0
	per_dataset: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
	per_dataset_metric_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

	with torch.inference_mode():
		for batch_index, batch in enumerate(selected_loader):
			if max_batches is not None and batch_index >= int(max_batches):
				break
			if not isinstance(batch, (tuple, list)) or len(batch) < 2:
				raise TypeError("Expected DataLoader batches with at least input and target tensors.")
			x_batch = batch[0].to(device, non_blocking=True)
			y_batch = batch[1].to(device, non_blocking=True)
			x_batch = apply_input_normalization(x_batch, input_normalizer, config)
			metadata_items = metadata_batch_to_list(batch[2], batch_size=int(x_batch.shape[0])) if len(batch) >= 3 else [{} for _ in range(int(x_batch.shape[0]))]
			with autocast_context(device, amp_dtype):
				y_pred = model(x_batch)
			if tuple(y_pred.shape) != tuple(y_batch.shape):
				raise ValueError(
					f"Prediction shape {tuple(y_pred.shape)} does not match target shape {tuple(y_batch.shape)}."
				)
			for sample_index, metadata in enumerate(metadata_items):
				dataset_name = str(metadata.get("dataset_name", f"dataset_{metadata.get('dataset_id', 'unknown')}"))
				sample_pred = y_pred[sample_index : sample_index + 1]
				sample_true = y_batch[sample_index : sample_index + 1]
				with autocast_context(device, amp_dtype):
					loss_result = criterion(sample_pred, sample_true)
				loss, loss_components = _coerce_loss_result(loss_result)
				metric_prediction, metric_target = _denormalize_target_tensors_for_metrics(
					selected_loader,
					sample_pred.detach(),
					sample_true.detach(),
				)
				sample_metrics = compute_metrics(metric_prediction, metric_target, config)

				aggregate_samples += 1
				aggregate_loss_total += float(loss.detach().item())
				per_dataset[dataset_name]["num_samples"] += 1.0
				per_dataset[dataset_name]["test_loss"] += float(loss.detach().item())
				for component_name, component_value in loss_components.items():
					aggregate_loss_component_totals[component_name] += float(component_value)
					per_dataset[dataset_name][f"test_{component_name}"] += float(component_value)
				for metric_name, metric_value in sample_metrics.items():
					metric_float = float(metric_value)
					if not math.isfinite(metric_float):
						continue
					metric_key = f"test_{metric_name}"
					aggregate_metric_totals[metric_name] += metric_float
					aggregate_metric_counts[metric_name] += 1
					per_dataset[dataset_name][metric_key] += metric_float
					per_dataset_metric_counts[dataset_name][metric_key] += 1

	if aggregate_samples == 0:
		raise ValueError(f"Selected split {split!r} produced no evaluation samples.")

	aggregate_results = {"test_loss": aggregate_loss_total / aggregate_samples}
	for component_name, total_value in aggregate_loss_component_totals.items():
		aggregate_results[f"test_{component_name}"] = total_value / aggregate_samples
	for metric_name, total_value in aggregate_metric_totals.items():
		metric_count = aggregate_metric_counts.get(metric_name, 0)
		if metric_count > 0:
			aggregate_results[f"test_{metric_name}"] = total_value / metric_count

	per_dataset_results: dict[str, dict[str, float]] = {}
	for dataset_name, totals in per_dataset.items():
		num_samples = max(int(totals.get("num_samples", 0.0)), 1)
		metric_counts = per_dataset_metric_counts.get(dataset_name, {})
		row: dict[str, float] = {"num_samples": float(num_samples)}
		for key, total_value in totals.items():
			if key == "num_samples":
				continue
			denominator = metric_counts.get(key, num_samples)
			row[key] = float(total_value) / max(int(denominator), 1)
		per_dataset_results[dataset_name] = row

	return {
		"checkpoint_path": str(resolved_checkpoint_path),
		"split": str(split).lower(),
		"num_samples": int(aggregate_samples),
		"sequence": _expected_sequence_metadata(config),
		"normalization": normalization_metadata,
		"aggregate_results": aggregate_results,
		"per_dataset_results": per_dataset_results,
	}
