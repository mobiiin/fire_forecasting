"""Shared checkpoint evaluation helpers for wildfire models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
import warnings

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.data.dataset import create_dataloaders, metadata_batch_to_list
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, validate_checkpoint_model_compatibility
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.hardware import autocast_context, choose_amp_dtype
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


def evaluate_checkpoint_on_split(
	config_path: str | Path,
	split: str = "test",
	checkpoint_path: str | Path | None = None,
	checkpoint_kind: str = "best",
	config_override: Mapping[str, Any] | None = None,
	max_batches: int | None = None,
	expected_architecture: str | None = None,
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

	if checkpoint_path is None:
		latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
		resolved_checkpoint_path = best_checkpoint_path if str(checkpoint_kind).lower() == "best" else latest_checkpoint_path
	else:
		resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
	if not resolved_checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

	checkpoint = load_checkpoint(resolved_checkpoint_path, map_location=device)
	_validate_checkpoint_architecture(checkpoint, expected_architecture, resolved_checkpoint_path)
	validate_checkpoint_model_compatibility(model, checkpoint, resolved_checkpoint_path)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	amp_dtype = choose_amp_dtype(config, device)

	aggregate_loss_total = 0.0
	aggregate_metric_totals: dict[str, float] = defaultdict(float)
	aggregate_loss_component_totals: dict[str, float] = defaultdict(float)
	aggregate_samples = 0
	per_dataset: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

	with torch.inference_mode():
		for batch_index, batch in enumerate(selected_loader):
			if max_batches is not None and batch_index >= int(max_batches):
				break
			if not isinstance(batch, (tuple, list)) or len(batch) < 2:
				raise TypeError("Expected DataLoader batches with at least input and target tensors.")
			x_batch = batch[0].to(device, non_blocking=True)
			y_batch = batch[1].to(device, non_blocking=True)
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
					aggregate_metric_totals[metric_name] += float(metric_value)
					per_dataset[dataset_name][f"test_{metric_name}"] += float(metric_value)

	if aggregate_samples == 0:
		raise ValueError(f"Selected split {split!r} produced no evaluation samples.")

	aggregate_results = {"test_loss": aggregate_loss_total / aggregate_samples}
	for component_name, total_value in aggregate_loss_component_totals.items():
		aggregate_results[f"test_{component_name}"] = total_value / aggregate_samples
	for metric_name, total_value in aggregate_metric_totals.items():
		aggregate_results[f"test_{metric_name}"] = total_value / aggregate_samples

	per_dataset_results: dict[str, dict[str, float]] = {}
	for dataset_name, totals in per_dataset.items():
		num_samples = max(int(totals.get("num_samples", 0.0)), 1)
		row: dict[str, float] = {"num_samples": float(num_samples)}
		for key, total_value in totals.items():
			if key == "num_samples":
				continue
			row[key] = float(total_value) / num_samples
		per_dataset_results[dataset_name] = row

	return {
		"checkpoint_path": str(resolved_checkpoint_path),
		"split": str(split).lower(),
		"num_samples": int(aggregate_samples),
		"aggregate_results": aggregate_results,
		"per_dataset_results": per_dataset_results,
	}
