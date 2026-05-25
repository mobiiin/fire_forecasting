"""Full training loop for the ConvLSTM U-Net wildfire model."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

try:
	from tqdm.auto import tqdm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	tqdm = None

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	from torch.cuda.amp import GradScaler, autocast  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = None
	GradScaler = None
	autocast = None

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint, save_checkpoint
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


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


def _get_device(config: Mapping[str, Any]) -> torch.device:
	"""Resolve the configured training device."""

	training_config = _get_section(config, "training")
	device_setting = str(training_config.get("device", config.get("device", "auto"))).lower()
	if device_setting == "auto":
		device_setting = "cuda" if torch.cuda.is_available() else "cpu"
	if device_setting in {"gpu"}:
		device_setting = "cuda"
	if device_setting == "cuda" and not torch.cuda.is_available():
		device_setting = "cpu"
	return torch.device(device_setting)


def _as_batch(batch: Any):
	"""Extract the model input and target tensors from a DataLoader batch."""

	if not isinstance(batch, (tuple, list)) or len(batch) < 2:
		raise TypeError(
			"Batches must be tuples/lists containing at least input and target tensors."
		)
	return batch[0], batch[1]


def _assert_batch_shapes(
	x: torch.Tensor,
	y: torch.Tensor,
	input_sequence_length: int,
	input_channels: int,
	output_channels: int,
) -> None:
	"""Validate the expected sequence-to-map tensor layout early and loudly."""

	if x.ndim != 5:
		raise ValueError(f"Expected x to have shape (B, T, C, H, W), got {tuple(x.shape)}.")
	if y.ndim != 4:
		raise ValueError(f"Expected y to have shape (B, C, H, W), got {tuple(y.shape)}.")
	if x.shape[1] != input_sequence_length:
		raise ValueError(
			f"Expected input_sequence_length={input_sequence_length}, got batch with T={x.shape[1]}."
		)
	if x.shape[2] != input_channels:
		raise ValueError(
			f"Expected input_channels={input_channels}, got batch with C={x.shape[2]}."
		)
	if y.shape[1] != output_channels:
		raise ValueError(
			f"Expected output_channels={output_channels}, got target batch with C={y.shape[1]}."
		)
	if x.shape[0] != y.shape[0]:
		raise ValueError(f"Batch size mismatch between x and y: {x.shape[0]} vs {y.shape[0]}.")
	if x.shape[-2:] != y.shape[-2:]:
		raise ValueError(f"Spatial size mismatch between x and y: {tuple(x.shape[-2:])} vs {tuple(y.shape[-2:])}.")


def _infer_input_channels_from_loader(train_loader) -> int:
	"""Infer the channel count from one training batch, falling back to the dataset."""

	try:
		first_batch = next(iter(train_loader))
	except StopIteration as exc:
		dataset_channels = getattr(getattr(train_loader, "dataset", None), "num_channels", None)
		if dataset_channels is not None:
			return int(dataset_channels)
		raise ValueError("Training DataLoader is empty; cannot infer input channels.") from exc

	x_batch, y_batch = _as_batch(first_batch)
	if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
		dataset_channels = getattr(getattr(train_loader, "dataset", None), "num_channels", None)
		if dataset_channels is not None:
			return int(dataset_channels)
		raise TypeError("Expected tensor batches from the training DataLoader.")

	if x_batch.ndim != 5:
		raise ValueError(f"Expected x batch to have shape (B, T, C, H, W), got {tuple(x_batch.shape)}.")
	if y_batch.ndim != 4:
		raise ValueError(f"Expected y batch to have shape (B, C, H, W), got {tuple(y_batch.shape)}.")

	return int(x_batch.shape[2])


def _maybe_autocast(enabled: bool):
	"""Return an autocast context manager when mixed precision is enabled."""

	if enabled and autocast is not None:
		return autocast()
	return nullcontext()


def _run_epoch(
	model: nn.Module,
	loader,
	criterion,
	config: Mapping[str, Any],
	device: torch.device,
	input_sequence_length: int,
	input_channels: int,
	output_channels: int,
	train: bool,
	optimizer=None,
	scaler=None,
	gradient_clip_norm: float | None = None,
	mixed_precision: bool = False,
) -> dict[str, float]:
	"""Execute one train or validation epoch and return averaged losses/metrics."""

	desc = "train" if train else "val"
	model.train(mode=train)

	total_samples = 0
	total_loss = 0.0
	metric_totals: dict[str, float] = defaultdict(float)
	progress_bar = tqdm(loader, desc=desc, total=len(loader), leave=False) if tqdm is not None else loader

	for batch in progress_bar:
		x_batch, y_batch = _as_batch(batch)
		if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
			raise TypeError("Expected tensor batches from the DataLoader.")

		_assert_batch_shapes(x_batch, y_batch, input_sequence_length, input_channels, output_channels)
		x_batch = x_batch.to(device, non_blocking=True)
		y_batch = y_batch.to(device, non_blocking=True)

		if train and optimizer is None:
			raise ValueError("An optimizer is required for training epochs.")

		if train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(train):
			with _maybe_autocast(mixed_precision and train):
				y_pred = model(x_batch)
				if y_pred.ndim != 4:
					raise ValueError(f"Model output must have shape (B, C, H, W), got {tuple(y_pred.shape)}.")
				if y_pred.shape[0] != x_batch.shape[0]:
					raise ValueError("Model output batch size does not match the input batch size.")
				if y_pred.shape[-2:] != y_batch.shape[-2:]:
					raise ValueError(
						f"Model output spatial size {tuple(y_pred.shape[-2:])} does not match target size {tuple(y_batch.shape[-2:])}."
					)
				if y_pred.shape[1] != y_batch.shape[1]:
					raise ValueError(
						f"Model output channels {y_pred.shape[1]} do not match target channels {y_batch.shape[1]}."
					)

				loss = criterion(y_pred, y_batch)

			if train:
				if scaler is not None and mixed_precision:
					scaler.scale(loss).backward()
					if gradient_clip_norm is not None:
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
					scaler.step(optimizer)
					scaler.update()
				else:
					loss.backward()
					if gradient_clip_norm is not None:
						torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
					optimizer.step()

		batch_size = int(x_batch.shape[0])
		total_samples += batch_size
		total_loss += float(loss.detach().item()) * batch_size

		batch_metrics = compute_metrics(y_pred.detach(), y_batch.detach(), config)
		for metric_name, metric_value in batch_metrics.items():
			metric_totals[metric_name] += float(metric_value) * batch_size

		if tqdm is not None and hasattr(progress_bar, "set_postfix"):
			postfix = {"loss": f"{float(loss.detach().item()):.5f}"}
			for metric_name, metric_value in batch_metrics.items():
				postfix[metric_name] = f"{float(metric_value):.5f}"
			progress_bar.set_postfix(postfix)

	if total_samples == 0:
		raise ValueError(f"The {desc} DataLoader produced no samples.")

	results = {f"{desc}_loss": total_loss / total_samples}
	for metric_name, total_value in metric_totals.items():
		results[f"{desc}_{metric_name}"] = total_value / total_samples
	return results


def _resolve_training_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
	"""Resolve the latest and best checkpoint locations."""

	checkpoint_config = _get_section(config, "checkpoint")
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	resolved_latest = _resolve_path(config_path, checkpoint_path)
	return latest_and_best_checkpoint_paths(resolved_latest)


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]):
	"""Construct the configured optimizer, defaulting to AdamW."""

	training_config = _get_section(config, "training")
	optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
	lr = float(training_config.get("learning_rate", config.get("learning_rate", 1e-4)))
	weight_decay = float(training_config.get("weight_decay", 0.0))

	if optimizer_name == "adam":
		return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	if optimizer_name == "adamw":
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	if optimizer_name == "sgd":
		momentum = float(training_config.get("momentum", 0.9))
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

	raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(optimizer, config: Mapping[str, Any], epochs: int):
	"""Construct the configured learning-rate scheduler."""

	training_config = _get_section(config, "training")
	scheduler_name = str(training_config.get("scheduler", training_config.get("scheduler_type", "reduce_on_plateau"))).lower()

	if scheduler_name in {"reduce_on_plateau", "plateau"}:
		factor = float(training_config.get("scheduler_factor", 0.5))
		patience = int(training_config.get("scheduler_patience", 5))
		min_lr = float(training_config.get("scheduler_min_lr", 0.0))
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=factor,
			patience=patience,
			min_lr=min_lr,
		)

	if scheduler_name in {"cosine", "cosineannealinglr", "cosine_annealing", "cosine_annealing_lr"}:
		eta_min = float(training_config.get("scheduler_eta_min", 0.0))
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)

	raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _log_to_csv(path: Path, row: Mapping[str, Any], append: bool) -> None:
	"""Append a row to the training CSV, creating the header when needed."""

	path.parent.mkdir(parents=True, exist_ok=True)
	mode = "a" if append else "w"
	with path.open(mode, newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
		if not append:
			writer.writeheader()
		writer.writerow(row)


def _current_lr(optimizer) -> float:
	"""Read the first learning rate from the optimizer state."""

	for param_group in optimizer.param_groups:
		return float(param_group.get("lr", 0.0))
	return 0.0


def _rename_result_prefix(results: Mapping[str, float], source_prefix: str, target_prefix: str) -> dict[str, float]:
	"""Rename metric prefixes for reuse across validation and test evaluation."""

	renamed: dict[str, float] = {}
	for key, value in results.items():
		if key.startswith(source_prefix):
			renamed[f"{target_prefix}{key[len(source_prefix):]}"] = float(value)
		else:
			renamed[key] = float(value)
	return renamed


def train_model(config_path: str | Path) -> dict[str, Any]:
	"""Train the ConvLSTM U-Net according to the provided YAML config."""

	if torch is None:
		raise ImportError("PyTorch is required to train the ConvLSTM U-Net model.")

	config = _ensure_config_path(load_config(config_path), config_path)
	training_config = _get_section(config, "training")
	logging_config = _get_section(config, "logging")
	checkpoint_config = _get_section(config, "checkpoint")

	seed = int(training_config.get("seed", config.get("seed", 42)))
	set_seed(seed)

	log_level = str(logging_config.get("level", "INFO"))
	log_dir = Path(logging_config.get("log_dir", "./artifacts/logs")).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_level, str(log_dir / "train_convlstm_unet.log"))
	logger.info("Loading dataloaders")

	train_loader, val_loader, test_loader = create_dataloaders(config)
	input_sequence_length = int(config.get("input_sequence_length", training_config.get("input_sequence_length", 1)))
	output_channels = int(_get_section(config, "model").get("output_channels", 1))

	input_channels = _infer_input_channels_from_loader(train_loader)
	configured_input_channels = int(_get_section(config, "model").get("input_channels", input_channels))
	if configured_input_channels != input_channels:
		logger.warning(
			"Overriding configured input_channels=%s with inferred input_channels=%s.",
			configured_input_channels,
			input_channels,
		)

	device = _get_device(config)
	logger.info("Using device: %s", device)

	model = build_model_from_config(config, input_channels=input_channels)
	model = model.to(device)

	criterion = get_loss_function(config)
	optimizer = _build_optimizer(model, config)
	epochs = int(config.get("epochs", training_config.get("epochs", 1)))
	scheduler = _build_scheduler(optimizer, config, epochs)

	use_mixed_precision = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
	scaler = GradScaler(enabled=use_mixed_precision) if use_mixed_precision and GradScaler is not None else None
	gradient_clip_norm_value = training_config.get("gradient_clip_norm", config.get("gradient_clip_norm", None))
	gradient_clip_norm = None if gradient_clip_norm_value in (None, "", 0, 0.0) else float(gradient_clip_norm_value)

	latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
	resume_enabled = bool(checkpoint_config.get("resume", True))
	start_epoch = 0
	best_val_loss = math.inf
	resumed_from_checkpoint = False

	if resume_enabled and latest_checkpoint_path.exists():
		logger.info("Resuming from checkpoint: %s", latest_checkpoint_path)
		checkpoint = load_checkpoint(latest_checkpoint_path, map_location="cpu")
		model.load_state_dict(checkpoint["model_state_dict"])
		if checkpoint.get("optimizer_state_dict") is not None:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if checkpoint.get("scheduler_state_dict") is not None and scheduler is not None:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		start_epoch = int(checkpoint.get("epoch", -1)) + 1
		best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
		resumed_from_checkpoint = True

	if start_epoch >= epochs:
		logger.info("Checkpoint already covers requested epochs (%s). Skipping training.", epochs)
		return {
			"start_epoch": start_epoch,
			"epochs": epochs,
			"best_val_loss": best_val_loss,
			"latest_checkpoint_path": str(latest_checkpoint_path),
			"best_checkpoint_path": str(best_checkpoint_path),
			"resumed_from_checkpoint": resumed_from_checkpoint,
		}

	training_log_path = Path("outputs/training_log.csv").resolve()
	training_log_path.parent.mkdir(parents=True, exist_ok=True)
	append_log = training_log_path.exists() and start_epoch > 0

	logger.info("Starting training for %s epochs", epochs)
	logger.info("Train samples: %s | Val samples: %s | Test samples: %s", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
	logger.info("Inferred input channels: %s", input_channels)
	logger.info("Model output channels: %s", output_channels)
	logger.info(
		"Patch mode | train=%s eval=%s patch_size=%s active_patch_probability=%s active_threshold=%s",
		bool(config.get("use_patches", False)),
		bool(config.get("use_patches_for_eval", False)),
		int(config.get("patch_size", 64)),
		float(config.get("active_patch_probability", 0.7)),
		float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
	)

	final_epoch_summary: dict[str, Any] = {}
	for epoch_index in range(start_epoch, epochs):
		epoch_number = epoch_index + 1
		logger.info("Epoch %s/%s", epoch_number, epochs)

		train_results = _run_epoch(
			model=model,
			loader=train_loader,
			criterion=criterion,
			config=config,
			device=device,
			input_sequence_length=input_sequence_length,
			input_channels=input_channels,
			output_channels=output_channels,
			train=True,
			optimizer=optimizer,
			scaler=scaler,
			gradient_clip_norm=gradient_clip_norm,
			mixed_precision=use_mixed_precision,
		)
		val_results = _run_epoch(
			model=model,
			loader=val_loader,
			criterion=criterion,
			config=config,
			device=device,
			input_sequence_length=input_sequence_length,
			input_channels=input_channels,
			output_channels=output_channels,
			train=False,
		)

		val_loss = float(val_results["val_loss"])
		if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
			scheduler.step(val_loss)
		else:
			scheduler.step()

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_checkpoint(
				best_checkpoint_path,
				config=config,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch_index,
				best_val_loss=best_val_loss,
				input_channels=input_channels,
				resumed_from_checkpoint=resumed_from_checkpoint,
			)
			if not best_checkpoint_path.exists():
				raise RuntimeError(f"Best checkpoint was not created at: {best_checkpoint_path}")

		save_checkpoint(
			latest_checkpoint_path,
			config=config,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			epoch=epoch_index,
			best_val_loss=best_val_loss,
			input_channels=input_channels,
			resumed_from_checkpoint=resumed_from_checkpoint,
		)
		if not latest_checkpoint_path.exists():
			raise RuntimeError(f"Latest checkpoint was not created at: {latest_checkpoint_path}")

		row = {
			"epoch": epoch_number,
			"learning_rate": _current_lr(optimizer),
			"train_loss": train_results["train_loss"],
			"val_loss": val_results["val_loss"],
			"best_val_loss": best_val_loss,
			"use_patches_train": int(bool(config.get("use_patches", False))),
			"use_patches_eval": int(bool(config.get("use_patches_for_eval", False))),
			"patch_size": int(config.get("patch_size", 64)),
		}
		for metric_name, metric_value in train_results.items():
			if metric_name != "train_loss":
				row[metric_name] = metric_value
		for metric_name, metric_value in val_results.items():
			if metric_name != "val_loss":
				row[metric_name] = metric_value

		_log_to_csv(training_log_path, row, append=append_log)
		if not training_log_path.exists():
			raise RuntimeError(f"Training log CSV was not created at: {training_log_path}")
		append_log = True
		final_epoch_summary = row

		logger.info(
			"Epoch %s summary | train_loss=%.6f | val_loss=%.6f | best_val_loss=%.6f",
			epoch_number,
			train_results["train_loss"],
			val_results["val_loss"],
			best_val_loss,
		)

	logger.info("Training complete. Loading best checkpoint for final test evaluation.")
	if best_checkpoint_path.exists():
		checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])

	test_results: dict[str, float] = {}
	if len(test_loader.dataset) > 0:
		test_results = _run_epoch(
			model=model,
			loader=test_loader,
			criterion=criterion,
			config=config,
			device=device,
			input_sequence_length=input_sequence_length,
			input_channels=input_channels,
			output_channels=output_channels,
			train=False,
		)
		logger.info("Test loss: %.6f", test_results["val_loss"])
		for metric_name, metric_value in test_results.items():
			if metric_name != "val_loss":
				logger.info("Test %s: %.6f", metric_name.removeprefix("val_"), metric_value)
	else:
		logger.info("Test split is empty; skipping final evaluation.")

	return {
		"start_epoch": start_epoch,
		"epochs": epochs,
		"best_val_loss": best_val_loss,
		"latest_checkpoint_path": str(latest_checkpoint_path),
		"best_checkpoint_path": str(best_checkpoint_path),
		"training_log_path": str(training_log_path),
		"final_epoch_summary": final_epoch_summary,
		"test_results": test_results,
	}


def evaluate_model_on_test_set(
	config_path: str | Path,
	checkpoint_path: str | Path | None = None,
	checkpoint_kind: str = "best",
) -> dict[str, Any]:
	"""Load a trained checkpoint and evaluate it on the configured test split."""

	if torch is None:
		raise ImportError("PyTorch is required to evaluate the ConvLSTM U-Net model.")

	config = _ensure_config_path(load_config(config_path), config_path)
	logging_config = _get_section(config, "logging")

	log_level = str(logging_config.get("level", "INFO"))
	log_dir = Path(logging_config.get("log_dir", "./artifacts/logs")).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_level, str(log_dir / "test_convlstm_unet.log"))

	train_loader, _, test_loader = create_dataloaders(config)
	if len(test_loader.dataset) == 0:
		raise ValueError("Test split is empty; cannot evaluate the model.")

	input_sequence_length = int(config.get("input_sequence_length", 1))
	output_channels = int(_get_section(config, "model").get("output_channels", 1))
	input_channels = _infer_input_channels_from_loader(train_loader)
	device = _get_device(config)

	model = build_model_from_config(config, input_channels=input_channels).to(device)
	criterion = get_loss_function(config)

	if checkpoint_path is None:
		latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
		checkpoint_selector = str(checkpoint_kind).lower()
		if checkpoint_selector == "best":
			resolved_checkpoint_path = best_checkpoint_path
		elif checkpoint_selector == "latest":
			resolved_checkpoint_path = latest_checkpoint_path
		else:
			raise ValueError(f"checkpoint_kind must be 'best' or 'latest', got {checkpoint_kind!r}.")
	else:
		resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()

	if not resolved_checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

	logger.info("Loading checkpoint for test evaluation: %s", resolved_checkpoint_path)
	checkpoint = load_checkpoint(resolved_checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])

	raw_results = _run_epoch(
		model=model,
		loader=test_loader,
		criterion=criterion,
		config=config,
		device=device,
		input_sequence_length=input_sequence_length,
		input_channels=input_channels,
		output_channels=output_channels,
		train=False,
	)
	test_results = _rename_result_prefix(raw_results, "val_", "test_")

	logger.info("Test evaluation complete on %s samples.", len(test_loader.dataset))
	for metric_name, metric_value in test_results.items():
		logger.info("%s=%.6f", metric_name, metric_value)

	return {
		"checkpoint_path": str(resolved_checkpoint_path),
		"num_test_samples": len(test_loader.dataset),
		"test_results": test_results,
	}


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the CLI argument parser."""

	parser = argparse.ArgumentParser(description="Train the ConvLSTM U-Net wildfire model.")
	parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	"""CLI entry point for training."""

	args = build_argument_parser().parse_args()
	train_model(args.config)


if __name__ == "__main__":
	main()
