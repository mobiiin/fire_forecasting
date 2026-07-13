"""Full training loop for wildfire forecasting models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

try:
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment-specific fallback
	matplotlib = None
	plt = None

try:
	from tqdm.auto import tqdm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	tqdm = None

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = None
	CudaGradScaler = None
	cuda_autocast = None

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.architecture_registry import resolve_model_architecture
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import (
	latest_and_best_checkpoint_paths,
	load_checkpoint,
	save_checkpoint,
	validate_checkpoint_model_compatibility,
)
from src.training.cuda_prefetcher import CUDAPrefetcher
from src.training.hardware import (
	autocast_context,
	cap_num_workers_by_slurm,
	choose_amp_dtype,
	configure_torch_backend,
	estimate_available_vram_gb,
	find_max_batch_size,
	get_cuda_device_info,
	get_performance_config,
)
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

try:
	import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency is already required by config.py
	yaml = None


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


def _tensor_on_device(tensor: torch.Tensor, device: torch.device) -> bool:
	tensor_device = tensor.device
	if tensor_device.type != device.type:
		return False
	if device.index is None:
		return True
	return tensor_device.index == device.index


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

	dataset = getattr(train_loader, "dataset", None)
	for attribute_name in ("total_input_channels", "input_channels_after_engineering", "num_channels"):
		dataset_channels = getattr(dataset, attribute_name, None)
		if dataset_channels is not None:
			return int(dataset_channels)

	try:
		first_batch = next(iter(train_loader))
	except StopIteration as exc:
		raise ValueError("Training DataLoader is empty; cannot infer input channels.") from exc

	x_batch, y_batch = _as_batch(first_batch)
	if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
		raise TypeError("Expected tensor batches from the training DataLoader.")

	if x_batch.ndim != 5:
		raise ValueError(f"Expected x batch to have shape (B, T, C, H, W), got {tuple(x_batch.shape)}.")
	if y_batch.ndim != 4:
		raise ValueError(f"Expected y batch to have shape (B, C, H, W), got {tuple(y_batch.shape)}.")

	return int(x_batch.shape[2])


def _maybe_autocast(device: torch.device, amp_dtype):
	"""Return an autocast context manager for the selected AMP dtype."""

	if amp_dtype is None:
		return nullcontext()
	return autocast_context(device, amp_dtype)


def _make_grad_scaler(enabled: bool):
	"""Create a CUDA AMP GradScaler while supporting old and new PyTorch APIs."""

	if not enabled or torch is None:
		return None
	if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
		try:
			return torch.amp.GradScaler("cuda", enabled=enabled)
		except TypeError:
			return torch.amp.GradScaler(enabled=enabled)
	if CudaGradScaler is not None:
		return CudaGradScaler(enabled=enabled)
	return None


def _auto_hardware_tuning_enabled(training_config: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
	auto_config = training_config.get("auto_hardware_tuning", {})
	if isinstance(auto_config, Mapping):
		return bool(auto_config.get("enabled", False)), auto_config
	return bool(auto_config), {}


def _architecture_auto_tuning_config(config: Mapping[str, Any], auto_config: Mapping[str, Any]) -> Mapping[str, Any]:
	architecture = resolve_model_architecture(config)
	architectures = auto_config.get("architectures", {})
	if isinstance(architectures, Mapping):
		architecture_config = architectures.get(architecture)
		if isinstance(architecture_config, Mapping):
			merged = dict(auto_config)
			merged.update(dict(architecture_config))
			return merged
	return auto_config


def _slurm_memory_limit_bytes() -> int | None:
	for variable_name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
		value = os.environ.get(variable_name)
		if value not in (None, ""):
			try:
				return int(value) * 1024 * 1024
			except ValueError:
				return None
	return None


def _slurm_cpus_per_task() -> int | None:
	value = os.environ.get("SLURM_CPUS_PER_TASK")
	if value in (None, ""):
		return None
	try:
		cpus = int(value)
	except ValueError:
		return None
	return cpus if cpus > 0 else None


def _apply_dataloader_worker_tuning(config: dict[str, Any], logger) -> None:
	"""Avoid oversubscribing CPU workers inside small Slurm allocations."""

	training_config = _get_section(config, "training")
	if not bool(training_config.get("auto_cap_num_workers_to_slurm_cpus", True)):
		return
	allocated_cpus = _slurm_cpus_per_task()
	if allocated_cpus is None:
		return
	configured_workers = training_config.get("num_workers", config.get("num_workers", 0))
	capped_workers = cap_num_workers_by_slurm(config, configured_workers)
	if str(configured_workers).lower() == "auto" or int(configured_workers) != capped_workers:
		training_config["num_workers"] = capped_workers
		config["num_workers"] = capped_workers
		logger.info(
			"Resolved DataLoader num_workers from %s to %s based on Slurm CPU allocation=%s.",
			configured_workers,
			capped_workers,
			allocated_cpus,
		)
	data_loader_config = _get_section(config, "data_loader")
	for split_name in ("train", "val", "test"):
		split_config = data_loader_config.get(split_name)
		if not isinstance(split_config, dict) or "num_workers" not in split_config:
			continue
		raw_workers = split_config["num_workers"]
		capped_split_workers = cap_num_workers_by_slurm(config, raw_workers)
		if str(raw_workers).lower() == "auto" or int(raw_workers) != capped_split_workers:
			split_config["num_workers"] = capped_split_workers
			logger.info(
				"Resolved data_loader.%s.num_workers from %s to %s based on Slurm CPU allocation=%s.",
				split_name,
				raw_workers,
				capped_split_workers,
				allocated_cpus,
			)


def _estimated_sample_bytes(config: Mapping[str, Any]) -> int:
	model_config = _get_section(config, "model")
	patching_config = _get_section(config, "patching")
	patch_size = int(config.get("patch_size", patching_config.get("patch_size", 64)))
	patch_height = int(patching_config.get("patch_height", patch_size))
	patch_width = int(patching_config.get("patch_width", patch_size))
	time_steps = int(config.get("input_sequence_length", 1))
	input_channels = int(model_config.get("input_channels", config.get("input_channel_count", 1)))
	output_channels = int(model_config.get("output_channels", 1))
	float32_bytes = 4
	input_bytes = time_steps * input_channels * patch_height * patch_width * float32_bytes
	target_bytes = output_channels * patch_height * patch_width * float32_bytes
	return max(1, input_bytes + target_bytes)


def _cap_batch_size_for_host_memory(
	config: Mapping[str, Any],
	batch_size: int,
	auto_config: Mapping[str, Any],
	logger,
) -> int:
	memory_limit = _slurm_memory_limit_bytes()
	if memory_limit is None:
		return batch_size
	training_config = _get_section(config, "training")
	num_workers = max(0, int(training_config.get("num_workers", config.get("num_workers", 0))))
	prefetch_factor = max(1, int(training_config.get("prefetch_factor", 2))) if num_workers > 0 else 1
	pin_memory = bool(training_config.get("pin_memory", torch.cuda.is_available() if torch is not None else False))
	resident_batches = max(1, num_workers * prefetch_factor + 1 + int(pin_memory))
	max_fraction = float(auto_config.get("max_host_memory_fraction", 0.65))
	sample_multiplier = max(
		1.0,
		float(auto_config.get("host_memory_sample_multiplier", training_config.get("host_memory_sample_multiplier", 3.0))),
	)
	sample_bytes = _estimated_sample_bytes(config)
	host_cap = int((memory_limit * max_fraction) / float(sample_bytes * resident_batches * sample_multiplier))
	host_cap = max(1, host_cap)
	if batch_size > host_cap and logger is not None:
		logger.info(
			"Auto hardware tuning capped batch_size from %s to %s based on SLURM memory, "
			"resident DataLoader batches=%s, and host_memory_sample_multiplier=%.2f.",
			batch_size,
			host_cap,
			resident_batches,
			sample_multiplier,
		)
	return min(batch_size, host_cap)


def _apply_auto_hardware_tuning(config: dict[str, Any], logger) -> None:
	"""Tune memory-sensitive training knobs from detected CUDA VRAM."""

	training_config = _get_section(config, "training")
	enabled, auto_config = _auto_hardware_tuning_enabled(training_config)
	if not enabled:
		return
	if torch is None or not torch.cuda.is_available():
		logger.info("Auto hardware tuning enabled, but CUDA is unavailable; keeping configured batch settings.")
		return

	device_index = torch.cuda.current_device()
	properties = torch.cuda.get_device_properties(device_index)
	total_vram_gb = float(properties.total_memory) / float(1024**3)
	architecture = resolve_model_architecture(config)
	auto_config = _architecture_auto_tuning_config(config, auto_config)
	if total_vram_gb >= 75.0:
		batch_size = int(auto_config.get("batch_size_80gb", 48))
	elif total_vram_gb >= 39.0:
		batch_size = int(auto_config.get("batch_size_40gb", 24))
	elif total_vram_gb >= 31.0:
		batch_size = int(auto_config.get("batch_size_32gb", 16))
	else:
		batch_size = int(auto_config.get("fallback_batch_size", config.get("batch_size", 8)))
	batch_size = max(1, batch_size)
	batch_size = _cap_batch_size_for_host_memory(config, batch_size, auto_config, logger)
	target_effective_batch = max(batch_size, int(auto_config.get("target_effective_batch_size", batch_size)))
	gradient_accumulation_steps = max(1, math.ceil(target_effective_batch / batch_size))

	config["batch_size"] = batch_size
	training_config["batch_size"] = batch_size
	training_config["gradient_accumulation_steps"] = gradient_accumulation_steps
	logger.info(
		"Auto hardware tuning | architecture=%s | gpu=%s | total_vram=%.1f GB | batch_size=%s | "
		"gradient_accumulation_steps=%s | effective_batch_size=%s",
		architecture,
		properties.name,
		total_vram_gb,
		batch_size,
		gradient_accumulation_steps,
		batch_size * gradient_accumulation_steps,
	)


def _denormalize_target_tensors_for_metrics(loader, y_pred: torch.Tensor, y_true: torch.Tensor):
	"""Convert normalized regression targets back to raw units for metric reporting."""

	dataset = getattr(loader, "dataset", None)
	normalize_target = bool(getattr(dataset, "normalize_target", False))
	target_mean = getattr(dataset, "target_mean", None)
	target_std = getattr(dataset, "target_std", None)
	task_type = str(getattr(dataset, "task_type", "regression")).lower()

	if not normalize_target or target_mean is None or target_std is None:
		return y_pred, y_true

	if task_type == "regression":
		mean_tensor = torch.as_tensor(float(target_mean), dtype=y_pred.dtype, device=y_pred.device)
		std_tensor = torch.as_tensor(max(float(target_std), 1e-6), dtype=y_pred.dtype, device=y_pred.device)
		return y_pred * std_tensor + mean_tensor, y_true * std_tensor + mean_tensor

	if task_type == "multitask":
		mean_array = torch.as_tensor(target_mean, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
		std_array = torch.as_tensor(target_std, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
		std_array = torch.clamp(std_array, min=1e-6)
		y_pred = y_pred.clone()
		y_true = y_true.clone()
		regression_channels = min(int(mean_array.shape[1]), 2)
		y_pred[:, :regression_channels] = y_pred[:, :regression_channels] * std_array[:, :regression_channels] + mean_array[:, :regression_channels]
		y_true[:, :regression_channels] = y_true[:, :regression_channels] * std_array[:, :regression_channels] + mean_array[:, :regression_channels]
		return y_pred, y_true

	return y_pred, y_true


def _coerce_loss_result(loss_result: Any) -> tuple[torch.Tensor, dict[str, float]]:
	"""Normalize a loss-module output into a scalar loss tensor plus loggable components."""

	if torch is None:
		raise ImportError("PyTorch is required to process training losses.")
	if torch.is_tensor(loss_result):
		return loss_result, {}
	if isinstance(loss_result, Mapping):
		if "total_loss" not in loss_result:
			raise KeyError("Loss mapping outputs must include a 'total_loss' tensor.")
		total_loss = loss_result["total_loss"]
		if not torch.is_tensor(total_loss):
			raise TypeError("loss_result['total_loss'] must be a tensor.")
		components: dict[str, float] = {}
		for key, value in loss_result.items():
			if key == "total_loss":
				continue
			if torch.is_tensor(value):
				components[str(key)] = float(value.detach().item())
			else:
				components[str(key)] = float(value)
		return total_loss, components
	raise TypeError(f"Unsupported loss result type: {type(loss_result)!r}.")


def _build_input_normalizer(loader, device: torch.device, input_channels: int):
	"""Build cached device tensors for per-channel input normalization."""

	dataset = getattr(loader, "dataset", None)
	if not bool(getattr(dataset, "input_normalization_on_device", False)):
		return None
	stats = getattr(dataset, "normalization_stats", None)
	if not isinstance(stats, Mapping):
		return None
	if "mean" not in stats or "std" not in stats:
		raise KeyError("Device-side input normalization requires normalization stats with mean and std.")

	mean_tensor = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device).flatten()
	std_tensor = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).flatten()
	if mean_tensor.numel() < input_channels or std_tensor.numel() < input_channels:
		raise ValueError(
			"Normalization stats channel count does not match model inputs. "
			f"Need at least {input_channels}, got mean={mean_tensor.numel()} std={std_tensor.numel()}."
		)
	if mean_tensor.numel() > input_channels:
		mean_tensor = mean_tensor[:input_channels]
	if std_tensor.numel() > input_channels:
		std_tensor = std_tensor[:input_channels]
	std_tensor = torch.clamp(std_tensor, min=1.0e-6)
	return {
		"mean": mean_tensor.reshape(1, 1, input_channels, 1, 1),
		"std": std_tensor.reshape(1, 1, input_channels, 1, 1),
	}


def _apply_input_normalizer(x_batch: torch.Tensor, normalizer) -> torch.Tensor:
	"""Normalize a batch in-place after it has been moved to the training device."""

	if normalizer is None:
		return x_batch
	with torch.no_grad():
		x_batch.sub_(normalizer["mean"])
		x_batch.div_(normalizer["std"])
	return x_batch


def _input_normalization_status(loader) -> str:
	"""Return a compact user/log facing normalization status for one loader."""

	dataset = getattr(loader, "dataset", None)
	if bool(getattr(dataset, "input_normalization_on_device", False)):
		return "device"
	if bool(getattr(dataset, "inputs_are_normalized", False)):
		return "dataset"
	if getattr(dataset, "normalization_stats", None) is None:
		return "none"
	return "unknown"


def _loader_summary(loader) -> dict[str, Any]:
	batch_sampler = getattr(loader, "batch_sampler", None)
	batch_size = getattr(loader, "batch_size", None)
	if batch_size is None and hasattr(batch_sampler, "batch_size"):
		batch_size = getattr(batch_sampler, "batch_size")
	return {
		"batch_size": batch_size,
		"num_workers": getattr(loader, "num_workers", None),
		"pin_memory": getattr(loader, "pin_memory", None),
		"persistent_workers": getattr(loader, "persistent_workers", None),
		"prefetch_factor": getattr(loader, "prefetch_factor", None),
		"sampler": type(getattr(loader, "sampler", None)).__name__,
		"batch_sampler": type(batch_sampler).__name__,
	}


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
	amp_dtype=None,
	gradient_accumulation_steps: int = 1,
	max_batches: int | None = None,
	logger=None,
	epoch_number: int | None = None,
	timing_csv_path: Path | None = None,
) -> dict[str, float]:
	"""Execute one train or validation epoch and return averaged losses/metrics."""

	desc = "train" if train else "val"
	model.train(mode=train)
	gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
	training_config = _get_section(config, "training")
	performance_config = get_performance_config(config)
	log_timing = bool(performance_config.get("log_timing", training_config.get("log_timing", False)))
	timing_interval = max(1, int(performance_config.get("timing_log_every_n_batches", training_config.get("timing_log_every_n_batches", 50))))
	synchronize_timing = bool(performance_config.get("synchronize_timing", False))
	compute_train_metrics_every_batch = bool(performance_config.get("compute_train_metrics_every_batch", training_config.get("compute_train_metrics_every_batch", False)))
	train_metrics_every_n_batches = max(
		1,
		int(performance_config.get("cheap_train_metrics_every_n_batches", training_config.get("train_metrics_every_n_batches", 100))),
	)
	compute_val_metrics = bool(performance_config.get("compute_val_metrics", training_config.get("compute_val_metrics", True)))
	non_blocking_transfer = bool(performance_config.get("non_blocking_transfer", True))
	use_cuda_prefetcher = bool(performance_config.get("prefetch_to_cuda", False)) and device.type == "cuda"

	total_samples = 0
	total_loss = 0.0
	metric_samples = 0
	metric_totals: dict[str, float] = defaultdict(float)
	loss_component_totals: dict[str, float] = defaultdict(float)
	timing_totals: dict[str, float] = defaultdict(float)
	timing_rows: list[dict[str, Any]] = []
	epoch_start_time = time.perf_counter()
	total_loader_batches = len(loader)
	total_batches = total_loader_batches if max_batches is None else min(total_loader_batches, int(max_batches))
	if total_batches <= 0:
		raise ValueError(f"The {desc} DataLoader produced no batches.")
	progress_bar = tqdm(range(total_batches), desc=desc, total=total_batches, leave=False) if tqdm is not None else range(total_batches)
	iterator_source = CUDAPrefetcher(loader, device, non_blocking=non_blocking_transfer) if use_cuda_prefetcher else loader
	iterator = iter(iterator_source)
	input_normalizer = _build_input_normalizer(loader, device, input_channels)

	def _sync_if_timing() -> None:
		if synchronize_timing and device.type == "cuda":
			torch.cuda.synchronize(device)

	for batch_offset in progress_bar:
		batch_number = int(batch_offset) + 1
		fetch_start_time = time.perf_counter()
		try:
			batch = next(iterator)
		except StopIteration:
			break
		fetch_end_time = time.perf_counter()
		data_wait_time = fetch_end_time - fetch_start_time
		x_batch, y_batch = _as_batch(batch)
		if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
			raise TypeError("Expected tensor batches from the DataLoader.")

		_assert_batch_shapes(x_batch, y_batch, input_sequence_length, input_channels, output_channels)
		h2d_start_time = time.perf_counter()
		if not _tensor_on_device(x_batch, device):
			x_batch = x_batch.to(device, non_blocking=non_blocking_transfer)
		if not _tensor_on_device(y_batch, device):
			y_batch = y_batch.to(device, non_blocking=non_blocking_transfer)
		_sync_if_timing()
		h2d_time = 0.0 if use_cuda_prefetcher else time.perf_counter() - h2d_start_time
		normalization_start_time = time.perf_counter()
		x_batch = _apply_input_normalizer(x_batch, input_normalizer)
		_sync_if_timing()
		normalization_time = time.perf_counter() - normalization_start_time

		if train and optimizer is None:
			raise ValueError("An optimizer is required for training epochs.")

		if train and (batch_number - 1) % gradient_accumulation_steps == 0:
			optimizer.zero_grad(set_to_none=True)

		forward_time = 0.0
		loss_time = 0.0
		backward_time = 0.0
		optimizer_time = 0.0
		metrics_time = 0.0
		batch_metrics: dict[str, float] = {}
		with torch.set_grad_enabled(train):
			with _maybe_autocast(device, amp_dtype):
				forward_start_time = time.perf_counter()
				y_pred = model(x_batch)
				_sync_if_timing()
				forward_time = time.perf_counter() - forward_start_time
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

				loss_start_time = time.perf_counter()
				loss_result = criterion(y_pred, y_batch)
				loss, batch_loss_components = _coerce_loss_result(loss_result)
				loss_for_backward = loss / float(gradient_accumulation_steps)
				_sync_if_timing()
				loss_time = time.perf_counter() - loss_start_time

			if train:
				should_step = (batch_number % gradient_accumulation_steps == 0) or (batch_number == total_batches)
				if scaler is not None:
					backward_start_time = time.perf_counter()
					scaler.scale(loss_for_backward).backward()
					if should_step and gradient_clip_norm is not None:
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
					_sync_if_timing()
					backward_time = time.perf_counter() - backward_start_time
					if should_step:
						optimizer_start_time = time.perf_counter()
						scaler.step(optimizer)
						scaler.update()
						_sync_if_timing()
						optimizer_time = time.perf_counter() - optimizer_start_time
				else:
					backward_start_time = time.perf_counter()
					loss_for_backward.backward()
					if should_step and gradient_clip_norm is not None:
						torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
					_sync_if_timing()
					backward_time = time.perf_counter() - backward_start_time
					if should_step:
						optimizer_start_time = time.perf_counter()
						optimizer.step()
						_sync_if_timing()
						optimizer_time = time.perf_counter() - optimizer_start_time

		batch_size = int(x_batch.shape[0])
		batch_loss_value = float(loss.detach().item())
		total_samples += batch_size
		total_loss += batch_loss_value * batch_size
		for component_name, component_value in batch_loss_components.items():
			loss_component_totals[component_name] += float(component_value) * batch_size

		should_compute_metrics = (
			(compute_train_metrics_every_batch or batch_number % train_metrics_every_n_batches == 0 or batch_number == total_batches)
			if train
			else compute_val_metrics
		)
		if should_compute_metrics:
			metrics_start_time = time.perf_counter()
			metric_prediction, metric_target = _denormalize_target_tensors_for_metrics(
				loader,
				y_pred.detach(),
				y_batch.detach(),
			)
			batch_metrics = compute_metrics(metric_prediction, metric_target, config)
			_sync_if_timing()
			metrics_time = time.perf_counter() - metrics_start_time
			metric_samples += batch_size
			for metric_name, metric_value in batch_metrics.items():
				metric_totals[metric_name] += float(metric_value) * batch_size

		batch_total_time = time.perf_counter() - fetch_start_time
		samples_per_second = batch_size / max(batch_total_time, 1.0e-9)
		gpu_mem_allocated_gb = 0.0
		gpu_mem_reserved_gb = 0.0
		if device.type == "cuda":
			gpu_mem_allocated_gb = float(torch.cuda.memory_allocated(device)) / float(1024**3)
			gpu_mem_reserved_gb = float(torch.cuda.memory_reserved(device)) / float(1024**3)
		timing_values = {
			"data_wait": data_wait_time,
			"h2d": h2d_time,
			"norm": normalization_time,
			"forward": forward_time,
			"loss": loss_time,
			"backward": backward_time,
			"optimizer": optimizer_time,
			"metrics": metrics_time,
			"total_step": batch_total_time,
		}
		for timing_name, timing_value in timing_values.items():
			timing_totals[timing_name] += float(timing_value)
		if timing_csv_path is not None:
			timing_rows.append(
				{
					"epoch": epoch_number if epoch_number is not None else "",
					"phase": desc,
					"batch": batch_number,
					"batch_size": batch_size,
					"samples_per_second": samples_per_second,
					"gpu_mem_allocated_gb": gpu_mem_allocated_gb,
					"gpu_mem_reserved_gb": gpu_mem_reserved_gb,
					**timing_values,
				}
			)
		if log_timing and (batch_number % timing_interval == 0 or batch_number == total_batches):
			message = (
				f"step {batch_number} | epoch={epoch_number if epoch_number is not None else '?'} {desc} "
				f"total={batch_total_time:.2f}s data={data_wait_time:.2f} h2d={h2d_time:.2f} "
				f"norm={normalization_time:.2f} fwd={forward_time:.2f} loss={loss_time:.2f} "
				f"bwd={backward_time:.2f} opt={optimizer_time:.2f} metrics={metrics_time:.2f} "
				f"mem={gpu_mem_allocated_gb:.1f}/{gpu_mem_reserved_gb:.1f}GB samples/s={samples_per_second:.2f}"
			)
			if logger is not None:
				logger.info(message)
			else:
				print(message)

		if tqdm is not None and hasattr(progress_bar, "set_postfix"):
			postfix = {"loss": f"{batch_loss_value:.5f}"}
			for component_name, component_value in batch_loss_components.items():
				postfix[component_name] = f"{float(component_value):.5f}"
			for metric_name, metric_value in batch_metrics.items():
				postfix[metric_name] = f"{float(metric_value):.5f}"
			progress_bar.set_postfix(postfix)

	if total_samples == 0:
		raise ValueError(f"The {desc} DataLoader produced no samples.")

	results = {f"{desc}_loss": total_loss / total_samples}
	for component_name, total_value in loss_component_totals.items():
		results[f"{desc}_{component_name}"] = total_value / total_samples
	for metric_name, total_value in metric_totals.items():
		results[f"{desc}_{metric_name}"] = total_value / max(metric_samples, 1)
	results[f"{desc}_samples"] = float(total_samples)
	results[f"{desc}_batches"] = float(batch_number)
	completed_batches = max(1, int(batch_number))
	epoch_wall_time = time.perf_counter() - epoch_start_time
	for timing_name, timing_total in timing_totals.items():
		results[f"{desc}_{timing_name}_avg"] = float(timing_total) / completed_batches
	results[f"{desc}_samples_per_second"] = float(total_samples) / max(epoch_wall_time, 1.0e-9)
	results[f"{desc}_patches_per_second"] = float(total_samples) / max(epoch_wall_time, 1.0e-9)
	if timing_csv_path is not None and timing_rows:
		_log_rows_to_csv(timing_csv_path, timing_rows, append=timing_csv_path.exists())
	avg_data_wait = results.get(f"{desc}_data_wait_avg", 0.0)
	avg_forward = results.get(f"{desc}_forward_avg", 0.0)
	avg_backward = results.get(f"{desc}_backward_avg", 0.0)
	if logger is not None:
		logger.info(
			"%s timing summary | data_wait=%.3fs h2d=%.3fs norm=%.3fs forward=%.3fs "
			"backward=%.3fs optimizer=%.3fs metrics=%.3fs samples/s=%.2f",
			desc,
			results.get(f"{desc}_data_wait_avg", 0.0),
			results.get(f"{desc}_h2d_avg", 0.0),
			results.get(f"{desc}_norm_avg", 0.0),
			avg_forward,
			avg_backward,
			results.get(f"{desc}_optimizer_avg", 0.0),
			results.get(f"{desc}_metrics_avg", 0.0),
			results[f"{desc}_samples_per_second"],
		)
		if avg_data_wait > (avg_forward + avg_backward):
			logger.info(
				"Data loading appears to be the bottleneck. Try increasing num_workers, "
				"shard_local_shuffle, or cache shard size."
			)
		if device.type == "cuda":
			device_info = get_cuda_device_info()
			total_memory_gb = float(device_info.get("total_memory_gb", 0.0) or 0.0)
			reserved_memory_gb = float(torch.cuda.memory_reserved(device)) / float(1024**3)
			if total_memory_gb > 0 and reserved_memory_gb < 0.5 * total_memory_gb and avg_data_wait <= (avg_forward + avg_backward):
				logger.info("GPU memory underused. Try increasing batch size or enabling auto_batch_size.")
	return results


def _run_external_test_epoch_with_spatial_handling(
	model: nn.Module,
	loader,
	criterion,
	config: Mapping[str, Any],
	device: torch.device,
) -> tuple[dict[str, float], dict[str, int]]:
	"""Evaluate an external test loader using direct inference or padded fallback."""

	model.eval()
	total_samples = 0
	total_loss = 0.0
	metric_totals: dict[str, float] = defaultdict(float)
	loss_component_totals: dict[str, float] = defaultdict(float)
	mode_counts: dict[str, int] = defaultdict(int)
	input_channels = int(getattr(getattr(loader, "dataset", None), "total_input_channels", 0))
	if input_channels <= 0:
		input_channels = int(_get_section(config, "model").get("input_channels", 0))
	input_normalizer = _build_input_normalizer(loader, device, input_channels) if input_channels > 0 else None

	for batch in loader:
		x_batch, y_batch = _as_batch(batch)
		if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
			raise TypeError("Expected tensor batches from the external test DataLoader.")

		x_batch = x_batch.to(device, non_blocking=True)
		y_batch = y_batch.to(device, non_blocking=True)
		x_batch = _apply_input_normalizer(x_batch, input_normalizer)
		with torch.no_grad():
			spatial_result = infer_with_external_test_spatial_handling(model, x_batch, config)
			y_pred = spatial_result["y_pred"]

		if tuple(y_pred.shape) != tuple(y_batch.shape):
			raise ValueError(
				"External test prediction shape does not match target shape after spatial handling. "
				f"Prediction={tuple(y_pred.shape)} target={tuple(y_batch.shape)} mode={spatial_result['mode_used']}."
			)

		loss_result = criterion(y_pred, y_batch)
		loss, batch_loss_components = _coerce_loss_result(loss_result)
		batch_size = int(x_batch.shape[0])
		total_samples += batch_size
		total_loss += float(loss.detach().item()) * batch_size
		for component_name, component_value in batch_loss_components.items():
			loss_component_totals[component_name] += float(component_value) * batch_size

		metric_prediction, metric_target = _denormalize_target_tensors_for_metrics(
			loader,
			y_pred.detach(),
			y_batch.detach(),
		)
		batch_metrics = compute_metrics(metric_prediction, metric_target, config)
		for metric_name, metric_value in batch_metrics.items():
			metric_totals[metric_name] += float(metric_value) * batch_size
		mode_counts[str(spatial_result["mode_used"])] += batch_size

	if total_samples == 0:
		raise ValueError("The external test DataLoader produced no samples.")

	results = {"test_loss": total_loss / total_samples}
	for component_name, total_value in loss_component_totals.items():
		results[f"test_{component_name}"] = total_value / total_samples
	for metric_name, total_value in metric_totals.items():
		results[f"test_{metric_name}"] = total_value / total_samples
	return results, dict(mode_counts)


def _resolve_training_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
	"""Resolve the latest and best checkpoint locations."""

	checkpoint_config = _get_section(config, "checkpoint")
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	resolved_latest = _resolve_path(config_path, checkpoint_path)
	best_override = checkpoint_config.get("best_path")
	if best_override:
		return resolved_latest, _resolve_path(config_path, best_override)
	return latest_and_best_checkpoint_paths(resolved_latest)


def _resolve_training_log_path(config: Mapping[str, Any]) -> Path:
	"""Resolve the CSV path used for epoch-level training logs."""

	logging_config = _get_section(config, "logging")
	configured = logging_config.get("training_log_path", "outputs/training_log.csv")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	return _resolve_path(config_path, configured)


def _run_name(config: Mapping[str, Any]) -> str:
	logging_config = _get_section(config, "logging")
	name = logging_config.get("run_name") or _get_section(config, "model").get("name") or resolve_model_architecture(config)
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "training_run"


def _resolve_timing_log_path(config: Mapping[str, Any]) -> Path:
	logging_config = _get_section(config, "logging")
	configured = logging_config.get("timing_log_path")
	if configured in (None, "", "null"):
		configured = f"./artifacts/logs/training_timing_{_run_name(config)}.csv"
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	return _resolve_path(config_path, configured)


def _positive_int_or_none(value: Any) -> int | None:
	if value in (None, "", "null", 0, 0.0):
		return None
	result = int(value)
	return result if result > 0 else None


def _resolve_max_batches(config: Mapping[str, Any], split: str) -> int | None:
	performance_config = get_performance_config(config)
	training_config = _get_section(config, "training")
	key = f"max_{split}_batches_per_epoch"
	return _positive_int_or_none(performance_config.get(key, training_config.get(key)))


def _model_parameter_count(model: nn.Module) -> int:
	return int(sum(parameter.numel() for parameter in model.parameters()))


def _current_git_commit() -> str | None:
	try:
		result = subprocess.run(
			["git", "rev-parse", "HEAD"],
			check=True,
			capture_output=True,
			text=True,
			cwd=Path(__file__).resolve().parents[2],
		)
	except Exception:
		return None
	return result.stdout.strip() or None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _save_resolved_run_artifacts(
	config: Mapping[str, Any],
	model: nn.Module,
	logger,
	normalization_stats_path: Path | None,
) -> dict[str, str]:
	"""Save reproducibility artifacts for the resolved training run."""

	run_name = _run_name(config)
	config_dir = Path("./artifacts/configs").expanduser().resolve()
	config_dir.mkdir(parents=True, exist_ok=True)
	resolved_config_path = config_dir / f"{run_name}_resolved.yaml"
	hardware_summary_path = config_dir / f"{run_name}_hardware.json"
	cache_manifest_copy_path = config_dir / f"{run_name}_cache_manifest.json"
	payload = dict(config)
	payload.setdefault("resolved_run", {})
	payload["resolved_run"] = {
		"run_name": run_name,
		"git_commit": _current_git_commit(),
		"normalization_stats_path": str(normalization_stats_path) if normalization_stats_path is not None else None,
		"model_parameter_count": _model_parameter_count(model),
		"cuda": get_cuda_device_info(),
	}
	if yaml is not None:
		with resolved_config_path.open("w", encoding="utf-8") as handle:
			yaml.safe_dump(payload, handle, sort_keys=False)
	else:
		_write_json(resolved_config_path.with_suffix(".json"), payload)
		resolved_config_path = resolved_config_path.with_suffix(".json")

	hardware_summary = {
		"cuda": get_cuda_device_info(),
		"available_vram_gb": estimate_available_vram_gb(),
		"backend": configure_torch_backend(config),
		"git_commit": payload["resolved_run"]["git_commit"],
		"model_parameter_count": payload["resolved_run"]["model_parameter_count"],
	}
	_write_json(hardware_summary_path, hardware_summary)

	cache_config = _get_section(config, "cache")
	if bool(cache_config.get("enabled", False)) and bool(cache_config.get("use_precomputed_patches", False)):
		try:
			from src.data.cache import MANIFEST_FILENAME, get_patch_cache_dir

			manifest_path = get_patch_cache_dir(config) / MANIFEST_FILENAME
			if manifest_path.exists():
				shutil.copyfile(manifest_path, cache_manifest_copy_path)
		except Exception as exc:
			logger.warning("Could not copy cache manifest for run metadata: %s", exc)

	logger.info("Saved resolved config: %s", resolved_config_path)
	logger.info("Saved hardware summary: %s", hardware_summary_path)
	return {
		"resolved_config_path": str(resolved_config_path),
		"hardware_summary_path": str(hardware_summary_path),
		"cache_manifest_copy_path": str(cache_manifest_copy_path) if cache_manifest_copy_path.exists() else "",
	}


def _resolve_existing_normalization_stats_path(config: Mapping[str, Any]) -> Path | None:
	"""Resolve normalization.path when the stats archive exists."""

	normalization_config = _get_section(config, "normalization")
	normalization_path = normalization_config.get("path")
	if not normalization_path:
		return None
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	resolved_path = _resolve_path(config_path, normalization_path)
	return resolved_path if resolved_path.exists() else None


def _build_optimizer_from_parameters(parameters, config: Mapping[str, Any]):
	"""Construct the configured optimizer for an iterable of parameters."""

	training_config = _get_section(config, "training")
	optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
	lr = float(training_config.get("learning_rate", config.get("learning_rate", 1e-4)))
	weight_decay = float(training_config.get("weight_decay", 0.0))

	if optimizer_name == "adam":
		return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
	if optimizer_name == "adamw":
		return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
	if optimizer_name == "sgd":
		momentum = float(training_config.get("momentum", 0.9))
		return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

	raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]):
	"""Construct the configured optimizer, defaulting to AdamW."""

	return _build_optimizer_from_parameters(model.parameters(), config)


def _maybe_probe_auto_batch_size(
	config: dict[str, Any],
	train_loader,
	input_channels: int,
	device: torch.device,
	logger,
) -> bool:
	"""Optionally run a CUDA memory probe and update batch-size config."""

	performance_config = get_performance_config(config)
	if not bool(performance_config.get("auto_batch_size", False)):
		return False
	if device.type != "cuda" or not torch.cuda.is_available():
		logger.info("auto_batch_size requested, but CUDA is unavailable; keeping configured batch size.")
		return False

	training_config = _get_section(config, "training")
	current_batch_size = int(training_config.get("batch_size", config.get("batch_size", 1)))
	max_batch_size = performance_config.get("auto_batch_max_batch_size")
	auto_config = _architecture_auto_tuning_config(config, _get_section(training_config, "auto_hardware_tuning"))
	if max_batch_size in (None, "", "null"):
		max_batch_size = max(current_batch_size, int(auto_config.get("target_effective_batch_size", current_batch_size)))
	original_host_safe_batch_size = training_config.get("_auto_batch_original_batch_size")
	if original_host_safe_batch_size not in (None, "", "null"):
		max_batch_size = min(int(max_batch_size), int(original_host_safe_batch_size))
	max_batch_size = _cap_batch_size_for_host_memory(config, int(max_batch_size), auto_config, logger)
	max_trials = int(performance_config.get("auto_batch_max_trials", 12))
	logger.info(
		"Running CUDA auto batch-size probe | initial=%s max=%s trials=%s",
		current_batch_size,
		max_batch_size,
		max_trials,
	)
	sample_batch = next(iter(train_loader))
	x_sample, y_sample = _as_batch(sample_batch)
	if not torch.is_tensor(x_sample) or not torch.is_tensor(y_sample):
		raise TypeError("Auto batch-size probing requires tensor batches.")

	probe_model = build_model_from_config(config, input_channels=input_channels).to(device)
	criterion = get_loss_function(config)
	amp_dtype = choose_amp_dtype(config, device)
	gradient_clip_norm_value = training_config.get("gradient_clip_norm", config.get("gradient_clip_norm", None))
	gradient_clip_norm = None if gradient_clip_norm_value in (None, "", 0, 0.0) else float(gradient_clip_norm_value)
	try:
		selected_batch_size = find_max_batch_size(
			model=probe_model,
			criterion=criterion,
			optimizer_factory=lambda parameters: _build_optimizer_from_parameters(parameters, config),
			sample_batch=(x_sample, y_sample),
			device=device,
			amp_dtype=amp_dtype,
			initial_batch_size=current_batch_size,
			max_batch_size=int(max_batch_size),
			gradient_accumulation_steps=max(1, int(training_config.get("gradient_accumulation_steps", 1))),
			gradient_clip_norm=gradient_clip_norm,
			logger=logger,
			max_trials=max_trials,
		)
	finally:
		del probe_model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	if selected_batch_size <= 0:
		return False
	host_safe_selected_batch_size = _cap_batch_size_for_host_memory(config, selected_batch_size, auto_config, logger)
	if host_safe_selected_batch_size < selected_batch_size:
		logger.info(
			"Auto batch-size probe capped selected batch_size from %s to %s for host/DataLoader memory.",
			selected_batch_size,
			host_safe_selected_batch_size,
		)
		selected_batch_size = host_safe_selected_batch_size
	original_batch_size = training_config.get("_auto_batch_original_batch_size")
	if selected_batch_size != current_batch_size or original_batch_size is not None:
		training_config["batch_size"] = selected_batch_size
		config["batch_size"] = selected_batch_size
		auto_config = _architecture_auto_tuning_config(config, _get_section(training_config, "auto_hardware_tuning"))
		target_effective_batch = max(
			selected_batch_size,
			int(performance_config.get("target_effective_batch_size", auto_config.get("target_effective_batch_size", selected_batch_size))),
		)
		training_config["gradient_accumulation_steps"] = max(1, math.ceil(target_effective_batch / selected_batch_size))
		logger.info(
			"Auto batch-size probe selected batch_size=%s gradient_accumulation_steps=%s effective_batch_size=%s",
			selected_batch_size,
			training_config["gradient_accumulation_steps"],
			selected_batch_size * int(training_config["gradient_accumulation_steps"]),
		)
		return selected_batch_size != current_batch_size
	logger.info("Auto batch-size probe kept configured batch_size=%s.", current_batch_size)
	return False


def train_model_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Train a model from an already-loaded configuration mapping."""

	if torch is None:
		raise ImportError("PyTorch is required to train the wildfire model.")

	config = dict(config)
	config["return_metadata"] = False
	training_config = _get_section(config, "training")
	performance_config = get_performance_config(config)
	logging_config = _get_section(config, "logging")
	checkpoint_config = _get_section(config, "checkpoint")
	checkpointing_config = _get_section(config, "checkpointing")
	architecture = resolve_model_architecture(config)

	seed = int(training_config.get("seed", config.get("seed", 42)))
	set_seed(seed)
	backend_summary = configure_torch_backend(config)

	log_level = str(logging_config.get("level", "INFO"))
	log_dir = Path(logging_config.get("log_dir", "./artifacts/logs")).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_level, str(log_dir / f"train_{architecture}.log"))
	logger.info("Torch backend: %s", backend_summary)
	_apply_dataloader_worker_tuning(config, logger)
	_apply_auto_hardware_tuning(config, logger)
	performance_config = get_performance_config(config)
	if bool(performance_config.get("auto_batch_size", False)) and torch.cuda.is_available():
		current_batch_size = int(training_config.get("batch_size", config.get("batch_size", 1)))
		probe_batch_size_value = performance_config.get("auto_batch_probe_batch_size", min(current_batch_size, 8))
		if probe_batch_size_value not in (None, "", "null"):
			probe_batch_size = max(1, int(probe_batch_size_value))
			if probe_batch_size < current_batch_size:
				training_config["_auto_batch_original_batch_size"] = current_batch_size
				training_config["batch_size"] = probe_batch_size
				config["batch_size"] = probe_batch_size
				logger.info(
					"auto_batch_size enabled; using initial probe DataLoader batch_size=%s before CUDA memory search.",
					probe_batch_size,
				)

	normalization_stats_path = _resolve_existing_normalization_stats_path(config)
	cache_config = _get_section(config, "cache")
	if bool(cache_config.get("enabled", False)) and bool(cache_config.get("use_precomputed_patches", False)):
		from src.data.cache import get_patch_cache_dir, validate_patch_cache

		try:
			cache_summary = validate_patch_cache(config, split=["train", "val"])
		except Exception as exc:
			if bool(cache_config.get("allow_dynamic_fallback", False)):
				logger.warning("Patch-cache validation failed; dynamic fallback is enabled: %s", exc)
			else:
				raise
		else:
			if not bool(cache_config.get("save_normalized_inputs", False)) and normalization_stats_path is None:
				raise RuntimeError(
					"Training is configured to load unnormalized precomputed patch shards, "
					"but normalization stats were not found. Run:\n"
					"python scripts/compute_normalization.py --config configs/default.yaml --from_cache"
				)
			logger.info(
				"Using precomputed patch cache: %s | train=%s | val=%s | normalization=%s",
				get_patch_cache_dir(config),
				cache_summary["splits"]["train"]["num_samples"],
				cache_summary["splits"]["val"]["num_samples"],
				normalization_stats_path,
			)
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

	if _maybe_probe_auto_batch_size(config, train_loader, input_channels, device, logger):
		logger.info("Rebuilding dataloaders with probed batch_size=%s", _get_section(config, "training").get("batch_size", config.get("batch_size")))
		train_loader, val_loader, test_loader = create_dataloaders(config)
		input_channels = _infer_input_channels_from_loader(train_loader)
		training_config = _get_section(config, "training")
		performance_config = get_performance_config(config)

	model = build_model_from_config(config, input_channels=input_channels)
	model = model.to(device)
	if bool(performance_config.get("torch_compile", training_config.get("torch_compile", False))):
		if not hasattr(torch, "compile"):
			logger.warning("torch_compile=true, but this PyTorch build does not provide torch.compile.")
		else:
			try:
				model = torch.compile(model)
				logger.info("Enabled torch.compile for the model.")
			except Exception as exc:  # pragma: no cover - backend-specific
				logger.warning("torch.compile failed; continuing without compilation: %s", exc)

	criterion = get_loss_function(config)
	optimizer = _build_optimizer(model, config)
	epochs = int(config.get("epochs", training_config.get("epochs", 1)))
	scheduler = _build_scheduler(optimizer, config, epochs)

	amp_dtype = choose_amp_dtype(config, device)
	scaler = _make_grad_scaler(amp_dtype is not None and amp_dtype is torch.float16)
	logger.info("Precision | amp_dtype=%s | grad_scaler=%s", amp_dtype, scaler is not None)
	gradient_clip_norm_value = training_config.get("gradient_clip_norm", config.get("gradient_clip_norm", None))
	gradient_clip_norm = None if gradient_clip_norm_value in (None, "", 0, 0.0) else float(gradient_clip_norm_value)
	gradient_accumulation_steps = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
	early_stopping_patience_value = training_config.get("early_stopping_patience", None)
	early_stopping_patience = None
	if early_stopping_patience_value not in (None, "", "null", 0, 0.0):
		early_stopping_patience = max(1, int(early_stopping_patience_value))

	latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
	resume_enabled = bool(checkpoint_config.get("resume", True))
	start_epoch = 0
	best_val_loss = math.inf
	resumed_from_checkpoint = False
	history_rows: list[dict[str, float | int]] = []

	if resume_enabled and latest_checkpoint_path.exists():
		logger.info("Resuming from checkpoint: %s", latest_checkpoint_path)
		checkpoint = load_checkpoint(latest_checkpoint_path, map_location="cpu")
		validate_checkpoint_model_compatibility(model, checkpoint, latest_checkpoint_path)
		model.load_state_dict(checkpoint["model_state_dict"])
		if checkpoint.get("optimizer_state_dict") is not None:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if checkpoint.get("scheduler_state_dict") is not None and scheduler is not None:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		start_epoch = int(checkpoint.get("epoch", -1)) + 1
		best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
		resumed_from_checkpoint = True
		history_rows = _coerce_history_rows(checkpoint.get("history", []))

	if start_epoch >= epochs:
		logger.info("Checkpoint already covers requested epochs (%s). Skipping training.", epochs)

	save_every_n_epochs = max(1, int(checkpointing_config.get("save_every_n_epochs", checkpoint_config.get("save_every_n_epochs", 1))))
	save_latest_checkpoint = bool(checkpointing_config.get("save_latest", True))
	save_best_checkpoint = bool(checkpointing_config.get("save_best", True))
	if bool(checkpointing_config.get("async_save", False)):
		logger.warning("checkpointing.async_save=true is not enabled in this safe path; saving checkpoints synchronously.")

	training_log_path = _resolve_training_log_path(config)
	training_log_path.parent.mkdir(parents=True, exist_ok=True)
	append_log = training_log_path.exists() and start_epoch > 0
	timing_log_path = _resolve_timing_log_path(config)
	timing_log_path.parent.mkdir(parents=True, exist_ok=True)
	run_artifact_paths = _save_resolved_run_artifacts(config, model, logger, normalization_stats_path)

	logger.info("Starting training for %s epochs", epochs)
	test_sample_count = 0 if test_loader is None else len(test_loader.dataset)
	logger.info("Train samples: %s | Val samples: %s | External test samples: %s", len(train_loader.dataset), len(val_loader.dataset), test_sample_count)
	logger.info("Train DataLoader: %s", _loader_summary(train_loader))
	logger.info("Val DataLoader: %s", _loader_summary(val_loader))
	if test_loader is not None:
		logger.info("Test DataLoader: %s", _loader_summary(test_loader))
	logger.info("Inferred input channels: %s", input_channels)
	logger.info("Model architecture: %s", architecture)
	logger.info("Model output channels: %s", output_channels)
	logger.info(
		"Input normalization | train=%s | val=%s | test=%s",
		_input_normalization_status(train_loader),
		_input_normalization_status(val_loader),
		"not_configured" if test_loader is None else _input_normalization_status(test_loader),
	)
	logger.info(
		"Patch mode | train=%s eval=%s patch_size=%s active_patch_probability=%s active_threshold=%s grad_accum=%s",
		bool(config.get("use_patches", False)),
		bool(config.get("use_patches_for_eval", False)),
		int(config.get("patch_size", 64)),
		float(config.get("active_patch_probability", 0.7)),
		float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		gradient_accumulation_steps,
	)

	final_epoch_summary: dict[str, Any] = dict(history_rows[-1]) if history_rows else {}
	epochs_without_validation_improvement = 0
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
			amp_dtype=amp_dtype,
			gradient_accumulation_steps=gradient_accumulation_steps,
			max_batches=_resolve_max_batches(config, "train"),
			logger=logger,
			epoch_number=epoch_number,
			timing_csv_path=timing_log_path,
		)
		max_val_batches = _resolve_max_batches(config, "val")
		full_validation_every_n_epochs = int(performance_config.get("full_validation_every_n_epochs", training_config.get("full_validation_every_n_epochs", 1)) or 0)
		run_full_validation = max_val_batches is None or (
			full_validation_every_n_epochs > 0 and epoch_number % full_validation_every_n_epochs == 0
		)
		val_max_batches = None if run_full_validation else max_val_batches
		if val_max_batches is not None:
			logger.info(
				"Validation capped at %s batch(es) for epoch %s; full validation every %s epoch(s).",
				val_max_batches,
				epoch_number,
				full_validation_every_n_epochs,
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
			max_batches=val_max_batches,
			logger=logger,
			epoch_number=epoch_number,
			amp_dtype=amp_dtype,
			timing_csv_path=timing_log_path,
		)

		val_loss = float(val_results["val_loss"])
		if math.isfinite(val_loss):
			if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				scheduler.step(val_loss)
			else:
				scheduler.step()
		else:
			logger.warning("Skipping scheduler step because validation loss is not finite: %s", val_loss)

		next_best_val_loss = min(best_val_loss, val_loss)
		row = {
			"epoch": epoch_number,
			"learning_rate": _current_lr(optimizer),
			"train_loss": train_results["train_loss"],
			"val_loss": val_results["val_loss"],
			"best_val_loss": next_best_val_loss,
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

		history_rows.append(row)
		is_best_epoch = val_loss < best_val_loss
		should_save_latest = save_latest_checkpoint and (
			epoch_number % save_every_n_epochs == 0 or epoch_number == epochs
		)
		if is_best_epoch and save_best_checkpoint:
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
				history=history_rows,
			)
			if not best_checkpoint_path.exists():
				raise RuntimeError(f"Best checkpoint was not created at: {best_checkpoint_path}")
		elif is_best_epoch:
			best_val_loss = val_loss

		if should_save_latest:
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
				history=history_rows,
			)
			if not latest_checkpoint_path.exists():
				raise RuntimeError(f"Latest checkpoint was not created at: {latest_checkpoint_path}")

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
		if early_stopping_patience is not None and math.isfinite(val_loss):
			if is_best_epoch:
				epochs_without_validation_improvement = 0
			else:
				epochs_without_validation_improvement += 1
			if epochs_without_validation_improvement >= early_stopping_patience:
				logger.info(
					"Early stopping after epoch %s: validation loss did not improve for %s epoch(s).",
					epoch_number,
					early_stopping_patience,
				)
				break

	test_results: dict[str, float] = {}
	test_plot_results: dict[str, float] = {}
	run_test_after_training = bool(training_config.get("run_test_after_training", config.get("run_test_after_training", False)))
	run_external_test_after_training = bool(training_config.get("run_external_test_after_training", config.get("run_external_test_after_training", False)))
	run_final_test_after_training = bool(run_test_after_training or run_external_test_after_training)
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	logger.info("Training complete. Best checkpoint selected using validation split.")
	if split_mode == "train_val_external_test":
		logger.info("Run scripts/test_model.py with test_data_dir configured for external testing.")
	else:
		logger.info("Run scripts/test_model.py to evaluate the combined internal test split.")
	if run_final_test_after_training and test_loader is not None and len(test_loader.dataset) > 0:
		logger.info("Loading best checkpoint for optional post-training test evaluation.")
		if best_checkpoint_path.exists():
			checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
			validate_checkpoint_model_compatibility(model, checkpoint, best_checkpoint_path)
			model.load_state_dict(checkpoint["model_state_dict"])
		if split_mode == "train_val_external_test":
			test_plot_results, spatial_mode_counts = _run_external_test_epoch_with_spatial_handling(
				model=model,
				loader=test_loader,
				criterion=criterion,
				config=config,
				device=device,
			)
		else:
			val_like_test_results = _run_epoch(
				model=model,
				loader=test_loader,
				criterion=criterion,
				config=config,
				device=device,
				input_sequence_length=input_sequence_length,
				input_channels=input_channels,
				output_channels=output_channels,
				train=False,
				amp_dtype=amp_dtype,
				timing_csv_path=timing_log_path,
			)
			test_plot_results = _rename_result_prefix(val_like_test_results, "val_", "test_")
			spatial_mode_counts = {}
		test_results = dict(test_plot_results)
		logger.info("Test loss: %.6f", test_plot_results["test_loss"])
		if spatial_mode_counts:
			logger.info("External test spatial mode counts: %s", spatial_mode_counts)
		for metric_name, metric_value in test_plot_results.items():
			if metric_name != "test_loss":
				logger.info("Test %s: %.6f", metric_name.removeprefix("test_"), metric_value)
	elif run_final_test_after_training:
		logger.info("Post-training test evaluation requested, but no test dataset is configured.")

	training_curve_paths: list[str] = []
	for checkpoint_path in (latest_checkpoint_path, best_checkpoint_path):
		figure_path = _save_training_curves_figure(checkpoint_path, history_rows, test_plot_results)
		if figure_path is not None and str(figure_path) not in training_curve_paths:
			training_curve_paths.append(str(figure_path))
			logger.info("Saved training curves: %s", figure_path)

	return {
		"start_epoch": start_epoch,
		"epochs": epochs,
		"best_val_loss": best_val_loss,
		"latest_checkpoint_path": str(latest_checkpoint_path),
		"best_checkpoint_path": str(best_checkpoint_path),
		"training_log_path": str(training_log_path),
		"timing_log_path": str(timing_log_path),
		"run_artifact_paths": run_artifact_paths,
		"training_curve_paths": training_curve_paths,
		"final_epoch_summary": final_epoch_summary,
		"history_rows": history_rows,
		"test_results": test_results,
	}


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


def _log_rows_to_csv(path: Path, rows: list[Mapping[str, Any]], append: bool) -> None:
	"""Append multiple rows to a CSV file."""

	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = list(rows[0].keys())
	mode = "a" if append else "w"
	with path.open(mode, newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		if not append:
			writer.writeheader()
		writer.writerows(rows)


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


def _coerce_history_rows(raw_history: Any) -> list[dict[str, float | int]]:
	"""Normalize checkpointed training history into a list of numeric row dictionaries."""

	if not isinstance(raw_history, list):
		return []

	history_rows: list[dict[str, float | int]] = []
	for raw_row in raw_history:
		if not isinstance(raw_row, dict):
			continue

		row: dict[str, float | int] = {}
		for key, value in raw_row.items():
			if isinstance(value, bool):
				row[str(key)] = int(value)
			elif isinstance(value, int):
				row[str(key)] = value
			elif isinstance(value, float):
				row[str(key)] = float(value)
		if row:
			history_rows.append(row)

	return history_rows


def _select_plot_metric_name(
	history_rows: list[dict[str, float | int]],
	test_results: Mapping[str, float],
) -> str | None:
	"""Choose the most useful non-loss metric to show beside the loss curves."""

	available_metrics: set[str] = set()
	for row in history_rows:
		for key in row:
			if key.startswith(("train_", "val_")) and not key.endswith("loss"):
				available_metrics.add(key.split("_", 1)[1])

	for key in test_results:
		if key.startswith("test_") and not key.endswith("loss"):
			available_metrics.add(key.split("_", 1)[1])

	preferred_metrics = (
		"accuracy",
		"mask_dice",
		"mask_iou",
		"surface_consumed_mae",
		"canopy_consumed_mae",
		"iou",
		"dice",
		"precision",
		"recall",
		"rmse",
		"mae",
		"active_region_mae",
		"active_mae",
	)
	for metric_name in preferred_metrics:
		if metric_name in available_metrics:
			return metric_name

	if not available_metrics:
		return None
	return sorted(available_metrics)[0]


def _save_training_curves_figure(
	checkpoint_path: Path,
	history_rows: list[dict[str, float | int]],
	test_results: Mapping[str, float],
) -> Path | None:
	"""Save a train/validation/test loss-and-metric figure beside a checkpoint."""

	if plt is None or not history_rows:
		return None

	epochs = [int(row["epoch"]) for row in history_rows if "epoch" in row]
	if not epochs:
		return None

	plot_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_training_curves.png")
	plot_path.parent.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, constrained_layout=True)

	loss_axis = axes[0]
	train_losses = [float(row["train_loss"]) for row in history_rows if "train_loss" in row]
	val_losses = [float(row["val_loss"]) for row in history_rows if "val_loss" in row]
	if train_losses:
		loss_axis.plot(epochs[: len(train_losses)], train_losses, color="tab:blue", linewidth=2.0, label="Train loss")
	if val_losses:
		loss_axis.plot(epochs[: len(val_losses)], val_losses, color="tab:orange", linewidth=2.0, label="Validation loss")
	if "test_loss" in test_results:
		loss_axis.axhline(
			float(test_results["test_loss"]),
			color="tab:green",
			linestyle="--",
			linewidth=1.8,
			label="Test loss",
		)
	loss_axis.set_title("Loss")
	loss_axis.set_xlabel("Epoch")
	loss_axis.set_ylabel("Loss")
	loss_axis.grid(True, alpha=0.3)
	loss_axis.legend()

	metric_axis = axes[1]
	metric_name = _select_plot_metric_name(history_rows, test_results)
	if metric_name is None:
		metric_axis.axis("off")
		metric_axis.text(
			0.5,
			0.5,
			"No accuracy or metric history available.",
			ha="center",
			va="center",
			fontsize=11,
		)
	else:
		train_metric_key = f"train_{metric_name}"
		val_metric_key = f"val_{metric_name}"
		test_metric_key = f"test_{metric_name}"

		train_metric = [float(row[train_metric_key]) for row in history_rows if train_metric_key in row]
		val_metric = [float(row[val_metric_key]) for row in history_rows if val_metric_key in row]
		if train_metric:
			metric_axis.plot(epochs[: len(train_metric)], train_metric, color="tab:blue", linewidth=2.0, label=f"Train {metric_name}")
		if val_metric:
			metric_axis.plot(epochs[: len(val_metric)], val_metric, color="tab:orange", linewidth=2.0, label=f"Validation {metric_name}")
		if test_metric_key in test_results:
			metric_axis.axhline(
				float(test_results[test_metric_key]),
				color="tab:green",
				linestyle="--",
				linewidth=1.8,
				label=f"Test {metric_name}",
			)
		if metric_name == "accuracy":
			metric_axis.set_ylim(0.0, 1.0)
		metric_axis.set_title("Accuracy" if metric_name == "accuracy" else metric_name.replace("_", " ").title())
		metric_axis.set_xlabel("Epoch")
		metric_axis.set_ylabel("Score")
		metric_axis.grid(True, alpha=0.3)
		metric_axis.legend()

	fig.suptitle("Training History", fontsize=14)
	fig.savefig(plot_path, bbox_inches="tight")
	plt.close(fig)
	return plot_path


def train_model(config_path: str | Path) -> dict[str, Any]:
	"""Train the forecasting model selected by the provided YAML config."""

	if torch is None:
		raise ImportError("PyTorch is required to train the wildfire forecasting model.")

	config = _ensure_config_path(load_config(config_path), config_path)
	return train_model_from_config(config)


def evaluate_model_on_test_set(
	config_path: str | Path,
	checkpoint_path: str | Path | None = None,
	checkpoint_kind: str = "best",
) -> dict[str, Any]:
	"""Load a trained checkpoint and evaluate it on the configured external test dataset."""

	if torch is None:
		raise ImportError("PyTorch is required to evaluate the ConvLSTM U-Net model.")

	config = _ensure_config_path(load_config(config_path), config_path)
	logging_config = _get_section(config, "logging")

	log_level = str(logging_config.get("level", "INFO"))
	log_dir = Path(logging_config.get("log_dir", "./artifacts/logs")).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_level, str(log_dir / "test_convlstm_unet.log"))

	train_loader, _, test_loader = create_dataloaders(config)
	if test_loader is None:
		raise ValueError(
			"No external test_data_dir configured. This project now uses data_dir only for train/val. "
			"Set test_data_dir in the config to evaluate on an external test dataset."
		)
	if len(test_loader.dataset) == 0:
		raise ValueError("External test dataset is empty; cannot evaluate the model.")

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
	validate_checkpoint_model_compatibility(model, checkpoint, resolved_checkpoint_path)
	model.load_state_dict(checkpoint["model_state_dict"])

	test_results, spatial_mode_counts = _run_external_test_epoch_with_spatial_handling(
		model=model,
		loader=test_loader,
		criterion=criterion,
		config=config,
		device=device,
	)

	logger.info("External test evaluation complete on %s samples.", len(test_loader.dataset))
	logger.info("External test spatial mode counts: %s", spatial_mode_counts)
	for metric_name, metric_value in test_results.items():
		logger.info("%s=%.6f", metric_name, metric_value)

	return {
		"checkpoint_path": str(resolved_checkpoint_path),
		"num_test_samples": len(test_loader.dataset),
		"test_results": test_results,
	}


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the CLI argument parser."""

	parser = argparse.ArgumentParser(description="Train the wildfire forecasting model selected by the config.")
	parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	"""CLI entry point for training."""

	args = build_argument_parser().parse_args()
	train_model(args.config)


if __name__ == "__main__":
	main()
