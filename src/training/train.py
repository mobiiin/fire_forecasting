"""Full training loop for wildfire forecasting models."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if "MPLCONFIGDIR" not in os.environ:
	_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_mplconfig"
	_mpl_config_dir.mkdir(parents=True, exist_ok=True)
	os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
	_xdg_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_xdg_cache"
	_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ["XDG_CACHE_HOME"] = str(_xdg_cache_dir)

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
	from torch.utils.data import DataLoader, Subset  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = None
	CudaGradScaler = None
	cuda_autocast = None

from src.config import compute_file_sha256, load_config
from src.data.cache import target_definition_version, temporal_target_offsets
from src.data.dataset import create_dataloaders
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.architecture_registry import resolve_model_architecture
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint, save_checkpoint, validate_checkpoint_model_compatibility
from src.training.cuda_prefetcher import CUDAPrefetcher
from src.training.early_stopping import build_early_stopping
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
from src.training.model_outputs import extract_prediction
from src.training.batch_utils import unpack_batch
from src.training.input_normalization import (
	apply_input_normalization,
	build_input_normalizer_for_loader,
	input_normalization_status,
	normalization_metadata_from_loader,
	resolve_input_normalization_stats_path,
)
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.run_manager import RunManager, get_training_checkpointing_config, get_training_output_config
from src.training.run_plots import save_training_run_figures
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


def _env_bool(name: str, default: bool) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "on"}:
		return True
	if normalized in {"0", "false", "no", "off", ""}:
		return False
	return default


def _env_int(name: str, default: int) -> int:
	value = os.environ.get(name)
	if value is None:
		return int(default)
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


def _env_float(name: str, default: float) -> float:
	value = os.environ.get(name)
	if value is None:
		return float(default)
	try:
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def _progress_log_interval(total_batches: int, percent_step: float) -> int | None:
	if percent_step <= 0.0:
		return None
	return max(1, int(math.ceil(float(total_batches) * min(percent_step, 100.0) / 100.0)))


def _format_duration(seconds: float) -> str:
	total_seconds = max(0, int(round(seconds)))
	hours, remainder = divmod(total_seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	if hours > 0:
		return f"{hours:d}:{minutes:02d}:{seconds:02d}"
	return f"{minutes:02d}:{seconds:02d}"


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


def apply_training_cli_overrides(
	config: Mapping[str, Any],
	run_name: str | None = None,
	output_root: str | Path | None = None,
	overwrite_run: bool = False,
	disable_early_stopping: bool = False,
	early_stopping_patience: int | None = None,
	early_stopping_monitor: str | None = None,
	early_stopping_min_delta: float | None = None,
) -> dict[str, Any]:
	"""Apply common training-output CLI overrides to a config mapping."""

	updated = dict(config)
	training_config = dict(updated.get("training", {})) if isinstance(updated.get("training"), Mapping) else {}
	if run_name not in (None, "", "null"):
		training_config["run_name"] = str(run_name)
	if output_root not in (None, "", "null"):
		output_config = dict(training_config.get("output", {})) if isinstance(training_config.get("output"), Mapping) else {}
		output_config["root_dir"] = str(output_root)
		training_config["output"] = output_config
	if overwrite_run:
		training_config["overwrite_run"] = True
	if disable_early_stopping or early_stopping_patience is not None or early_stopping_monitor not in (None, "", "null") or early_stopping_min_delta is not None:
		early_config = dict(training_config.get("early_stopping", {})) if isinstance(training_config.get("early_stopping"), Mapping) else {}
		if disable_early_stopping:
			early_config["enabled"] = False
		if early_stopping_patience is not None:
			early_config["patience"] = int(early_stopping_patience)
			early_config["enabled"] = True
		if early_stopping_monitor not in (None, "", "null"):
			early_config["monitor"] = str(early_stopping_monitor)
		if early_stopping_min_delta is not None:
			early_config["min_delta"] = float(early_stopping_min_delta)
		training_config["early_stopping"] = early_config
	updated["training"] = training_config
	return updated


def add_early_stopping_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	"""Add shared early-stopping override flags to a training CLI parser."""

	parser.add_argument("--disable_early_stopping", action="store_true", help="Disable training.early_stopping for this run.")
	parser.add_argument("--early_stopping_patience", type=int, default=None, help="Override training.early_stopping.patience.")
	parser.add_argument("--early_stopping_monitor", default=None, help="Override training.early_stopping.monitor.")
	parser.add_argument("--early_stopping_min_delta", type=float, default=None, help="Override training.early_stopping.min_delta.")
	return parser


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

	x_batch, y_batch, _batch_extra = unpack_batch(first_batch)
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


def _host_memory_loader_settings(config: Mapping[str, Any], split: str = "train") -> tuple[int, int, bool]:
	"""Resolve DataLoader settings that affect resident host-memory batches."""

	training_config = _get_section(config, "training")
	performance_config = get_performance_config(config)
	data_loader_config = _get_section(config, "data_loader")
	split_config = data_loader_config.get(split, {})
	if not isinstance(split_config, Mapping):
		split_config = {}
	raw_workers = split_config.get(
		"num_workers",
		data_loader_config.get("num_workers", training_config.get("num_workers", config.get("num_workers", 0))),
	)
	num_workers = cap_num_workers_by_slurm(config, raw_workers)
	pin_memory_default = bool(torch.cuda.is_available()) if torch is not None else False
	pin_memory = bool(
		split_config.get(
			"pin_memory",
			data_loader_config.get("pin_memory", training_config.get("pin_memory", pin_memory_default)),
		)
	)
	if not bool(performance_config.get("non_blocking_transfer", True)):
		pin_memory = False
	prefetch_factor = 1
	if num_workers > 0:
		raw_prefetch_factor = split_config.get(
			"prefetch_factor",
			data_loader_config.get("prefetch_factor", training_config.get("prefetch_factor", 2)),
		)
		prefetch_factor = 2 if raw_prefetch_factor in (None, "", "null") else max(1, int(raw_prefetch_factor))
	return max(0, int(num_workers)), int(prefetch_factor), bool(pin_memory)


def _cap_batch_size_for_host_memory(
	config: Mapping[str, Any],
	batch_size: int,
	auto_config: Mapping[str, Any],
	logger,
) -> int:
	memory_limit = _slurm_memory_limit_bytes()
	if memory_limit is None:
		return batch_size
	num_workers, prefetch_factor, pin_memory = _host_memory_loader_settings(config, split="train")
	resident_batches = max(1, num_workers * prefetch_factor + 1 + int(pin_memory))
	max_fraction = float(auto_config.get("max_host_memory_fraction", 0.65))
	sample_multiplier = max(
		1.0,
		float(auto_config.get("host_memory_sample_multiplier", _get_section(config, "training").get("host_memory_sample_multiplier", 3.0))),
	)
	sample_bytes = _estimated_sample_bytes(config)
	host_cap = int((memory_limit * max_fraction) / float(sample_bytes * resident_batches * sample_multiplier))
	host_cap = max(1, host_cap)
	if batch_size > host_cap and logger is not None:
		logger.info(
			"Auto hardware tuning capped batch_size from %s to %s based on SLURM memory, "
			"train_workers=%s, prefetch_factor=%s, pin_memory=%s, resident DataLoader batches=%s, "
			"and host_memory_sample_multiplier=%.2f.",
			batch_size,
			host_cap,
			num_workers,
			prefetch_factor,
			pin_memory,
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

	return build_input_normalizer_for_loader(loader, device, input_channels)


def _apply_input_normalizer(x_batch: torch.Tensor, normalizer) -> torch.Tensor:
	"""Normalize a batch in-place after it has been moved to the training device."""

	return apply_input_normalization(x_batch, normalizer)


def _input_normalization_status(loader) -> str:
	"""Return a compact user/log facing normalization status for one loader."""

	return input_normalization_status(loader)


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


SUPPORTED_VALIDATION_MODES = {"full_every_epoch", "fixed_subset_every_epoch", "random_subset_every_epoch"}


def _coerce_optional_positive_int(value: Any, *, name: str) -> int | None:
	if value in (None, "", "null", "None"):
		return None
	resolved = int(value)
	if resolved <= 0:
		raise ValueError(f"{name} must be null or a positive integer, got {value!r}.")
	return resolved


def _resolve_validation_seed(value: Any, *, default: int = 42, random_when_missing: bool = False) -> int:
	"""Resolve a validation-subset seed, optionally generating a fresh run seed."""

	if value in (None, "", "null", "None", "random", "Random", "RANDOM"):
		if random_when_missing:
			return int.from_bytes(os.urandom(8), "little") % (2**63 - 1)
		return int(default)
	return int(value)


def resolve_validation_policy(config: Mapping[str, Any], val_loader=None, logger=None) -> dict[str, Any]:
	"""Resolve the training validation protocol.

	Supported modes:
	- full_every_epoch: use the complete validation loader every epoch.
	- fixed_subset_every_epoch: choose one deterministic batch-index subset once.
	- random_subset_every_epoch: sample a fresh deterministic-random validation subset each epoch.
	"""

	training_config = _get_section(config, "training")
	validation_config = dict(training_config.get("validation", {})) if isinstance(training_config.get("validation"), Mapping) else {}
	performance_config = get_performance_config(config)

	deprecated_values = []
	if "full_validation_every_n_epochs" in config:
		deprecated_values.append("full_validation_every_n_epochs")
	if "full_validation_every_n_epochs" in training_config:
		deprecated_values.append("training.full_validation_every_n_epochs")
	if "full_validation_every_n_epochs" in performance_config:
		deprecated_values.append("training.performance.full_validation_every_n_epochs")
	if deprecated_values and logger is not None:
		logger.warning(
			"full_validation_every_n_epochs is deprecated and ignored. Use training.validation.mode instead."
		)

	mode = str(validation_config.get("mode", "fixed_subset_every_epoch")).strip().lower()
	if mode not in SUPPORTED_VALIDATION_MODES:
		raise ValueError(
			"Unsupported training.validation.mode. "
			"Expected one of: full_every_epoch, fixed_subset_every_epoch, random_subset_every_epoch. "
			f"Got {mode!r}."
		)

	if mode == "full_every_epoch":
		if validation_config.get("max_val_batches_per_epoch") not in (None, "", "null", "None") and logger is not None:
			logger.warning(
				"training.validation.max_val_batches_per_epoch is set but ignored because "
				"training.validation.mode=full_every_epoch."
			)
		total_batches = len(val_loader) if val_loader is not None else None
		return {
			"validation_mode": mode,
			"validation_scope": "full",
			"max_val_batches_per_epoch": None,
			"fixed_subset_seed": None,
			"fixed_subset_shuffle": False,
			"selected_batch_indices": None,
			"validation_batches_total": total_batches,
			"validation_batches_used": total_batches,
			"is_full_validation": True,
			"use_same_metric_for_checkpointing": bool(validation_config.get("use_same_metric_for_checkpointing", True)),
		}

	max_samples = _coerce_optional_positive_int(validation_config.get("max_val_samples_per_epoch"), name="training.validation.max_val_samples_per_epoch")
	batch_size = getattr(val_loader, "batch_size", None) if val_loader is not None else None
	if batch_size is None and val_loader is not None and hasattr(getattr(val_loader, "batch_sampler", None), "batch_size"):
		batch_size = getattr(val_loader.batch_sampler, "batch_size")
	max_batches_from_samples = None
	if max_samples is not None:
		if batch_size is None:
			raise ValueError("training.validation.max_val_samples_per_epoch requires a validation DataLoader with a known batch_size.")
		max_batches_from_samples = max(1, int(math.ceil(float(max_samples) / float(batch_size))))
	max_batches_value = validation_config.get("max_val_batches_per_epoch", performance_config.get("max_val_batches_per_epoch", 50))
	max_batches = _coerce_optional_positive_int(max_batches_value, name="training.validation.max_val_batches_per_epoch")
	if max_batches_from_samples is not None:
		max_batches = max_batches_from_samples
	if max_batches is None:
		max_batches = len(val_loader) if val_loader is not None else None
	if max_batches is None:
		raise ValueError(f"{mode} requires max_val_batches_per_epoch, max_val_samples_per_epoch, or a validation loader.")

	fixed_subset_shuffle = bool(validation_config.get("fixed_subset_shuffle", False))
	fixed_subset_seed = _resolve_validation_seed(
		validation_config.get("fixed_subset_seed", 42),
		default=42,
		random_when_missing=(mode == "fixed_subset_every_epoch" and fixed_subset_shuffle),
	)
	random_subset_seed = _resolve_validation_seed(
		validation_config.get("random_subset_seed", validation_config.get("fixed_subset_seed", 42)),
		default=fixed_subset_seed,
		random_when_missing=(mode == "random_subset_every_epoch"),
	)

	total_batches = len(val_loader) if val_loader is not None else None
	if total_batches is None:
		used_batches = int(max_batches)
		selected_batch_indices = list(range(used_batches)) if mode == "fixed_subset_every_epoch" else None
	elif int(total_batches) <= int(max_batches):
		used_batches = int(total_batches)
		selected_batch_indices = list(range(used_batches)) if mode == "fixed_subset_every_epoch" else None
	else:
		used_batches = int(max_batches)
		if mode == "random_subset_every_epoch":
			selected_batch_indices = None
		elif fixed_subset_shuffle:
			generator = torch.Generator()
			generator.manual_seed(fixed_subset_seed)
			selected_batch_indices = sorted(torch.randperm(int(total_batches), generator=generator)[:used_batches].tolist())
		else:
			selected_batch_indices = list(range(used_batches))

	return {
		"validation_mode": mode,
		"validation_scope": "random_subset" if mode == "random_subset_every_epoch" else "fixed_subset",
		"max_val_batches_per_epoch": int(max_batches),
		"max_val_samples_per_epoch": max_samples,
		"fixed_subset_seed": fixed_subset_seed,
		"fixed_subset_shuffle": fixed_subset_shuffle,
		"random_subset_seed": random_subset_seed,
		"selected_batch_indices": selected_batch_indices,
		"validation_batches_total": total_batches,
		"validation_batches_used": used_batches,
		"is_full_validation": False,
		"use_same_metric_for_checkpointing": bool(validation_config.get("use_same_metric_for_checkpointing", True)),
	}


def _validation_subset_metadata(policy: Mapping[str, Any], val_loader) -> dict[str, Any]:
	return {
		"validation_mode": policy["validation_mode"],
		"validation_scope": policy["validation_scope"],
		"fixed_subset_seed": policy.get("fixed_subset_seed"),
		"fixed_subset_shuffle": policy.get("fixed_subset_shuffle"),
		"max_val_batches_per_epoch": policy.get("max_val_batches_per_epoch"),
		"max_val_samples_per_epoch": policy.get("max_val_samples_per_epoch"),
		"random_subset_seed": policy.get("random_subset_seed"),
		"selected_batch_indices": policy.get("selected_batch_indices"),
		"selected_sample_indices": None,
		"validation_dataset_length": len(getattr(val_loader, "dataset", [])),
		"validation_batches_total": policy.get("validation_batches_total", len(val_loader)),
		"validation_batches_used": policy.get("validation_batches_used"),
		"is_full_validation": bool(policy.get("is_full_validation", False)),
		"use_same_metric_for_checkpointing": bool(policy.get("use_same_metric_for_checkpointing", True)),
		"created_at": datetime.now(timezone.utc).isoformat(),
	}


def save_validation_subset_metadata(run_manager: RunManager, policy: Mapping[str, Any], val_loader) -> Path | None:
	"""Persist validation subset metadata when using fixed-subset validation."""

	if str(policy.get("validation_mode")) != "fixed_subset_every_epoch":
		return None
	path = run_manager.metadata_dir / "validation_subset.json"
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(_validation_subset_metadata(policy, val_loader), handle, indent=2, sort_keys=True, default=str)
	return path


def validation_batch_indices_for_epoch(policy: Mapping[str, Any], epoch_number: int | None) -> list[int] | None:
	"""Return the validation batch-index subset for one epoch."""

	if str(policy.get("validation_mode")) != "random_subset_every_epoch":
		selected = policy.get("selected_batch_indices")
		return None if selected is None else [int(index) for index in selected]
	if policy.get("max_val_samples_per_epoch") is not None:
		return None
	total_batches = policy.get("validation_batches_total")
	used_batches = int(policy.get("validation_batches_used", 0))
	if total_batches is None:
		return list(range(used_batches))
	total_batches = int(total_batches)
	if total_batches <= used_batches:
		return list(range(total_batches))
	seed = int(policy.get("random_subset_seed", 42)) + max(0, int(epoch_number or 0))
	generator = torch.Generator()
	generator.manual_seed(seed)
	return sorted(torch.randperm(total_batches, generator=generator)[:used_batches].tolist())


def validation_loader_for_epoch(val_loader, policy: Mapping[str, Any], epoch_number: int | None):
	"""Return an exact random validation sample-subset DataLoader for one epoch when configured."""

	if str(policy.get("validation_mode")) != "random_subset_every_epoch" or policy.get("max_val_samples_per_epoch") is None:
		return val_loader, None
	dataset = getattr(val_loader, "dataset", None)
	if dataset is None:
		raise ValueError("Random validation sample subset requires val_loader.dataset.")
	dataset_length = len(dataset)
	if dataset_length <= 0:
		raise ValueError("Random validation sample subset requires a non-empty validation dataset.")
	requested_samples = int(policy["max_val_samples_per_epoch"])
	used_samples = min(dataset_length, requested_samples)
	seed = int(policy.get("random_subset_seed", 42)) + max(0, int(epoch_number or 0))
	generator = torch.Generator()
	generator.manual_seed(seed)
	selected_sample_indices = sorted(torch.randperm(dataset_length, generator=generator)[:used_samples].tolist())
	subset = Subset(dataset, selected_sample_indices)
	num_workers = int(getattr(val_loader, "num_workers", 0) or 0)
	loader_kwargs = {
		"batch_size": getattr(val_loader, "batch_size", None),
		"shuffle": False,
		"num_workers": num_workers,
		"collate_fn": getattr(val_loader, "collate_fn", None),
		"pin_memory": bool(getattr(val_loader, "pin_memory", False)),
		"drop_last": False,
	}
	if num_workers > 0:
		prefetch_factor = getattr(val_loader, "prefetch_factor", None)
		if prefetch_factor is not None:
			loader_kwargs["prefetch_factor"] = prefetch_factor
		loader_kwargs["persistent_workers"] = False
	return DataLoader(subset, **loader_kwargs), selected_sample_indices


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
	batch_indices: Iterable[int] | None = None,
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
	timing_interval = max(
		0,
		_env_int(
			"FIRE_FORECASTING_TIMING_LOG_EVERY_N_BATCHES",
			int(performance_config.get("timing_log_every_n_batches", training_config.get("timing_log_every_n_batches", 50))),
		),
	)
	synchronize_timing = bool(performance_config.get("synchronize_timing", False))
	compute_train_metrics_every_batch = bool(performance_config.get("compute_train_metrics_every_batch", training_config.get("compute_train_metrics_every_batch", False)))
	train_metrics_every_n_batches = max(
		1,
		int(performance_config.get("cheap_train_metrics_every_n_batches", training_config.get("train_metrics_every_n_batches", 100))),
	)
	compute_val_metrics = bool(performance_config.get("compute_val_metrics", training_config.get("compute_val_metrics", True)))
	non_blocking_transfer = bool(performance_config.get("non_blocking_transfer", True))
	use_cuda_prefetcher = bool(performance_config.get("prefetch_to_cuda", False)) and device.type == "cuda"
	show_progress_bar = _env_bool(
		"FIRE_FORECASTING_PROGRESS_BAR",
		bool(performance_config.get("show_progress_bar", training_config.get("show_progress_bar", True))),
	)
	progress_log_percent_step = max(
		0.0,
		_env_float(
			"FIRE_FORECASTING_PROGRESS_PERCENT",
			float(performance_config.get("progress_log_percent_step", training_config.get("progress_log_percent_step", 0.0))),
		),
	)

	total_samples = 0
	total_loss = 0.0
	metric_samples = 0
	metric_totals: dict[str, float] = defaultdict(float)
	loss_component_totals: dict[str, float] = defaultdict(float)
	timing_totals: dict[str, float] = defaultdict(float)
	timing_rows: list[dict[str, Any]] = []
	epoch_start_time = time.perf_counter()
	total_loader_batches = len(loader)
	selected_batch_indices = None if batch_indices is None else sorted({int(index) for index in batch_indices})
	if selected_batch_indices is not None:
		if not selected_batch_indices:
			raise ValueError(f"The {desc} batch-index subset is empty.")
		invalid_batch_indices = [index for index in selected_batch_indices if index < 0 or index >= total_loader_batches]
		if invalid_batch_indices:
			raise ValueError(
				f"The {desc} batch-index subset contains out-of-range index/indices. "
				f"Valid range is [0, {total_loader_batches - 1}], got {invalid_batch_indices[:10]}."
			)
		total_batches = len(selected_batch_indices)
	else:
		total_batches = total_loader_batches if max_batches is None else min(total_loader_batches, int(max_batches))
	if total_batches <= 0:
		raise ValueError(f"The {desc} DataLoader produced no batches.")
	progress_log_batch_interval = _progress_log_interval(total_batches, progress_log_percent_step)
	next_progress_log_batch = progress_log_batch_interval
	use_progress_bar = bool(show_progress_bar and tqdm is not None)
	progress_bar = tqdm(range(total_batches), desc=desc, total=total_batches, leave=False) if use_progress_bar else range(total_batches)
	iterator_source = CUDAPrefetcher(loader, device, non_blocking=non_blocking_transfer) if use_cuda_prefetcher else loader
	selected_batch_index_set = None if selected_batch_indices is None else set(selected_batch_indices)
	iterator = enumerate(iterator_source)
	input_normalizer = _build_input_normalizer(loader, device, input_channels)

	def _sync_if_timing() -> None:
		if synchronize_timing and device.type == "cuda":
			torch.cuda.synchronize(device)

	batch_number = 0
	for _batch_offset in progress_bar:
		fetch_start_time = time.perf_counter()
		for loader_batch_index, batch in iterator:
			if selected_batch_index_set is None or int(loader_batch_index) in selected_batch_index_set:
				break
		else:
			break
		batch_number += 1
		fetch_end_time = time.perf_counter()
		data_wait_time = fetch_end_time - fetch_start_time
		x_batch, y_batch, batch_extra = unpack_batch(batch)
		terrain_batch = batch_extra.get("terrain")
		if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
			raise TypeError("Expected tensor batches from the DataLoader.")

		_assert_batch_shapes(x_batch, y_batch, input_sequence_length, input_channels, output_channels)
		h2d_start_time = time.perf_counter()
		if not _tensor_on_device(x_batch, device):
			x_batch = x_batch.to(device, non_blocking=non_blocking_transfer)
		if not _tensor_on_device(y_batch, device):
			y_batch = y_batch.to(device, non_blocking=non_blocking_transfer)
		if terrain_batch is not None:
			terrain_batch = terrain_batch.to(device, non_blocking=non_blocking_transfer)
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
				model_output = model(x_batch, terrain=terrain_batch) if terrain_batch is not None else model(x_batch)
				y_pred = extract_prediction(model_output)
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

				if batch_number == 1:
					with torch.no_grad():
						debug_parts = [
							f"{desc} first batch debug",
							f"x_shape={tuple(x_batch.shape)}",
							f"terrain_shape={tuple(terrain_batch.shape) if terrain_batch is not None else None}",
							f"prediction_shape={tuple(y_pred.shape)}",
						]
						if y_pred.shape[1] >= 3:
							mask_logits = y_pred[:, 2].detach().float()
							finite_mask_logits = mask_logits[torch.isfinite(mask_logits)]
							if finite_mask_logits.numel():
								debug_parts.append(
									f"mask_logits_minmax=({float(finite_mask_logits.min().item()):.6g}, {float(finite_mask_logits.max().item()):.6g})"
								)
						if y_batch.shape[1] >= 3:
							target_mask = y_batch[:, 2].detach().float()
							finite_target_mask = target_mask[torch.isfinite(target_mask)]
							if finite_target_mask.numel():
								debug_parts.append(
									f"target_mask_minmax=({float(finite_target_mask.min().item()):.6g}, {float(finite_target_mask.max().item()):.6g})"
								)
								debug_parts.append(f"target_mask_active_fraction={float((target_mask > 0.5).float().mean().item()):.6g}")
					message = " | ".join(debug_parts)
					if logger is not None:
						logger.info(message)
					else:
						print(message)

				loss_start_time = time.perf_counter()
				loss_result = criterion(model_output, y_batch)
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
		if log_timing and timing_interval > 0 and (batch_number % timing_interval == 0 or batch_number == total_batches):
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

		if use_progress_bar and hasattr(progress_bar, "set_postfix"):
			postfix = {"loss": f"{batch_loss_value:.5f}"}
			for component_name, component_value in batch_loss_components.items():
				postfix[component_name] = f"{float(component_value):.5f}"
			for metric_name, metric_value in batch_metrics.items():
				postfix[metric_name] = f"{float(metric_value):.5f}"
			progress_bar.set_postfix(postfix)
		if progress_log_batch_interval is not None and next_progress_log_batch is not None:
			if batch_number >= next_progress_log_batch or batch_number == total_batches:
				percent_complete = min(100.0, 100.0 * float(batch_number) / float(total_batches))
				elapsed_time = time.perf_counter() - epoch_start_time
				remaining_batches = max(0, total_batches - batch_number)
				estimated_remaining_time = elapsed_time * float(remaining_batches) / float(max(1, batch_number))
				message = (
					f"{desc} progress | epoch={epoch_number if epoch_number is not None else '?'} "
					f"{percent_complete:.0f}% | batch={batch_number}/{total_batches} "
					f"elapsed={_format_duration(elapsed_time)} remaining={_format_duration(estimated_remaining_time)} "
					f"loss={batch_loss_value:.5f} samples/s={samples_per_second:.2f}"
				)
				if logger is not None:
					logger.info(message)
				else:
					print(message)
				while next_progress_log_batch is not None and next_progress_log_batch <= batch_number:
					next_progress_log_batch += progress_log_batch_interval

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
		x_batch, y_batch, batch_extra = unpack_batch(batch)
		terrain_batch = batch_extra.get("terrain")
		if not torch.is_tensor(x_batch) or not torch.is_tensor(y_batch):
			raise TypeError("Expected tensor batches from the external test DataLoader.")
		if terrain_batch is not None:
			raise ValueError("External spatial test handling does not support terrain batches; use the processed evaluation path.")

		x_batch = x_batch.to(device, non_blocking=True)
		y_batch = y_batch.to(device, non_blocking=True)
		x_batch = _apply_input_normalizer(x_batch, input_normalizer)
		with torch.no_grad():
			spatial_result = infer_with_external_test_spatial_handling(model, x_batch, config)
			y_pred = extract_prediction(spatial_result["y_pred"])

		if tuple(y_pred.shape) != tuple(y_batch.shape):
			raise ValueError(
				"External test prediction shape does not match target shape after spatial handling. "
				f"Prediction={tuple(y_pred.shape)} target={tuple(y_batch.shape)} mode={spatial_result['mode_used']}."
			)

		loss_result = criterion(spatial_result["y_pred"], y_batch)
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


def _slurm_environment_summary() -> dict[str, Any]:
	"""Capture a compact set of Slurm environment variables for run metadata."""

	keys = [
		"SLURM_JOB_ID",
		"SLURM_JOB_NAME",
		"SLURM_NODELIST",
		"SLURM_CPUS_PER_TASK",
		"SLURM_MEM_PER_NODE",
		"SLURM_MEM_PER_CPU",
		"SLURM_GPUS",
		"SLURM_JOB_GPUS",
		"SLURM_SUBMIT_DIR",
	]
	return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def _build_hardware_summary(config: Mapping[str, Any], backend_summary: Mapping[str, Any], amp_dtype: Any) -> dict[str, Any]:
	"""Build the hardware and runtime metadata saved with each run."""

	cuda_info = get_cuda_device_info()
	summary: dict[str, Any] = {
		"hostname": socket.gethostname(),
		"cuda": cuda_info,
		"cuda_available": bool(cuda_info.get("available", False)),
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"torch_version": getattr(torch, "__version__", None) if torch is not None else None,
		"torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None,
		"cudnn_version": torch.backends.cudnn.version() if torch is not None and hasattr(torch.backends, "cudnn") else None,
		"selected_precision": str(amp_dtype),
		"tf32_matmul_allowed": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
		if torch is not None and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul")
		else None,
		"tf32_cudnn_allowed": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if torch is not None and hasattr(torch.backends, "cudnn") else None,
		"backend": dict(backend_summary),
		"slurm": _slurm_environment_summary(),
		"conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
		"python": sys.version,
		"available_vram_gb": estimate_available_vram_gb(),
	}
	summary["gpu_name"] = cuda_info.get("name")
	summary["gpu_total_vram_gb"] = cuda_info.get("total_memory_gb")
	device_entries = cuda_info.get("devices")
	if isinstance(device_entries, list) and device_entries:
		first_device = device_entries[0]
		if isinstance(first_device, Mapping):
			summary["gpu_name"] = first_device.get("name")
			total_memory_gb = first_device.get("total_memory_gb", first_device.get("memory_total_gb"))
			if total_memory_gb is not None:
				summary["gpu_total_vram_gb"] = total_memory_gb
	return summary


def _input_shape_metadata(config: Mapping[str, Any], input_channels: int) -> list[Any]:
	training_config = _get_section(config, "training")
	patching_config = _get_section(config, "patching")
	sequence_length = int(config.get("input_sequence_length", training_config.get("input_sequence_length", 1)))
	patch_height = int(patching_config.get("patch_height", patching_config.get("patch_size", config.get("patch_size", 64))))
	patch_width = int(patching_config.get("patch_width", patching_config.get("patch_size", config.get("patch_size", 64))))
	return ["batch", sequence_length, int(input_channels), patch_height, patch_width]


def _output_shape_metadata(config: Mapping[str, Any], output_channels: int) -> list[Any]:
	patching_config = _get_section(config, "patching")
	patch_height = int(patching_config.get("patch_height", patching_config.get("patch_size", config.get("patch_size", 64))))
	patch_width = int(patching_config.get("patch_width", patching_config.get("patch_size", config.get("patch_size", 64))))
	return ["batch", int(output_channels), patch_height, patch_width]


def _sequence_target_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
	training_config = _get_section(config, "training")
	input_sequence_length = int(config.get("input_sequence_length", training_config.get("input_sequence_length", 1)))
	prediction_horizon = int(config.get("prediction_horizon", training_config.get("prediction_horizon", 10)))
	offsets = temporal_target_offsets(
		{
			"input_sequence_length": input_sequence_length,
			"prediction_horizon": prediction_horizon,
		}
	)
	return {
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_offset_from_start": int(offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
	}


def _save_resolved_run_artifacts(
	config: Mapping[str, Any],
	original_config: Mapping[str, Any],
	model: nn.Module,
	logger,
	normalization_stats_path: Path | None,
	run_manager: RunManager,
	backend_summary: Mapping[str, Any],
	amp_dtype: Any,
	input_channels: int,
	output_channels: int,
	loader_summaries: Mapping[str, Any],
	normalization_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
	"""Save reproducibility artifacts for the resolved training run."""

	payload = dict(config)
	existing_resolved_run = dict(payload.get("resolved_run", {})) if isinstance(payload.get("resolved_run"), Mapping) else {}
	payload["resolved_run"] = {
		"architecture": run_manager.architecture,
		"run_name": run_manager.run_name,
		"run_dir": str(run_manager.run_dir),
		"git_commit": _current_git_commit(),
		"normalization_stats_path": str(normalization_stats_path) if normalization_stats_path is not None else None,
		"model_parameter_count": _model_parameter_count(model),
		"cuda": get_cuda_device_info(),
		"sequence": _sequence_target_metadata(config),
		"input_shape": _input_shape_metadata(config, input_channels),
		"output_shape": _output_shape_metadata(config, output_channels),
		"loader_summaries": dict(loader_summaries),
		"normalization": dict(normalization_metadata or {}),
	}
	if "validation" in existing_resolved_run:
		payload["resolved_run"]["validation"] = existing_resolved_run["validation"]
	paths = run_manager.save_configs(original_config=original_config, resolved_config=payload)
	if bool(run_manager.output_config.get("save_hardware_summary", True)):
		hardware_summary = _build_hardware_summary(config, backend_summary, amp_dtype)
		hardware_summary["git_commit"] = payload["resolved_run"]["git_commit"]
		hardware_summary["model_parameter_count"] = payload["resolved_run"]["model_parameter_count"]
		hardware_summary["sequence"] = payload["resolved_run"]["sequence"]
		hardware_summary["input_shape"] = payload["resolved_run"]["input_shape"]
		hardware_summary["output_shape"] = payload["resolved_run"]["output_shape"]
		hardware_summary_path = run_manager.save_metadata("hardware_summary.json", hardware_summary)
		paths["hardware_summary_path"] = str(hardware_summary_path)
	paths["git_info_path"] = str(run_manager.save_git_info())

	cache_config = _get_section(config, "cache")
	if bool(cache_config.get("enabled", False)) and bool(cache_config.get("use_precomputed_patches", False)):
		try:
			from src.data.cache import MANIFEST_FILENAME, get_patch_cache_dir

			manifest_path = get_patch_cache_dir(config) / MANIFEST_FILENAME
			if manifest_path.exists():
				if bool(run_manager.output_config.get("save_cache_manifest_copy", True)):
					cache_manifest_copy_path = run_manager.copy_metadata_file(manifest_path, "cache_manifest.json")
					if cache_manifest_copy_path is not None:
						paths["cache_manifest_copy_path"] = str(cache_manifest_copy_path)
				path_record = run_manager.record_path_metadata("cache_manifest_path.txt", manifest_path)
				if path_record is not None:
					paths["cache_manifest_path_record"] = str(path_record)
		except Exception as exc:
			logger.warning("Could not copy cache manifest for run metadata: %s", exc)

	if normalization_stats_path is not None:
		path_record = run_manager.record_path_metadata("normalization_stats_path.txt", normalization_stats_path)
		if path_record is not None:
			paths["normalization_stats_path_record"] = str(path_record)
		normalization_npz_path: Path | None = None
		if normalization_stats_path.suffix.lower() == ".json" and normalization_stats_path.exists():
			normalization_copy_path = run_manager.copy_metadata_file(normalization_stats_path, "normalization_stats.json")
			if normalization_copy_path is not None:
				paths["normalization_stats_copy_path"] = str(normalization_copy_path)
			try:
				normalization_payload = json.loads(normalization_stats_path.read_text(encoding="utf-8"))
				paths_payload = normalization_payload.get("paths", {}) if isinstance(normalization_payload, Mapping) else {}
				npz_value = paths_payload.get("npz_path") if isinstance(paths_payload, Mapping) else None
				if npz_value not in (None, "", "null"):
					normalization_npz_path = Path(str(npz_value)).expanduser()
					if not normalization_npz_path.is_absolute():
						normalization_npz_path = (normalization_stats_path.parent / normalization_npz_path).resolve()
			except Exception as exc:
				logger.warning("Could not parse normalization JSON metadata for NPZ path: %s", exc)
		elif normalization_stats_path.suffix.lower() == ".npz":
			normalization_npz_path = normalization_stats_path
			if bool(run_manager.output_config.get("save_normalization_stats_copy", False)):
				normalization_copy_path = run_manager.copy_metadata_file(normalization_stats_path, "normalization_stats.npz")
				if normalization_copy_path is not None:
					paths["normalization_stats_copy_path"] = str(normalization_copy_path)
		if normalization_npz_path is not None:
			npz_record = run_manager.record_path_metadata("normalization_npz_path.txt", normalization_npz_path)
			if npz_record is not None:
				paths["normalization_npz_path_record"] = str(npz_record)
			if normalization_npz_path.exists():
				hash_path = run_manager.metadata_path("normalization_npz_sha256.txt")
				hash_path.write_text(compute_file_sha256(normalization_npz_path) + "\n", encoding="utf-8")
				paths["normalization_npz_sha256_path"] = str(hash_path)
			if bool(run_manager.output_config.get("copy_normalization_npz", False)):
				npz_copy_path = run_manager.copy_metadata_file(normalization_npz_path, "normalization_stats.npz")
				if npz_copy_path is not None:
					paths["normalization_npz_copy_path"] = str(npz_copy_path)

	logger.info("Saved resolved config: %s", paths.get("resolved_config_path"))
	if "hardware_summary_path" in paths:
		logger.info("Saved hardware summary: %s", paths["hardware_summary_path"])
	return paths


def _resolve_existing_normalization_stats_path(config: Mapping[str, Any]) -> Path | None:
	"""Resolve normalization.path when the stats archive exists."""

	return resolve_input_normalization_stats_path(config, must_exist=False)


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
	auto_batch_max_memory_fraction_value = performance_config.get(
		"auto_batch_max_memory_fraction",
		auto_config.get("auto_batch_max_memory_fraction", 0.85),
	)
	auto_batch_max_memory_fraction = (
		None
		if auto_batch_max_memory_fraction_value in (None, "", "null", 0, 0.0)
		else float(auto_batch_max_memory_fraction_value)
	)
	logger.info(
		"Running CUDA auto batch-size probe | initial=%s max=%s trials=%s max_memory_fraction=%s",
		current_batch_size,
		max_batch_size,
		max_trials,
		auto_batch_max_memory_fraction,
	)
	sample_batch = next(iter(train_loader))
	x_sample, y_sample = _as_batch(sample_batch)
	if not torch.is_tensor(x_sample) or not torch.is_tensor(y_sample):
		raise TypeError("Auto batch-size probing requires tensor batches.")
	probe_input_normalizer = _build_input_normalizer(train_loader, device, input_channels)

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
			max_memory_fraction=auto_batch_max_memory_fraction,
			logger=logger,
			max_trials=max_trials,
			input_transform=lambda batch: _apply_input_normalizer(batch, probe_input_normalizer),
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
	original_config = deepcopy(config)
	config["return_metadata"] = False
	training_config = _get_section(config, "training")
	performance_config = get_performance_config(config)
	logging_config = _get_section(config, "logging")
	checkpoint_config = _get_section(config, "checkpoint")
	checkpointing_config = get_training_checkpointing_config(config)
	output_config = get_training_output_config(config)
	architecture = resolve_model_architecture(config)
	run_manager = RunManager(config, architecture)
	run_manager.create_run_dir()
	training_config["run_name"] = run_manager.run_name
	training_config["run_dir"] = str(run_manager.run_dir)
	config["training"] = training_config
	config["run"] = {
		"architecture": run_manager.architecture,
		"run_name": run_manager.run_name,
		"run_dir": str(run_manager.run_dir),
	}
	checkpoint_config["path"] = str(run_manager.checkpoint_path("latest"))
	checkpoint_config["best_path"] = str(run_manager.checkpoint_path("best"))
	config["checkpoint"] = checkpoint_config
	logging_config["log_dir"] = str(run_manager.log_dir)
	logging_config["training_log_path"] = str(run_manager.log_path("training"))
	logging_config["validation_log_path"] = str(run_manager.log_path("validation"))
	logging_config["timing_log_path"] = str(run_manager.log_path("timing"))
	logging_config["metrics_log_path"] = str(run_manager.log_path("metrics"))
	config["logging"] = logging_config

	seed = int(training_config.get("seed", config.get("seed", 42)))
	set_seed(seed)
	backend_summary = configure_torch_backend(config)

	log_level = str(logging_config.get("level", "INFO"))
	log_dir = Path(logging_config.get("log_dir", run_manager.log_dir)).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_level, str(run_manager.log_path("process")))
	logger.info("Run outputs: %s", run_manager.run_dir)
	logger.info("Run checkpoints: %s", run_manager.checkpoint_dir)
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
	processed_mode = str(_get_section(config, "dataloader").get("source", "")).lower() == "processed_full_frames"
	if processed_mode:
		data_config = _get_section(config, "dataloader")
		processed_config = _get_section(config, "processed_dataset")
		logger.info("Data source: processed_full_frames | root=%s | pattern=%s | target_root=%s | normalization=%s | input_sequence_length=%s | prediction_horizon=%s", processed_config.get("root"), data_config.get("sample_pattern"), data_config.get("target_root", "<dataset_root>/targets"), normalization_stats_path, config.get("input_sequence_length", training_config.get("input_sequence_length")), config.get("prediction_horizon", training_config.get("prediction_horizon")))
	cache_config = _get_section(config, "cache")
	if not processed_mode and bool(cache_config.get("enabled", False)) and bool(cache_config.get("use_precomputed_patches", False)):
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
	epochs = int(training_config.get("max_epochs", config.get("max_epochs", config.get("epochs", training_config.get("epochs", 1)))))
	scheduler = _build_scheduler(optimizer, config, epochs)

	amp_dtype = choose_amp_dtype(config, device)
	scaler = _make_grad_scaler(amp_dtype is not None and amp_dtype is torch.float16)
	logger.info("Precision | amp_dtype=%s | grad_scaler=%s", amp_dtype, scaler is not None)
	gradient_clip_norm_value = training_config.get("gradient_clip_norm", config.get("gradient_clip_norm", None))
	gradient_clip_norm = None if gradient_clip_norm_value in (None, "", 0, 0.0) else float(gradient_clip_norm_value)
	gradient_accumulation_steps = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
	early_stopper = build_early_stopping(config)
	early_stopping_config = training_config.get("early_stopping", {}) if isinstance(training_config.get("early_stopping"), Mapping) else {}
	early_stopping_checkpoint_best = bool(early_stopping_config.get("checkpoint_best", True))
	early_stopping_save_latest_on_stop = bool(early_stopping_config.get("save_latest_on_stop", True))
	early_stopping_verbose = bool(early_stopping_config.get("verbose", True))

	latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
	resume_enabled = bool(checkpoint_config.get("resume", True))
	start_epoch = 0
	best_val_loss = math.inf
	best_epoch: int | None = None
	global_step = 0
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
		best_epoch_value = checkpoint.get("best_epoch")
		if best_epoch_value not in (None, "", "null"):
			best_epoch = int(best_epoch_value)
		global_step = int(checkpoint.get("global_step", 0))
		resumed_from_checkpoint = True
		history_rows = _coerce_history_rows(checkpoint.get("history", []))
		early_stopper.load_state_dict(checkpoint.get("early_stopping"))

	if start_epoch >= epochs:
		logger.info("Checkpoint already covers requested epochs (%s). Skipping training.", epochs)

	save_every_value = checkpointing_config.get("save_every_n_epochs", checkpoint_config.get("save_every_n_epochs", None))
	save_every_n_epochs = 1 if save_every_value in (None, "", "null", 0, 0.0) else max(1, int(save_every_value))
	save_latest_checkpoint = bool(output_config.get("save_latest_checkpoint", True)) and bool(checkpointing_config.get("save_latest", True))
	save_best_checkpoint = bool(output_config.get("save_best_checkpoint", True)) and bool(checkpointing_config.get("save_best", True))
	save_epoch_checkpoints = bool(output_config.get("save_epoch_checkpoints", False))
	keep_last_n_epoch_checkpoints = checkpointing_config.get("keep_last_n_epoch_checkpoints", 3)
	if bool(checkpointing_config.get("async_save", False)):
		logger.warning("checkpointing.async_save=true is not enabled in this safe path; saving checkpoints synchronously.")

	training_log_path = _resolve_training_log_path(config)
	training_log_path.parent.mkdir(parents=True, exist_ok=True)
	append_log = training_log_path.exists() and start_epoch > 0
	validation_log_path = Path(logging_config.get("validation_log_path", run_manager.log_path("validation"))).expanduser().resolve()
	validation_log_path.parent.mkdir(parents=True, exist_ok=True)
	append_validation_log = validation_log_path.exists() and start_epoch > 0
	metrics_log_path = Path(logging_config.get("metrics_log_path", run_manager.log_path("metrics"))).expanduser().resolve()
	metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
	append_metrics_log = metrics_log_path.exists() and start_epoch > 0
	timing_log_path = _resolve_timing_log_path(config)
	timing_log_path.parent.mkdir(parents=True, exist_ok=True)
	loader_summaries = {
		"train": _loader_summary(train_loader),
		"val": _loader_summary(val_loader),
		"test": {} if test_loader is None else _loader_summary(test_loader),
	}
	normalization_metadata = {
		"train": normalization_metadata_from_loader(train_loader, config, input_channels, normalization_stats_path),
		"val": normalization_metadata_from_loader(val_loader, config, input_channels, normalization_stats_path),
		"test": None if test_loader is None else normalization_metadata_from_loader(test_loader, config, input_channels, normalization_stats_path),
	}
	validation_policy = resolve_validation_policy(config, val_loader=val_loader, logger=logger)
	validation_subset_path = save_validation_subset_metadata(run_manager, validation_policy, val_loader)
	validation_protocol_metadata = _validation_subset_metadata(validation_policy, val_loader)
	if validation_policy["validation_mode"] == "full_every_epoch":
		logger.info("Validation mode: full_every_epoch")
		logger.info("Validation batches this epoch: all")
	elif validation_policy["validation_mode"] == "random_subset_every_epoch":
		logger.info("Validation mode: random_subset_every_epoch")
		logger.info("Random validation subset: %s batch(es) per epoch", validation_policy["validation_batches_used"])
		logger.info("Random validation base seed: %s", validation_policy["random_subset_seed"])
	else:
		logger.info("Validation mode: fixed_subset_every_epoch")
		logger.info("Fixed validation subset: %s batch(es)", validation_policy["validation_batches_used"])
		logger.info("Fixed validation seed: %s", validation_policy["fixed_subset_seed"])
		if validation_subset_path is not None:
			logger.info("Saved validation subset metadata: %s", validation_subset_path)
	checkpoint_monitor = str(checkpointing_config.get("monitor", checkpoint_config.get("monitor", "val_loss")))
	if early_stopper.enabled:
		logger.info(
			"Early stopping | monitor=%s mode=%s patience=%s min_delta=%s start_epoch=%s stop_on_nan=%s",
			early_stopper.monitor,
			early_stopper.mode,
			early_stopper.patience,
			early_stopper.min_delta,
			early_stopper.start_epoch,
			early_stopper.stop_on_nan,
		)
		if early_stopper.monitor != checkpoint_monitor:
			logger.warning(
				"Early stopping monitor %r differs from checkpoint monitor %r. "
				"Best checkpoint and early stopping may track different metrics.",
				early_stopper.monitor,
				checkpoint_monitor,
			)
	else:
		logger.info("Early stopping: disabled")
	config["resolved_run"] = {
		"architecture": architecture,
		"run_name": run_manager.run_name,
		"run_dir": str(run_manager.run_dir),
		"checkpoint_dir": str(run_manager.checkpoint_dir),
		"log_dir": str(run_manager.log_dir),
		"figure_dir": str(run_manager.figure_dir),
		"input_channels": input_channels,
		"output_channels": output_channels,
		"sequence": _sequence_target_metadata(config),
		"input_shape": _input_shape_metadata(config, input_channels),
		"output_shape": _output_shape_metadata(config, output_channels),
		"effective_batch_size": int(training_config.get("batch_size", config.get("batch_size", 1)))
		* int(training_config.get("gradient_accumulation_steps", 1)),
		"normalization_stats_path": str(normalization_stats_path) if normalization_stats_path is not None else None,
		"normalization": normalization_metadata,
		"validation": validation_protocol_metadata,
	}
	run_artifact_paths = _save_resolved_run_artifacts(
		config,
		original_config,
		model,
		logger,
		normalization_stats_path,
		run_manager,
		backend_summary,
		amp_dtype,
		input_channels,
		output_channels,
		loader_summaries,
		normalization_metadata,
	)
	if validation_subset_path is not None:
		run_artifact_paths["validation_subset_path"] = str(validation_subset_path)

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
		"Temporal target | input_sequence_length=%s prediction_horizon=%s target_offset_from_start=%s target_definition_version=%s",
		input_sequence_length,
		int(config.get("prediction_horizon", training_config.get("prediction_horizon", 10))),
		input_sequence_length - 1 + int(config.get("prediction_horizon", training_config.get("prediction_horizon", 10))),
		target_definition_version(config),
	)
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

	partial_training_result: dict[str, Any] = {
		"start_epoch": start_epoch,
		"epochs": epochs,
		"best_val_loss": best_val_loss,
		"best_epoch": best_epoch,
		"best_metric_name": str(checkpointing_config.get("monitor", checkpoint_config.get("monitor", "val_loss"))),
		"global_step": global_step,
		"latest_checkpoint_path": str(latest_checkpoint_path),
		"best_checkpoint_path": str(best_checkpoint_path),
		"training_log_path": str(training_log_path),
		"validation_log_path": str(validation_log_path),
		"metrics_log_path": str(metrics_log_path),
		"timing_log_path": str(timing_log_path),
		"run_dir": str(run_manager.run_dir),
		"run_name": run_manager.run_name,
		"architecture": architecture,
		"run_artifact_paths": run_artifact_paths,
		"sequence": _sequence_target_metadata(config),
		"normalization": normalization_metadata,
		"history_rows": history_rows,
		"final_epoch_summary": {},
		"num_epochs_completed": start_epoch,
		"validation": validation_protocol_metadata,
		"early_stopping": early_stopper.state_dict(),
		"stopped_early": False,
		"stop_reason": "",
		"stop_epoch": None,
	}
	run_completed = False

	def _write_incomplete_run_summary() -> None:
		if run_completed:
			return
		try:
			run_manager.finalize(
				partial_training_result,
				status="failed",
				error_message="Training process exited before successful completion.",
			)
		except Exception:
			pass

	atexit.register(_write_incomplete_run_summary)

	def _checkpoint_extra_state() -> dict[str, Any]:
		cawfe_latte_metadata = {}
		if architecture in {"cawfe_latte", "cawfe_latte_v1_1", "cawfe_latte_v1_2"}:
			cawfe_config = _get_section(config, "cawfe_latte_v1_1") if architecture == "cawfe_latte_v1_1" else (_get_section(config, "cawfe_latte_v1_2") if architecture == "cawfe_latte_v1_2" else _get_section(config, "cawfe_latte"))
			loss_config_for_meta = _get_section(_get_section(config, "training"), "loss")
			cawfe_latte_metadata = {
				"version": cawfe_config.get("version", "v1_end_to_end"),
				"encoder_dim": int(cawfe_config.get("output_dim", 64)),
				"backbone": {"type": _get_section(cawfe_config, "backbone").get("type", "temporal_cnn")},
				"decoder": {"type": _get_section(cawfe_config, "decoder").get("type", "shallow_cnn")},
				"auxiliary": {
					"fire_support_head": {
						"enabled": bool(_get_section(_get_section(cawfe_config, "auxiliary"), "fire_support_head").get("enabled", True)),
					}
				},
				"loss_weights": {
					"surface": float(_get_section(loss_config_for_meta, "surface").get("weight", 1.0)),
					"canopy": float(_get_section(loss_config_for_meta, "canopy").get("weight", 1.0)),
					"mask": float(_get_section(loss_config_for_meta, "mask").get("weight", 5.0)),
					"energy": float(_get_section(loss_config_for_meta, "energy").get("weight", 1.0)),
					"aux_fire_support": float(_get_section(loss_config_for_meta, "auxiliary_fire_support").get("weight", 0.2)),
				},
			}
		return {
			"architecture": architecture,
			"run_name": run_manager.run_name,
			"run_dir": str(run_manager.run_dir),
			"global_step": global_step,
			"best_metric": best_val_loss,
			"best_metric_name": str(checkpointing_config.get("monitor", checkpoint_config.get("monitor", "val_loss"))),
			"best_epoch": best_epoch,
			"input_channels": input_channels,
			**_sequence_target_metadata(config),
			"input_shape": _input_shape_metadata(config, input_channels),
			"output_shape": _output_shape_metadata(config, output_channels),
			"normalization_stats": str(normalization_stats_path) if normalization_stats_path is not None else None,
			"normalization": normalization_metadata.get("train", {}),
			"validation": validation_protocol_metadata,
			"cache_manifest_path": run_artifact_paths.get("cache_manifest_copy_path")
			or run_artifact_paths.get("cache_manifest_path_record"),
			"resolved_config_path": run_artifact_paths.get("resolved_config_path"),
			"resumed_from_checkpoint": resumed_from_checkpoint,
			"history": history_rows,
			"cawfe_latte": cawfe_latte_metadata,
			"early_stopping": {
				**early_stopper.state_dict(),
				"restore_best_weights": bool(early_stopping_config.get("restore_best_weights", False)),
				"checkpoint_best": bool(early_stopping_checkpoint_best),
				"save_latest_on_stop": bool(early_stopping_save_latest_on_stop),
			},
		}

	final_epoch_summary: dict[str, Any] = dict(history_rows[-1]) if history_rows else {}
	stopped_early = False
	stop_reason = ""

	def _early_stopping_summary() -> dict[str, Any]:
		summary = early_stopper.state_dict()
		summary["restore_best_weights"] = bool(early_stopping_config.get("restore_best_weights", False))
		summary["checkpoint_best"] = bool(early_stopping_checkpoint_best)
		summary["save_latest_on_stop"] = bool(early_stopping_save_latest_on_stop)
		summary["stopped_early"] = bool(stopped_early or early_stopper.should_stop)
		return summary

	partial_training_result["early_stopping"] = _early_stopping_summary()

	for epoch_index in range(start_epoch, epochs):
		epoch_number = epoch_index + 1
		epoch_start_time = time.perf_counter()
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
		validation_epoch_loader, validation_sample_indices = validation_loader_for_epoch(val_loader, validation_policy, epoch_number)
		validation_batch_indices = validation_batch_indices_for_epoch(validation_policy, epoch_number)
		if validation_policy["validation_mode"] == "fixed_subset_every_epoch":
			logger.info("Validation using fixed subset: %s batch(es)", validation_policy["validation_batches_used"])
		elif validation_policy["validation_mode"] == "random_subset_every_epoch":
			if validation_sample_indices is not None:
				logger.info("Validation using random sample subset: %s sample(s), %s batch(es)", len(validation_sample_indices), len(validation_epoch_loader))
			else:
				logger.info("Validation using random batch subset: %s batch(es)", validation_policy["validation_batches_used"])
		else:
			logger.info("Validation batches this epoch: all")
		val_results = _run_epoch(
			model=model,
			loader=validation_epoch_loader,
			criterion=criterion,
			config=config,
			device=device,
			input_sequence_length=input_sequence_length,
			input_channels=input_channels,
			output_channels=output_channels,
			train=False,
			max_batches=None,
			batch_indices=validation_batch_indices,
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

		global_step += int(math.ceil(float(train_results.get("train_batches", 0.0)) / float(gradient_accumulation_steps)))
		next_best_val_loss = min(best_val_loss, val_loss)
		is_best_epoch = val_loss < best_val_loss
		if is_best_epoch:
			best_epoch = epoch_number
		epoch_time_sec = time.perf_counter() - epoch_start_time
		gpu_memory_allocated_gb = 0.0
		gpu_memory_reserved_gb = 0.0
		if device.type == "cuda":
			gpu_memory_allocated_gb = float(torch.cuda.memory_allocated(device)) / float(1024**3)
			gpu_memory_reserved_gb = float(torch.cuda.memory_reserved(device)) / float(1024**3)
		train_samples = float(train_results.get("train_samples", 0.0))
		row = {
			"epoch": epoch_number,
			"global_step": global_step,
			"learning_rate": _current_lr(optimizer),
			"train_loss": train_results["train_loss"],
			"val_loss": val_results["val_loss"],
			"best_val_loss": next_best_val_loss,
			"best_metric_so_far": next_best_val_loss,
			"is_best": int(is_best_epoch),
			"epoch_time_sec": epoch_time_sec,
			"samples_per_sec": train_samples / max(epoch_time_sec, 1.0e-9),
			"gpu_memory_allocated_gb": gpu_memory_allocated_gb,
			"gpu_memory_reserved_gb": gpu_memory_reserved_gb,
			"use_patches_train": int(bool(config.get("use_patches", False))),
			"use_patches_eval": int(bool(config.get("use_patches_for_eval", False))),
			"patch_size": int(config.get("patch_size", 64)),
			"validation_mode": validation_policy["validation_mode"],
			"validation_scope": validation_policy["validation_scope"],
			"validation_batches_used": int(val_results.get("val_batches", 0.0)),
			"validation_samples_used": int(val_results.get("val_samples", 0.0)),
			"is_full_validation": bool(validation_policy["is_full_validation"]),
		}
		for metric_name, metric_value in train_results.items():
			if metric_name != "train_loss":
				row[metric_name] = metric_value
		for metric_name, metric_value in val_results.items():
			if metric_name != "val_loss":
				row[metric_name] = metric_value
		early_stopping_state = early_stopper.step(epoch_number, row)
		row.update(early_stopping_state)
		if early_stopper.should_stop:
			stopped_early = True
			stop_reason = early_stopper.stop_reason
			if early_stopping_verbose:
				logger.info("Early stopping triggered at epoch %s: %s", epoch_number, stop_reason)

		history_rows.append(row)
		should_save_latest = save_latest_checkpoint and (
			epoch_number % save_every_n_epochs == 0
			or epoch_number == epochs
			or (early_stopper.should_stop and early_stopping_save_latest_on_stop)
		)
		if is_best_epoch and save_best_checkpoint and early_stopping_checkpoint_best:
			best_val_loss = val_loss
			save_checkpoint(
				best_checkpoint_path,
				config=config,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch_index,
				best_val_loss=best_val_loss,
				**_checkpoint_extra_state(),
			)
			if not best_checkpoint_path.exists():
				raise RuntimeError(f"Best checkpoint was not created at: {best_checkpoint_path}")
			for compatibility_path in run_manager.copy_checkpoint_to_compatibility(best_checkpoint_path, "best"):
				logger.info("Updated compatibility best checkpoint: %s", compatibility_path)
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
				**_checkpoint_extra_state(),
			)
			if not latest_checkpoint_path.exists():
				raise RuntimeError(f"Latest checkpoint was not created at: {latest_checkpoint_path}")
			for compatibility_path in run_manager.copy_checkpoint_to_compatibility(latest_checkpoint_path, "latest"):
				logger.info("Updated compatibility latest checkpoint: %s", compatibility_path)

		if save_epoch_checkpoints and (epoch_number % save_every_n_epochs == 0 or epoch_number == epochs):
			epoch_checkpoint_path = run_manager.checkpoint_path("epoch", epoch=epoch_number)
			save_checkpoint(
				epoch_checkpoint_path,
				config=config,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch_index,
				best_val_loss=best_val_loss,
				**_checkpoint_extra_state(),
			)
			run_manager.prune_epoch_checkpoints(keep_last_n_epoch_checkpoints)

		_log_to_csv(training_log_path, row, append=append_log)
		if not training_log_path.exists():
			raise RuntimeError(f"Training log CSV was not created at: {training_log_path}")
		append_log = True
		validation_row = {
			key: value
			for key, value in row.items()
			if key in {"epoch", "global_step", "val_loss", "best_val_loss", "best_metric_so_far", "is_best"}
			or key.startswith("val_")
		}
		if validation_row:
			_log_to_csv(validation_log_path, validation_row, append=append_validation_log)
			append_validation_log = True
		metrics_row = {
			key: value
			for key, value in row.items()
			if key in {"epoch", "global_step"}
			or (
				key.startswith(("train_", "val_"))
				and not key.endswith("loss")
				and not key.endswith("_avg")
				and key not in {"train_samples", "train_batches", "val_samples", "val_batches"}
			)
		}
		if len(metrics_row) > 2:
			_log_to_csv(metrics_log_path, metrics_row, append=append_metrics_log)
			append_metrics_log = True
		final_epoch_summary = row
		partial_training_result.update(
			{
				"best_val_loss": best_val_loss,
				"best_epoch": best_epoch,
				"global_step": global_step,
				"history_rows": history_rows,
				"final_epoch_summary": final_epoch_summary,
				"num_epochs_completed": epoch_number,
				"validation": validation_protocol_metadata,
				"early_stopping": _early_stopping_summary(),
				"stopped_early": bool(stopped_early),
				"stop_reason": stop_reason,
				"stop_epoch": early_stopper.stop_epoch,
			}
		)

		logger.info(
			"Epoch %s summary | train_loss=%.6f | val_loss=%.6f | best_val_loss=%.6f",
			epoch_number,
			train_results["train_loss"],
			val_results["val_loss"],
			best_val_loss,
		)
		if early_stopper.should_stop:
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
	if (
		bool(output_config.get("save_training_curves", True))
		or bool(output_config.get("save_metric_curves", True))
		or bool(output_config.get("save_timing_plots", True))
	):
		try:
			figure_groups = save_training_run_figures(
				run_manager.run_dir,
				architecture=architecture,
				run_name=run_manager.run_name,
				test_results=test_plot_results,
			)
			for group_name, paths in figure_groups.items():
				if group_name in {"loss_curves", "learning_rate_curve"} and not bool(output_config.get("save_training_curves", True)):
					continue
				if group_name == "timing_breakdown" and not bool(output_config.get("save_timing_plots", True)):
					continue
				if group_name == "metric_curves" and not bool(output_config.get("save_metric_curves", True)):
					continue
				for figure_path in paths:
					if figure_path not in training_curve_paths:
						training_curve_paths.append(figure_path)
						logger.info("Saved training figure: %s", figure_path)
		except Exception as exc:
			logger.warning("Could not save training figures for run %s: %s", run_manager.run_name, exc)

	result = {
		"start_epoch": start_epoch,
		"epochs": epochs,
		"best_val_loss": best_val_loss,
		"best_epoch": best_epoch,
		"best_metric_name": str(checkpointing_config.get("monitor", checkpoint_config.get("monitor", "val_loss"))),
		"global_step": global_step,
		"num_epochs_completed": int(final_epoch_summary.get("epoch", start_epoch)) if final_epoch_summary else start_epoch,
		"latest_checkpoint_path": str(latest_checkpoint_path),
		"best_checkpoint_path": str(best_checkpoint_path),
		"training_log_path": str(training_log_path),
		"validation_log_path": str(validation_log_path),
		"metrics_log_path": str(metrics_log_path),
		"timing_log_path": str(timing_log_path),
		"run_dir": str(run_manager.run_dir),
		"run_name": run_manager.run_name,
		"architecture": architecture,
		"run_artifact_paths": run_artifact_paths,
		"training_curve_paths": training_curve_paths,
		"final_epoch_summary": final_epoch_summary,
		"history_rows": history_rows,
		"normalization": normalization_metadata,
		"validation": validation_protocol_metadata,
		"test_results": test_results,
		"early_stopping": _early_stopping_summary(),
		"stopped_early": bool(stopped_early),
		"stop_reason": stop_reason,
		"stop_epoch": early_stopper.stop_epoch,
		"epochs_completed": int(final_epoch_summary.get("epoch", start_epoch)) if final_epoch_summary else start_epoch,
	}
	hardware_summary_path = run_artifact_paths.get("hardware_summary_path")
	if hardware_summary_path:
		try:
			with Path(hardware_summary_path).open("r", encoding="utf-8") as handle:
				hardware_summary = json.load(handle)
			result["gpu_name"] = hardware_summary.get("gpu_name")
			result["gpu_total_vram_gb"] = hardware_summary.get("gpu_total_vram_gb")
		except Exception:
			pass
	run_manager.finalize(result, status="completed")
	run_completed = True
	try:
		atexit.unregister(_write_incomplete_run_summary)
	except Exception:
		pass
	logger.info("Training outputs saved under: %s", run_manager.run_dir)
	logger.info("Best checkpoint: %s", best_checkpoint_path)
	return result


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
	row = dict(row)
	if append and path.exists():
		with path.open("r", newline="", encoding="utf-8") as handle:
			reader = csv.DictReader(handle)
			existing_fieldnames = list(reader.fieldnames or [])
			existing_rows = list(reader)
		fieldnames = list(existing_fieldnames)
		for key in row:
			if key not in fieldnames:
				fieldnames.append(key)
		if fieldnames != existing_fieldnames:
			with path.open("w", newline="", encoding="utf-8") as handle:
				writer = csv.DictWriter(handle, fieldnames=fieldnames)
				writer.writeheader()
				writer.writerows(existing_rows)
				writer.writerow(row)
			return
		with path.open("a", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writerow(row)
		return

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
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


def train_model(
	config_path: str | Path,
	run_name: str | None = None,
	output_root: str | Path | None = None,
	overwrite_run: bool = False,
	disable_early_stopping: bool = False,
	early_stopping_patience: int | None = None,
	early_stopping_monitor: str | None = None,
	early_stopping_min_delta: float | None = None,
) -> dict[str, Any]:
	"""Train the forecasting model selected by the provided YAML config."""

	if torch is None:
		raise ImportError("PyTorch is required to train the wildfire forecasting model.")

	config = _ensure_config_path(load_config(config_path), config_path)
	config = apply_training_cli_overrides(
		config,
		run_name=run_name,
		output_root=output_root,
		overwrite_run=overwrite_run,
		disable_early_stopping=disable_early_stopping,
		early_stopping_patience=early_stopping_patience,
		early_stopping_monitor=early_stopping_monitor,
		early_stopping_min_delta=early_stopping_min_delta,
	)
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
	parser.add_argument("--run_name", default=None, help="Optional explicit run name for artifacts/runs/<architecture>/<run_name>.")
	parser.add_argument("--output_root", default=None, help="Override training.output.root_dir for run artifacts.")
	parser.add_argument("--overwrite_run", action="store_true", help="Allow writing into an existing explicit run directory.")
	return parser


def main() -> None:
	"""CLI entry point for training."""

	args = build_argument_parser().parse_args()
	train_model(args.config, run_name=args.run_name, output_root=args.output_root, overwrite_run=args.overwrite_run)


if __name__ == "__main__":
	main()
