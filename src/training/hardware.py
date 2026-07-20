"""Hardware and precision helpers for the shared training pipeline."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Mapping

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional training dependency
	torch = None


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def get_performance_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
	"""Return merged training performance settings with legacy fallbacks."""

	training = _get_section(config, "training")
	performance = dict(training.get("performance", {})) if isinstance(training.get("performance"), Mapping) else {}

	defaults = {
		"precision": "auto" if bool(training.get("mixed_precision", False)) else "fp32",
		"allow_tf32": bool(training.get("allow_tf32", True)),
		"cudnn_benchmark": bool(training.get("cudnn_benchmark", False)),
		"torch_compile": bool(training.get("torch_compile", False)),
		"log_timing": bool(training.get("log_timing", False)),
		"timing_log_every_n_batches": int(training.get("timing_log_every_n_batches", 50)),
		"synchronize_timing": bool(training.get("synchronize_timing", False)),
		"auto_batch_size": False,
		"auto_batch_max_memory_fraction": float(training.get("auto_batch_max_memory_fraction", 0.85)),
		"prefetch_to_cuda": False,
		"non_blocking_transfer": True,
		"max_train_batches_per_epoch": training.get("max_train_batches_per_epoch"),
		"max_val_batches_per_epoch": training.get("max_val_batches_per_epoch"),
		"cheap_train_metrics_every_n_batches": int(training.get("train_metrics_every_n_batches", 100)),
		"compute_train_metrics_every_batch": bool(training.get("compute_train_metrics_every_batch", False)),
		"compute_val_metrics": bool(training.get("compute_val_metrics", True)),
	}
	for key, value in defaults.items():
		performance.setdefault(key, value)
	return performance


def get_cuda_device_info(device: int | None = None) -> dict[str, Any]:
	"""Return a compact CUDA hardware summary without requiring CUDA."""

	if torch is None or not torch.cuda.is_available():
		return {
			"available": False,
			"device_count": 0,
			"name": None,
			"capability": None,
			"total_memory_gb": 0.0,
		}

	device_index = torch.cuda.current_device() if device is None else int(device)
	properties = torch.cuda.get_device_properties(device_index)
	capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
	return {
		"available": True,
		"device_count": int(torch.cuda.device_count()),
		"current_device": device_index,
		"name": str(properties.name),
		"capability": capability,
		"total_memory_gb": float(properties.total_memory) / float(1024**3),
		"multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
	}


def configure_torch_backend(config: Mapping[str, Any] | None) -> dict[str, Any]:
	"""Apply backend options such as TF32 and cuDNN benchmarking."""

	performance = get_performance_config(config)
	summary = {
		"torch_available": torch is not None,
		"cuda_available": bool(torch is not None and torch.cuda.is_available()),
		"allow_tf32": bool(performance.get("allow_tf32", True)),
		"cudnn_benchmark": bool(performance.get("cudnn_benchmark", False)),
	}
	if torch is None:
		return summary

	allow_tf32 = bool(performance.get("allow_tf32", True))
	if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
		torch.backends.cuda.matmul.allow_tf32 = allow_tf32
	if hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.allow_tf32 = allow_tf32
		if torch.backends.cudnn.is_available():
			torch.backends.cudnn.benchmark = bool(performance.get("cudnn_benchmark", False))
			if bool(performance.get("cudnn_benchmark", False)):
				torch.backends.cudnn.deterministic = False
	return summary


def _cuda_supports_bf16(device: int | None = None) -> bool:
	if torch is None or not torch.cuda.is_available():
		return False
	device_index = torch.cuda.current_device() if device is None else int(device)
	if hasattr(torch.cuda, "is_bf16_supported"):
		try:
			return bool(torch.cuda.is_bf16_supported())
		except TypeError:
			pass
	major, _minor = torch.cuda.get_device_capability(device_index)
	return int(major) >= 8


def choose_amp_dtype(config: Mapping[str, Any] | None, device: str | Any = "cuda"):
	"""Choose the autocast dtype from ``training.performance.precision``."""

	if torch is None:
		return None
	device_type = getattr(device, "type", str(device)).lower()
	if device_type != "cuda" or not torch.cuda.is_available():
		return None

	performance = get_performance_config(config)
	precision = str(performance.get("precision", "auto")).lower()
	if precision in {"none", "off", "false", "fp32", "float32"}:
		return None
	if precision in {"bf16", "bfloat16"}:
		return torch.bfloat16
	if precision in {"fp16", "float16", "half"}:
		return torch.float16
	if precision == "auto":
		return torch.bfloat16 if _cuda_supports_bf16() else torch.float16
	raise ValueError(f"Unsupported training.performance.precision: {precision!r}")


def autocast_context(device: str | Any, amp_dtype):
	"""Return an autocast context manager for the selected device and dtype."""

	if torch is None or amp_dtype is None:
		return nullcontext()
	device_type = getattr(device, "type", str(device)).lower()
	if device_type != "cuda":
		return nullcontext()
	if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
		return torch.amp.autocast("cuda", dtype=amp_dtype)
	return torch.cuda.amp.autocast(dtype=amp_dtype)  # type: ignore[attr-defined]


def estimate_available_vram_gb(device: int | None = None) -> float:
	"""Estimate currently free CUDA memory in GiB."""

	if torch is None or not torch.cuda.is_available():
		return 0.0
	device_index = torch.cuda.current_device() if device is None else int(device)
	try:
		free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
		return float(free_bytes) / float(1024**3)
	except Exception:
		properties = torch.cuda.get_device_properties(device_index)
		reserved = torch.cuda.memory_reserved(device_index)
		return max(0.0, float(properties.total_memory - reserved) / float(1024**3))


def get_slurm_cpu_count() -> int | None:
	"""Return the CPUs allocated to this Slurm task when available."""

	for variable_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
		value = os.environ.get(variable_name)
		if value in (None, ""):
			continue
		first_value = str(value).split(",", 1)[0].split("(", 1)[0]
		try:
			cpus = int(first_value)
		except ValueError:
			continue
		if cpus > 0:
			return cpus
	return None


def cap_num_workers_by_slurm(config: Mapping[str, Any] | None, requested_workers: int | str) -> int:
	"""Cap DataLoader workers to Slurm CPU allocation when requested."""

	training = _get_section(config, "training")
	data_loader = _get_section(config, "data_loader")
	auto_cap = bool(training.get("auto_cap_num_workers_to_slurm_cpus", True))
	if "auto_cap_num_workers_to_slurm_cpus" in data_loader:
		auto_cap = bool(data_loader["auto_cap_num_workers_to_slurm_cpus"])
	if isinstance(requested_workers, str) and requested_workers.lower() == "auto":
		cpu_count = get_slurm_cpu_count()
		return max(0, int(cpu_count if cpu_count is not None else os.cpu_count() or 0))
	workers = max(0, int(requested_workers))
	if not auto_cap:
		return workers
	cpu_count = get_slurm_cpu_count()
	if cpu_count is None:
		return workers
	return min(workers, max(0, cpu_count))


def _resize_batch_tensor(tensor, target_batch_size: int):
	if int(tensor.shape[0]) == target_batch_size:
		return tensor
	if int(tensor.shape[0]) > target_batch_size:
		return tensor[:target_batch_size]
	repeats = int(math.ceil(target_batch_size / int(tensor.shape[0])))
	repeat_shape = [repeats] + [1] * (tensor.ndim - 1)
	return tensor.repeat(*repeat_shape)[:target_batch_size]


def _is_cuda_oom(exc: BaseException) -> bool:
	message = str(exc).lower()
	return "out of memory" in message or "cuda error: out of memory" in message


def find_max_batch_size(
	model,
	criterion,
	optimizer_factory: Callable[[Any], Any],
	sample_batch: tuple[Any, Any],
	device,
	amp_dtype=None,
	initial_batch_size: int = 1,
	max_batch_size: int | None = None,
	gradient_accumulation_steps: int = 1,
	gradient_clip_norm: float | None = None,
	max_memory_fraction: float | None = None,
	logger=None,
	max_trials: int = 12,
) -> int:
	"""Probe CUDA memory using exponential growth followed by binary search."""

	if torch is None:
		raise ImportError("PyTorch is required for automatic batch-size probing.")
	if getattr(device, "type", str(device)).lower() != "cuda" or not torch.cuda.is_available():
		return max(1, int(initial_batch_size))

	x_sample, y_sample = sample_batch
	if not torch.is_tensor(x_sample) or not torch.is_tensor(y_sample):
		raise TypeError("find_max_batch_size expects tensor sample batches.")
	initial_batch_size = max(1, int(initial_batch_size))
	max_batch_size = max_batch_size if max_batch_size is not None else max(initial_batch_size, int(x_sample.shape[0]) * 4)
	max_batch_size = max(1, int(max_batch_size))
	max_trials = max(1, int(max_trials))
	if max_memory_fraction in (None, 0, 0.0):
		memory_fraction = None
	else:
		raw_memory_fraction = float(max_memory_fraction)
		memory_fraction = None if raw_memory_fraction <= 0.0 else min(raw_memory_fraction, 1.0)
	peak_limit_gb = None
	if memory_fraction is not None:
		properties = torch.cuda.get_device_properties(device)
		peak_limit_gb = float(properties.total_memory) * memory_fraction / float(1024**3)
	optimizer = optimizer_factory(model.parameters())
	best = 0
	low = initial_batch_size
	high = None
	trials = 0

	def log(message: str, *args: Any) -> None:
		if logger is not None:
			logger.info(message, *args)

	def trial(batch_size: int) -> bool:
		nonlocal trials
		trials += 1
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.reset_peak_memory_stats(device)
		optimizer.zero_grad(set_to_none=True)
		model.train()
		try:
			x_batch = _resize_batch_tensor(x_sample, batch_size).to(device, non_blocking=True)
			y_batch = _resize_batch_tensor(y_sample, batch_size).to(device, non_blocking=True)
			start = time.perf_counter()
			with autocast_context(device, amp_dtype):
				y_pred = model(x_batch)
				loss = criterion(y_pred, y_batch)
				if isinstance(loss, Mapping):
					loss = loss["total_loss"]
				loss = loss / float(max(1, gradient_accumulation_steps))
			loss.backward()
			if gradient_clip_norm is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
			torch.cuda.synchronize(device)
			peak_allocated_gb = float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
			peak_reserved_gb = float(torch.cuda.max_memory_reserved(device)) / float(1024**3)
			peak_used_gb = max(peak_allocated_gb, peak_reserved_gb)
			if peak_limit_gb is not None and peak_used_gb > peak_limit_gb:
				log(
					"auto_batch trial over memory margin | batch_size=%s | peak_allocated=%.2f GB | peak_reserved=%.2f GB | limit=%.2f GB | fraction=%.2f",
					batch_size,
					peak_allocated_gb,
					peak_reserved_gb,
					peak_limit_gb,
					memory_fraction,
				)
				return False
			log(
				"auto_batch trial ok | batch_size=%s | peak_allocated=%.2f GB | peak_reserved=%.2f GB | time=%.2fs",
				batch_size,
				peak_allocated_gb,
				peak_reserved_gb,
				time.perf_counter() - start,
			)
			return True
		except RuntimeError as exc:
			if not _is_cuda_oom(exc):
				raise
			log("auto_batch trial OOM | batch_size=%s", batch_size)
			return False
		finally:
			optimizer.zero_grad(set_to_none=True)
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	current = low
	while trials < max_trials and current <= max_batch_size:
		if trial(current):
			best = current
			if current >= max_batch_size:
				break
			current = min(max_batch_size, current * 2)
		else:
			high = current - 1
			break

	if high is None:
		high = max_batch_size
		low = best + 1
	else:
		low = best + 1
	while trials < max_trials and low <= high:
		mid = (low + high) // 2
		if trial(mid):
			best = mid
			low = mid + 1
		else:
			high = mid - 1

	if best <= 0:
		best = max(1, initial_batch_size // 2)
	log("auto_batch selected batch_size=%s after %s trial(s)", best, trials)
	return int(best)
