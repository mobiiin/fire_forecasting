"""Debug trained wildfire model predictions against target channel semantics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.config import load_config
from src.data.cache import MANIFEST_FILENAME, get_patch_cache_dir
from src.data.dataset import create_dataloaders, metadata_batch_to_list
from src.data.patching import resolve_patching_config
from src.models.evaluation import _validate_checkpoint_sequence
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, validate_checkpoint_model_compatibility
from src.training.hardware import autocast_context, choose_amp_dtype
from src.training.train import (
	_apply_input_normalizer,
	_build_input_normalizer,
	_ensure_config_path,
	_get_device,
	_infer_input_channels_from_loader,
	_input_normalization_status,
	_loader_summary,
	_resolve_existing_normalization_stats_path,
)


STATS_FIELDS = [
	"min",
	"p01",
	"p05",
	"p25",
	"mean",
	"median",
	"p75",
	"p95",
	"p99",
	"max",
	"std",
	"frac_nonzero_abs_gt_1e-8",
	"frac_positive",
	"frac_negative",
	"frac_nan",
	"frac_inf",
]
STAT_TENSOR_NAMES = [
	"y_surface",
	"y_canopy",
	"y_mask",
	"y_energy_log",
	"y_energy_mw",
	"pred_surface",
	"pred_canopy",
	"pred_mask_logits",
	"pred_mask_prob",
	"pred_energy_log",
	"pred_energy_mw_safe",
	"abs_error_surface",
	"abs_error_canopy",
	"abs_error_energy_log",
	"abs_error_energy_mw_safe",
]
METRIC_FIELDS = [
	"surface_mae",
	"canopy_mae",
	"mask_dice",
	"mask_iou",
	"mask_precision",
	"mask_recall",
	"energy_log_mae",
	"energy_mw_mae_safe",
	"active_energy_mw_mae_safe",
]


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Debug trained wildfire model prediction scales, channels, and checkpoint metadata."
	)
	parser.add_argument("--config", required=True, help="YAML config used to build the dataloader/model.")
	parser.add_argument("--model_architecture", required=True, help="Requested model architecture.")
	parser.add_argument("--checkpoint", required=True, help="Path to checkpoint, usually checkpoints/best_model.pt.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to inspect.")
	parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to run.")
	parser.add_argument("--num_samples_to_plot", type=int, default=4, help="Maximum sample figures to save.")
	parser.add_argument(
		"--output_dir",
		default=None,
		help="Output directory. Defaults to artifacts/debug_predictions/<architecture>/<timestamp>.",
	)
	parser.add_argument("--batch_size", type=int, default=None, help="Optional DataLoader batch-size override.")
	parser.add_argument("--threshold", type=float, default=0.5, help="Threshold applied to sigmoid(mask logits).")
	parser.add_argument("--energy_active_threshold_mw", type=float, default=1.0e-3)
	parser.add_argument("--consumed_active_threshold", type=float, default=1.0e-3)
	parser.add_argument("--save_npz", action="store_true", help="Save raw y/pred arrays for plotted samples.")
	parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:0, etc.")
	parser.add_argument("--allow_architecture_mismatch", action="store_true", help="Continue despite checkpoint architecture mismatch.")
	parser.add_argument(
		"--allow_sequence_mismatch",
		action="store_true",
		help="Continue despite checkpoint T/horizon metadata mismatch with the config.",
	)
	return parser


def _to_jsonable(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(key): _to_jsonable(nested) for key, nested in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return _to_jsonable(value.tolist())
	if isinstance(value, np.generic):
		return _to_jsonable(value.item())
	if torch.is_tensor(value):
		if value.ndim == 0:
			return _to_jsonable(value.detach().cpu().item())
		return _to_jsonable(value.detach().cpu().tolist())
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


def save_json(path: str | Path, payload: Any) -> None:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def _csv_cell(value: Any) -> Any:
	if isinstance(value, float) and not math.isfinite(value):
		return ""
	if isinstance(value, (dict, list, tuple)):
		return json.dumps(_to_jsonable(value), sort_keys=True)
	return value


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if fieldnames is None:
		ordered: list[str] = []
		seen: set[str] = set()
		for row in rows:
			for key in row:
				key_text = str(key)
				if key_text not in seen:
					seen.add(key_text)
					ordered.append(key_text)
		fieldnames = ordered
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
		writer.writeheader()
		for row in rows:
			writer.writerow({key: _csv_cell(row.get(key, "")) for key in fieldnames})


def _format_float(value: Any) -> str:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return "nan"
	if not math.isfinite(number):
		return "nan"
	if abs(number) >= 1.0e4 or (abs(number) < 1.0e-3 and number != 0.0):
		return f"{number:.4e}"
	return f"{number:.6f}"


def _mean_from_stats(rows: Sequence[Mapping[str, Any]], name: str) -> float:
	for row in rows:
		if str(row.get("name")) == name:
			try:
				return float(row.get("mean", math.nan))
			except (TypeError, ValueError):
				return math.nan
	return math.nan


def _max_from_stats(rows: Sequence[Mapping[str, Any]], name: str) -> float:
	for row in rows:
		if str(row.get("name")) == name:
			try:
				return float(row.get("max", math.nan))
			except (TypeError, ValueError):
				return math.nan
	return math.nan


def _min_from_stats(rows: Sequence[Mapping[str, Any]], name: str) -> float:
	for row in rows:
		if str(row.get("name")) == name:
			try:
				return float(row.get("min", math.nan))
			except (TypeError, ValueError):
				return math.nan
	return math.nan


def _finite_mean(tensor: torch.Tensor) -> float:
	values = tensor.detach().float()
	values = values[torch.isfinite(values)]
	return float(values.mean().item()) if values.numel() else math.nan


def tensor_stats(name: str, tensor: Any) -> dict[str, Any]:
	"""Return robust scalar diagnostics for a tensor, tolerating NaN and Inf."""

	if not torch.is_tensor(tensor):
		tensor = torch.as_tensor(tensor)
	values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
	total = int(values.numel())
	row: dict[str, Any] = {"name": str(name), "numel": total}
	if total == 0:
		for field in STATS_FIELDS:
			row[field] = math.nan
		return row

	is_nan = torch.isnan(values)
	is_inf = torch.isinf(values)
	is_finite = torch.isfinite(values)
	finite_values = values[is_finite]
	row["frac_nan"] = float(is_nan.sum().item()) / total
	row["frac_inf"] = float(is_inf.sum().item()) / total
	row["frac_nonzero_abs_gt_1e-8"] = float((torch.abs(values[~is_nan]) > 1.0e-8).sum().item()) / total
	row["frac_positive"] = float((values[~is_nan] > 0).sum().item()) / total
	row["frac_negative"] = float((values[~is_nan] < 0).sum().item()) / total

	if finite_values.numel() == 0:
		for field in ("min", "p01", "p05", "p25", "mean", "median", "p75", "p95", "p99", "max", "std"):
			row[field] = math.nan
		return row

	quantiles = torch.quantile(
		finite_values,
		torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float32),
	)
	row.update(
		{
			"min": float(finite_values.min().item()),
			"p01": float(quantiles[0].item()),
			"p05": float(quantiles[1].item()),
			"p25": float(quantiles[2].item()),
			"mean": float(finite_values.mean().item()),
			"median": float(quantiles[3].item()),
			"p75": float(quantiles[4].item()),
			"p95": float(quantiles[5].item()),
			"p99": float(quantiles[6].item()),
			"max": float(finite_values.max().item()),
			"std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() > 1 else 0.0,
		}
	)
	return row


def safe_expm1_log_energy(log_tensor: torch.Tensor, max_log: float | None = 20.0) -> torch.Tensor:
	"""Convert log1p energy to MW with nonnegative clamp and optional overflow cap."""

	energy_log = torch.clamp(log_tensor, min=0.0)
	if max_log is not None:
		energy_log = torch.clamp(energy_log, max=float(max_log))
	return torch.expm1(energy_log)


def compute_mask_metrics(pred_prob: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
	"""Compute binary mask overlap metrics with clear empty-mask behavior."""

	pred_binary = pred_prob.detach() > float(threshold)
	target_binary = target_mask.detach() > 0.5
	true_positive = float(torch.logical_and(pred_binary, target_binary).sum().item())
	false_positive = float(torch.logical_and(pred_binary, torch.logical_not(target_binary)).sum().item())
	false_negative = float(torch.logical_and(torch.logical_not(pred_binary), target_binary).sum().item())
	pred_count = true_positive + false_positive
	target_count = true_positive + false_negative
	union = true_positive + false_positive + false_negative
	dice_denominator = pred_count + target_count
	dice = 1.0 if dice_denominator == 0 else (2.0 * true_positive) / dice_denominator
	iou = 1.0 if union == 0 else true_positive / union
	precision = 1.0 if pred_count == 0 and target_count == 0 else (true_positive / pred_count if pred_count > 0 else 0.0)
	recall = 1.0 if pred_count == 0 and target_count == 0 else (true_positive / target_count if target_count > 0 else 0.0)
	return {
		"mask_dice": float(dice),
		"mask_iou": float(iou),
		"mask_precision": float(precision),
		"mask_recall": float(recall),
	}


def _prediction_tensors(pred: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
	y_surface = y[:, 0]
	y_canopy = y[:, 1]
	y_mask = y[:, 2]
	y_energy_log = y[:, 3]
	pred_surface = pred[:, 0]
	pred_canopy = pred[:, 1]
	pred_mask_logits = pred[:, 2]
	pred_energy_log = pred[:, 3]
	pred_mask_prob = torch.sigmoid(pred_mask_logits)
	y_energy_mw = safe_expm1_log_energy(y_energy_log, max_log=None)
	pred_energy_mw_safe = safe_expm1_log_energy(pred_energy_log, max_log=20.0)
	return {
		"y_surface": y_surface,
		"y_canopy": y_canopy,
		"y_mask": y_mask,
		"y_energy_log": y_energy_log,
		"y_energy_mw": y_energy_mw,
		"pred_surface": pred_surface,
		"pred_canopy": pred_canopy,
		"pred_mask_logits": pred_mask_logits,
		"pred_mask_prob": pred_mask_prob,
		"pred_energy_log": pred_energy_log,
		"pred_energy_mw_safe": pred_energy_mw_safe,
		"abs_error_surface": torch.abs(pred_surface - y_surface),
		"abs_error_canopy": torch.abs(pred_canopy - y_canopy),
		"abs_error_energy_log": torch.abs(pred_energy_log - y_energy_log),
		"abs_error_energy_mw_safe": torch.abs(pred_energy_mw_safe - y_energy_mw),
	}


def compute_debug_metrics(
	pred: torch.Tensor,
	y: torch.Tensor,
	threshold: float = 0.5,
	energy_active_threshold_mw: float = 1.0e-3,
	consumed_active_threshold: float = 1.0e-3,
) -> dict[str, float]:
	"""Compute quick prediction-vs-target diagnostics for the four output channels."""

	tensors = _prediction_tensors(pred, y)
	metrics = {
		"surface_mae": _finite_mean(tensors["abs_error_surface"]),
		"canopy_mae": _finite_mean(tensors["abs_error_canopy"]),
		"energy_log_mae": _finite_mean(tensors["abs_error_energy_log"]),
		"energy_mw_mae_safe": _finite_mean(tensors["abs_error_energy_mw_safe"]),
	}
	metrics.update(compute_mask_metrics(tensors["pred_mask_prob"], tensors["y_mask"], threshold=threshold))
	active = (
		(tensors["y_mask"] > 0.5)
		| (tensors["y_energy_mw"] > float(energy_active_threshold_mw))
		| (tensors["y_surface"] > float(consumed_active_threshold))
		| (tensors["y_canopy"] > float(consumed_active_threshold))
	)
	if bool(active.any().item()):
		metrics["active_energy_mw_mae_safe"] = _finite_mean(tensors["abs_error_energy_mw_safe"][active])
	else:
		metrics["active_energy_mw_mae_safe"] = math.nan
	return {key: float(value) for key, value in metrics.items()}


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
	raise ValueError(f"Unsupported split: {split!r}")


def _resolve_device(config: Mapping[str, Any], device_arg: str) -> torch.device:
	device_text = str(device_arg).lower()
	if device_text == "auto":
		return _get_device(config)
	if device_text == "gpu":
		device_text = "cuda"
	if device_text == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("--device cuda was requested, but CUDA is not available.")
	return torch.device(device_text)


def _apply_architecture_override(config: Mapping[str, Any], architecture: str) -> dict[str, Any]:
	updated = dict(config)
	model_config = dict(updated.get("model", {})) if isinstance(updated.get("model"), Mapping) else {}
	model_config["architecture"] = str(architecture).lower()
	updated["model"] = model_config
	return updated


def _apply_batch_size_override(config: Mapping[str, Any], batch_size: int | None) -> dict[str, Any]:
	if batch_size is None:
		return dict(config)
	if int(batch_size) <= 0:
		raise ValueError("--batch_size must be positive when provided.")
	updated = dict(config)
	updated["batch_size"] = int(batch_size)
	data_loader_config = dict(updated.get("data_loader", {})) if isinstance(updated.get("data_loader"), Mapping) else {}
	data_loader_config["batch_size"] = int(batch_size)
	for split in ("train", "val", "test"):
		split_config = dict(data_loader_config.get(split, {})) if isinstance(data_loader_config.get(split), Mapping) else {}
		split_config["batch_size"] = int(batch_size)
		data_loader_config[split] = split_config
	updated["data_loader"] = data_loader_config
	return updated


def checkpoint_metadata_payload(
	checkpoint: Mapping[str, Any],
	checkpoint_path: str | Path,
	requested_architecture: str,
	allow_architecture_mismatch: bool = False,
) -> dict[str, Any]:
	"""Build checkpoint metadata and enforce architecture compatibility."""

	resolved_path = Path(checkpoint_path).expanduser().resolve()
	checkpoint_architecture = checkpoint.get("architecture")
	requested = str(requested_architecture).lower()
	if checkpoint_architecture not in (None, "") and str(checkpoint_architecture).lower() != requested:
		message = (
			"Checkpoint architecture mismatch: "
			f"checkpoint={checkpoint_architecture!r}, requested={requested!r}, path={resolved_path}. "
			"Pass --allow_architecture_mismatch to continue."
		)
		if not allow_architecture_mismatch:
			raise ValueError(message)
	stat = resolved_path.stat()
	return {
		"checkpoint_path": str(resolved_path),
		"checkpoint_keys": sorted(str(key) for key in checkpoint.keys()),
		"architecture": checkpoint_architecture,
		"requested_architecture": requested,
		"run_name": checkpoint.get("run_name"),
		"run_dir": checkpoint.get("run_dir"),
		"epoch": checkpoint.get("epoch"),
		"best_epoch": checkpoint.get("best_epoch"),
		"best_metric": checkpoint.get("best_metric", checkpoint.get("best_val_loss")),
		"global_step": checkpoint.get("global_step"),
		"input_sequence_length": checkpoint.get("input_sequence_length"),
		"prediction_horizon": checkpoint.get("prediction_horizon"),
		"target_offset_from_start": checkpoint.get("target_offset_from_start"),
		"target_offset_from_last_input": checkpoint.get("target_offset_from_last_input"),
		"target_definition_version": checkpoint.get("target_definition_version"),
		"file_size_bytes": int(stat.st_size),
		"modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
	}


def _state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
	for key in ("model_state_dict", "state_dict"):
		state = checkpoint.get(key)
		if isinstance(state, Mapping):
			return state
	if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
		return checkpoint
	raise KeyError("Checkpoint does not contain model_state_dict or state_dict.")


def _extract_prediction_and_aux(model_output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
	if torch.is_tensor(model_output):
		return model_output, {}
	if isinstance(model_output, (tuple, list)):
		if not model_output:
			raise TypeError("Model returned an empty tuple/list.")
		prediction = model_output[0]
		if not torch.is_tensor(prediction):
			raise TypeError(f"First model output must be a tensor, got {type(prediction)!r}.")
		aux = model_output[1] if len(model_output) > 1 else {}
		if isinstance(aux, Mapping):
			aux_payload = {str(key): _aux_value_summary(value) for key, value in aux.items()}
		else:
			aux_payload = {"aux_type": type(aux).__name__}
		return prediction, aux_payload
	if isinstance(model_output, Mapping):
		for key in ("pred", "prediction", "y_pred", "output", "logits"):
			value = model_output.get(key)
			if torch.is_tensor(value):
				aux_payload = {str(aux_key): _aux_value_summary(aux_value) for aux_key, aux_value in model_output.items() if aux_key != key}
				return value, aux_payload
	raise TypeError(f"Unsupported model output type: {type(model_output)!r}.")


def _aux_value_summary(value: Any) -> Any:
	if torch.is_tensor(value):
		return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
	if isinstance(value, Mapping):
		return {str(key): _aux_value_summary(nested) for key, nested in value.items()}
	if isinstance(value, (list, tuple)):
		if len(value) <= 8:
			return [_aux_value_summary(item) for item in value]
		return {"type": type(value).__name__, "length": len(value)}
	return _to_jsonable(value)


def _validate_shapes(x_batch: torch.Tensor, y_batch: torch.Tensor, pred: torch.Tensor) -> None:
	if x_batch.ndim != 5:
		raise ValueError(f"Expected x shape (B,T,C,H,W), got {tuple(x_batch.shape)}.")
	if y_batch.ndim != 4:
		raise ValueError(f"Expected target y shape (B,4,H,W), got {tuple(y_batch.shape)}.")
	if pred.ndim != 4:
		raise ValueError(f"Expected prediction shape (B,4,H,W), got {tuple(pred.shape)}.")
	if int(y_batch.shape[1]) != 4:
		raise ValueError(f"Expected target y to have 4 output channels, got {tuple(y_batch.shape)}.")
	if int(pred.shape[1]) != 4:
		raise ValueError(f"Expected prediction to have 4 output channels, got {tuple(pred.shape)}.")
	if tuple(pred.shape) != tuple(y_batch.shape):
		raise ValueError(f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(y_batch.shape)}.")


def _metadata_items(batch: Sequence[Any], batch_size: int) -> list[dict[str, Any]]:
	if len(batch) < 3:
		return [{} for _ in range(batch_size)]
	try:
		items = metadata_batch_to_list(batch[2], batch_size=batch_size)
	except Exception:
		return [{} for _ in range(batch_size)]
	if len(items) < batch_size:
		items.extend({} for _ in range(batch_size - len(items)))
	return items[:batch_size]


def _metadata_csv_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
	keys = [
		"dataset_name",
		"fire_name",
		"sample_index",
		"current_idx",
		"current_index",
		"future_idx",
		"future_index",
		"start_idx",
		"input_indices",
		"last_input_idx",
		"target_idx",
		"input_sequence_length",
		"prediction_horizon",
		"target_offset_from_start",
		"target_offset_from_last_input",
		"target_definition_version",
		"patch",
		"patch_top",
		"patch_left",
		"patch_bottom",
		"patch_right",
		"cache_shard_path",
		"cache_local_index",
	]
	return {key: metadata.get(key) for key in keys if key in metadata}


def _metadata_title(metadata: Mapping[str, Any]) -> str:
	fire_name = metadata.get("fire_name", metadata.get("dataset_name", "unknown"))
	sample_index = metadata.get("sample_index", metadata.get("cache_local_index", "unknown"))
	patch = metadata.get("patch")
	if isinstance(patch, Mapping):
		patch_text = f"patch=({patch.get('y0')}:{patch.get('y1')},{patch.get('x0')}:{patch.get('x1')})"
	elif {"patch_top", "patch_bottom", "patch_left", "patch_right"}.issubset(metadata.keys()):
		patch_text = (
			f"patch=({metadata.get('patch_top')}:{metadata.get('patch_bottom')},"
			f"{metadata.get('patch_left')}:{metadata.get('patch_right')})"
		)
	else:
		patch_text = "patch=unknown"
	target_text = ""
	if "target_idx" in metadata or "prediction_horizon" in metadata:
		target_text = f" | target={metadata.get('target_idx', 'unknown')} H={metadata.get('prediction_horizon', 'unknown')}"
	return f"fire={fire_name} | sample={sample_index}{target_text} | {patch_text}"


def _robust_vmax(*arrays: Any, percentile: float = 99.0, minimum: float = 1.0) -> float:
	values = []
	for array in arrays:
		np_array = np.asarray(array, dtype=np.float64)
		finite = np_array[np.isfinite(np_array)]
		if finite.size:
			values.append(finite.reshape(-1))
	if not values:
		return float(minimum)
	stacked = np.concatenate(values)
	value = float(np.nanpercentile(stacked, percentile))
	if not math.isfinite(value) or value <= 0.0:
		return float(minimum)
	return value


def _imshow_panel(ax, data: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "inferno") -> None:
	image = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
	ax.set_title(title, fontsize=9)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def plot_sample_debug(
	output_path: str | Path,
	pred_sample: torch.Tensor,
	y_sample: torch.Tensor,
	metadata: Mapping[str, Any] | None = None,
	title_prefix: str = "",
	threshold: float = 0.5,
) -> None:
	"""Save the 12-panel sample diagnostic figure."""

	metadata = metadata or {}
	pred_cpu = pred_sample.detach().float().cpu()
	y_cpu = y_sample.detach().float().cpu()
	if pred_cpu.shape != y_cpu.shape or pred_cpu.ndim != 3 or int(pred_cpu.shape[0]) != 4:
		raise ValueError(f"Expected pred/y sample shapes (4,H,W), got pred={tuple(pred_cpu.shape)} y={tuple(y_cpu.shape)}.")

	target_surface = y_cpu[0].numpy()
	pred_surface = pred_cpu[0].numpy()
	target_canopy = y_cpu[1].numpy()
	pred_canopy = pred_cpu[1].numpy()
	target_mask = y_cpu[2].numpy()
	pred_mask_prob = torch.sigmoid(pred_cpu[2]).numpy()
	pred_mask_binary = (pred_mask_prob > float(threshold)).astype(np.float32)
	target_energy_mw = safe_expm1_log_energy(y_cpu[3], max_log=None).numpy()
	pred_energy_mw = safe_expm1_log_energy(pred_cpu[3], max_log=20.0).numpy()

	surface_error = np.abs(pred_surface - target_surface)
	canopy_error = np.abs(pred_canopy - target_canopy)
	mask_error = np.abs(pred_mask_binary - (target_mask > 0.5).astype(np.float32))
	energy_error = np.abs(pred_energy_mw - target_energy_mw)

	fuel_vmax = _robust_vmax(target_surface, pred_surface, target_canopy, pred_canopy)
	surface_error_vmax = _robust_vmax(surface_error)
	canopy_error_vmax = _robust_vmax(canopy_error)
	energy_vmax = _robust_vmax(target_energy_mw, pred_energy_mw)
	energy_error_vmax = _robust_vmax(energy_error)

	fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)
	panels = [
		(target_surface, "target surface", 0.0, fuel_vmax, "inferno"),
		(pred_surface, "pred surface", 0.0, fuel_vmax, "inferno"),
		(surface_error, "abs surface error", 0.0, surface_error_vmax, "magma"),
		(target_canopy, "target canopy", 0.0, fuel_vmax, "inferno"),
		(pred_canopy, "pred canopy", 0.0, fuel_vmax, "inferno"),
		(canopy_error, "abs canopy error", 0.0, canopy_error_vmax, "magma"),
		(target_mask, "target mask", 0.0, 1.0, "gray_r"),
		(pred_mask_prob, "pred mask prob", 0.0, 1.0, "viridis"),
		(mask_error, "mask binary error", 0.0, 1.0, "magma"),
		(target_energy_mw, "target energy MW", 0.0, energy_vmax, "inferno"),
		(pred_energy_mw, "pred energy MW safe", 0.0, energy_vmax, "inferno"),
		(energy_error, "abs energy MW error", 0.0, energy_error_vmax, "magma"),
	]
	for ax, (data, title, vmin, vmax, cmap) in zip(axes.flat, panels):
		_imshow_panel(ax, np.asarray(data), title, vmin=vmin, vmax=vmax, cmap=cmap)
	fig.suptitle(f"{title_prefix} | {_metadata_title(metadata)}", fontsize=11)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _hist_values(tensor: torch.Tensor, max_points: int = 500_000) -> np.ndarray:
	values = tensor.detach().float().cpu().reshape(-1)
	values = values[torch.isfinite(values)]
	if values.numel() > max_points:
		indices = torch.linspace(0, values.numel() - 1, steps=max_points).long()
		values = values[indices]
	return values.numpy()


def _save_overlay_histogram(path: Path, target: torch.Tensor, pred: torch.Tensor, title: str, xlabel: str) -> None:
	target_values = _hist_values(target)
	pred_values = _hist_values(pred)
	fig, ax = plt.subplots(figsize=(7, 4))
	if target_values.size:
		ax.hist(target_values, bins=80, alpha=0.55, label="target", density=True)
	if pred_values.size:
		ax.hist(pred_values, bins=80, alpha=0.55, label="pred", density=True)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("density")
	ax.legend()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def _save_single_histogram(path: Path, values_tensor: torch.Tensor, title: str, xlabel: str) -> None:
	values = _hist_values(values_tensor)
	fig, ax = plt.subplots(figsize=(7, 4))
	if values.size:
		ax.hist(values, bins=80, alpha=0.8)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("count")
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def _save_histograms(figures_dir: Path, aggregate_tensors: Mapping[str, torch.Tensor]) -> None:
	_save_overlay_histogram(
		figures_dir / "hist_surface_target_vs_pred.png",
		aggregate_tensors["y_surface"],
		aggregate_tensors["pred_surface"],
		"Surface consumed fuel: target vs prediction",
		"surface consumed fuel",
	)
	_save_overlay_histogram(
		figures_dir / "hist_canopy_target_vs_pred.png",
		aggregate_tensors["y_canopy"],
		aggregate_tensors["pred_canopy"],
		"Canopy consumed fuel: target vs prediction",
		"canopy consumed fuel",
	)
	_save_overlay_histogram(
		figures_dir / "hist_energy_log_target_vs_pred.png",
		aggregate_tensors["y_energy_log"],
		aggregate_tensors["pred_energy_log"],
		"Energy log1p: target vs prediction",
		"log1p energy",
	)
	_save_overlay_histogram(
		figures_dir / "hist_energy_mw_target_vs_pred.png",
		aggregate_tensors["y_energy_mw"],
		aggregate_tensors["pred_energy_mw_safe"],
		"Energy MW: target vs safe prediction",
		"energy MW",
	)
	_save_single_histogram(figures_dir / "hist_mask_prob.png", aggregate_tensors["pred_mask_prob"], "Predicted mask probabilities", "probability")


def _stats_rows(scope: str, tensors: Mapping[str, torch.Tensor], batch_index: int | None = None) -> list[dict[str, Any]]:
	rows = []
	for name in STAT_TENSOR_NAMES:
		row = tensor_stats(name, tensors[name])
		row["scope"] = scope
		if batch_index is not None:
			row["batch_index"] = int(batch_index)
		rows.append(row)
	return rows


def flatten_stats_to_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
	return [dict(row) for row in rows]


def _metric_summary(metrics_by_batch: Sequence[Mapping[str, Any]], aggregate_metrics: Mapping[str, float]) -> dict[str, Any]:
	summary: dict[str, Any] = {"aggregate": dict(aggregate_metrics), "num_batches": len(metrics_by_batch)}
	for field in METRIC_FIELDS:
		values = []
		for row in metrics_by_batch:
			try:
				value = float(row.get(field, math.nan))
			except (TypeError, ValueError):
				value = math.nan
			if math.isfinite(value):
				values.append(value)
		if values:
			array = np.asarray(values, dtype=np.float64)
			summary[field] = {
				"mean_by_batch": float(array.mean()),
				"min_by_batch": float(array.min()),
				"max_by_batch": float(array.max()),
			}
		else:
			summary[field] = {"mean_by_batch": math.nan, "min_by_batch": math.nan, "max_by_batch": math.nan}
	return summary


def _is_much_larger(pred_mean: float, target_mean: float) -> bool:
	if not math.isfinite(pred_mean) or not math.isfinite(target_mean):
		return False
	if target_mean <= 1.0e-8:
		return pred_mean > 1.0e-3
	return pred_mean > max(10.0 * target_mean, target_mean + 1.0e-5)


def _diagnostic_warnings(
	aggregate_stats: Sequence[Mapping[str, Any]],
	aggregate_tensors: Mapping[str, torch.Tensor],
	aggregate_metrics: Mapping[str, float],
	threshold: float,
) -> list[str]:
	warnings: list[str] = []
	if _is_much_larger(_mean_from_stats(aggregate_stats, "pred_surface"), _mean_from_stats(aggregate_stats, "y_surface")):
		warnings.append("WARNING: predicted surface consumed fuel mean is much larger than target mean.")
	if _is_much_larger(_mean_from_stats(aggregate_stats, "pred_canopy"), _mean_from_stats(aggregate_stats, "y_canopy")):
		warnings.append("WARNING: predicted canopy consumed fuel mean is much larger than target mean.")
	if _max_from_stats(aggregate_stats, "pred_energy_log") > 20.0:
		warnings.append("WARNING: pred energy_log max > 20; expm1 energy may explode.")
	if _is_much_larger(_mean_from_stats(aggregate_stats, "pred_energy_mw_safe"), _mean_from_stats(aggregate_stats, "y_energy_mw")):
		warnings.append("WARNING: predicted safe energy MW mean is much larger than target mean.")

	mask_prob = aggregate_tensors["pred_mask_prob"]
	mask_positive_fraction = float((mask_prob > float(threshold)).float().mean().item())
	if mask_positive_fraction >= 0.99:
		warnings.append("WARNING: mask probabilities are nearly all above threshold.")
	elif mask_positive_fraction <= 0.01:
		warnings.append("WARNING: mask probabilities are nearly all below threshold.")

	pred_tensor = aggregate_tensors["pred_surface"]
	for name in ("pred_canopy", "pred_mask_logits", "pred_energy_log"):
		pred_tensor = torch.cat([pred_tensor.reshape(-1), aggregate_tensors[name].reshape(-1)])
	if bool(torch.isnan(pred_tensor).any().item()) or bool(torch.isinf(pred_tensor).any().item()):
		warnings.append("WARNING: prediction contains NaN or Inf values.")
	target_tensor = aggregate_tensors["y_surface"]
	for name in ("y_canopy", "y_mask", "y_energy_log"):
		target_tensor = torch.cat([target_tensor.reshape(-1), aggregate_tensors[name].reshape(-1)])
	if bool(torch.isnan(target_tensor).any().item()) or bool(torch.isinf(target_tensor).any().item()):
		warnings.append("WARNING: target contains NaN or Inf values.")
	if not math.isfinite(float(aggregate_metrics.get("active_energy_mw_mae_safe", math.nan))):
		warnings.append("WARNING: no active pixels were found for active_energy_mw_mae_safe.")
	if _min_from_stats(aggregate_stats, "pred_surface") < 0 or _min_from_stats(aggregate_stats, "pred_canopy") < 0 or _min_from_stats(aggregate_stats, "pred_energy_log") < 0:
		warnings.append("WARNING: one or more constrained output channels contain negative values.")
	return warnings


def _normalization_debug(config: Mapping[str, Any], loader, input_channels: int, split: str) -> dict[str, Any]:
	dataset = getattr(loader, "dataset", None)
	stats = getattr(dataset, "normalization_stats", None)
	normalization_shapes: dict[str, Any] = {}
	channel_count_matches = None
	if isinstance(stats, Mapping):
		for key in ("mean", "std"):
			if key in stats:
				array = np.asarray(stats[key])
				normalization_shapes[f"{key}_shape"] = list(array.shape)
		if "mean" in stats and "std" in stats:
			mean_count = int(np.asarray(stats["mean"]).reshape(-1).shape[0])
			std_count = int(np.asarray(stats["std"]).reshape(-1).shape[0])
			channel_count_matches = bool(mean_count >= int(input_channels) and std_count >= int(input_channels))

	normalization_stats_path = _resolve_existing_normalization_stats_path(config)
	cache_manifest_path = None
	try:
		manifest_path = get_patch_cache_dir(config) / MANIFEST_FILENAME
		if manifest_path.exists():
			cache_manifest_path = str(manifest_path)
	except Exception:
		cache_manifest_path = None

	patching = resolve_patching_config(config)
	model_config = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
	return {
		"input_normalization_device": _input_normalization_status(loader),
		"normalization_stats_path": str(normalization_stats_path) if normalization_stats_path is not None else None,
		"normalization_stats_shapes": normalization_shapes,
		"normalization_channel_count_matches_input_channels": channel_count_matches,
		"cache_manifest_path": cache_manifest_path,
		"patch_size": patching.get("patch_size"),
		"input_sequence_length": config.get("input_sequence_length"),
		"prediction_horizon": config.get("prediction_horizon"),
		"input_channels": int(input_channels),
		"output_channels": int(model_config.get("output_channels", 4)),
		"split": str(split),
		"dataloader": _loader_summary(loader),
	}


def _summary_text(
	output_dir: Path,
	checkpoint_metadata: Mapping[str, Any],
	config_debug: Mapping[str, Any],
	aggregate_stats: Sequence[Mapping[str, Any]],
	metrics_summary: Mapping[str, Any],
	warnings: Sequence[str],
) -> str:
	lines = [
		"Model Prediction Debug Summary",
		"==============================",
		f"Output directory: {output_dir}",
		f"Checkpoint: {checkpoint_metadata.get('checkpoint_path')}",
		f"Requested architecture: {checkpoint_metadata.get('requested_architecture')}",
		f"Checkpoint architecture: {checkpoint_metadata.get('architecture')}",
		f"Run name: {checkpoint_metadata.get('run_name')}",
		f"Epoch: {checkpoint_metadata.get('epoch')}",
		f"Best epoch: {checkpoint_metadata.get('best_epoch')}",
		f"Best metric: {checkpoint_metadata.get('best_metric')}",
		"",
		"Config / normalization",
		"----------------------",
	]
	for key, value in config_debug.items():
		lines.append(f"{key}: {_to_jsonable(value)}")

	lines.extend(["", "Aggregate metrics", "-----------------"])
	aggregate_metrics = metrics_summary.get("aggregate", {}) if isinstance(metrics_summary.get("aggregate"), Mapping) else {}
	for field in METRIC_FIELDS:
		lines.append(f"{field}: {_format_float(aggregate_metrics.get(field, math.nan))}")

	lines.extend(["", "Aggregate channel stats", "-----------------------"])
	for name in STAT_TENSOR_NAMES:
		mn = _mean_from_stats(aggregate_stats, name)
		mx = _max_from_stats(aggregate_stats, name)
		mi = _min_from_stats(aggregate_stats, name)
		lines.append(f"{name}: min={_format_float(mi)} mean={_format_float(mn)} max={_format_float(mx)}")

	pred_surface_min = _min_from_stats(aggregate_stats, "pred_surface")
	pred_canopy_min = _min_from_stats(aggregate_stats, "pred_canopy")
	pred_energy_log_min = _min_from_stats(aggregate_stats, "pred_energy_log")
	any_negative = bool(pred_surface_min < 0 or pred_canopy_min < 0 or pred_energy_log_min < 0)
	lines.extend(
		[
			"",
			"Output constraint check",
			"-----------------------",
			f"pred_surface min: {_format_float(pred_surface_min)}",
			f"pred_canopy min: {_format_float(pred_canopy_min)}",
			f"pred_energy_log min: {_format_float(pred_energy_log_min)}",
			f"any constrained channels negative: {any_negative}",
		]
	)

	lines.extend(["", "Warnings", "--------"])
	if warnings:
		lines.extend(warnings)
	else:
		lines.append("No compact interpretation warnings triggered.")
	return "\n".join(lines) + "\n"


def _default_output_dir(architecture: str) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return PROJECT_ROOT / "artifacts" / "debug_predictions" / str(architecture).lower() / timestamp


def _print_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
	print("Checkpoint metadata")
	print("-------------------")
	print(f"path: {metadata.get('checkpoint_path')}")
	print(f"keys: {metadata.get('checkpoint_keys')}")
	for key in (
		"architecture",
		"requested_architecture",
		"run_name",
		"epoch",
		"best_epoch",
		"best_metric",
		"global_step",
		"input_sequence_length",
		"prediction_horizon",
		"target_offset_from_start",
		"target_offset_from_last_input",
		"target_definition_version",
	):
		print(f"{key}: {metadata.get(key)}")


def _print_stats(scope: str, rows: Sequence[Mapping[str, Any]]) -> None:
	print(f"\n{scope} channel stats")
	print("-" * (len(scope) + 14))
	for row in rows:
		print(
			f"{row.get('name'):<26} "
			f"min={_format_float(row.get('min')):>12} "
			f"mean={_format_float(row.get('mean')):>12} "
			f"max={_format_float(row.get('max')):>12} "
			f"nan={_format_float(row.get('frac_nan')):>10} "
			f"inf={_format_float(row.get('frac_inf')):>10}"
		)


def main() -> None:
	args = build_argument_parser().parse_args()
	if int(args.num_batches) <= 0:
		raise ValueError("--num_batches must be positive.")
	if int(args.num_samples_to_plot) < 0:
		raise ValueError("--num_samples_to_plot must be nonnegative.")

	output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(args.model_architecture)
	figures_dir = output_dir / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)

	config = _ensure_config_path(load_config(args.config), args.config)
	config = _apply_architecture_override(config, args.model_architecture)
	config = _apply_batch_size_override(config, args.batch_size)
	config["return_metadata"] = True

	device = _resolve_device(config, args.device)
	checkpoint_path = Path(args.checkpoint).expanduser().resolve()
	checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
	checkpoint_metadata = checkpoint_metadata_payload(
		checkpoint,
		checkpoint_path,
		requested_architecture=args.model_architecture,
		allow_architecture_mismatch=bool(args.allow_architecture_mismatch),
	)
	_validate_checkpoint_sequence(
		checkpoint,
		config,
		checkpoint_path,
		allow_sequence_mismatch=bool(args.allow_sequence_mismatch),
	)
	_print_checkpoint_metadata(checkpoint_metadata)
	save_json(output_dir / "checkpoint_metadata.json", checkpoint_metadata)

	with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
		yaml.safe_dump(_to_jsonable(config), handle, sort_keys=False)

	train_loader, val_loader, test_loader = create_dataloaders(config)
	selected_loader = _select_loader(train_loader, val_loader, test_loader, args.split)
	if len(selected_loader.dataset) == 0:
		raise ValueError(f"Selected split {args.split!r} is empty.")

	input_channels = _infer_input_channels_from_loader(train_loader)
	model = build_model_from_config(config, input_channels=input_channels).to(device)
	validate_checkpoint_model_compatibility(model, checkpoint, checkpoint_path)
	model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
	model.eval()
	amp_dtype = choose_amp_dtype(config, device)
	normalizer = _build_input_normalizer(selected_loader, device, input_channels)
	config_debug = _normalization_debug(config, selected_loader, input_channels, args.split)

	print("\nConfig / normalization debug")
	print("----------------------------")
	for key, value in config_debug.items():
		print(f"{key}: {_to_jsonable(value)}")

	stats_rows: list[dict[str, Any]] = []
	metrics_by_batch: list[dict[str, Any]] = []
	pred_batches: list[torch.Tensor] = []
	y_batches: list[torch.Tensor] = []
	aux_summaries: list[dict[str, Any]] = []
	metadata_rows: list[dict[str, Any]] = []
	plotted_count = 0

	with torch.inference_mode():
		for batch_index, batch in enumerate(selected_loader):
			if batch_index >= int(args.num_batches):
				break
			if not isinstance(batch, (tuple, list)) or len(batch) < 2:
				raise TypeError("Expected DataLoader batches with at least input and target tensors.")
			x_batch = batch[0].to(device, non_blocking=True)
			y_batch = batch[1].to(device, non_blocking=True)
			x_batch = _apply_input_normalizer(x_batch, normalizer)
			with autocast_context(device, amp_dtype):
				model_output = model(x_batch)
			pred, aux = _extract_prediction_and_aux(model_output)
			pred = pred.float()
			y_batch = y_batch.float()
			_validate_shapes(x_batch, y_batch, pred)

			batch_tensors = _prediction_tensors(pred.detach(), y_batch.detach())
			batch_stats = _stats_rows("batch", batch_tensors, batch_index=batch_index)
			stats_rows.extend(batch_stats)
			_print_stats(f"batch {batch_index}", batch_stats)

			metrics = compute_debug_metrics(
				pred.detach(),
				y_batch.detach(),
				threshold=float(args.threshold),
				energy_active_threshold_mw=float(args.energy_active_threshold_mw),
				consumed_active_threshold=float(args.consumed_active_threshold),
			)
			metrics_row: dict[str, Any] = {"batch_index": int(batch_index), "batch_size": int(pred.shape[0])}
			metrics_row.update(metrics)
			metrics_by_batch.append(metrics_row)
			if not math.isfinite(metrics["active_energy_mw_mae_safe"]):
				print(f"WARNING: batch {batch_index} has no active pixels for active_energy_mw_mae_safe.")

			aux_summaries.append({"batch_index": int(batch_index), "aux": aux})
			pred_cpu = pred.detach().cpu()
			y_cpu = y_batch.detach().cpu()
			pred_batches.append(pred_cpu)
			y_batches.append(y_cpu)

			items = _metadata_items(batch, int(pred.shape[0]))
			for sample_index, metadata in enumerate(items):
				metadata_rows.append(
					{
						"batch_index": int(batch_index),
						"sample_index_in_batch": int(sample_index),
						**_metadata_csv_fields(metadata),
					}
				)
				if plotted_count >= int(args.num_samples_to_plot):
					continue
				figure_name = f"sample_{plotted_count:03d}_batch{batch_index}_idx{sample_index}.png"
				plot_sample_debug(
					figures_dir / figure_name,
					pred_cpu[sample_index],
					y_cpu[sample_index],
					metadata=metadata,
					title_prefix=f"{args.model_architecture} | split={args.split}",
					threshold=float(args.threshold),
				)
				if bool(args.save_npz):
					np.savez_compressed(
						figures_dir / f"sample_{plotted_count:03d}_batch{batch_index}_idx{sample_index}.npz",
						pred=pred_cpu[sample_index].numpy(),
						y=y_cpu[sample_index].numpy(),
						metadata=json.dumps(_to_jsonable(metadata), sort_keys=True),
					)
				plotted_count += 1

	if not pred_batches:
		raise ValueError(f"Selected split {args.split!r} produced no batches.")

	pred_all = torch.cat(pred_batches, dim=0)
	y_all = torch.cat(y_batches, dim=0)
	aggregate_tensors = _prediction_tensors(pred_all, y_all)
	aggregate_stats = _stats_rows("aggregate", aggregate_tensors)
	stats_rows.extend(aggregate_stats)
	_print_stats("aggregate", aggregate_stats)

	aggregate_metrics = compute_debug_metrics(
		pred_all,
		y_all,
		threshold=float(args.threshold),
		energy_active_threshold_mw=float(args.energy_active_threshold_mw),
		consumed_active_threshold=float(args.consumed_active_threshold),
	)
	metrics_summary = _metric_summary(metrics_by_batch, aggregate_metrics)
	warnings = _diagnostic_warnings(aggregate_stats, aggregate_tensors, aggregate_metrics, threshold=float(args.threshold))
	for warning in warnings:
		print(warning)

	_write_csv(
		output_dir / "channel_stats.csv",
		flatten_stats_to_csv(stats_rows),
		fieldnames=["scope", "batch_index", "name", "numel", *STATS_FIELDS],
	)
	_write_csv(output_dir / "metrics_by_batch.csv", metrics_by_batch, fieldnames=["batch_index", "batch_size", *METRIC_FIELDS])
	_write_csv(output_dir / "sample_metadata.csv", metadata_rows)
	save_json(output_dir / "metrics_summary.json", metrics_summary)
	save_json(output_dir / "aux_summaries.json", aux_summaries)
	_save_histograms(figures_dir, aggregate_tensors)

	summary = _summary_text(output_dir, checkpoint_metadata, config_debug, aggregate_stats, metrics_summary, warnings)
	(output_dir / "diagnostics_summary.txt").write_text(summary, encoding="utf-8")
	print("\n" + summary)


if __name__ == "__main__":
	main()
