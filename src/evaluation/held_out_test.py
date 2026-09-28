"""Locked, complete held-out test evaluation shared by all Table 2 models."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from src.data.dataset import metadata_batch_to_list
from src.evaluation.fire_activity import (
	ACTIVE_FRACTION_BIN_NAMES,
	FIRE_MASK_CHANNEL,
	PREDICTED_FIRE_THRESHOLD,
	active_fraction_bin_index,
	classify_fire_masks,
)
from src.evaluation.full_validation import FullValidationAccumulator, _sample_metric_rows
from src.evaluation.validation_subset import ensure_qualitative_validation_samples
from src.training.batch_utils import unpack_batch
from src.training.hardware import autocast_context
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.model_outputs import extract_prediction


DEFAULT_QUALITATIVE_TEST_INDEX = Path(
	"artifacts/table2_baselines/shared/qualitative_test_samples.json"
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	columns: list[str] = []
	for row in rows:
		for key in row:
			if key not in columns:
				columns.append(str(key))
	if not columns:
		raise ValueError(f"Cannot write empty CSV schema: {path}")
	temporary = path.with_suffix(path.suffix + ".tmp")
	with temporary.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)
	temporary.replace(path)


def _test_metrics(accumulator: FullValidationAccumulator) -> dict[str, Any]:
	metrics = accumulator.finalize()
	converted = {
		("test_" + key[len("full_val_") :]) if key.startswith("full_val_") else key: value
		for key, value in metrics.items()
	}
	converted["test_f1"] = converted.get("test_dice")
	converted["test_mask_f1"] = converted.get("test_dice")
	return converted


def _per_fire_row(
	model_name: str,
	seed: int | None,
	fire_name: str,
	accumulator: FullValidationAccumulator,
) -> dict[str, Any]:
	metrics = _test_metrics(accumulator)
	return {
		"model": model_name,
		"model_name": model_name,
		"seed": seed,
		"fire_name": fire_name,
		"sample_count": metrics["test_total_patch_count"],
		"fire_patch_count": metrics["test_fire_patch_count"],
		"no_fire_patch_count": metrics["test_no_fire_patch_count"],
		"dice": metrics["test_dice"],
		"iou": metrics["test_iou"],
		"precision": metrics["test_precision"],
		"recall": metrics["test_recall"],
		"f1": metrics["test_f1"],
		"surface_mae": metrics["test_surface_mae"],
		"surface_rmse": metrics["test_surface_rmse"],
		"canopy_mae": metrics["test_canopy_mae"],
		"canopy_rmse": metrics["test_canopy_rmse"],
		"energy_log_mae": metrics["test_energy_log_mae"],
		"energy_log_rmse": metrics["test_energy_log_rmse"],
		"energy_mw_mae": metrics["test_energy_mw_mae"],
		"energy_mw_rmse": metrics["test_energy_mw_rmse"],
		"active_canopy_mae": metrics["test_active_canopy_mae"],
		"no_fire_surface_mae": metrics["test_no_fire_surface_mae"],
		"no_fire_canopy_mae": metrics["test_no_fire_canopy_mae"],
		"no_fire_energy_log_mae": metrics["test_no_fire_energy_log_mae"],
		"no_fire_surface_signed_mean": metrics["test_no_fire_surface_pred_mean"],
		"no_fire_surface_abs_mean": metrics["test_no_fire_surface_abs_pred_mean"],
		"no_fire_canopy_signed_mean": metrics["test_no_fire_canopy_pred_mean"],
		"no_fire_canopy_abs_mean": metrics["test_no_fire_canopy_abs_pred_mean"],
		"no_fire_energy_log_signed_mean": metrics["test_no_fire_energy_log_pred_mean"],
		"no_fire_energy_log_abs_mean": metrics["test_no_fire_energy_log_abs_pred_mean"],
		"no_fire_surface_rmse": metrics["test_no_fire_surface_pred_rmse"],
		"no_fire_canopy_rmse": metrics["test_no_fire_canopy_pred_rmse"],
		"no_fire_energy_log_rmse": metrics["test_no_fire_energy_log_pred_rmse"],
		"no_fire_pixel_fp_rate": metrics["test_no_fire_mask_false_positive_rate"],
		"no_fire_patch_fp_rate": metrics["test_no_fire_patch_false_positive_rate"],
		"no_fire_mean_mask_probability": metrics["test_no_fire_mask_prob_mean"],
	}


def _save_sample_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, str]:
	"""Save sample metrics with a dependency-free CSV fallback.

	Parquet is emitted when pandas plus a Parquet engine is installed.  The
	canonical environment currently has neither pyarrow nor fastparquet, so CSV
	is always retained as the portable source of truth.
	"""

	csv_path = path.with_suffix(".csv")
	_atomic_csv(csv_path, rows)
	try:
		import pandas as pd  # type: ignore[import-not-found]
		frame = pd.DataFrame(rows)
		frame.to_parquet(path, index=False)
		return path, "parquet"
	except (ImportError, ModuleNotFoundError):
		return csv_path, "csv"


def _model_call(model: torch.nn.Module, inputs: torch.Tensor, terrain: torch.Tensor | None):
	return model(inputs) if terrain is None else model(inputs, terrain=terrain)


def evaluate_locked_test(
	*,
	test_loader: Any,
	config: Mapping[str, Any],
	run_dir: str | Path,
	model_name: str,
	seed: int | None,
	model: torch.nn.Module | None = None,
	deterministic_predictor: Callable[[Sequence[Mapping[str, Any]]], torch.Tensor] | None = None,
	device: torch.device | None = None,
	amp_dtype: Any = None,
	checkpoint_path: str | Path | None = None,
	checkpoint_epoch: int | None = None,
	efficiency_metadata: Mapping[str, Any] | None = None,
	logger: Any = None,
) -> dict[str, Any]:
	"""Evaluate every held-out test patch exactly once and persist Table 2 data."""

	if (model is None) == (deterministic_predictor is None):
		raise ValueError("Provide exactly one of model or deterministic_predictor.")
	if bool(getattr(test_loader, "drop_last", False)):
		raise RuntimeError("Locked test evaluation requires drop_last=false.")
	if not isinstance(getattr(test_loader, "sampler", None), SequentialSampler):
		raise RuntimeError("Locked test evaluation requires sequential, no-shuffle sampling.")
	dataset = test_loader.dataset
	if str(getattr(dataset, "split", "")) != "test":
		raise RuntimeError(f"Locked evaluator requires the test dataset, got split={getattr(dataset, 'split', None)!r}.")
	expected_samples = len(dataset)
	if expected_samples <= 0:
		raise RuntimeError("Held-out test dataset is empty.")
	records = getattr(dataset, "records", None)
	if isinstance(records, Sequence):
		wrong_split = [record.get("sample_id", index) for index, record in enumerate(records) if record.get("split") != "test"]
		if wrong_split:
			raise RuntimeError(f"Test dataset contains non-test records: {wrong_split[:3]}")

	run_path = Path(run_dir).expanduser().resolve()
	evaluation_path = run_path / "evaluation"
	evaluation_path.mkdir(parents=True, exist_ok=True)
	qualitative_config = config.get("table2", {}) if isinstance(config.get("table2"), Mapping) else {}
	qualitative_path = qualitative_config.get("qualitative_test_index_path", DEFAULT_QUALITATIVE_TEST_INDEX)
	qualitative = ensure_qualitative_validation_samples(
		dataset,
		config,
		output_path=qualitative_path,
		per_group=int(qualitative_config.get("qualitative_samples_per_group", 4)),
		seed=int(qualitative_config.get("qualitative_seed", 97531)),
		logger=logger,
	)
	qualitative_by_id = {str(item["sample_id"]): item for item in qualitative["selected_samples"]}

	resolved_device = device or torch.device("cpu")
	if model is not None:
		model.eval()
		normalizer = build_input_normalizer_for_loader(
			test_loader,
			resolved_device,
			int(config.get("model", {}).get("input_channels", 0)),
		)
	else:
		normalizer = None

	overall = FullValidationAccumulator()
	per_fire: dict[str, FullValidationAccumulator] = {}
	per_activity = {name: FullValidationAccumulator() for name in ACTIVE_FRACTION_BIN_NAMES}
	sample_rows: list[dict[str, Any]] = []
	seen_sample_ids: set[str] = set()
	qualitative_arrays: dict[str, dict[str, Any]] = {}
	warmup_batches = min(5, max(0, len(test_loader) - 1))
	timed_seconds = 0.0
	timed_samples = 0
	timed_batches = 0
	if resolved_device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(resolved_device)

	with torch.inference_mode():
		for batch_index, batch in enumerate(test_loader, start=1):
			x_raw, y_raw, extra = unpack_batch(batch)
			y = y_raw.to(resolved_device, non_blocking=True).float()
			metadata_batch = extra.get("metadata")
			metadata_items = metadata_batch_to_list(metadata_batch, batch_size=int(y.shape[0])) if isinstance(metadata_batch, Mapping) else []
			if len(metadata_items) != int(y.shape[0]):
				raise RuntimeError("Locked test evaluation requires metadata for every sample.")
			if resolved_device.type == "cuda" and batch_index > warmup_batches:
				torch.cuda.synchronize(resolved_device)
			start = time.perf_counter() if batch_index > warmup_batches else None
			if model is not None:
				x = apply_input_normalization(x_raw.to(resolved_device, non_blocking=True), normalizer)
				terrain_raw = extra.get("terrain")
				terrain = terrain_raw.to(resolved_device, non_blocking=True) if terrain_raw is not None else None
				with autocast_context(resolved_device, amp_dtype):
					prediction = extract_prediction(_model_call(model, x, terrain)).float()
			else:
				assert deterministic_predictor is not None
				prediction = deterministic_predictor(metadata_items).to(resolved_device).float()
			if prediction.shape != y.shape:
				raise RuntimeError(f"Test prediction shape {tuple(prediction.shape)} does not match target {tuple(y.shape)}.")
			if start is not None:
				if resolved_device.type == "cuda":
					torch.cuda.synchronize(resolved_device)
				timed_seconds += time.perf_counter() - start
				timed_samples += int(y.shape[0])
				timed_batches += 1

			overall.update(prediction, y)
			classification = classify_fire_masks(y[:, FIRE_MASK_CHANNEL])
			grouped_fire: dict[str, list[int]] = defaultdict(list)
			grouped_activity: dict[str, list[int]] = defaultdict(list)
			for item_index, metadata in enumerate(metadata_items):
				sample_id = str(metadata.get("sample_id", item_index))
				if sample_id in seen_sample_ids:
					raise RuntimeError(f"Held-out test sample visited more than once: {sample_id}")
				seen_sample_ids.add(sample_id)
				fire_name = str(metadata.get("fire_name", metadata.get("fire", "unknown")))
				grouped_fire[fire_name].append(item_index)
				fraction = float(classification["active_fraction"][item_index].item())
				grouped_activity[ACTIVE_FRACTION_BIN_NAMES[active_fraction_bin_index(fraction)]].append(item_index)
				if sample_id in qualitative_by_id:
					target_item = y[item_index].detach().cpu().numpy().astype(np.float32, copy=False)
					prediction_item = prediction[item_index].detach().cpu()
					qualitative_arrays[sample_id] = {
						"sample_id": sample_id,
						"fire_name": fire_name,
						"activity_bin": str(qualitative_by_id[sample_id]["activity_bin"]),
						"target": target_item,
						"prediction": prediction_item.numpy().astype(np.float32, copy=False),
						"mask_probability": torch.sigmoid(prediction_item[2]).numpy().astype(np.float32, copy=False),
					}
			for fire_name, indices in grouped_fire.items():
				per_fire.setdefault(fire_name, FullValidationAccumulator()).update(prediction[indices], y[indices])
			for activity_name, indices in grouped_activity.items():
				per_activity[activity_name].update(prediction[indices], y[indices])
			rows = _sample_metric_rows(
				prediction,
				y,
				metadata_items,
				model_name=model_name,
				seed=-1 if seed is None else int(seed),
			)
			for row in rows:
				row["model"] = model_name
				row["seed"] = seed
			sample_rows.extend(rows)
			if logger is not None and (batch_index == 1 or batch_index % 100 == 0 or batch_index == len(test_loader)):
				logger.info("Locked test progress: %s/%s batches", batch_index, len(test_loader))

	metrics = _test_metrics(overall)
	evaluated = int(metrics["test_total_patch_count"])
	if evaluated != expected_samples or len(sample_rows) != expected_samples or len(seen_sample_ids) != expected_samples:
		raise RuntimeError(
			f"Incomplete held-out test evaluation: dataset={expected_samples} evaluated={evaluated} "
			f"sample_rows={len(sample_rows)} unique_ids={len(seen_sample_ids)}."
		)
	if int(metrics["test_fire_patch_count"]) + int(metrics["test_no_fire_patch_count"]) != evaluated:
		raise RuntimeError("Held-out fire/no-fire counts do not sum to total_test_count.")
	per_fire_rows = [
		_per_fire_row(model_name, seed, fire_name, accumulator)
		for fire_name, accumulator in sorted(per_fire.items())
	]
	if sum(int(row["sample_count"]) for row in per_fire_rows) != evaluated:
		raise RuntimeError("Held-out per-fire counts do not sum to total_test_count.")
	activity_payload = {name: _test_metrics(per_activity[name]) for name in ACTIVE_FRACTION_BIN_NAMES}
	if sum(int(values["test_total_patch_count"]) for values in activity_payload.values()) != evaluated:
		raise RuntimeError("Held-out activity-bin counts do not sum to total_test_count.")

	metrics.update(
		{
			"test_inference_time_seconds": timed_seconds,
			"test_inference_ms_per_sample": 1000.0 * timed_seconds / timed_samples if timed_samples else None,
			"test_samples_per_second": timed_samples / timed_seconds if timed_seconds > 0 else None,
			"test_timing_warmup_batches": warmup_batches,
			"test_peak_gpu_memory_mb": (
				float(torch.cuda.max_memory_allocated(resolved_device)) / (1024.0 ** 2)
				if resolved_device.type == "cuda"
				else None
			),
		}
	)
	metrics.update(dict(efficiency_metadata or {}))

	per_fire_path = evaluation_path / "test_per_fire_metrics.csv"
	_atomic_csv(per_fire_path, per_fire_rows)
	sample_path, sample_format = _save_sample_rows(evaluation_path / "test_sample_metrics.parquet", sample_rows)
	selected_ids = [str(item["sample_id"]) for item in qualitative["selected_samples"]]
	missing_qualitative = [sample_id for sample_id in selected_ids if sample_id not in qualitative_arrays]
	if missing_qualitative:
		raise RuntimeError(f"Held-out evaluation missed qualitative samples: {missing_qualitative[:10]}")
	qualitative_path_out = evaluation_path / "qualitative_test_predictions.npz"
	ordered = [qualitative_arrays[sample_id] for sample_id in selected_ids]
	temporary_npz = qualitative_path_out.with_name(qualitative_path_out.name + ".tmp.npz")
	np.savez_compressed(
		temporary_npz,
		sample_id=np.asarray([item["sample_id"] for item in ordered]),
		fire_name=np.asarray([item["fire_name"] for item in ordered]),
		activity_bin=np.asarray([item["activity_bin"] for item in ordered]),
		target=np.stack([item["target"] for item in ordered]),
		prediction=np.stack([item["prediction"] for item in ordered]),
		mask_probability=np.stack([item["mask_probability"] for item in ordered]),
	)
	temporary_npz.replace(qualitative_path_out)

	payload = {
		"schema_version": 1,
		"created_at": datetime.now(timezone.utc).isoformat(),
		"metric_scope": "complete_locked_held_out_test",
		"split": "test",
		"test_used_for_model_selection": False,
		"model": model_name,
		"seed": seed,
		"checkpoint": None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve()),
		"checkpoint_epoch": checkpoint_epoch,
		"dataset_sample_count": expected_samples,
		"evaluated_test_samples": evaluated,
		"unique_sample_id_count": len(seen_sample_ids),
		"fire_count": int(metrics["test_fire_patch_count"]),
		"no_fire_count": int(metrics["test_no_fire_patch_count"]),
		"prediction_mask_semantics": "logits_thresholded_at_sigmoid_0.5",
		"qualitative_test_index": str(Path(qualitative_path).expanduser().resolve()),
		"sample_metrics_path": str(sample_path),
		"sample_metrics_format": sample_format,
		"per_fire_metrics_path": str(per_fire_path),
		"qualitative_predictions_path": str(qualitative_path_out),
		"metrics": metrics,
		"by_activity_bin": activity_payload,
	}
	metrics_path = evaluation_path / "test_metrics.json"
	_atomic_json(metrics_path, payload)
	return {
		"artifact": payload,
		"metrics": metrics,
		"metrics_path": str(metrics_path),
		"per_fire_metrics_path": str(per_fire_path),
		"sample_metrics_path": str(sample_path),
		"qualitative_predictions_path": str(qualitative_path_out),
	}


__all__ = ["DEFAULT_QUALITATIVE_TEST_INDEX", "evaluate_locked_test"]
