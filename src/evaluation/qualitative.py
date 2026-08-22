"""Qualitative evaluation helpers for wildfire forecast samples."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

try:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
	plt = None


QUANTITIES = (
	("surface", "Surface", "viridis"),
	("canopy", "Canopy", "viridis"),
	("mask", "Mask", "magma"),
	("energy_log", "Energy Log", "inferno"),
)


def select_qualitative_sample_indices(dataset_length: int, num_samples: int, seed: int) -> list[int]:
	"""Return reproducible unique dataset indices for qualitative inspection."""

	length = int(dataset_length)
	requested = int(num_samples)
	if length <= 0:
		raise ValueError("Selected split is empty; qualitative evaluation needs at least one sample.")
	if requested <= 0:
		raise ValueError("--num_samples must be positive for qualitative evaluation.")
	count = min(requested, length)
	return [int(index) for index in random.Random(int(seed)).sample(range(length), count)]


def _jsonable(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(key): _jsonable(nested) for key, nested in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return [_jsonable(item) for item in value.tolist()]
	if isinstance(value, np.generic):
		return value.item()
	return value


def _tensor_to_numpy(value: Any) -> np.ndarray:
	if hasattr(value, "detach"):
		value = value.detach().cpu().numpy()
	return np.asarray(value, dtype=np.float32)


def coerce_chw4(value: Any) -> np.ndarray:
	"""Coerce a prediction or target to the first four ``(C, H, W)`` channels."""

	array = _tensor_to_numpy(value)
	if array.ndim == 4 and int(array.shape[0]) == 1:
		array = array[0]
	if array.ndim == 3 and int(array.shape[0]) >= 4:
		return np.asarray(array[:4], dtype=np.float32)
	if array.ndim == 3 and int(array.shape[-1]) >= 4:
		return np.asarray(np.moveaxis(array[..., :4], -1, 0), dtype=np.float32)
	raise ValueError(f"Expected an array with at least four channels, got shape {tuple(array.shape)}.")


def sigmoid_logits(logits: Any) -> np.ndarray:
	values = np.clip(np.asarray(logits, dtype=np.float32), -60.0, 60.0)
	return (1.0 / (1.0 + np.exp(-values))).astype(np.float32, copy=False)


def display_maps(chw4: Any, *, target: bool = False) -> dict[str, np.ndarray]:
	array = coerce_chw4(chw4)
	mask = np.clip(array[2], 0.0, 1.0) if target else sigmoid_logits(array[2])
	return {
		"surface": np.asarray(array[0], dtype=np.float32),
		"canopy": np.asarray(array[1], dtype=np.float32),
		"mask": np.asarray(mask, dtype=np.float32),
		"energy_log": np.asarray(array[3], dtype=np.float32),
	}


def _patch_from_metadata(metadata: Mapping[str, Any]) -> dict[str, int] | None:
	patch = metadata.get("patch")
	if isinstance(patch, Mapping):
		keys = ("y0", "y1", "x0", "x1")
		if all(patch.get(key) is not None for key in keys):
			return {key: int(patch[key]) for key in keys}
	required = ("patch_top", "patch_left", "patch_bottom", "patch_right")
	if all(metadata.get(key) is not None for key in required):
		return {
			"y0": int(metadata["patch_top"]),
			"y1": int(metadata["patch_bottom"]),
			"x0": int(metadata["patch_left"]),
			"x1": int(metadata["patch_right"]),
		}
	if metadata.get("patch_top") is not None and metadata.get("patch_left") is not None and metadata.get("patch_size") is not None:
		y0 = int(metadata["patch_top"])
		x0 = int(metadata["patch_left"])
		size = int(metadata["patch_size"])
		return {"y0": y0, "y1": y0 + size, "x0": x0, "x1": x0 + size}
	return None


def selected_sample_record(
	*,
	sample_number: int,
	dataset_index: int,
	metadata: Mapping[str, Any],
	split: str,
	input_sequence_length: int,
	prediction_horizon: int,
) -> dict[str, Any]:
	fire_name = metadata.get("fire_name", metadata.get("dataset_name"))
	local_sample_index = metadata.get("sample_index", metadata.get("local_sample_index", metadata.get("start_idx")))
	input_indices = metadata.get("original_input_indices", metadata.get("input_indices"))
	record = {
		"sample_number": int(sample_number),
		"dataset_index": int(dataset_index),
		"fire_name": None if fire_name in (None, "") else str(fire_name),
		"local_sample_index": _jsonable(local_sample_index),
		"original_input_indices": _jsonable(input_indices),
		"input_indices": _jsonable(input_indices),
		"last_input_index": _jsonable(metadata.get("last_input_idx", metadata.get("current_idx", metadata.get("current_index")))),
		"target_index": _jsonable(metadata.get("target_idx", metadata.get("future_idx", metadata.get("future_index")))),
		"patch_coords": _patch_from_metadata(metadata),
		"split": str(split).lower(),
		"input_sequence_length": int(metadata.get("input_sequence_length", input_sequence_length)),
		"prediction_horizon": int(metadata.get("prediction_horizon", prediction_horizon)),
	}
	for optional_key in ("dataset_name", "dataset_id", "cache_shard_path", "cache_local_index", "start_idx"):
		if optional_key in metadata:
			record[optional_key] = _jsonable(metadata[optional_key])
	return record


def load_qualitative_samples(
	dataset: Any,
	indices: Sequence[int],
	*,
	split: str,
	input_sequence_length: int,
	prediction_horizon: int,
) -> list[dict[str, Any]]:
	samples: list[dict[str, Any]] = []
	for sample_number, dataset_index in enumerate(indices):
		item = dataset[int(dataset_index)]
		if isinstance(item, Mapping):
			if "y" not in item:
				raise KeyError("Qualitative mapping dataset items must contain a 'y' target tensor.")
			metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), Mapping) else {}
			target_source = item["y"]
		elif isinstance(item, (tuple, list)) and len(item) >= 2:
			metadata = dict(item[2]) if len(item) >= 3 and isinstance(item[2], Mapping) else {}
			target_source = item[1]
		else:
			raise TypeError("Qualitative dataset items must contain at least input and target tensors or mapping keys 'x'/'y'.")
		target = coerce_chw4(target_source)
		record = selected_sample_record(
			sample_number=sample_number,
			dataset_index=int(dataset_index),
			metadata=metadata,
			split=split,
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
		)
		samples.append(
			{
				"sample_number": int(sample_number),
				"dataset_index": int(dataset_index),
				"target": target,
				"metadata": metadata,
				"record": record,
			}
		)
	return samples


def _finite_percentile(values: Sequence[np.ndarray], percentile: float = 99.0, fallback: float = 1.0) -> float:
	finite_parts = [np.asarray(value, dtype=np.float32)[np.isfinite(value)] for value in values]
	finite_parts = [part for part in finite_parts if part.size > 0]
	if not finite_parts:
		return float(fallback)
	merged = np.concatenate(finite_parts)
	if merged.size == 0:
		return float(fallback)
	value = float(np.nanpercentile(merged, percentile))
	if not np.isfinite(value) or value <= 0.0:
		return float(fallback)
	return value


def _sample_title(sample_record: Mapping[str, Any]) -> str:
	fire = sample_record.get("fire_name") or sample_record.get("dataset_name") or "unknown fire"
	local_sample = sample_record.get("local_sample_index")
	target_index = sample_record.get("target_index")
	return (
		f"{fire} | dataset index {sample_record.get('dataset_index')} | "
		f"sample {local_sample} | target {target_index}"
	)


def _imshow(ax: Any, data: np.ndarray, title: str, *, cmap: str, vmin: float, vmax: float) -> None:
	clean = np.asarray(data, dtype=np.float32)
	clean = np.where(np.isfinite(clean), clean, 0.0)
	ax.imshow(clean, cmap=cmap, vmin=float(vmin), vmax=float(vmax), interpolation="nearest")
	ax.set_title(title, fontsize=9)
	ax.axis("off")


def _save_single_model_figure(
	*,
	target: np.ndarray,
	prediction: np.ndarray,
	model: Mapping[str, Any],
	sample_record: Mapping[str, Any],
	output_path: Path,
	dpi: int,
) -> None:
	target_maps = display_maps(target, target=True)
	prediction_maps = display_maps(prediction, target=False)
	figure, axes = plt.subplots(4, 3, figsize=(10.5, 12.0), squeeze=False)
	for row_index, (key, label, cmap) in enumerate(QUANTITIES):
		value_vmax = _finite_percentile([target_maps[key], prediction_maps[key]])
		error = np.abs(prediction_maps[key] - target_maps[key])
		error_vmax = _finite_percentile([error])
		_imshow(axes[row_index, 0], target_maps[key], f"GT {label}", cmap=cmap, vmin=0.0, vmax=value_vmax)
		_imshow(axes[row_index, 1], prediction_maps[key], f"Prediction {label}", cmap=cmap, vmin=0.0, vmax=value_vmax)
		_imshow(axes[row_index, 2], error, f"Abs Error {label}", cmap="magma", vmin=0.0, vmax=error_vmax)
	figure.suptitle(f"{model.get('display_name', model.get('key', 'model'))} | {_sample_title(sample_record)}", fontsize=12)
	figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=int(dpi))
	plt.close(figure)


def _save_multi_model_figure(
	*,
	target: np.ndarray,
	predictions: Mapping[str, np.ndarray],
	models: Sequence[Mapping[str, Any]],
	sample_record: Mapping[str, Any],
	output_path: Path,
	dpi: int,
) -> None:
	target_maps = display_maps(target, target=True)
	prediction_maps = {str(key): display_maps(value, target=False) for key, value in predictions.items()}
	successful_models = [model for model in models if str(model.get("key")) in prediction_maps]
	row_count = 1 + len(successful_models)
	figure, axes = plt.subplots(row_count, 4, figsize=(14.0, max(3.2, 2.25 * row_count)), squeeze=False)
	scales: dict[str, float] = {}
	for key, _label, _cmap in QUANTITIES:
		values = [target_maps[key], *[prediction_maps[str(model.get("key"))][key] for model in successful_models]]
		scales[key] = _finite_percentile(values)
	for col_index, (key, label, cmap) in enumerate(QUANTITIES):
		_imshow(axes[0, col_index], target_maps[key], f"GT {label}", cmap=cmap, vmin=0.0, vmax=scales[key])
	for row_index, model in enumerate(successful_models, start=1):
		model_key = str(model.get("key"))
		model_maps = prediction_maps[model_key]
		for col_index, (key, label, cmap) in enumerate(QUANTITIES):
			_imshow(
				axes[row_index, col_index],
				model_maps[key],
				f"{model.get('display_name', model_key)} {label}",
				cmap=cmap,
				vmin=0.0,
				vmax=scales[key],
			)
	figure.suptitle(_sample_title(sample_record), fontsize=12)
	figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=int(dpi))
	plt.close(figure)


def save_qualitative_summary_image(
	*,
	target: Any,
	predictions: Mapping[str, Any],
	models: Sequence[Mapping[str, Any]],
	sample_record: Mapping[str, Any],
	output_path: str | Path,
	dpi: int = 180,
) -> Path:
	"""Save one qualitative summary image for one selected sample."""

	if plt is None:
		raise ImportError("matplotlib is required for qualitative evaluation plots.")
	path = Path(output_path)
	chw_target = coerce_chw4(target)
	chw_predictions = {str(key): coerce_chw4(value) for key, value in predictions.items()}
	successful_models = [model for model in models if str(model.get("key")) in chw_predictions]
	if not successful_models:
		raise ValueError("No successful model predictions are available for qualitative plotting.")
	if len(successful_models) == 1:
		model_key = str(successful_models[0].get("key"))
		_save_single_model_figure(
			target=chw_target,
			prediction=chw_predictions[model_key],
			model=successful_models[0],
			sample_record=sample_record,
			output_path=path,
			dpi=dpi,
		)
	else:
		_save_multi_model_figure(
			target=chw_target,
			predictions=chw_predictions,
			models=successful_models,
			sample_record=sample_record,
			output_path=path,
			dpi=dpi,
		)
	return path


def save_individual_model_images(
	*,
	target: Any,
	predictions: Mapping[str, Any],
	models: Sequence[Mapping[str, Any]],
	sample_record: Mapping[str, Any],
	output_root: str | Path,
	output_format: str,
	dpi: int = 180,
) -> list[Path]:
	paths: list[Path] = []
	root = Path(output_root)
	for model in models:
		model_key = str(model.get("key"))
		if model_key not in predictions:
			continue
		path = root / model_key / f"sample_{int(sample_record.get('sample_number', 0)):03d}.{output_format}"
		_save_single_model_figure(
			target=coerce_chw4(target),
			prediction=coerce_chw4(predictions[model_key]),
			model=model,
			sample_record=sample_record,
			output_path=path,
			dpi=dpi,
		)
		paths.append(path)
	return paths
