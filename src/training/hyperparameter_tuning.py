"""Utilities for CAWFE-Latte hyperparameter tuning."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


LOSS_PARAM_ALIASES = {
	"training.loss.surface_consumed_weight": "multitask.surface_loss_weight",
	"training.loss.canopy_consumed_weight": "multitask.canopy_loss_weight",
	"training.loss.mask_weight": "multitask.segmentation_loss_weight",
	"training.loss.energy_weight": "multitask.energy_loss_weight",
}


DEFAULT_SEARCH_SPACE = {
	"training.learning_rate": [1e-4, 3e-4, 5e-4],
	"training.weight_decay": [1e-5, 1e-4],
	"cawfe_latte.backbone_dim": [64, 96, 128],
	"cawfe_latte.backbone_depths": [[1, 1], [2, 2]],
	"cawfe_latte.fire_gate_strength": [0.5, 1.0, 1.5],
	"cawfe_latte.neural_operator_depth": [1, 2],
	"cawfe_latte.neural_operator_type": ["afno"],
	"cawfe_latte.drop_path": [0.0, 0.1],
	"training.loss.surface_consumed_weight": [1.0],
	"training.loss.canopy_consumed_weight": [1.0],
	"training.loss.mask_weight": [1.0, 2.0],
	"training.loss.energy_weight": [1.0, 2.0],
}


def repo_root() -> Path:
	"""Return the repository root used for durable artifacts."""

	return Path(__file__).resolve().parents[2]


def _base_config_parent(base_config: Mapping[str, Any]) -> Path:
	config_path = base_config.get("config_path", base_config.get("_config_path"))
	if config_path:
		return Path(str(config_path)).expanduser().resolve().parent
	return repo_root()


def _resolve_from_base_config(base_config: Mapping[str, Any], configured_path: Any) -> str:
	path = Path(str(configured_path)).expanduser()
	if path.is_absolute():
		return str(path.resolve())
	return str((_base_config_parent(base_config) / path).resolve())


def make_portable_tuned_config_paths(config: dict[str, Any], base_config: Mapping[str, Any]) -> None:
	"""Absolutize paths that would break after saving best_config.yaml outside configs/."""

	fire_index = config.get("fire_dataset_index_json")
	if fire_index:
		config["fire_dataset_index_json"] = _resolve_from_base_config(base_config, fire_index)

	normalization = config.get("normalization")
	if isinstance(normalization, dict) and normalization.get("path"):
		normalization["path"] = _resolve_from_base_config(base_config, normalization["path"])


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
	"""Read a nested value using dot notation."""

	current: Any = config
	for part in str(dotted_key).split("."):
		if not isinstance(current, Mapping) or part not in current:
			return default
		current = current[part]
	return current


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
	"""Set a nested value using dot notation, creating dictionaries as needed."""

	parts = str(dotted_key).split(".")
	current = config
	for part in parts[:-1]:
		next_value = current.get(part)
		if not isinstance(next_value, dict):
			next_value = {}
			current[part] = next_value
		current = next_value
	current[parts[-1]] = value


def delete_nested(config: dict[str, Any], dotted_key: str) -> None:
	"""Delete a nested value if it exists."""

	parts = str(dotted_key).split(".")
	current: Any = config
	for part in parts[:-1]:
		if not isinstance(current, dict) or part not in current:
			return
		current = current[part]
	if isinstance(current, dict):
		current.pop(parts[-1], None)


def force_cawfe_latte(config: dict[str, Any]) -> dict[str, Any]:
	"""Force a config to train the main CAWFE-Latte architecture."""

	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	model_config["name"] = "cawfe_latte"
	config["model"] = model_config
	return config


def _normalize_param_key(dotted_key: str) -> str:
	return LOSS_PARAM_ALIASES.get(str(dotted_key), str(dotted_key))


def _apply_backbone_dim_consistency(config: dict[str, Any], backbone_dim: int) -> None:
	section = config.setdefault("cawfe_latte", {})
	backbone_dim = int(backbone_dim)
	section["backbone_dim"] = backbone_dim
	section["fused_dim"] = backbone_dim
	section["bottleneck_dim"] = 2 * backbone_dim
	section["decoder_channels"] = [2 * backbone_dim, backbone_dim, 64]
	if backbone_dim == 64:
		section["atm_embed_dim"] = 32
		section["fire_embed_dim"] = 32
	elif backbone_dim in {96, 128}:
		section["atm_embed_dim"] = 48
		section["fire_embed_dim"] = 48


def apply_params(config: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
	"""Apply tuning params to a deep copy of ``config``."""

	tuned = force_cawfe_latte(deepcopy(dict(config)))
	for raw_key, value in params.items():
		key = _normalize_param_key(str(raw_key))
		set_nested(tuned, key, deepcopy(value))
	if "cawfe_latte.backbone_dim" in params:
		_apply_backbone_dim_consistency(tuned, int(params["cawfe_latte.backbone_dim"]))
	return tuned


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
	"""Recursively merge two mappings."""

	result = deepcopy(dict(base))
	for key, value in overrides.items():
		if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
			result[key] = deep_merge(result[key], value)
		else:
			result[key] = deepcopy(value)
	return result


def get_tuning_config(config: Mapping[str, Any]) -> dict[str, Any]:
	section = config.get("hparam_tuning", {})
	return dict(section) if isinstance(section, Mapping) else {}


def build_search_space(config: Mapping[str, Any]) -> dict[str, list[Any]]:
	tuning_config = get_tuning_config(config)
	raw_space = tuning_config.get("search_space", DEFAULT_SEARCH_SPACE)
	if not isinstance(raw_space, Mapping):
		raise TypeError("hparam_tuning.search_space must be a mapping of dotted keys to value lists.")
	search_space: dict[str, list[Any]] = {}
	for key, values in raw_space.items():
		if not isinstance(values, list) or not values:
			raise ValueError(f"Search-space entry {key!r} must be a non-empty list.")
		search_space[str(key)] = deepcopy(values)
	return search_space


def generate_trial_params(
	search_space: Mapping[str, list[Any]],
	method: str,
	num_trials: int,
	seed: int,
) -> list[dict[str, Any]]:
	"""Generate trial parameter dictionaries using grid or random search."""

	method = str(method).lower()
	num_trials = max(1, int(num_trials))
	keys = list(search_space)
	if method == "grid":
		all_trials = [dict(zip(keys, values)) for values in itertools.product(*(search_space[key] for key in keys))]
		return all_trials[:num_trials]
	if method in {"random", "optuna"}:
		rng = random.Random(int(seed))
		trials = []
		for _ in range(num_trials):
			trials.append({key: deepcopy(rng.choice(search_space[key])) for key in keys})
		return trials
	raise ValueError(f"Unsupported tuning method: {method!r}. Expected random, grid, or optuna.")


def make_trial_config(
	base_config: Mapping[str, Any],
	params: Mapping[str, Any],
	trial_id: int,
	output_dir: str | Path,
	trial_max_epochs: int,
	max_train_batches_per_epoch: int | None,
	max_val_batches_per_epoch: int | None,
) -> dict[str, Any]:
	"""Create one resolved trial config."""

	config = apply_params(base_config, params)
	tuning_config = get_tuning_config(base_config)
	trial_overrides = tuning_config.get("trial_overrides", {})
	if isinstance(trial_overrides, Mapping):
		config = deep_merge(config, trial_overrides)

	trial_dir = Path(output_dir) / f"trial_{trial_id:03d}"
	training = config.setdefault("training", {})
	training["epochs"] = int(trial_max_epochs)
	config["epochs"] = int(trial_max_epochs)
	training["run_test_after_training"] = False
	training["run_external_test_after_training"] = False
	training["run_name"] = f"cawfe_latte_hparam_trial_{trial_id:03d}"
	if "trial_early_stopping_patience" in tuning_config:
		training["early_stopping_patience"] = int(tuning_config["trial_early_stopping_patience"])

	performance = training.setdefault("performance", {})
	if max_train_batches_per_epoch is not None:
		performance["max_train_batches_per_epoch"] = int(max_train_batches_per_epoch)
	if max_val_batches_per_epoch is not None:
		performance["max_val_batches_per_epoch"] = int(max_val_batches_per_epoch)
	if "full_validation_every_n_epochs" in performance:
		training["full_validation_every_n_epochs"] = performance["full_validation_every_n_epochs"]

	root = repo_root()
	checkpoint = config.setdefault("checkpoint", {})
	checkpoint["path"] = str(root / "artifacts" / "checkpoints" / "cawfe_latte_hparam" / f"trial_{trial_id:03d}" / "latest_model.pt")
	checkpoint["best_path"] = str(root / "artifacts" / "checkpoints" / "cawfe_latte_hparam" / f"trial_{trial_id:03d}" / "best_model.pt")
	checkpoint["resume"] = False

	checkpointing = config.setdefault("checkpointing", {})
	checkpointing["save_best"] = True
	checkpointing["save_latest"] = False

	cache = config.setdefault("cache", {})
	if bool(cache.get("use_precomputed_patches", config.get("use_precomputed_patches", False))):
		cache["allow_config_hash_mismatch"] = True

	logging = config.setdefault("logging", {})
	logging["run_name"] = f"cawfe_latte_hparam_trial_{trial_id:03d}"
	logging["training_log_path"] = str((root / trial_dir / "training_log.csv").resolve() if not trial_dir.is_absolute() else trial_dir / "training_log.csv")
	logging["timing_log_path"] = str((root / trial_dir / "training_timing.csv").resolve() if not trial_dir.is_absolute() else trial_dir / "training_timing.csv")
	return config


def compute_composite_score(metrics: Mapping[str, Any]) -> float | None:
	"""Compute the optional validation composite score from available metric fields."""

	required = (
		"val_surface_consumed_mae",
		"val_canopy_consumed_mae",
		"val_energy_log_mae",
		"val_energy_MW_active_mae",
		"val_mask_dice",
	)
	if not all(key in metrics and metrics[key] is not None for key in required):
		return None
	return (
		float(metrics["val_surface_consumed_mae"])
		+ float(metrics["val_canopy_consumed_mae"])
		+ float(metrics["val_energy_log_mae"])
		+ float(metrics["val_energy_MW_active_mae"])
		- 0.1 * float(metrics["val_mask_dice"])
	)


def resolve_metric_value(metrics: Mapping[str, Any], selection_metric: str) -> float | None:
	"""Resolve a selection metric from a training-history row."""

	metric = str(selection_metric)
	if metric == "val_composite_score":
		return compute_composite_score(metrics)
	if metric == "val_multitask_loss":
		metric = "val_loss"
	value = metrics.get(metric)
	if value is None:
		return None
	try:
		value_float = float(value)
	except (TypeError, ValueError):
		return None
	return value_float if math.isfinite(value_float) else None


def select_best_history_row(
	history_rows: Iterable[Mapping[str, Any]],
	selection_metric: str,
	selection_mode: str,
) -> tuple[dict[str, Any] | None, float | None, str]:
	"""Select the best history row using validation-only metrics."""

	mode = str(selection_mode).lower()
	if mode not in {"min", "max"}:
		raise ValueError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}.")
	best_row = None
	best_score = None
	metric_used = selection_metric
	for row in history_rows:
		row_metric_used = selection_metric
		score = resolve_metric_value(row, selection_metric)
		if score is None and selection_metric != "val_multitask_loss":
			score = resolve_metric_value(row, "val_multitask_loss")
			row_metric_used = "val_multitask_loss"
		if score is None:
			continue
		if best_score is None or (mode == "min" and score < best_score) or (mode == "max" and score > best_score):
			best_score = score
			best_row = dict(row)
			metric_used = row_metric_used
	return best_row, best_score, metric_used


def save_yaml(path: str | Path, config: Mapping[str, Any]) -> Path:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		yaml.safe_dump(dict(config), handle, sort_keys=False)
	return path


def load_json(path: str | Path) -> dict[str, Any]:
	with Path(path).open("r", encoding="utf-8") as handle:
		value = json.load(handle)
	if not isinstance(value, dict):
		raise ValueError(f"Expected JSON object in {path}.")
	return value


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
	return path


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def append_csv(path: str | Path, row: Mapping[str, Any]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not path.exists()
	fieldnames = list(row.keys())
	with path.open("a", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def make_final_config_from_best_params(
	base_config: Mapping[str, Any],
	best_params: Mapping[str, Any],
	keep_trial_epochs: bool = False,
) -> dict[str, Any]:
	"""Apply best params to the full-training config and remove trial-only limits."""

	params = best_params.get("params", {}) if isinstance(best_params.get("params"), Mapping) else {}
	config = apply_params(base_config, params)
	if not keep_trial_epochs:
		base_training = dict(base_config.get("training", {})) if isinstance(base_config.get("training"), Mapping) else {}
		base_epochs = int(base_config.get("epochs", base_training.get("epochs", 1)))
		config["epochs"] = base_epochs
		config.setdefault("training", {})["epochs"] = base_epochs

	training = config.setdefault("training", {})
	performance = training.setdefault("performance", {})
	performance["max_train_batches_per_epoch"] = None
	delete_nested(config, "training.performance.max_val_batches_per_epoch")
	base_training = base_config.get("training", {})
	if isinstance(base_training, Mapping) and "max_val_batches_per_epoch" in base_training:
		training["max_val_batches_per_epoch"] = base_config["training"]["max_val_batches_per_epoch"]
	training.pop("early_stopping_patience", None)

	cache = config.setdefault("cache", {})
	if bool(cache.get("use_precomputed_patches", config.get("use_precomputed_patches", False))):
		cache["allow_config_hash_mismatch"] = True

	make_portable_tuned_config_paths(config, base_config)

	root = repo_root()
	checkpoint = config.setdefault("checkpoint", {})
	checkpoint["path"] = str(root / "artifacts" / "checkpoints" / "cawfe_latte" / "latest_model.pt")
	checkpoint["best_path"] = str(root / "artifacts" / "checkpoints" / "cawfe_latte" / "best_model.pt")
	checkpoint["resume"] = True
	logging = config.setdefault("logging", {})
	logging["run_name"] = "cawfe_latte"
	logging["training_log_path"] = str(root / "artifacts" / "logs" / "cawfe_latte_training_log.csv")
	logging["timing_log_path"] = str(root / "artifacts" / "logs" / "cawfe_latte_timing_log.csv")
	return force_cawfe_latte(config)
