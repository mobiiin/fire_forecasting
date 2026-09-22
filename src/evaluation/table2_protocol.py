"""Configuration validation and compatibility identity for Table 2 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.config import compute_file_sha256, load_config
from src.data.cache import target_definition_version


CANONICAL_FINALIST_CONFIG = Path("configs/final_training/cawfe_latte_baseline_full.yaml")
LEARNED_BASELINES = {"convlstm_unet", "earthformer_lite", "cawfe_st_mamba"}
DETERMINISTIC_BASELINES = {"persistence", "linear_extrapolation"}
ARCHITECTURE_BY_BASELINE = {
	"convlstm_unet": "convlstm_unet",
	"earthformer_lite": "earthformer_lite",
	"cawfe_st_mamba": "st_mamba_lite",
	"persistence": "persistence",
	"linear_extrapolation": "linear_extrapolation",
}


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
	value = config.get(name)
	return dict(value) if isinstance(value, Mapping) else {}


def _resolved_path(value: Any) -> Path | None:
	if value in (None, "", "null"):
		return None
	return Path(str(value)).expanduser().resolve()


def _file_record(path: Path | None, *, required: bool = True) -> dict[str, Any] | None:
	if path is None:
		if required:
			raise FileNotFoundError("Required compatibility file path is not configured.")
		return None
	if not path.is_file():
		if required:
			raise FileNotFoundError(f"Required compatibility file is missing: {path}")
		return {"path": str(path), "sha256": None}
	return {"path": str(path), "sha256": compute_file_sha256(path)}


def _stable_hash(value: Any) -> str:
	encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def table2_protocol_identity(config: Mapping[str, Any], baseline_name: str, seed: int | None) -> dict[str, Any]:
	baseline = str(baseline_name)
	if baseline not in LEARNED_BASELINES | DETERMINISTIC_BASELINES | {"cawfe_latte_baseline", "cawfe_latte_final"}:
		raise ValueError(f"Unknown Table 2 baseline identity: {baseline!r}")
	dataloader = _section(config, "dataloader")
	processed = _section(config, "processed_dataset")
	normalization = _section(config, "normalization")
	training = _section(config, "training")
	root = _resolved_path(dataloader.get("dataset_root", processed.get("root")))
	if root is None:
		raise FileNotFoundError("Canonical processed dataset root is not configured.")
	pattern = str(dataloader.get("sample_pattern", ""))
	sample_index = root / "indices" / "temporal" / f"samples_{pattern}.jsonl"
	model_config = _section(config, "model")
	architecture = str(model_config.get("architecture", ""))
	architecture_section_name = "st_mamba_lite" if baseline == "cawfe_st_mamba" else architecture
	if baseline.startswith("cawfe_latte_"):
		architecture_section_name = "cawfe_latte"
	model_definition = {
		"model": model_config,
		"architecture_config": _section(config, architecture_section_name),
	}
	identity: dict[str, Any] = {
		"schema_version": 1,
		"baseline_name": baseline,
		"seed": None if seed is None else int(seed),
		"learned": baseline not in DETERMINISTIC_BASELINES,
		"model_architecture": architecture,
		"model_config_sha256": _stable_hash(model_definition),
		"dataset_root": str(root),
		"dataset_version": str(processed.get("version", "")),
		"sample_pattern": pattern,
		"input_sequence_length": int(config.get("input_sequence_length", model_config.get("input_sequence_length", 0))),
		"prediction_horizon": int(config.get("prediction_horizon", training.get("prediction_horizon", 0))),
		"input_channels": int(model_config.get("input_channels", 0)),
		"output_channels": int(model_config.get("output_channels", 0)),
		"target_definition_version": target_definition_version(config),
		"target_order": ["surface_consumed", "canopy_consumed", "fire_mask", "energy_log"],
		"dataset_manifest": _file_record(root / "dataset_manifest.json"),
		"split_manifest": _file_record(root / "split_manifest.json"),
		"sample_index": _file_record(sample_index),
		"target_manifest": _file_record(root / "targets" / "h10" / "target_manifest.json"),
		"normalization_json": _file_record(_resolved_path(normalization.get("stats_path"))),
		"normalization_npz": _file_record(_resolved_path(normalization.get("npz_path"))),
		"mask_thresholds": _section(_section(config, "target_construction"), "fire_mask"),
	}
	if identity["learned"]:
		identity["training_protocol"] = {
			"batch_size": int(training.get("batch_size", 0)),
			"max_train_batches": int(training.get("max_train_batches", 0)),
			"max_epochs": int(training.get("max_epochs", 0)),
			"gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
			"optimizer": str(training.get("optimizer", "")),
			"learning_rate": float(training.get("learning_rate", 0.0)),
			"weight_decay": float(training.get("weight_decay", 0.0)),
			"scheduler": str(training.get("scheduler", "")),
			"scheduler_factor": float(training.get("scheduler_factor", 0.0)),
			"scheduler_patience": int(training.get("scheduler_patience", 0)),
			"gradient_clip_norm": float(training.get("gradient_clip_norm", 0.0)),
			"loss_sha256": _stable_hash(training.get("loss", {})),
			"early_stopping": training.get("early_stopping", {}),
			"checkpointing": training.get("checkpointing", {}),
		}
	identity["identity_sha256"] = _stable_hash(identity)
	return identity


def validate_table2_config(config: Mapping[str, Any], baseline_name: str, seed: int | None) -> None:
	baseline = str(baseline_name)
	if baseline not in LEARNED_BASELINES | DETERMINISTIC_BASELINES:
		raise ValueError(f"Unknown Table 2 baseline: {baseline!r}")
	canonical = load_config(CANONICAL_FINALIST_CONFIG)
	problems: list[str] = []
	dataloader = _section(config, "dataloader")
	canonical_loader = _section(canonical, "dataloader")
	processed = _section(config, "processed_dataset")
	canonical_processed = _section(canonical, "processed_dataset")
	model = _section(config, "model")
	training = _section(config, "training")
	canonical_training = _section(canonical, "training")
	table2 = _section(config, "table2")

	checks = {
		"dataloader.source": (dataloader.get("source"), canonical_loader.get("source")),
		"dataloader.sample_pattern": (dataloader.get("sample_pattern"), "sparse5_h10"),
		"processed_dataset.root": (str(processed.get("root")), str(canonical_processed.get("root"))),
		"processed_dataset.version": (processed.get("version"), canonical_processed.get("version")),
		"input_sequence_length": (int(config.get("input_sequence_length", 0)), 5),
		"prediction_horizon": (int(config.get("prediction_horizon", 0)), 10),
		"model.architecture": (str(model.get("architecture")), ARCHITECTURE_BY_BASELINE[baseline]),
		"model.input_channels": (int(model.get("input_channels", 0)), 129),
		"model.output_channels": (int(model.get("output_channels", 0)), 4),
		"table2.baseline_name": (table2.get("baseline_name"), baseline),
		"table2.locked_test_split": (bool(table2.get("locked_test_split", False)), True),
	}
	for label, (actual, expected) in checks.items():
		if actual != expected:
			problems.append(f"{label}={actual!r}, expected {expected!r}")
	if _section(config, "multitask") != _section(canonical, "multitask"):
		problems.append("multitask target/loss weights differ from the finalist protocol")
	if _section(config, "energy_release") != _section(canonical, "energy_release"):
		problems.append("energy_release target semantics differ from the finalist protocol")
	if _section(config, "target_construction") != _section(canonical, "target_construction"):
		problems.append("target_construction differs from the canonical dataset definition")
	if _section(config, "normalization") != _section(canonical, "normalization"):
		problems.append("normalization differs from the finalist protocol")
	if _section(config, "patching") != _section(canonical, "patching"):
		problems.append("patching/cropping policy differs from the finalist protocol")
	if baseline in LEARNED_BASELINES:
		learned_checks = {
			"training.seed": (int(training.get("seed", -1)), int(seed) if seed is not None else None),
			"training.batch_size": (int(training.get("batch_size", 0)), 8),
			"training.max_train_batches": (int(training.get("max_train_batches", 0)), 7500),
			"training.max_epochs": (int(training.get("max_epochs", 0)), 60),
			"training.gradient_accumulation_steps": (int(training.get("gradient_accumulation_steps", 0)), 1),
			"training.optimizer": (training.get("optimizer"), canonical_training.get("optimizer")),
			"training.learning_rate": (float(training.get("learning_rate", 0.0)), float(canonical_training.get("learning_rate", 0.0))),
			"training.weight_decay": (float(training.get("weight_decay", 0.0)), float(canonical_training.get("weight_decay", 0.0))),
			"training.scheduler": (training.get("scheduler"), canonical_training.get("scheduler")),
			"training.scheduler_factor": (float(training.get("scheduler_factor", 0.0)), float(canonical_training.get("scheduler_factor", 0.0))),
			"training.scheduler_patience": (int(training.get("scheduler_patience", 0)), int(canonical_training.get("scheduler_patience", 0))),
			"training.gradient_clip_norm": (float(training.get("gradient_clip_norm", 0.0)), float(canonical_training.get("gradient_clip_norm", 0.0))),
			"training.loss": (training.get("loss"), canonical_training.get("loss")),
			"training.early_stopping": (training.get("early_stopping"), canonical_training.get("early_stopping")),
			"training.run_test_after_training": (bool(training.get("run_test_after_training", False)), False),
			"training.run_external_test_after_training": (bool(training.get("run_external_test_after_training", False)), False),
		}
		for label, (actual, expected) in learned_checks.items():
			if actual != expected:
				problems.append(f"{label} differs from finalist protocol")
		data_loader = _section(config, "data_loader")
		if int(data_loader.get("batch_size", 0)) != 8:
			problems.append("data_loader.batch_size must remain 8")
		for split in ("train", "val", "test"):
			split_loader = _section(data_loader, split)
			if bool(split_loader.get("drop_last", False)):
				problems.append(f"data_loader.{split}.drop_last must be false")
		performance = _section(training, "performance")
		if performance.get("max_train_batches_per_epoch") not in (None, "null"):
			problems.append("legacy performance.max_train_batches_per_epoch must be disabled")
		if bool(_section(training, "auto_hardware_tuning").get("enabled", False)):
			problems.append("automatic hardware batch-size tuning must be disabled")
		if bool(performance.get("auto_batch_size", False)):
			problems.append("automatic batch-size probing must be disabled")
		validation = _section(training, "validation")
		if not bool(_section(validation, "screening").get("enabled", False)):
			problems.append("fixed screening validation must be enabled")
		full = _section(validation, "full")
		if not bool(full.get("enabled", False)) or full.get("checkpoint") != "best":
			problems.append("complete validation of the best checkpoint must be enabled")
	if baseline == "cawfe_st_mamba":
		if str(_section(config, "st_mamba_lite").get("mamba_backend")) != "mamba_ssm":
			problems.append("CAWFE-ST-Mamba must require the official mamba_ssm backend")
	if problems:
		raise ValueError("Unsafe Table 2 config:\n- " + "\n- ".join(problems))


def identities_match(expected: Mapping[str, Any], actual: Mapping[str, Any] | None) -> bool:
	return isinstance(actual, Mapping) and str(expected.get("identity_sha256")) == str(actual.get("identity_sha256"))


__all__ = [
	"ARCHITECTURE_BY_BASELINE",
	"DETERMINISTIC_BASELINES",
	"LEARNED_BASELINES",
	"identities_match",
	"table2_protocol_identity",
	"validate_table2_config",
]
