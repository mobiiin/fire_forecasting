#!/usr/bin/env python3
"""Train/reuse one Table 2 baseline and evaluate its frozen model once on test."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
import yaml

from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor, ProcessedTargetOnlyDataset
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.full_validation import evaluate_full_validation
from src.evaluation.held_out_test import evaluate_locked_test
from src.evaluation.table2_protocol import (
	DETERMINISTIC_BASELINES,
	LEARNED_BASELINES,
	identities_match,
	table2_protocol_identity,
	validate_table2_config,
)
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible, validate_checkpoint_model_compatibility
from src.training.hardware import choose_amp_dtype
from src.training.train import train_model_from_config


REGISTRY_PATH = Path("configs/table2_baselines/table2_baselines.yaml")
OUTPUT_ROOT = Path("artifacts/table2_baselines")


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
	temporary.replace(path)


def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
	payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
	if not isinstance(payload, Mapping) or not isinstance(payload.get("baselines"), Mapping):
		raise ValueError(f"Invalid Table 2 registry: {path}")
	return dict(payload)


def load_entry(name: str, registry_path: str | Path = REGISTRY_PATH) -> tuple[dict[str, Any], list[int]]:
	registry = load_registry(registry_path)
	if name not in registry["baselines"]:
		raise ValueError(f"Unknown Table 2 baseline {name!r}. Expected one of: {', '.join(registry['baselines'])}.")
	return dict(registry["baselines"][name]), [int(seed) for seed in registry["seeds"]]


def output_parent(name: str, seed: int | None, root: Path = OUTPUT_ROOT) -> Path:
	return root / name if seed is None else root / name / f"seed_{int(seed)}"


def _resolved_config_path(run_dir: Path) -> Path | None:
	for candidate in (run_dir / "resolved_config.yaml", run_dir / "configs" / "resolved_config.yaml"):
		if candidate.is_file():
			return candidate
	return None


def _candidate_dirs(parent: Path) -> list[Path]:
	if not parent.is_dir():
		return []
	candidates = [child for child in parent.iterdir() if child.is_dir() and _resolved_config_path(child) is not None]
	if _resolved_config_path(parent) is not None:
		candidates.append(parent)
	return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _candidate_identity(run_dir: Path, baseline: str, seed: int | None) -> dict[str, Any] | None:
	metadata_path = run_dir / "metadata" / "table2_protocol_identity.json"
	if metadata_path.is_file():
		try:
			return json.loads(metadata_path.read_text(encoding="utf-8"))
		except (json.JSONDecodeError, OSError):
			return None
	config_path = _resolved_config_path(run_dir)
	if config_path is None:
		return None
	try:
		return table2_protocol_identity(load_config(config_path), baseline, seed)
	except (KeyError, ValueError, FileNotFoundError, TypeError):
		return None


def _compatible_candidates(parent: Path, baseline: str, seed: int | None, expected: Mapping[str, Any]) -> list[Path]:
	return [
		run_dir
		for run_dir in _candidate_dirs(parent)
		if identities_match(expected, _candidate_identity(run_dir, baseline, seed))
	]


def _complete_test_artifact(run_dir: Path) -> bool:
	path = run_dir / "evaluation" / "test_metrics.json"
	if not path.is_file():
		return False
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
		total = int(payload["dataset_sample_count"])
		evaluated = int(payload["evaluated_test_samples"])
		unique = int(payload["unique_sample_id_count"])
		fire = int(payload["fire_count"])
		no_fire = int(payload["no_fire_count"])
		evaluation_dir = run_dir / "evaluation"
		sample_metrics_exist = (evaluation_dir / "test_sample_metrics.parquet").is_file() or (
			evaluation_dir / "test_sample_metrics.csv"
		).is_file()
		return (
			payload.get("metric_scope") == "complete_locked_held_out_test"
			and payload.get("split") == "test"
			and payload.get("test_used_for_model_selection") is False
			and total > 0
			and evaluated == total
			and unique == total
			and fire + no_fire == total
			and (evaluation_dir / "test_per_fire_metrics.csv").is_file()
			and sample_metrics_exist
			and (evaluation_dir / "qualitative_test_predictions.npz").is_file()
		)
	except (KeyError, TypeError, ValueError, json.JSONDecodeError):
		return False


def _has_best_checkpoint(run_dir: Path) -> bool:
	return (run_dir / "checkpoints" / "best_model.pt").is_file()


def _has_complete_validation(run_dir: Path) -> bool:
	return (run_dir / "full_validation_metrics.json").is_file() or (run_dir / "evaluation" / "full_validation_metrics.json").is_file()


def _runtime_config(config_path: Path, baseline: str, seed: int, run_id: str, parent: Path) -> dict[str, Any]:
	config = load_config(config_path)
	training = dict(config.get("training", {}))
	training.update(
		{
			"seed": int(seed),
			"max_epochs": 60,
			"epochs": 60,
			"max_train_batches": 7500,
			"max_train_batches_per_epoch": None,
			"run_name": str(run_id),
			"overwrite_run": True,
			"run_external_test_after_training": False,
			"run_test_after_training": False,
		}
	)
	performance = dict(training.get("performance", {}))
	performance["max_train_batches_per_epoch"] = None
	performance["auto_batch_size"] = False
	training["performance"] = performance
	output = dict(training.get("output", {}))
	output.update(
		{
			"root_dir": str(parent),
			"flat_run_layout": True,
			"update_architecture_latest_checkpoint": False,
		}
	)
	training["output"] = output
	config["training"] = training
	config["seed"] = int(seed)
	config["return_metadata"] = True
	dataloader = dict(config.get("dataloader", {}))
	dataloader["return_metadata"] = True
	config["dataloader"] = dataloader
	final_training = dict(config.get("final_training", {}))
	final_training["finalist"] = baseline
	config["final_training"] = final_training
	return config


def _read_history_rows(run_dir: Path) -> list[dict[str, Any]]:
	for path in (run_dir / "history" / "epoch_history.csv", run_dir / "training_history.csv"):
		if path.is_file():
			with path.open(newline="", encoding="utf-8") as handle:
				return [dict(row) for row in csv.DictReader(handle)]
	return []


def _float_values(rows: list[Mapping[str, Any]], key: str) -> list[float]:
	values: list[float] = []
	for row in rows:
		try:
			value = float(row.get(key, ""))
		except (TypeError, ValueError):
			continue
		if math.isfinite(value):
			values.append(value)
	return values


def _efficiency_metadata(model: torch.nn.Module, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
	parameters = list(model.parameters())
	parameter_count = sum(parameter.numel() for parameter in parameters)
	trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
	model_size_mb = sum(parameter.numel() * parameter.element_size() for parameter in parameters) / (1024.0 ** 2)
	rows = _read_history_rows(run_dir)
	epoch_seconds = _float_values(rows, "epoch_total_seconds") or _float_values(rows, "epoch_time_sec")
	peak_memory = _float_values(rows, "peak_gpu_memory_gb")
	return {
		"parameter_count": int(parameter_count),
		"trainable_parameter_count": int(trainable_count),
		"model_size_mb": float(model_size_mb),
		"checkpoint_size_mb": checkpoint.stat().st_size / (1024.0 ** 2),
		"mean_training_seconds_per_epoch": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
		"total_training_time_seconds": sum(epoch_seconds) if epoch_seconds else None,
		"training_peak_gpu_memory_gb": max(peak_memory) if peak_memory else None,
	}


def _device(config: Mapping[str, Any]) -> torch.device:
	requested = str(config.get("device", "auto")).lower()
	if requested in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
		return torch.device("cuda")
	if requested in {"cuda", "gpu"} and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested for a learned Table 2 baseline but is unavailable.")
	return torch.device("cpu")


def _copy_validation_artifact(run_dir: Path) -> None:
	source = run_dir / "full_validation_metrics.json"
	destination = run_dir / "evaluation" / "full_validation_metrics.json"
	if source.is_file():
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, destination)


def _write_run_metadata(
	*,
	run_dir: Path,
	baseline: str,
	seed: int | None,
	identity: Mapping[str, Any],
	model: torch.nn.Module | None,
	checkpoint: Path | None,
) -> None:
	metadata_dir = run_dir / "metadata"
	metadata_dir.mkdir(parents=True, exist_ok=True)
	_json_write(metadata_dir / "table2_protocol_identity.json", identity)
	is_existing_cawfe_run = baseline in {"cawfe_latte_baseline", "cawfe_latte_final"}
	metadata_name = "table2_test_metadata.json" if is_existing_cawfe_run else "run_metadata.json"
	summary_name = "table2_model_summary.txt" if is_existing_cawfe_run else "model_summary.txt"
	parameter_count = None if model is None else sum(parameter.numel() for parameter in model.parameters())
	trainable_count = None if model is None else sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	payload = {
		"schema_version": 1,
		"created_at": datetime.now(timezone.utc).isoformat(),
		"model": baseline,
		"seed": seed,
		"run_dir": str(run_dir),
		"best_checkpoint": None if checkpoint is None else str(checkpoint),
		"parameter_count": parameter_count,
		"trainable_parameter_count": trainable_count,
		"test_split_used_for_selection": False,
		"protocol_identity_sha256": identity["identity_sha256"],
	}
	# Finalist runs predate Table 2 and contain paper-analysis metadata that must
	# remain byte-for-byte intact. Store the held-out-test additions separately.
	_json_write(metadata_dir / metadata_name, payload)
	if model is not None:
		(metadata_dir / summary_name).write_text(
			f"Model: {baseline}\nSeed: {seed}\nParameters: {parameter_count}\n"
			f"Trainable parameters: {trainable_count}\n\n{model}\n",
			encoding="utf-8",
		)


def evaluate_learned_run(
	*,
	run_dir: Path,
	config: dict[str, Any],
	baseline: str,
	seed: int,
	identity: Mapping[str, Any],
	require_full_validation: bool = True,
) -> dict[str, Any]:
	checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"Best validation-selected checkpoint is missing: {checkpoint_path}")
	config["return_metadata"] = True
	dataloader = dict(config.get("dataloader", {}))
	dataloader["return_metadata"] = True
	config["dataloader"] = dataloader
	train_loader, val_loader, test_loader = create_dataloaders(config)
	if str(getattr(test_loader.dataset, "split", "")) != "test":
		raise RuntimeError("Table 2 evaluator did not receive the held-out test dataset.")
	device = _device(config)
	model = build_model_from_config(config, input_channels=int(config["model"]["input_channels"]))
	model = model.to(device)
	checkpoint = load_checkpoint(checkpoint_path, map_location=device)
	validate_checkpoint_model_compatibility(model, checkpoint, checkpoint_path)
	load_model_state_dict_compatible(model, checkpoint, checkpoint_path)
	if baseline == "cawfe_st_mamba" and getattr(model, "mamba_backend_used", None) != "mamba_ssm":
		raise RuntimeError(f"Publishable CAWFE-ST-Mamba run used backend={getattr(model, 'mamba_backend_used', None)!r}.")
	amp_dtype = choose_amp_dtype(config, device)
	if require_full_validation and not _has_complete_validation(run_dir):
		evaluate_full_validation(
			model=model,
			val_loader=val_loader,
			config=config,
			device=device,
			amp_dtype=amp_dtype,
			run_dir=run_dir,
			checkpoint_path=checkpoint_path,
			checkpoint_epoch=checkpoint.get("epoch"),
		)
	_copy_validation_artifact(run_dir)
	final_path = run_dir / "checkpoints" / "final_model.pt"
	if baseline in LEARNED_BASELINES:
		# For new external baselines, final_model is explicitly the
		# validation-selected best checkpoint, never merely the last epoch.
		shutil.copy2(checkpoint_path, final_path)
	elif not final_path.is_file():
		# Existing CAWFE finalist runs are never overwritten. This branch only
		# repairs an absent alias for an already frozen best checkpoint.
		shutil.copy2(checkpoint_path, final_path)
	resolved_source = _resolved_config_path(run_dir)
	if resolved_source is not None and resolved_source.resolve() != (run_dir / "resolved_config.yaml").resolve():
		shutil.copy2(resolved_source, run_dir / "resolved_config.yaml")
	efficiency = _efficiency_metadata(model, checkpoint_path, run_dir)
	_write_run_metadata(
		run_dir=run_dir,
		baseline=baseline,
		seed=seed,
		identity=identity,
		model=model,
		checkpoint=checkpoint_path,
	)
	result = evaluate_locked_test(
		test_loader=test_loader,
		config=config,
		run_dir=run_dir,
		model_name=baseline,
		seed=seed,
		model=model,
		device=device,
		amp_dtype=amp_dtype,
		checkpoint_path=checkpoint_path,
		checkpoint_epoch=checkpoint.get("epoch"),
		efficiency_metadata=efficiency,
	)
	del train_loader, val_loader, test_loader, model, checkpoint
	return result


def evaluate_existing_cawfe_run(run_dir: str | Path, table2_name: str, seed: int) -> dict[str, Any]:
	"""Evaluate one frozen CAWFE-Latte finalist checkpoint with the same test evaluator."""

	run_path = Path(run_dir).expanduser().resolve()
	config_path = _resolved_config_path(run_path)
	if config_path is None:
		raise FileNotFoundError(f"Resolved finalist config is missing: {run_path}")
	config = load_config(config_path)
	expected_finalist = "baseline" if table2_name == "cawfe_latte_baseline" else "GA_Q2"
	if str(config.get("final_training", {}).get("finalist")) != expected_finalist:
		raise ValueError(f"Finalist mismatch for {table2_name}: {config.get('final_training', {}).get('finalist')!r}")
	protocol = config.get("training", {}).get("sampling_protocol", {})
	if (
		str(protocol.get("protocol_id")) != "epoch_random_subset_without_replacement_v1"
		or int(protocol.get("batches_per_epoch", 0)) != 7500
		or int(protocol.get("batch_size", 0)) != 8
	):
		raise ValueError(f"CAWFE-Latte finalist does not use the required shortened training protocol: {run_path}")
	identity = table2_protocol_identity(config, table2_name, seed)
	if _complete_test_artifact(run_path) and identities_match(
		identity,
		_candidate_identity(run_path, table2_name, seed),
	):
		print(f"REUSING COMPLETE RUN: {run_path}")
		return json.loads((run_path / "evaluation" / "test_metrics.json").read_text(encoding="utf-8"))
	return evaluate_learned_run(
		run_dir=run_path,
		config=config,
		baseline=table2_name,
		seed=seed,
		identity=identity,
		require_full_validation=False,
	)


def _deterministic_record_order(record: Mapping[str, Any]) -> tuple[Any, ...]:
	"""Group identical full-frame predictions while preserving deterministic patch order."""

	patch = record.get("patch", {})
	return (
		str(record["fire_name"]),
		tuple(int(value) for value in record["input_indices"]),
		int(record["current_index"]),
		int(patch.get("y0", 0)),
		int(patch.get("x0", 0)),
		int(patch.get("height", 0)),
		int(patch.get("width", 0)),
		str(record.get("sample_id", "")),
	)


def _evaluate_deterministic(config: dict[str, Any], baseline: str, identity: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
	if _complete_test_artifact(run_dir) and identities_match(identity, _candidate_identity(run_dir, baseline, None)):
		print(f"REUSING COMPLETE RUN: {run_dir}")
		return json.loads((run_dir / "evaluation" / "test_metrics.json").read_text(encoding="utf-8"))
	dataloader = config["dataloader"]
	root = Path(str(dataloader.get("dataset_root", config["processed_dataset"]["root"]))).expanduser().resolve()
	pattern = str(dataloader["sample_pattern"])
	dataset = ProcessedTargetOnlyDataset(root, root / "indices" / "temporal" / f"samples_{pattern}.jsonl", "test")
	# The canonical index interleaves prediction keys. Grouping records here keeps
	# SequentialSampler/no-shuffle semantics while allowing the predictor cache to
	# compute each full-frame deterministic prediction once. Membership and every
	# sample ID are unchanged; the locked evaluator still enforces exact-once use.
	dataset.records.sort(key=_deterministic_record_order)
	loader = DataLoader(
		dataset,
		batch_size=int(config.get("training", {}).get("batch_size", 8)),
		shuffle=False,
		drop_last=False,
		num_workers=int(config.get("training", {}).get("num_workers", 0)),
		pin_memory=False,
	)
	predictor = ProcessedHistoryBaselinePredictor(baseline, root, config)
	run_dir.mkdir(parents=True, exist_ok=True)
	(run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
	_write_run_metadata(run_dir=run_dir, baseline=baseline, seed=None, identity=identity, model=None, checkpoint=None)
	return evaluate_locked_test(
		test_loader=loader,
		config=config,
		run_dir=run_dir,
		model_name=baseline,
		seed=None,
		deterministic_predictor=predictor,
		device=torch.device("cpu"),
		efficiency_metadata={
			"parameter_count": 0,
			"trainable_parameter_count": 0,
			"model_size_mb": 0.0,
			"mean_training_seconds_per_epoch": None,
			"total_training_time_seconds": 0.0,
			"training_peak_gpu_memory_gb": None,
			"deterministic_evaluation_order": "prediction_key_grouped_sequential",
		},
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("baseline", nargs="?")
	parser.add_argument("seed", nargs="?", type=int)
	parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
	parser.add_argument("--run-id", default=None)
	parser.add_argument("--list-baselines", action="store_true")
	parser.add_argument("--list-seeds", action="store_true")
	parser.add_argument("--print-config", action="store_true")
	parser.add_argument("--print-output-parent", action="store_true")
	parser.add_argument("--audit-only", action="store_true")
	args = parser.parse_args()

	registry = load_registry(args.registry)
	if args.list_baselines:
		print("\n".join(registry["baselines"]))
		return
	if args.list_seeds:
		print("\n".join(str(int(seed)) for seed in registry["seeds"]))
		return
	if args.baseline is None:
		parser.error("baseline is required")
	entry, registered_seeds = load_entry(args.baseline, args.registry)
	config_path = Path(str(entry["config_path"]))
	learned = bool(entry["learned"])
	if args.print_config:
		print(config_path)
		return
	if learned:
		if args.seed is None:
			parser.error(f"seed is required for learned baseline {args.baseline}")
		if int(args.seed) not in registered_seeds:
			raise ValueError(f"Seed {args.seed} is not registered: {registered_seeds}")
		seed: int | None = int(args.seed)
	else:
		if args.seed is not None:
			raise ValueError(f"Deterministic baseline {args.baseline} does not accept a seed.")
		seed = None
	parent = output_parent(args.baseline, seed)
	if args.print_output_parent:
		print(parent)
		return

	if learned:
		run_id = args.run_id or f"slurm{os.environ.get('SLURM_JOB_ID', 'local')}"
		config = _runtime_config(config_path, args.baseline, int(seed), run_id, parent)
	else:
		config = load_config(config_path)
		config["return_metadata"] = True
	validate_table2_config(config, args.baseline, seed)
	identity = table2_protocol_identity(config, args.baseline, seed)
	compatible = _compatible_candidates(parent, args.baseline, seed, identity) if learned else []
	complete = [run_dir for run_dir in compatible if _has_best_checkpoint(run_dir) and _has_complete_validation(run_dir) and _complete_test_artifact(run_dir)]
	best_only = [run_dir for run_dir in compatible if _has_best_checkpoint(run_dir) and not _complete_test_artifact(run_dir)]
	print(
		json.dumps(
			{
				"baseline": args.baseline,
				"seed": seed,
				"protocol_identity_sha256": identity["identity_sha256"],
				"compatible_complete_runs": [str(path) for path in complete],
				"compatible_best_checkpoint_runs_missing_test": [str(path) for path in best_only],
			},
			indent=2,
		)
	)
	if args.audit_only:
		return
	if not learned:
		_evaluate_deterministic(config, args.baseline, identity, parent)
		print(parent)
		return
	if complete:
		print(f"REUSING COMPLETE RUN: {complete[0]}")
		print(complete[0])
		return
	if best_only:
		run_dir = best_only[0]
		resolved = _resolved_config_path(run_dir)
		assert resolved is not None
		evaluation_config = load_config(resolved)
		validate_table2_config(evaluation_config, args.baseline, seed)
		print(f"REUSING BEST CHECKPOINT; RUNNING MISSING VALIDATION/TEST ONLY: {run_dir}")
		evaluate_learned_run(
			run_dir=run_dir,
			config=evaluation_config,
			baseline=args.baseline,
			seed=int(seed),
			identity=identity,
		)
		print(run_dir)
		return

	result = train_model_from_config(config)
	run_dir = Path(str(result["run_dir"])).expanduser().resolve()
	if not _has_best_checkpoint(run_dir):
		raise RuntimeError(f"Training did not produce a validation-selected best checkpoint: {run_dir}")
	evaluate_learned_run(
		run_dir=run_dir,
		config=config,
		baseline=args.baseline,
		seed=int(seed),
		identity=identity,
	)
	print(run_dir)


if __name__ == "__main__":
	main()
