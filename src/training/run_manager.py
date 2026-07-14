"""Run-directory management for training jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
	import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency is already required by config.py
	yaml = None


DEFAULT_OUTPUT_CONFIG: dict[str, Any] = {
	"root_dir": "artifacts/runs",
	"checkpoint_root": "artifacts/checkpoints",
	"log_root": "artifacts/logs",
	"save_compatibility_checkpoints": True,
	"update_architecture_latest_checkpoint": True,
	"save_epoch_checkpoints": False,
	"save_best_checkpoint": True,
	"save_latest_checkpoint": True,
	"save_training_curves": True,
	"save_metric_curves": True,
	"save_timing_plots": True,
	"save_resolved_config": True,
	"save_original_config": True,
	"save_run_summary": True,
	"save_hardware_summary": True,
	"save_cache_manifest_copy": True,
	"save_normalization_stats_copy": True,
}

DEFAULT_CHECKPOINTING_CONFIG: dict[str, Any] = {
	"monitor": "val_loss",
	"mode": "min",
	"save_best": True,
	"save_latest": True,
	"save_every_n_epochs": None,
	"keep_last_n_epoch_checkpoints": 3,
}


def sanitize_run_component(value: Any, fallback: str = "run") -> str:
	"""Return a filesystem-friendly run-name component."""

	sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
	return sanitized or fallback


def _to_builtin(value: Any) -> Any:
	"""Convert nested values into JSON/YAML-friendly builtins."""

	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): _to_builtin(nested_value) for key, nested_value in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_builtin(item) for item in value]
	return value


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
	value = config.get(name)
	return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(_config: Mapping[str, Any], configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	repo_root = Path(__file__).resolve().parents[2]
	return (repo_root / path).resolve()


def get_training_output_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Return training.output with defaults applied."""

	training_config = _section(config, "training")
	output_config = dict(DEFAULT_OUTPUT_CONFIG)
	configured = training_config.get("output", {})
	if isinstance(configured, Mapping):
		output_config.update(dict(configured))
	return output_config


def get_training_checkpointing_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Merge legacy top-level checkpointing and training.checkpointing config."""

	checkpointing_config = dict(DEFAULT_CHECKPOINTING_CONFIG)
	legacy_config = config.get("checkpointing", {})
	if isinstance(legacy_config, Mapping):
		checkpointing_config.update(dict(legacy_config))
	training_config = _section(config, "training")
	nested_config = training_config.get("checkpointing", {})
	if isinstance(nested_config, Mapping):
		checkpointing_config.update(dict(nested_config))
	return checkpointing_config


def generated_run_name(architecture: str) -> str:
	"""Build the default unique run name for local or Slurm execution."""

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	slurm_job_id = os.environ.get("SLURM_JOB_ID")
	if slurm_job_id:
		suffix = f"slurm{sanitize_run_component(slurm_job_id, 'job')}"
	else:
		suffix = f"local{os.getpid()}"
	return f"{sanitize_run_component(architecture)}_{timestamp}_{suffix}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(_to_builtin(dict(payload)), handle, indent=2, sort_keys=True, default=str)
	return path


def _write_yaml_or_json(path: Path, payload: Mapping[str, Any]) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	if yaml is not None:
		with path.open("w", encoding="utf-8") as handle:
			yaml.safe_dump(_to_builtin(dict(payload)), handle, sort_keys=False)
		return path
	json_path = path.with_suffix(".json")
	_write_json(json_path, payload)
	return json_path


def _git_info(repo_root: Path) -> dict[str, Any]:
	info: dict[str, Any] = {"repo_root": str(repo_root)}
	commands = {
		"commit": ["git", "rev-parse", "HEAD"],
		"branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
	}
	for key, command in commands.items():
		try:
			result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=repo_root)
		except Exception:
			info[key] = None
		else:
			info[key] = result.stdout.strip() or None
	try:
		status_result = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True, cwd=repo_root)
	except Exception:
		info["dirty"] = None
		info["status_short"] = ""
	else:
		status = status_result.stdout.strip()
		info["dirty"] = bool(status)
		info["status_short"] = status
	return info


class RunManager:
	"""Create and manage per-run training outputs."""

	def __init__(self, config: Mapping[str, Any], architecture: str, run_name: str | None = None) -> None:
		self.config = deepcopy(dict(config))
		self.architecture = sanitize_run_component(architecture, fallback="architecture")
		self.output_config = get_training_output_config(config)
		self.checkpointing_config = get_training_checkpointing_config(config)
		self.root_dir = _resolve_path(config, self.output_config["root_dir"])
		self.checkpoint_root = _resolve_path(config, self.output_config["checkpoint_root"])
		self.repo_root = Path(__file__).resolve().parents[2]
		self.start_time = datetime.now(timezone.utc)
		training_config = _section(config, "training")
		explicit_run_name = run_name if run_name not in (None, "", "null") else training_config.get("run_name")
		if explicit_run_name in (None, "", "null"):
			self.requested_run_name = generated_run_name(self.architecture)
		else:
			self.requested_run_name = sanitize_run_component(explicit_run_name, fallback=self.architecture)
		self.overwrite_run = bool(training_config.get("overwrite_run", False))
		self.run_name = self.requested_run_name
		self.run_dir = self.root_dir / self.architecture / self.run_name
		self.checkpoint_dir = self.run_dir / "checkpoints"
		self.log_dir = self.run_dir / "logs"
		self.figure_dir = self.run_dir / "figures"
		self.config_dir = self.run_dir / "configs"
		self.metadata_dir = self.run_dir / "metadata"
		self.created = False

	def create_run_dir(self) -> Path:
		"""Create the unique run directory and subdirectories."""

		if self.created:
			return self.run_dir

		base_run_name = self.requested_run_name
		if self.overwrite_run:
			self.run_name = base_run_name
			self.run_dir = self.root_dir / self.architecture / self.run_name
			self.run_dir.mkdir(parents=True, exist_ok=True)
		else:
			for suffix_index in range(1, 1000):
				candidate_name = base_run_name if suffix_index == 1 else f"{base_run_name}_v{suffix_index}"
				candidate_dir = self.root_dir / self.architecture / candidate_name
				try:
					candidate_dir.mkdir(parents=True, exist_ok=False)
				except FileExistsError:
					continue
				self.run_name = candidate_name
				self.run_dir = candidate_dir
				break
			else:  # pragma: no cover - defensive guard for pathological collisions
				raise FileExistsError(f"Could not create a unique run directory under {self.root_dir / self.architecture}.")

		self.checkpoint_dir = self.run_dir / "checkpoints"
		self.log_dir = self.run_dir / "logs"
		self.figure_dir = self.run_dir / "figures"
		self.config_dir = self.run_dir / "configs"
		self.metadata_dir = self.run_dir / "metadata"
		for directory in (self.checkpoint_dir, self.log_dir, self.figure_dir, self.config_dir, self.metadata_dir):
			directory.mkdir(parents=True, exist_ok=True)
		self.created = True
		return self.run_dir

	def checkpoint_path(self, kind: str, epoch: int | None = None) -> Path:
		"""Return a checkpoint path for best/latest/epoch checkpoints."""

		self.create_run_dir()
		kind_text = str(kind).lower()
		if kind_text == "best":
			return self.checkpoint_dir / "best_model.pt"
		if kind_text == "latest":
			return self.checkpoint_dir / "latest_model.pt"
		if kind_text == "epoch":
			if epoch is None:
				raise ValueError("epoch checkpoint paths require epoch=.")
			return self.checkpoint_dir / f"epoch_{int(epoch):03d}.pt"
		if kind_text.startswith("epoch_"):
			return self.checkpoint_dir / f"{sanitize_run_component(kind_text)}.pt"
		return self.checkpoint_dir / f"{sanitize_run_component(kind_text)}.pt"

	def log_path(self, name: str) -> Path:
		"""Return a log path within the run log directory."""

		self.create_run_dir()
		mapping = {
			"training": "training_log.csv",
			"train": "training_log.csv",
			"validation": "validation_log.csv",
			"val": "validation_log.csv",
			"timing": "timing_log.csv",
			"metrics": "metrics_log.csv",
			"process": "training_process.log",
		}
		filename = mapping.get(str(name).lower(), str(name))
		if "." not in Path(filename).name:
			filename = f"{filename}.csv"
		return self.log_dir / filename

	def figure_path(self, name: str) -> Path:
		"""Return a figure path within the run figure directory."""

		self.create_run_dir()
		filename = str(name)
		if "." not in Path(filename).name:
			filename = f"{filename}.png"
		return self.figure_dir / filename

	def metadata_path(self, name: str) -> Path:
		"""Return a metadata path within the run metadata directory."""

		self.create_run_dir()
		filename = str(name)
		if "." not in Path(filename).name:
			filename = f"{filename}.json"
		return self.metadata_dir / filename

	def config_path(self, name: str) -> Path:
		"""Return a config artifact path within the run config directory."""

		self.create_run_dir()
		filename = str(name)
		if "." not in Path(filename).name:
			filename = f"{filename}.yaml"
		return self.config_dir / filename

	def save_configs(
		self,
		original_config: Mapping[str, Any] | None = None,
		resolved_config: Mapping[str, Any] | None = None,
	) -> dict[str, str]:
		"""Save original and resolved config artifacts."""

		self.create_run_dir()
		paths: dict[str, str] = {}
		if original_config is not None and bool(self.output_config.get("save_original_config", True)):
			path = _write_yaml_or_json(self.config_path("original_config.yaml"), dict(original_config))
			paths["original_config_path"] = str(path)
		if resolved_config is not None and bool(self.output_config.get("save_resolved_config", True)):
			path = _write_yaml_or_json(self.config_path("resolved_config.yaml"), dict(resolved_config))
			paths["resolved_config_path"] = str(path)
		return paths

	def save_metadata(self, name: str, payload: Mapping[str, Any]) -> Path:
		"""Save one JSON metadata payload."""

		return _write_json(self.metadata_path(name), dict(payload))

	def save_git_info(self) -> Path:
		"""Save git commit, branch, and dirty-worktree metadata."""

		return self.save_metadata("git_info.json", _git_info(self.repo_root))

	def record_path_metadata(self, name: str, source_path: str | Path | None) -> Path | None:
		"""Write a small text metadata file containing a source path."""

		if source_path is None:
			return None
		path = self.metadata_path(name)
		if path.suffix == ".json":
			path = path.with_suffix(".txt")
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(str(source_path) + "\n", encoding="utf-8")
		return path

	def copy_metadata_file(self, source_path: str | Path | None, destination_name: str) -> Path | None:
		"""Copy a metadata file into the run metadata directory if it exists."""

		if source_path is None:
			return None
		source = Path(source_path).expanduser()
		if not source.exists() or not source.is_file():
			return None
		destination = self.metadata_path(destination_name)
		if destination.suffix == ".json" and source.suffix:
			destination = destination.with_suffix(source.suffix)
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(source, destination)
		return destination

	def compatibility_checkpoint_path(self, kind: str) -> Path:
		"""Return the run-specific compatibility checkpoint path."""

		filename = "best_model.pt" if str(kind).lower() == "best" else "latest_model.pt"
		return self.checkpoint_root / self.architecture / self.run_name / filename

	def architecture_latest_checkpoint_path(self, kind: str) -> Path:
		"""Return the legacy architecture-level compatibility checkpoint path."""

		filename = "best_model.pt" if str(kind).lower() == "best" else "latest_model.pt"
		return self.checkpoint_root / self.architecture / filename

	def copy_checkpoint_to_compatibility(self, source_path: str | Path, kind: str) -> list[Path]:
		"""Atomically copy a checkpoint into compatibility locations."""

		if not bool(self.output_config.get("save_compatibility_checkpoints", True)):
			return []
		source = Path(source_path).expanduser().resolve()
		if not source.exists():
			raise FileNotFoundError(f"Cannot copy missing checkpoint: {source}")

		destinations = [self.compatibility_checkpoint_path(kind)]
		if bool(self.output_config.get("update_architecture_latest_checkpoint", True)):
			destinations.append(self.architecture_latest_checkpoint_path(kind))

		written: list[Path] = []
		for destination in destinations:
			destination.parent.mkdir(parents=True, exist_ok=True)
			temp_path = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
			shutil.copy2(source, temp_path)
			os.replace(temp_path, destination)
			written.append(destination)
		return written

	def prune_epoch_checkpoints(self, keep_last: int | None) -> list[Path]:
		"""Remove old epoch checkpoints from this run only."""

		if keep_last is None or int(keep_last) <= 0:
			return []
		self.create_run_dir()
		epoch_paths = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
		to_remove = epoch_paths[: max(0, len(epoch_paths) - int(keep_last))]
		for path in to_remove:
			path.unlink(missing_ok=True)
		return to_remove

	def finalize(
		self,
		training_result: Mapping[str, Any],
		status: str = "completed",
		error_message: str = "",
		notes: str = "",
	) -> Path | None:
		"""Write the final run summary JSON."""

		if not bool(self.output_config.get("save_run_summary", True)):
			return None
		self.create_run_dir()
		end_time = datetime.now(timezone.utc)
		final_epoch_summary = training_result.get("final_epoch_summary", {})
		if not isinstance(final_epoch_summary, Mapping):
			final_epoch_summary = {}
		summary = {
			"architecture": self.architecture,
			"run_name": self.run_name,
			"run_dir": str(self.run_dir),
			"start_time": self.start_time.isoformat(),
			"end_time": end_time.isoformat(),
			"duration_sec": max(0.0, (end_time - self.start_time).total_seconds()),
			"status": status,
			"error_message": error_message,
			"best_epoch": training_result.get("best_epoch"),
			"best_metric_name": training_result.get("best_metric_name", "val_loss"),
			"best_metric_value": training_result.get("best_val_loss"),
			"best_checkpoint_path": training_result.get("best_checkpoint_path"),
			"latest_checkpoint_path": training_result.get("latest_checkpoint_path"),
			"final_train_loss": final_epoch_summary.get("train_loss"),
			"final_val_loss": final_epoch_summary.get("val_loss"),
			"num_epochs_completed": training_result.get("num_epochs_completed"),
			"global_steps": training_result.get("global_step"),
			"config_path": self.config.get("config_path", self.config.get("_config_path")),
			"resolved_config_path": training_result.get("run_artifact_paths", {}).get("resolved_config_path")
			if isinstance(training_result.get("run_artifact_paths"), Mapping)
			else None,
			"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
			"slurm_nodelist": os.environ.get("SLURM_NODELIST"),
			"gpu_name": training_result.get("gpu_name"),
			"gpu_total_vram_gb": training_result.get("gpu_total_vram_gb"),
			"hostname": socket.gethostname(),
			"python": sys.version,
			"notes": notes,
		}
		summary.update({key: value for key, value in training_result.items() if key in {"test_results", "training_curve_paths"}})
		return self.save_metadata("run_summary.json", summary)
