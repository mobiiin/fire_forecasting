"""Inspect train-only input normalization across loaders and device application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import load_config
from src.data.cache import compute_dataset_index_hash, get_patch_cache_dir
from src.data.dataset import create_dataloaders
from src.data.preprocessing import load_normalization_stats
from src.training.input_normalization import (
	apply_input_normalization,
	build_input_normalizer_for_loader,
	input_batch_summary,
	normalization_metadata_from_loader,
	resolve_input_normalization_stats_path,
	validate_normalization_stats,
)
from src.training.train import _ensure_config_path, _get_device, _infer_input_channels_from_loader


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inspect input-normalization consistency for train/val/test loaders.")
	parser.add_argument("--config", default="configs/default.yaml", help="YAML config to inspect.")
	parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all", help="Split to inspect.")
	parser.add_argument("--device", default=None, help="Override device used for the device-side normalization sample.")
	parser.add_argument("--output_json", default=None, help="Optional path for a JSON inspection report.")
	return parser


def _select_splits(args_split: str, loaders: Mapping[str, Any]) -> list[str]:
	if args_split == "all":
		return [name for name, loader in loaders.items() if loader is not None]
	return [args_split]


def _config_name(config: Mapping[str, Any], config_path: Path) -> str:
	experiment = config.get("experiment", {}) if isinstance(config.get("experiment"), Mapping) else {}
	name = experiment.get("name")
	return str(name) if name not in (None, "", "null") else config_path.stem


def _normalization_output_dir(config: Mapping[str, Any], stats_path: Path | None) -> str | None:
	normalization = config.get("normalization", {}) if isinstance(config.get("normalization"), Mapping) else {}
	output_dir = normalization.get("output_dir")
	if output_dir not in (None, "", "null"):
		return str(output_dir)
	paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
	if paths.get("normalization_root") not in (None, "", "null"):
		return str(paths["normalization_root"])
	return str(stats_path.parent) if stats_path is not None else None


def _read_normalization_json(stats_path: Path | None) -> dict[str, Any]:
	if stats_path is None or stats_path.suffix.lower() != ".json" or not stats_path.exists():
		return {}
	try:
		with stats_path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
	except Exception as exc:
		return {"error": f"Could not read normalization JSON: {exc}"}
	return payload if isinstance(payload, dict) else {"error": "Normalization JSON did not contain an object."}


def _normalization_provenance(config: Mapping[str, Any], config_path: Path, stats_path: Path | None, input_channels: int | None = None) -> dict[str, Any]:
	payload = _read_normalization_json(stats_path)
	paths_payload = payload.get("paths", {}) if isinstance(payload.get("paths"), Mapping) else {}
	config_payload = payload.get("config", {}) if isinstance(payload.get("config"), Mapping) else {}
	data_payload = payload.get("data", {}) if isinstance(payload.get("data"), Mapping) else {}
	cache_payload = payload.get("cache", {}) if isinstance(payload.get("cache"), Mapping) else {}
	stats_payload = payload.get("stats", {}) if isinstance(payload.get("stats"), Mapping) else {}
	npz_path_value = paths_payload.get("npz_path")
	npz_path = None
	if npz_path_value not in (None, "", "null"):
		npz_path = Path(str(npz_path_value)).expanduser()
		if not npz_path.is_absolute() and stats_path is not None:
			npz_path = (stats_path.parent / npz_path).resolve()
	elif stats_path is not None and stats_path.suffix.lower() == ".npz":
		npz_path = stats_path
	current_cache = config.get("cache", {}) if isinstance(config.get("cache"), Mapping) else {}
	current_dataset_hash = compute_dataset_index_hash(config)
	warnings: list[str] = []
	current_config_name = _config_name(config, config_path)
	if config_payload.get("config_name") not in (None, "", current_config_name):
		warnings.append(f"stats config_name {config_payload.get('config_name')!r} does not match current config name {current_config_name!r}")
	if cache_payload.get("cache_version") not in (None, "", current_cache.get("cache_version")):
		warnings.append(f"stats cache_version {cache_payload.get('cache_version')!r} does not match current config cache_version {current_cache.get('cache_version')!r}")
	if paths_payload.get("dataset_index_hash") not in (None, "", current_dataset_hash):
		warnings.append("stats dataset_index_hash does not match current config")
	if stats_path is None or not stats_path.exists():
		warnings.append("stats path is missing")
	if npz_path is None or not npz_path.exists():
		warnings.append("NPZ file is missing")
	if input_channels is not None and stats_payload.get("num_channels") not in (None, "", int(input_channels)):
		warnings.append(f"stats num_channels {stats_payload.get('num_channels')!r} does not match input_channels {int(input_channels)}")
	latest_target = None
	if stats_path is not None and stats_path.is_symlink():
		latest_target = str(stats_path.resolve())
	return {
		"config_name": current_config_name,
		"config_path": str(config_path),
		"normalization_output_dir": _normalization_output_dir(config, stats_path),
		"stats_path": str(stats_path) if stats_path is not None else None,
		"npz_path": str(npz_path) if npz_path is not None else None,
		"latest_alias_target": latest_target,
		"timestamped_json_path": paths_payload.get("json_path"),
		"timestamped_npz_path": paths_payload.get("npz_path"),
		"fit_split": data_payload.get("fit_split", payload.get("fit_split", payload.get("split_used"))),
		"cache_version": cache_payload.get("cache_version", current_cache.get("cache_version")),
		"cache_dir": paths_payload.get("cache_dir", str(get_patch_cache_dir(config))),
		"dataset_index_hash": paths_payload.get("dataset_index_hash"),
		"input_channels": data_payload.get("input_channels", stats_payload.get("num_channels")),
		"warnings": warnings,
	}


def _inspect_loader(name: str, loader, config: Mapping[str, Any], device: torch.device, input_channels: int) -> dict[str, Any]:
	metadata = normalization_metadata_from_loader(loader, config, input_channels)
	dataset = getattr(loader, "dataset", None)
	stats = getattr(dataset, "normalization_stats", None)
	if isinstance(stats, Mapping):
		validate_normalization_stats(stats, input_channels, config)

	batch = next(iter(loader))
	if not isinstance(batch, (tuple, list)) or len(batch) < 2:
		raise TypeError(f"{name} loader did not return an input/target batch tuple.")
	x_raw = batch[0]
	normalizer = build_input_normalizer_for_loader(loader, device, input_channels, config)
	x_device = x_raw.to(device, non_blocking=True)
	raw_summary = input_batch_summary(x_device, prefix="raw_x")
	x_model = apply_input_normalization(x_device, normalizer, config)
	model_summary = input_batch_summary(x_model, prefix="model_x")
	return {
		"split": name,
		"num_samples": int(len(loader.dataset)),
		"batch_shape": list(x_raw.shape),
		"normalization": metadata,
		"raw_input_stats": raw_summary,
		"model_input_stats": model_summary,
	}


def main() -> None:
	args = build_arg_parser().parse_args()
	config = _ensure_config_path(load_config(args.config), args.config)
	if args.device:
		config.setdefault("training", {})
		config["training"]["device"] = args.device
	device = _get_device(config)
	train_loader, val_loader, test_loader = create_dataloaders(config)
	input_channels = _infer_input_channels_from_loader(train_loader)
	stats_path = resolve_input_normalization_stats_path(config, must_exist=False)
	loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
	try:
		stats = load_normalization_stats(stats_path) if stats_path is not None else {}
	except Exception:
		stats = {}

	report = {
		"config": str(Path(args.config).expanduser().resolve()),
		"device": str(device),
		"normalization_stats_path": str(stats_path) if stats_path is not None else None,
		"input_channels": int(input_channels),
		"normalization_provenance": _normalization_provenance(config, Path(args.config).expanduser().resolve(), stats_path, int(input_channels)),
		"loaded_stats_keys": sorted(str(key) for key in stats.keys()),
		"splits": [],
	}
	for split_name in _select_splits(args.split, loaders):
		loader = loaders.get(split_name)
		if loader is None:
			continue
		report["splits"].append(_inspect_loader(split_name, loader, config, device, input_channels))

	print(json.dumps(report, indent=2, sort_keys=True))
	if args.output_json:
		output_path = Path(args.output_json).expanduser().resolve()
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
