"""Build split-preserving, full-frame engineered tensors (no targets)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm.auto import tqdm

from src.config import compute_file_sha256, load_config
from src.data.dataset import (
	_resolve_input_channel_indices,
	_sort_chronologically,
	build_engineered_features,
	count_atmospheric_engineered_channels,
	_count_engineered_channels,
	resolve_engineered_feature_slices,
)
from src.data.fire_index import load_fire_dataset_index
from src.data.geometry import load_fire_geometry
from src.data.terrain import compute_terrain_features, find_terrain_file, parse_terrain_file, validate_terrain_features
from src.data.processed_dataset import SPLITS, frame_npz_roundtrip, validate_split_assignments


def _path(config_path: Path, value: str | Path) -> Path:
	p = Path(str(value)).expanduser()
	return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()


def _index_path(config_path: Path, config: Mapping[str, Any]) -> Path:
	value = config.get("fire_dataset_index_json")
	if value is None and isinstance(config.get("data"), Mapping):
		value = config["data"].get("fire_dataset_index")
	if value is None:
		value = "fire_dataset_index.json"
	return _path(config_path, value)


def _processed_config(config: Mapping[str, Any]) -> dict[str, Any]:
	section = config.get("processed_dataset", {})
	return dict(section) if isinstance(section, Mapping) else {}


def _split_config(config: Mapping[str, Any]) -> dict[str, list[str]]:
	for candidate in (config.get("manual_fire_split"), config.get("data", {}).get("manual_fire_split") if isinstance(config.get("data"), Mapping) else None, config.get("splits")):
		if isinstance(candidate, Mapping) and any(k in candidate for k in ("train_fires", "val_fires", "test_fires", "train", "val", "test")):
			return {s: [str(x) for x in candidate.get(f"{s}_fires", candidate.get(s, []))] for s in SPLITS}
	raise ValueError("No explicit train_fires/val_fires/test_fires split found in config")


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _channel_manifest(config: Mapping[str, Any], raw_count: int) -> dict[str, Any]:
	entries = [{"index": i, "name": f"raw_{i:03d}", "source": "raw", "description": "Raw CAWFE channel"} for i in range(raw_count)]
	slices = resolve_engineered_feature_slices(config, raw_count)
	groups = []
	for group, slc in slices.items():
		if group in {"wind_dir_cos", "wind_dir_sin"}:
			continue
		groups.extend([group] * len(range(*slc.indices(raw_count + _count_engineered_channels(config) + 1))))
	# The slice order is authoritative and matches build_engineered_features.
	engineered_names: list[str] = []
	for group, slc in slices.items():
		if group in {"wind_dir_cos", "wind_dir_sin"}:
			continue
		engineered_names.extend([f"{group}_{i}" for i in range(slc.start, slc.stop)])
	for i, name in enumerate(engineered_names, raw_count):
		entries.append({"index": i, "name": name, "source": "engineered", "description": f"Current training feature group: {name.split('_')[0]}"})
	return {"array_layout": "C,H,W", "num_raw_channels": raw_count, "num_engineered_channels": len(entries) - raw_count, "num_total_channels": len(entries), "channels": entries}


def _terrain_quicklook(path: Path, height: np.ndarray, features: np.ndarray, fire: str, metadata: Mapping[str, Any]) -> None:
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
	images = [(height, "raw elevation", None, None), (features[0], "relative elevation [0,1]", 0.0, 1.0), (features[1], "slope magnitude [0,1]", 0.0, 1.0), (features[2], "slope_x [-1,1]", -1.0, 1.0), (features[3], "slope_y [-1,1]", -1.0, 1.0)]
	for ax, (image, label, vmin, vmax) in zip(axes.flat, images):
		im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax); ax.set_title(label); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
	axes.flat[-1].axis("off")
	fig.suptitle(f"{fire} | terrain shape={tuple(height.shape)} | dx={metadata.get('dx')} dy={metadata.get('dy')}")
	fig.savefig(path, dpi=120); plt.close(fig)


def _quicklook(path: Path, x: np.ndarray, fire: str, split: str, local: int, original: int) -> None:
	import matplotlib.pyplot as plt

	channels = [84, 85, 80, 81, 82, 83]
	images = []
	labels = ["surface fuel", "canopy fuel", "surface sensible flux", "surface latent flux", "canopy sensible flux", "canopy latent flux"]
	for c in channels:
		if c < x.shape[0]:
			images.append((x[c], labels[len(images)]))
	if x.shape[0] > 83:
		images.append((x[80] + x[81] + x[82] + x[83], "total flux"))
	if x.shape[0] >= 3:
		images.append((np.sqrt(x[0] ** 2 + x[1] ** 2), "low-level wind speed"))
		images.append((np.maximum(x[2], 0), "updraft level 0"))
	fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
	for ax, (image, label) in zip(axes.flat, images):
		im = ax.imshow(image, cmap="viridis")
		ax.set_title(f"{label}\nmin={np.nanmin(image):.3g} max={np.nanmax(image):.3g}")
		ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
	for ax in axes.flat[len(images):]: ax.axis("off")
	fig.suptitle(f"{fire} | {split} | local={local} original={original} | shape={tuple(x.shape)}")
	fig.savefig(path, dpi=120); plt.close(fig)


def build(args: argparse.Namespace) -> dict[str, Any]:
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	pc = _processed_config(config)
	output = Path(args.output_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
	if not output.is_absolute(): output = (config_path.parent / output).resolve()
	output.mkdir(parents=True, exist_ok=True)
	log_path = output / "logs" / "build_engineered_frame_dataset.log"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

	index_path = _index_path(config_path, config)
	index = load_fire_dataset_index(index_path)
	index_fires = index.get("fires", index)
	splits = _split_config(config)
	assignments = validate_split_assignments(index_fires, splits["train"], splits["val"], splits["test"])
	selected = {fire for names in assignments.values() for fire in names}
	if args.fires != "all": selected &= {x.strip() for x in args.fires.split(",")}
	selected_splits = set(args.splits)
	raw_count = int(config.get("input_channel_count", 86))
	input_indices = _resolve_input_channel_indices(config, raw_count)
	if len(input_indices) != raw_count: raw_count = len(input_indices)
	channel_manifest = _channel_manifest(config, raw_count)
	_write_json(output / "channel_manifest.json", channel_manifest)

	for split in SPLITS:
		_write_json(output / "indices" / "splits" / f"{split}_fires.json", assignments[split])
	_write_json(output / "indices" / "splits" / "split_summary.json", {"split_mode": config.get("split_mode", "manual_fire_holdout"), "splits": assignments, "not_selected_by_split": sorted(set(index_fires) - selected)})
	_write_json(output / "split_manifest.json", {"split_mode": config.get("split_mode", "manual_fire_holdout"), "splits": assignments})

	fires_manifest: dict[str, Any] = {}
	selected_fire_rows = [(split, fire_name) for split, fire_names in assignments.items() if split in selected_splits for fire_name in fire_names if fire_name in selected]
	for split, fire_name in tqdm(selected_fire_rows, desc="Fires", unit="fire"):
		record = index_fires[fire_name]
		data_dir = Path(str(record["data_dir"])).expanduser()
		files = _sort_chronologically(list(data_dir.glob(str(record.get("file_pattern", config.get("file_pattern", "*.npy"))))))
		trim = record.get("temporal_trim", {}) if isinstance(record.get("temporal_trim"), Mapping) else {}
		start = int(trim.get("trim_start_index", 0)); end = int(trim.get("trim_end_index", len(files) - 1))
		selected_files = files[start:end + 1]
		if args.max_frames_per_fire: selected_files = selected_files[:args.max_frames_per_fire]
		if not selected_files: raise ValueError(f"No frames remain after trimming for {fire_name}")
		first = np.asarray(np.load(selected_files[0], mmap_mode="r", allow_pickle=False))
		if first.ndim != 3 or first.shape[-1] < max(input_indices) + 1: raise ValueError(f"Expected H,W,C raw frame for {fire_name}, got {first.shape}")
		h, w = first.shape[:2]
		fire_root = output / "fires" / fire_name
		frames_dir, quick_dir, geom_dir = fire_root / "frames", fire_root / "quicklooks", fire_root / "geometry"
		terrain_dir = fire_root / "terrain"
		for d in (frames_dir, geom_dir): d.mkdir(parents=True, exist_ok=True)
		terrain_config = config.get("terrain", {}) if isinstance(config.get("terrain"), Mapping) else {}
		terrain_enabled = bool(terrain_config.get("enabled", False))
		terrain_required = bool(terrain_config.get("required", False))
		terain_manifest = {"available": False}
		if terrain_enabled:
			source_terrain = Path(str(record["terrain_path"])).expanduser() if record.get("terrain_path") else find_terrain_file(data_dir)
			if source_terrain is None:
				if terrain_required: raise FileNotFoundError(f"Required .terrain file was not found for {fire_name}: {data_dir}")
				logging.warning("No terrain file found for %s; continuing without terrain", fire_name)
			else:
				terrain_dir.mkdir(parents=True, exist_ok=True)
				height, terrain_meta = parse_terrain_file(source_terrain, expected_shape=(h, w))
				feature_config = terrain_config.get("features", {}) if isinstance(terrain_config.get("features"), Mapping) else {}
				features, feature_meta = compute_terrain_features(height, dx=terrain_meta.get("dx"), dy=terrain_meta.get("dy"), normalization_config=feature_config, x_axis=int(terrain_meta.get("x_axis", 1)), y_axis=int(terrain_meta.get("y_axis", 0)))
				if tuple(height.shape) != (h, w) or tuple(features.shape) != (4, h, w): raise ValueError(f"Terrain shape mismatch for {fire_name}: height={height.shape}, features={features.shape}, frame={(h, w)}")
				validate_terrain_features(features.astype(np.float32, copy=False), expected_shape=(h, w), context=f"terrain_features for {fire_name}")
				original_path = terrain_dir / "original.terrain"
				if bool(terrain_config.get("copy_original_file", True)): shutil.copy2(source_terrain, original_path)
				np.save(terrain_dir / "terrain_height.npy", height.astype(np.float32)); np.save(terrain_dir / "terrain_features.npy", features.astype(np.float32))
				terrain_meta.update(feature_meta); terrain_meta["original_terrain_path"] = str(original_path.relative_to(fire_root)); terrain_meta["height_path"] = "terrain_height.npy"; terrain_meta["features_path"] = "terrain_features.npy"
				_write_json(terrain_dir / "terrain_metadata.json", terrain_meta)
				quick_cfg = terrain_config.get("quicklook", {}) if isinstance(terrain_config.get("quicklook"), Mapping) else {}
				if bool(quick_cfg.get("enabled", True)): _terrain_quicklook(terrain_dir / "terrain_quicklook.png", height, features, fire_name, terrain_meta)
				terrain_manifest = {"available": True, "original_terrain_path": str(original_path.relative_to(fire_root)) if original_path.exists() else None, "height_path": "terrain/terrain_height.npy", "features_path": "terrain/terrain_features.npy", "metadata_path": "terrain/terrain_metadata.json", "feature_shape": list(features.shape), "feature_channels": feature_meta["feature_channels"]}
		# fire_dataset_index records the orientation discovered during indexing.
		# Some CAWFE .geom grids are transposed relative to their H,W frame
		# tensors (for example THOMAS/1204). Honor that metadata locally rather
		# than requiring every experiment config to enable a global transpose.
		geometry_config = dict(config)
		geometry_section = dict(config.get("geometry", {})) if isinstance(config.get("geometry"), Mapping) else {}
		if bool(record.get("geom_requires_transpose", False)):
			geometry_section["allow_area_transpose_if_needed"] = True
		geometry_config["geometry"] = geometry_section
		area = load_fire_geometry(data_dir, geometry_config, geom_path=Path(str(record["geom_path"])) if record.get("geom_path") else None, terrain_path=Path(str(record["terrain_path"])) if record.get("terrain_path") else None, expected_shape=(h, w))["area_2d_m2"]
		if bool(pc.get("save_area_2d", True)): np.save(geom_dir / "area_2d.npy", area.astype(np.float32))
		_write_json(geom_dir / "geom_metadata.json", {"shape": list(area.shape), "dtype": "float32", "source_geom": record.get("geom_path"), "area_units": "m^2"})
		frame_rows = []
		for local, frame_path in enumerate(tqdm(selected_files, desc=f"{split}/{fire_name} frames", unit="frame", leave=False)):
			out_path = frames_dir / f"frame_{local:06d}.npz"
			if args.skip_existing and out_path.exists():
				with np.load(out_path, allow_pickle=False) as z: x = np.asarray(z["x_engineered"])
			else:
				raw = np.asarray(np.load(frame_path, allow_pickle=False), dtype=np.float32)
				base = raw[:, :, input_indices]
				eng = build_engineered_features(np.expand_dims(raw, 0), files, start_index=start + local, config=config, energy_geometry={"area_2d_m2": area})[0]
				x = np.concatenate([base, eng], axis=-1).transpose(2, 0, 1).astype(np.float32)
				if x.shape[0] != channel_manifest["num_total_channels"]: raise ValueError(f"Channel count mismatch for {fire_name}: {x.shape[0]} vs {channel_manifest['num_total_channels']}")
				if not np.isfinite(x).all(): logging.warning("NaN/Inf detected in %s", frame_path)
				frame_npz_roundtrip(out_path, x, raw.transpose(2, 0, 1) if bool(pc.get("save_raw_channels", True)) else None, frame_index_local=local, frame_index_original=start + local, fire_name=fire_name, split=split)
			quick_path = None
			quick_cfg = pc.get("quicklook", {}) if isinstance(pc.get("quicklook"), Mapping) else {}
			if (args.quicklooks and bool(quick_cfg.get("enabled", True)) and (local == 0 or local == len(selected_files) - 1 or local % int(quick_cfg.get("every_n_frames", 100)) == 0)) and sum(1 for row in frame_rows if row.get("quicklook_path")) < int(quick_cfg.get("max_per_fire", 20)):
				quick_dir.mkdir(parents=True, exist_ok=True); quick_path = quick_dir / f"frame_{local:06d}_core.png"; _quicklook(quick_path, x, fire_name, split, local, start + local)
			frame_rows.append({"local_index": local, "original_index": start + local, "path": str(out_path.relative_to(output)), "source_raw_file": str(frame_path), "quicklook_path": str(quick_path.relative_to(output)) if quick_path else None})
		_write_json(fire_root / "fire_manifest.json", {"fire_name": fire_name, "split": split, "source_fire_path": str(data_dir), "num_raw_frames": len(files), "num_processed_frames": len(selected_files), "temporal_trim": dict(trim), "frame_shape": [channel_manifest["num_total_channels"], h, w], "raw_shape": [86, h, w], "frame_format": "npz", "array_layout": "C,H,W", "frames_dir": "frames", "geometry": {"area_2d_path": str((geom_dir / "area_2d.npy").relative_to(fire_root)), "height": h, "width": w}, "terrain": terrain_manifest})
		(fire_root / "frame_manifest.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in frame_rows) + "\n", encoding="utf-8")
		fires_manifest[fire_name] = {"split": split, "manifest": str((fire_root / "fire_manifest.json").relative_to(output)), "num_processed_frames": len(selected_files), "shape": [channel_manifest["num_total_channels"], h, w], "terrain": terrain_manifest}

	config_text = config_path.read_text(encoding="utf-8")
	_write_json(output / "dataset_manifest.json", {"dataset_version": pc.get("version", "cawfe_engineered_v1"), "created_at": datetime.now(timezone.utc).isoformat(), "created_by": "scripts/build_engineered_frame_dataset.py", "config": {"config_path": str(config_path), "config_path_absolute": str(config_path), "config_sha256": compute_file_sha256(config_path), "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()}, "source": {"fire_dataset_index": str(index_path), "fire_dataset_index_hash": compute_file_sha256(index_path)}, "processed_dataset": {"root": str(output), "frame_format": "npz", "array_layout": "C,H,W", "dtype": "float32", "compression": "compressed"}, "channels": {"channel_manifest_path": "channel_manifest.json", "num_raw_channels": raw_count, "num_engineered_total_channels": channel_manifest["num_total_channels"]}, "splits": {"split_mode": config.get("split_mode", "manual_fire_holdout"), **assignments}, "fires": fires_manifest})
	(output / "processing_config.yaml").write_text(config_text, encoding="utf-8")
	(output / "indices" / "temporal").mkdir(parents=True, exist_ok=True)
	(output / "indices" / "temporal" / "README.md").write_text("# Temporal sample indices\n\nTemporal sample indices will be built later. They will reference fire_name, patch_id, input_indices, current_index, target_index, horizon, and pattern. No targets or X/y samples are created by this stage.\n", encoding="utf-8")
	return {"output_root": str(output), "fires": len(fires_manifest), "channels": channel_manifest["num_total_channels"]}


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--output_root"); p.add_argument("--fires", default="all"); p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS)); p.add_argument("--overwrite", action="store_true"); p.add_argument("--skip_existing", action="store_true"); p.add_argument("--max_frames_per_fire", type=int); p.add_argument("--no_quicklooks", dest="quicklooks", action="store_false", default=True); p.add_argument("--dtype", default="float32"); p.add_argument("--compression", default="compressed"); p.add_argument("--include_unsplit_fires", action="store_true")
	args = p.parse_args(); print(json.dumps(build(args), indent=2))


if __name__ == "__main__": main()
