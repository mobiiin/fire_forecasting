"""Inspect and preview precomputed wildfire patch-cache shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.config import load_config
from src.data.cache import get_patch_cache_dir, validate_patch_cache
from src.data.cached_patch_dataset import CachedPatchDataset


def _get_pyplot():
	try:
		import matplotlib
		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except ImportError:  # pragma: no cover - optional diagnostics
		return None
	return plt


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
	resolved_path = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved_path)
	config["_config_path"] = str(resolved_path)
	return config


def _selected_splits(split: str) -> list[str]:
	if split == "all":
		return ["train", "val", "test"]
	return [split]


def _created_age_days(manifest: Mapping[str, Any]) -> float | None:
	created_at = manifest.get("created_at")
	if not created_at:
		return None
	try:
		created = datetime.fromisoformat(str(created_at))
	except ValueError:
		return None
	if created.tzinfo is None:
		created = created.replace(tzinfo=timezone.utc)
	return (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 86400.0


def _save_preview(x_tensor, y_tensor, metadata: Mapping[str, Any], output_path: Path) -> None:
	plt = _get_pyplot()
	if plt is None:
		return
	x_array = x_tensor.detach().cpu().numpy()
	y_array = y_tensor.detach().cpu().numpy()
	latest_x = x_array[-1]
	panels = [
		("input ch0", latest_x[0]),
		("input last ch", latest_x[-1]),
		("surface consumed", y_array[0]),
		("canopy consumed", y_array[1] if y_array.shape[0] > 1 else y_array[0]),
		("mask", y_array[2] if y_array.shape[0] > 2 else y_array[0]),
		("log1p energy", y_array[3] if y_array.shape[0] > 3 else y_array[-1]),
	]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=140, constrained_layout=True)
	for axis, (title, array) in zip(axes.ravel(), panels):
		image = axis.imshow(array, cmap="viridis")
		axis.set_title(title, fontsize=8)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	fig.suptitle(
		f"{metadata.get('fire_name', metadata.get('dataset_name', 'fire'))} "
		f"sample={metadata.get('sample_index')} patch={metadata.get('patch')}",
		fontsize=9,
	)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inspect a precomputed wildfire patch cache.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Split to inspect.")
	parser.add_argument("--num-previews", type=int, default=8, help="Number of random preview PNGs to save.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for preview selection.")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = _ensure_config_path(load_config(config_path), config_path)
	cache_dir = get_patch_cache_dir(config)
	summary = validate_patch_cache(config, split=args.split)
	manifest = summary["manifest"]
	age_days = _created_age_days(manifest)

	print(f"cache_dir: {cache_dir}")
	print(f"cache_version: {manifest.get('cache_version')}")
	print(f"created_at: {manifest.get('created_at')}")
	if age_days is not None:
		print(f"cache_age_days: {age_days:.2f}")
		if age_days > 25.0:
			print("WARNING: cache is older than 25 days; /scratch files may be close to purge age.")
	print(f"config_hash: {manifest.get('config_hash')}")
	print(
		f"input: T={manifest.get('input_sequence_length')} C={manifest.get('input_channels')} "
		f"horizon={manifest.get('prediction_horizon')}"
	)
	print(
		"target offsets: "
		f"from_start={manifest.get('target_offset_from_start')} "
		f"from_last_input={manifest.get('target_offset_from_last_input')}"
	)
	print(f"target_definition_version: {manifest.get('target_definition_version')}")
	print(f"target channels: {manifest.get('output_channels')}")
	print(f"patch: {manifest.get('patch_height')}x{manifest.get('patch_width')}")
	print(f"include_border_patches: {manifest.get('include_border_patches')}")
	print(f"patch_modes: {manifest.get('patch_modes')}")
	print(f"strides: {manifest.get('strides')}")
	print(f"save_normalized_inputs: {manifest.get('save_normalized_inputs')}")
	for split, split_summary in summary["splits"].items():
		print(
			f"{split}: patch_mode={split_summary['patch_mode']} stride={split_summary['stride']} "
			f"samples={split_summary['num_samples']} shards={split_summary['num_shards']} "
			f"first_X={split_summary['x_shape']} first_y={split_summary['y_shape']}"
		)

	if args.num_previews <= 0:
		return
	plt = _get_pyplot()
	if plt is None:
		print("Matplotlib is unavailable; skipping preview images.")
		return
	rng = np.random.default_rng(int(args.seed))
	for split in _selected_splits(args.split):
		dataset = CachedPatchDataset(
			cache_dir=cache_dir,
			split=split,
			config=config,
			normalization_stats=None,
			return_metadata=True,
		)
		num_previews = min(int(args.num_previews), len(dataset))
		if num_previews <= 0:
			continue
		indices = rng.choice(len(dataset), size=num_previews, replace=False)
		for preview_index, sample_index in enumerate(indices):
			x_tensor, y_tensor, metadata = dataset[int(sample_index)]
			output_path = cache_dir / "previews" / "inspect" / split / f"preview_{preview_index:04d}_idx_{int(sample_index):07d}.png"
			_save_preview(x_tensor, y_tensor, metadata, output_path)
		print(f"{split}: saved {num_previews} preview image(s) under {cache_dir / 'previews' / 'inspect' / split}")


if __name__ == "__main__":
	main()
