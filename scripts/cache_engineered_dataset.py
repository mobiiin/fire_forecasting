"""Optionally cache base + engineered per-timestep feature tensors to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.dataset import build_engineered_features, _resolve_input_channel_indices, _sort_chronologically


def _resolve_path(base_path: Path, configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (base_path.parent / path).resolve()


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Cache base + engineered per-timestep wildfire tensors.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--output_dir", required=True, help="Directory for cached engineered tensors.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	data_dir = _resolve_path(config_path, config["data_dir"])
	files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{config['file_pattern']}'.")

	output_dir = Path(args.output_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	input_channel_count = int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0)))
	input_channel_indices = _resolve_input_channel_indices(config, input_channel_count)

	print("Caching engineered tensors with current/previous-frame features only.")
	print("Labels remain dynamic during training and are not cached here.")

	for file_index, file_path in enumerate(files):
		raw_frame = np.load(file_path, mmap_mode="r", allow_pickle=False)
		raw_frame = np.asarray(raw_frame, dtype=np.float32)
		base_frame = raw_frame[:, :, input_channel_indices]
		engineered = build_engineered_features(
			input_frames=np.expand_dims(raw_frame, axis=0),
			file_paths=files,
			start_index=file_index,
			config=config,
		)[0]
		cached_frame = np.concatenate([base_frame, engineered], axis=-1).astype(np.float32, copy=False)
		output_path = output_dir / file_path.name
		np.save(output_path, cached_frame)
		if file_index < 3:
			print(f"cached: {output_path} shape={cached_frame.shape}")

	print(f"cached files: {len(files)}")
	print(f"output_dir: {output_dir}")


if __name__ == "__main__":
	main()
