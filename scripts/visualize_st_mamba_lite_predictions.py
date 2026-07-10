"""Visualize predictions from an ST-Mamba-Lite checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import yaml

from src.config import load_config
from scripts.visualize_predictions import visualize_predictions


def _override_and_write_temp_config(config_path: str | Path) -> str:
	config = load_config(config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "st_mamba_lite"
	model_config["name"] = "st_mamba_lite"
	config["model"] = model_config
	checkpoint_config = dict(config.get("checkpoint", {}))
	checkpoint_config["path"] = "./artifacts/checkpoints/st_mamba_lite/latest_model.pt"
	checkpoint_config["best_path"] = "./artifacts/checkpoints/st_mamba_lite/best_model.pt"
	config["checkpoint"] = checkpoint_config
	with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
		yaml.safe_dump(config, handle, sort_keys=False)
		return handle.name


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visualize ST-Mamba-Lite wildfire predictions.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Which split to visualize.")
	parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	temp_config_path = _override_and_write_temp_config(args.config)
	visualize_predictions(temp_config_path, num_samples=args.num_samples, split=args.split)


if __name__ == "__main__":
	main()
