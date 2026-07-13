"""Train the ConvLSTM U-Net wildfire model through the shared pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.training.train import _ensure_config_path, train_model_from_config


def _convlstm_unet_config(config_path: str | Path) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "convlstm_unet"
	model_config["name"] = "convlstm_unet"
	config["model"] = model_config
	return config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the ConvLSTM U-Net wildfire model.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	train_model_from_config(_convlstm_unet_config(args.config))


if __name__ == "__main__":
	main()
