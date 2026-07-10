"""Train the Earthformer-lite wildfire model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.training.train import _ensure_config_path, train_model_from_config


def _earthformer_config(config_path: str | Path) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "earthformer_lite"
	model_config["name"] = "earthformer_lite"
	config["model"] = model_config

	checkpoint_config = dict(config.get("checkpoint", {}))
	checkpoint_config["path"] = "./artifacts/checkpoints/earthformer_lite/latest_model.pt"
	checkpoint_config["best_path"] = "./artifacts/checkpoints/earthformer_lite/best_model.pt"
	config["checkpoint"] = checkpoint_config

	logging_config = dict(config.get("logging", {}))
	logging_config["training_log_path"] = "./artifacts/logs/earthformer_lite_training_log.csv"
	config["logging"] = logging_config
	return config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the Earthformer-lite wildfire model.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	train_model_from_config(_earthformer_config(args.config))


if __name__ == "__main__":
	main()
