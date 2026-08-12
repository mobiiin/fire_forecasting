"""Train the ConvLSTM U-Net wildfire model through the shared pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.training.train import _ensure_config_path, add_early_stopping_cli_args, apply_training_cli_overrides, train_model_from_config


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
	parser.add_argument("--run_name", default=None, help="Optional explicit run name.")
	parser.add_argument("--output_root", default=None, help="Override training.output.root_dir.")
	parser.add_argument("--overwrite_run", action="store_true", help="Allow writing into an existing explicit run directory.")
	add_early_stopping_cli_args(parser)
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config = apply_training_cli_overrides(
		_convlstm_unet_config(args.config),
		run_name=args.run_name,
		output_root=args.output_root,
		overwrite_run=args.overwrite_run,
		disable_early_stopping=args.disable_early_stopping,
		early_stopping_patience=args.early_stopping_patience,
		early_stopping_monitor=args.early_stopping_monitor,
		early_stopping_min_delta=args.early_stopping_min_delta,
	)
	train_model_from_config(config)


if __name__ == "__main__":
	main()
