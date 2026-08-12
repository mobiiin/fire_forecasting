"""Train the WeatherFormer-lite wildfire model."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.models.model_factory import build_model_from_config
from src.training.train import _ensure_config_path, add_early_stopping_cli_args, apply_training_cli_overrides, train_model_from_config


def _weatherformer_config(config_path: str | Path) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "weatherformer_lite"
	model_config["name"] = "weatherformer_lite"
	config["model"] = model_config

	checkpoint_config = dict(config.get("checkpoint", {}))
	checkpoint_config["path"] = "./artifacts/checkpoints/weatherformer_lite/latest_model.pt"
	checkpoint_config["best_path"] = "./artifacts/checkpoints/weatherformer_lite/best_model.pt"
	config["checkpoint"] = checkpoint_config

	logging_config = dict(config.get("logging", {}))
	logging_config["training_log_path"] = "./artifacts/logs/weatherformer_lite_training_log.csv"
	config["logging"] = logging_config
	return config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the WeatherFormer-lite wildfire model.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--run_name", default=None, help="Optional explicit run name.")
	parser.add_argument("--output_root", default=None, help="Override training.output.root_dir.")
	parser.add_argument("--overwrite_run", action="store_true", help="Allow writing into an existing explicit run directory.")
	add_early_stopping_cli_args(parser)
	return parser


def _print_model_summary(config: dict) -> None:
	if torch is None:
		return
	input_channels = int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 129)))
	model = build_model_from_config(config, input_channels=input_channels)
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	section = config.get("weatherformer_lite", {})
	sequence_length = int(section.get("input_sequence_length", config.get("input_sequence_length", 5)))
	patch_size = int(section.get("patch_size", config.get("patch_size", 64)))
	output_channels = int(config.get("model", {}).get("output_channels", 4))
	print("architecture: weatherformer_lite")
	print(f"parameter_count: {parameter_count}")
	print(f"patch_size: {patch_size}")
	print(f"window_size: {section.get('window_size', 8)}")
	print(f"num_heads: {section.get('num_heads', [4, 8])}")
	print(f"depths: {section.get('depths', [2, 2])}")
	print(f"input_shape: (B, {sequence_length}, {input_channels}, {patch_size}, {patch_size})")
	print(f"output_shape: (B, {output_channels}, {patch_size}, {patch_size})")


def main() -> None:
	args = build_argument_parser().parse_args()
	config = apply_training_cli_overrides(
		_weatherformer_config(args.config),
		run_name=args.run_name,
		output_root=args.output_root,
		overwrite_run=args.overwrite_run,
		disable_early_stopping=args.disable_early_stopping,
		early_stopping_patience=args.early_stopping_patience,
		early_stopping_monitor=args.early_stopping_monitor,
		early_stopping_min_delta=args.early_stopping_min_delta,
	)
	_print_model_summary(config)
	train_model_from_config(config)


if __name__ == "__main__":
	main()
