"""Train the CAWFE-Latte-Lite wildfire model."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.models.model_factory import build_model_from_config
from src.training.train import _ensure_config_path, apply_training_cli_overrides, train_model_from_config


def _cawfe_latte_config(config_path: str | Path) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte_lite"
	model_config["name"] = "cawfe_latte_lite"
	config["model"] = model_config

	checkpoint_config = dict(config.get("checkpoint", {}))
	checkpoint_config["path"] = "./artifacts/checkpoints/cawfe_latte_lite/latest_model.pt"
	checkpoint_config["best_path"] = "./artifacts/checkpoints/cawfe_latte_lite/best_model.pt"
	config["checkpoint"] = checkpoint_config

	logging_config = dict(config.get("logging", {}))
	logging_config["training_log_path"] = "./artifacts/logs/cawfe_latte_lite_training_log.csv"
	config["logging"] = logging_config
	return config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the CAWFE-Latte-Lite wildfire model.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--run_name", default=None, help="Optional explicit run name.")
	parser.add_argument("--output_root", default=None, help="Override training.output.root_dir.")
	parser.add_argument("--overwrite_run", action="store_true", help="Allow writing into an existing explicit run directory.")
	return parser


def _print_model_summary(config: dict) -> None:
	if torch is None:
		return
	input_channels = int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 129)))
	model = build_model_from_config(config, input_channels=input_channels)
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	section = config.get("cawfe_latte_lite", {})
	sequence_length = int(section.get("input_sequence_length", config.get("input_sequence_length", 5)))
	patch_size = int(section.get("patch_size", config.get("patch_size", 64)))
	output_channels = int(config.get("model", {}).get("output_channels", 4))
	print("architecture: cawfe_latte_lite")
	print(f"parameter_count: {parameter_count}")
	print(f"enabled_modules: {getattr(model, 'enabled_modules', lambda: {})()}")
	print(f"vertical_encoder_type: {section.get('vertical_encoder_type', 'attention')}")
	print(f"backbone_type: {section.get('backbone_type', 'hybrid_transformer_mamba')}")
	print(f"mamba_backend: {section.get('mamba_backend', 'auto')}")
	print(f"mamba_backend_used: {getattr(model, 'mamba_backend_used', 'unknown')}")
	print(f"patch_size: {patch_size}")
	print(f"input_shape: (B, {sequence_length}, {input_channels}, {patch_size}, {patch_size})")
	print(f"output_shape: (B, {output_channels}, {patch_size}, {patch_size})")


def main() -> None:
	args = build_argument_parser().parse_args()
	config = apply_training_cli_overrides(
		_cawfe_latte_config(args.config),
		run_name=args.run_name,
		output_root=args.output_root,
		overwrite_run=args.overwrite_run,
	)
	_print_model_summary(config)
	train_model_from_config(config)


if __name__ == "__main__":
	main()
