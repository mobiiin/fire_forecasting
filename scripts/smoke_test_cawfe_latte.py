"""Smoke-test full CAWFE-Latte with a fake batch."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function
from src.training.train import _ensure_config_path


def _config_override(config_path: str) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	model_config["name"] = "cawfe_latte"
	config["model"] = model_config
	return config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run a fake-batch smoke test for full CAWFE-Latte.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config = _config_override(args.config)
	section = config.get("cawfe_latte", {})
	sequence_length = int(section.get("input_sequence_length", config.get("input_sequence_length", 5)))
	input_channels = int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 129)))
	output_channels = int(config.get("model", {}).get("output_channels", 4))
	patch_size = int(section.get("patch_size", config.get("patch_size", 64)))
	model = build_model_from_config(config, input_channels=input_channels)
	x = torch.randn(2, sequence_length, input_channels, patch_size, patch_size, dtype=torch.float32)
	pred = model(x)
	assert tuple(pred.shape) == (2, output_channels, patch_size, patch_size)
	pred_aux, aux = model(x, return_aux=True)
	assert tuple(pred_aux.shape) == (2, output_channels, patch_size, patch_size)
	assert "module_enabled_flags" in aux
	if bool(section.get("use_fire_front_gate", True)):
		assert aux["fire_gate_map"] is not None
	if bool(section.get("use_wind_guided_directional_module", True)):
		assert aux["wind_guidance_map"] is not None
		assert aux["wind_direction_summary"]["wind_speed"] is not None
	if bool(section.get("use_neural_operator_bottleneck", True)):
		assert "operator_residual_norm" in aux["neural_operator_energy"]
	assert "temporal_attention_weights" in aux

	criterion = get_loss_function(config)
	target = torch.randn(2, output_channels, patch_size, patch_size, dtype=torch.float32)
	target[:, 2].copy_(torch.randint(0, 2, size=(2, patch_size, patch_size), dtype=torch.int64).to(torch.float32))
	loss_result = criterion(pred, target)
	loss = loss_result["total_loss"] if isinstance(loss_result, dict) else loss_result
	loss.backward()
	assert torch.isfinite(loss)
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	parameter_bytes = parameter_count * 4
	if torch.cuda.is_available():
		torch.cuda.synchronize()
		peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
	else:
		peak_memory_mb = 0.0
	print(f"output_shape: {tuple(pred.shape)}")
	print(f"parameter_count: {parameter_count}")
	print(f"parameter_memory_mb_fp32: {parameter_bytes / (1024 ** 2):.2f}")
	print(f"enabled_modules: {getattr(model, 'enabled_modules', lambda: {})()}")
	print(f"mamba_backend_used: {getattr(model, 'mamba_backend_used', 'unknown')}")
	print(f"neural_operator_type: {section.get('neural_operator_type', 'afno')}")
	print(f"loss: {float(loss.detach().item()):.6f}")
	if torch.cuda.is_available():
		print(f"peak_gpu_memory_mb: {peak_memory_mb:.2f}")
	print("smoke test passed")


if __name__ == "__main__":
	main()
