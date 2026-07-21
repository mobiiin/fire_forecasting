"""Lightweight diagnostics for implemented model architectures."""

from __future__ import annotations

import argparse

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.models.architecture_registry import get_architecture_spec
from src.models.model_factory import build_model_from_config
from src.training.train import _ensure_config_path


def _build_config_for_architecture(base_config: dict, architecture: str) -> dict:
	config = dict(base_config)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = architecture
	model_config["name"] = architecture
	config["model"] = model_config
	if architecture == "st_mamba_lite":
		checkpoint_config = dict(config.get("checkpoint", {}))
		checkpoint_config["path"] = "./artifacts/checkpoints/st_mamba_lite/latest_model.pt"
		checkpoint_config["best_path"] = "./artifacts/checkpoints/st_mamba_lite/best_model.pt"
		config["checkpoint"] = checkpoint_config
	elif architecture == "weatherformer_lite":
		checkpoint_config = dict(config.get("checkpoint", {}))
		checkpoint_config["path"] = "./artifacts/checkpoints/weatherformer_lite/latest_model.pt"
		checkpoint_config["best_path"] = "./artifacts/checkpoints/weatherformer_lite/best_model.pt"
		config["checkpoint"] = checkpoint_config
	elif architecture == "earthformer_lite":
		checkpoint_config = dict(config.get("checkpoint", {}))
		checkpoint_config["path"] = "./artifacts/checkpoints/earthformer_lite/latest_model.pt"
		checkpoint_config["best_path"] = "./artifacts/checkpoints/earthformer_lite/best_model.pt"
		config["checkpoint"] = checkpoint_config
	elif architecture == "cawfe_latte_lite":
		checkpoint_config = dict(config.get("checkpoint", {}))
		checkpoint_config["path"] = "./artifacts/checkpoints/cawfe_latte_lite/latest_model.pt"
		checkpoint_config["best_path"] = "./artifacts/checkpoints/cawfe_latte_lite/best_model.pt"
		config["checkpoint"] = checkpoint_config
	elif architecture == "cawfe_latte":
		checkpoint_config = dict(config.get("checkpoint", {}))
		checkpoint_config["path"] = "./artifacts/checkpoints/cawfe_latte/latest_model.pt"
		checkpoint_config["best_path"] = "./artifacts/checkpoints/cawfe_latte/best_model.pt"
		config["checkpoint"] = checkpoint_config
	return config


def _smoke_forward(config: dict, architecture: str) -> dict[str, str]:
	if torch is None:
		return {"status": "skipped", "detail": "PyTorch not installed"}
	section = config.get(architecture, {})
	sequence_length = int(section.get("input_sequence_length", config.get("input_sequence_length", 5))) if isinstance(section, dict) else int(config.get("input_sequence_length", 5))
	input_channels = int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 129)))
	output_channels = int(config.get("model", {}).get("output_channels", 4))
	patch_size = int(section.get("patch_size", config.get("patch_size", 64))) if isinstance(section, dict) else int(config.get("patch_size", 64))
	model = build_model_from_config(config, input_channels=input_channels)
	x = torch.randn(1, sequence_length, input_channels, patch_size, patch_size)
	if torch.cuda.is_available():
		model = model.to("cuda")
		x = x.to("cuda")
		torch.cuda.reset_peak_memory_stats()
	y = model(x)
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	peak_memory_mb = 0.0
	if torch.cuda.is_available():
		torch.cuda.synchronize()
		peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
	assert tuple(y.shape) == (1, output_channels, patch_size, patch_size)
	return {
		"status": "ok",
		"detail": f"output_shape={tuple(y.shape)} params={parameter_count} peak_memory_mb={peak_memory_mb:.2f}",
	}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Diagnose implemented model architectures.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	base_config = _ensure_config_path(load_config(args.config), args.config)
	for architecture in ("convlstm_unet", "earthformer_lite", "st_mamba_lite", "weatherformer_lite", "cawfe_latte_lite", "cawfe_latte"):
		config = _build_config_for_architecture(base_config, architecture)
		spec = get_architecture_spec(architecture)
		result = _smoke_forward(config, architecture)
		model = build_model_from_config(config, input_channels=int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 129)))) if torch is not None else None
		parameter_count = sum(parameter.numel() for parameter in model.parameters()) if model is not None else 0
		print(f"[{architecture}]")
		print(f"expected_input_shape: {spec.expected_input_shape}")
		print(f"expected_output_shape: {spec.expected_output_shape}")
		print(f"patch_divisibility: {spec.patch_divisibility}")
		print(f"supports_patch_cache: {spec.supports_patch_cache}")
		print(f"supports_tiled_inference: {spec.supports_tiled_inference}")
		print(f"custom_architecture: {spec.custom_architecture}")
		print(f"ablation_ready: {spec.ablation_ready}")
		print(f"paper_main_model: {spec.paper_main_model}")
		print(f"parameter_count: {parameter_count}")
		print(f"smoke_forward: {result['status']} | {result['detail']}")


if __name__ == "__main__":
	main()
