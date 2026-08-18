"""Smoke-test the fresh CAWFE-Latte encoders and fusion stem."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import load_config
from src.models.model_factory import build_model_from_config


def main() -> None:
	parser = argparse.ArgumentParser(description="Smoke-test CAWFE-Latte encoders + fusion.")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--height", type=int, default=64)
	parser.add_argument("--width", type=int, default=64)
	args = parser.parse_args()

	config = load_config(args.config)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	config["model"] = model_config
	section = dict(config.get("cawfe_latte", {}))
	section["debug_prediction_head"] = False
	config["cawfe_latte"] = section

	input_channels = int(section.get("input_channels", model_config.get("input_channels", 129)))
	time_steps = int(section.get("input_sequence_length", config.get("input_sequence_length", 5)))
	model = build_model_from_config(config, input_channels=input_channels).eval()
	x = torch.randn(1, time_steps, input_channels, int(args.height), int(args.width))
	with torch.no_grad():
		features = model(x, return_features=True, return_attention=True)
	for key in ("atmosphere", "wind", "fire_fuel", "flux_energy", "fused"):
		print(f"{key}: {tuple(features[key].shape)}")
	attention = features["fusion_attention"]
	print(f"fusion_attention: {tuple(attention.shape)} min={attention.min().item():.4f} max={attention.max().item():.4f} mean={attention.mean().item():.4f}")


if __name__ == "__main__":
	main()
