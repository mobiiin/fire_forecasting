"""Smoke-test CAWFE-Latte v1 forward, loss, and backward."""

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
from src.training.losses import get_loss_function, extract_prediction, extract_aux_outputs


def main() -> None:
	parser = argparse.ArgumentParser(description="Smoke-test trainable CAWFE-Latte v1.")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--height", type=int, default=64)
	parser.add_argument("--width", type=int, default=64)
	args = parser.parse_args()

	config = load_config(args.config)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	config["model"] = model_config
	section = dict(config.get("cawfe_latte", {}))
	section.setdefault("version", "v1_end_to_end")
	config["cawfe_latte"] = section

	input_channels = int(section.get("input_channels", model_config.get("input_channels", 129)))
	time_steps = int(section.get("input_sequence_length", config.get("input_sequence_length", 5)))
	model = build_model_from_config(config, input_channels=input_channels)
	criterion = get_loss_function(config)
	x = torch.randn(1, time_steps, input_channels, int(args.height), int(args.width))
	y = torch.randn(1, 4, int(args.height), int(args.width)) * 0.1
	y[:, 2:3] = (torch.rand(1, 1, int(args.height), int(args.width)) > 0.7).float()
	y[:, 3:4] = torch.rand(1, 1, int(args.height), int(args.width))
	output = model(x)
	prediction = extract_prediction(output)
	aux = extract_aux_outputs(output)
	losses = criterion(output, y)
	loss = losses["total_loss"]
	loss.backward()
	print(f"prediction shape: {tuple(prediction.shape)}")
	print(f"aux shape: {tuple(aux['aux_fire_support_logits'].shape) if 'aux_fire_support_logits' in aux else None}")
	print(f"total loss: {float(loss.detach().item()):.6f}")
	for key, value in sorted(losses.items()):
		print(f"{key}: {float(value.detach().item()):.6f}")
	print("CAWFE-Latte v1 smoke test passed")


if __name__ == "__main__":
	main()
