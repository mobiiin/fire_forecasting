"""Generate full CAWFE-Latte ablation configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from src.config import load_config


ABLATIONS = {
	"cawfe_latte_full.yaml": {},
	"cawfe_latte_no_vertical.yaml": {"use_vertical_atmosphere_encoder": False, "vertical_encoder_type": "flatten_conv"},
	"cawfe_latte_no_fire_encoder.yaml": {"use_fire_fuel_encoder": False},
	"cawfe_latte_no_fire_gate.yaml": {"use_fire_front_gate": False},
	"cawfe_latte_no_wind_guidance.yaml": {"use_wind_guided_directional_module": False},
	"cawfe_latte_no_operator.yaml": {"use_neural_operator_bottleneck": False, "neural_operator_type": "none"},
	"cawfe_latte_transformer_only.yaml": {"backbone_type": "transformer_only"},
	"cawfe_latte_mamba_only.yaml": {"backbone_type": "mamba_only"},
	"cawfe_latte_no_constraints.yaml": {"use_physical_output_constraints": False},
	"cawfe_latte_lite_equivalent.yaml": {"use_wind_guided_directional_module": False, "use_neural_operator_bottleneck": False, "neural_operator_type": "none"},
}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate full CAWFE-Latte ablation YAML configs.")
	parser.add_argument("--base_config", default="configs/default.yaml", help="Base YAML config.")
	parser.add_argument("--output_dir", default="configs/ablations/cawfe_latte/", help="Directory for generated configs.")
	parser.add_argument("--run", action="store_true", help="Print training commands with a run prefix. Does not launch training.")
	return parser


def _ablation_config(base_config: dict, overrides: dict) -> dict:
	config = deepcopy(base_config)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	model_config["name"] = "cawfe_latte"
	config["model"] = model_config
	section = dict(config.get("cawfe_latte", {}))
	section.update(overrides)
	config["cawfe_latte"] = section
	return config


def main() -> None:
	args = build_argument_parser().parse_args()
	base_config = load_config(args.base_config)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	for filename, overrides in ABLATIONS.items():
		config = _ablation_config(base_config, overrides)
		path = output_dir / filename
		with path.open("w", encoding="utf-8") as handle:
			yaml.safe_dump(config, handle, sort_keys=False)
		command = f"python scripts/train_cawfe_latte.py --config {path}"
		print(command if args.run else f"generated: {path}")
		if not args.run:
			print(f"train: {command}")


if __name__ == "__main__":
	main()
