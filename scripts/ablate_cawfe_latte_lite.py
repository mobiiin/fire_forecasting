"""Generate CAWFE-Latte-Lite ablation configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from src.config import load_config


ABLATIONS = {
	"cawfe_latte_lite_full.yaml": {},
	"cawfe_latte_lite_no_vertical.yaml": {
		"use_vertical_atmosphere_encoder": False,
		"vertical_encoder_type": "flatten_conv",
	},
	"cawfe_latte_lite_no_fire_encoder.yaml": {
		"use_fire_fuel_encoder": False,
	},
	"cawfe_latte_lite_no_fire_gate.yaml": {
		"use_fire_front_gate": False,
	},
	"cawfe_latte_lite_transformer_only.yaml": {
		"backbone_type": "transformer_only",
	},
	"cawfe_latte_lite_mamba_only.yaml": {
		"backbone_type": "mamba_only",
	},
	"cawfe_latte_lite_no_constraints.yaml": {
		"use_physical_output_constraints": False,
	},
	"cawfe_latte_lite_conv_only.yaml": {
		"backbone_type": "conv_only",
	},
}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate CAWFE-Latte-Lite ablation YAML configs.")
	parser.add_argument("--base_config", default="configs/default.yaml", help="Base YAML config.")
	parser.add_argument("--output_dir", default="configs/ablations/cawfe_latte_lite/", help="Directory for generated configs.")
	parser.add_argument("--run", action="store_true", help="Print training commands with a run prefix. Does not launch training.")
	return parser


def _ablation_config(base_config: dict, overrides: dict) -> dict:
	config = deepcopy(base_config)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte_lite"
	model_config["name"] = "cawfe_latte_lite"
	config["model"] = model_config
	section = dict(config.get("cawfe_latte_lite", {}))
	section.update(overrides)
	config["cawfe_latte_lite"] = section
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
		command = f"python scripts/train_cawfe_latte_lite.py --config {path}"
		print(command if args.run else f"generated: {path}")
		if not args.run:
			print(f"train: {command}")


if __name__ == "__main__":
	main()
