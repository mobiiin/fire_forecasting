"""Generate and optionally run full CAWFE-Latte ablation configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.config import load_config
from src.training.hyperparameter_tuning import force_cawfe_latte, make_portable_tuned_config_paths, save_yaml
from src.training.train import _ensure_config_path, train_model_from_config


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


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate and optionally run full CAWFE-Latte ablation YAML configs.")
	parser.add_argument("--base_config", default="configs/default.yaml", help="Base YAML config.")
	parser.add_argument("--output_dir", default="configs/ablations/cawfe_latte/", help="Directory for generated configs.")
	parser.add_argument("--run", action="store_true", help="Run each generated ablation sequentially through the shared trainer.")
	return parser


def _ablation_name(filename: str) -> str:
	return Path(filename).stem


def _ablation_config(base_config: dict, filename: str, overrides: dict) -> dict:
	name = _ablation_name(filename)
	config = force_cawfe_latte(deepcopy(base_config))
	section = dict(config.get("cawfe_latte", {}))
	section.update(overrides)
	config["cawfe_latte"] = section
	make_portable_tuned_config_paths(config, base_config)

	checkpoint_dir = REPO_ROOT / "artifacts" / "checkpoints" / "cawfe_latte_ablations" / name
	checkpoint_config = dict(config.get("checkpoint", {}))
	checkpoint_config["path"] = str(checkpoint_dir / "latest_model.pt")
	checkpoint_config["best_path"] = str(checkpoint_dir / "best_model.pt")
	config["checkpoint"] = checkpoint_config

	log_dir = REPO_ROOT / "artifacts" / "logs" / "cawfe_latte_ablations" / name
	logging_config = dict(config.get("logging", {}))
	logging_config["run_name"] = f"cawfe_latte_ablation_{name}"
	logging_config["training_log_path"] = str(log_dir / "training_log.csv")
	logging_config["timing_log_path"] = str(log_dir / "timing_log.csv")
	config["logging"] = logging_config

	training_config = dict(config.get("training", {}))
	training_config["run_name"] = f"cawfe_latte_ablation_{name}"
	training_config["run_test_after_training"] = False
	training_config["run_external_test_after_training"] = False
	output_config = dict(training_config.get("output", {}))
	output_config["update_architecture_latest_checkpoint"] = False
	training_config["output"] = output_config
	config["training"] = training_config
	return config


def main() -> None:
	args = build_argument_parser().parse_args()
	base_config = _ensure_config_path(load_config(args.base_config), args.base_config)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	for filename, overrides in ABLATIONS.items():
		config = _ablation_config(base_config, filename, overrides)
		path = output_dir / filename
		save_yaml(path, config)
		command = f"python scripts/train_forecasting_model.py --config {path}"
		if args.run:
			print(f"running: {path}")
			train_model_from_config(config)
		else:
			print(f"generated: {path}")
			print(f"train: {command}")


if __name__ == "__main__":
	main()
