"""Apply CAWFE-Latte best hyperparameters to a full-training config."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.training.hyperparameter_tuning import load_json, make_final_config_from_best_params, save_yaml
from src.training.train import _ensure_config_path


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Apply tuned CAWFE-Latte hyperparameters to a base config.")
	parser.add_argument("--base_config", default="configs/default.yaml", help="Base YAML config.")
	parser.add_argument("--best_params", required=True, help="Path to best_params.json from tuning.")
	parser.add_argument("--output_config", required=True, help="Path for the resolved tuned YAML config.")
	parser.add_argument("--keep_trial_epochs", action="store_true", help="Keep trial epoch limits instead of restoring full base epochs.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	base_config = _ensure_config_path(load_config(args.base_config), args.base_config)
	best_params = load_json(args.best_params)
	config = make_final_config_from_best_params(
		base_config=base_config,
		best_params=best_params,
		keep_trial_epochs=bool(args.keep_trial_epochs),
	)
	save_yaml(args.output_config, config)
	print(f"Wrote tuned CAWFE-Latte config: {args.output_config}")


if __name__ == "__main__":
	main()
