#!/usr/bin/env python3
"""Fail-fast safety validation for every canonical Table 2 baseline config."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import load_config
from src.evaluation.table2_protocol import DETERMINISTIC_BASELINES, validate_table2_config


REGISTRY = Path("configs/table2_baselines/table2_baselines.yaml")


def main() -> None:
	registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
	seeds = [int(seed) for seed in registry["seeds"]]
	for baseline, entry in registry["baselines"].items():
		config = load_config(entry["config_path"])
		if baseline in DETERMINISTIC_BASELINES:
			validate_table2_config(config, baseline, None)
		else:
			for seed in seeds:
				resolved = dict(config)
				training = dict(config.get("training", {}))
				training["seed"] = seed
				resolved["training"] = training
				resolved["seed"] = seed
				validate_table2_config(resolved, baseline, seed)
	print("Table 2 baseline config safety check: PASS")
	print("Baselines: " + ", ".join(registry["baselines"]))
	print("Learned seeds: " + ", ".join(str(seed) for seed in seeds))
	print("Locked test is disabled during training and reserved for frozen-checkpoint evaluation.")


if __name__ == "__main__":
	main()
