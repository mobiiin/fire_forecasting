from __future__ import annotations

import os
from pathlib import Path

from scripts.ablate_cawfe_latte import ABLATIONS, _ablation_config
from src.training.hyperparameter_tuning import (
	apply_params,
	generate_trial_params,
	make_final_config_from_best_params,
	make_trial_config,
)


def _base_config(tmp_path: Path) -> dict:
	config_path = tmp_path / "configs" / "default.yaml"
	return {
		"config_path": str(config_path),
		"_config_path": str(config_path),
		"fire_dataset_index_json": "../fire_dataset_index.json",
		"model": {"architecture": "convlstm_unet", "name": "convlstm_unet"},
		"training": {
			"epochs": 20,
			"performance": {"max_train_batches_per_epoch": None},
			"validation": {
				"mode": "fixed_subset_every_epoch",
				"max_val_batches_per_epoch": 200,
				"fixed_subset_seed": 42,
				"fixed_subset_shuffle": False,
				"use_same_metric_for_checkpointing": True,
			},
			"early_stopping_patience": 3,
		},
		"multitask": {
			"surface_loss_weight": 1.0,
			"canopy_loss_weight": 1.0,
			"segmentation_loss_weight": 5.0,
			"energy_loss_weight": 1.0,
		},
		"normalization": {"path": "./artifacts/normalization_stats.npz"},
		"cache": {"use_precomputed_patches": True, "allow_config_hash_mismatch": False},
		"cawfe_latte": {
			"backbone_dim": 96,
			"fused_dim": 96,
			"bottleneck_dim": 192,
			"decoder_channels": [192, 96, 64],
			"atm_embed_dim": 48,
			"fire_embed_dim": 48,
			"neural_operator_depth": 2,
			"neural_operator_type": "afno",
			"use_neural_operator_bottleneck": True,
		},
		"hparam_tuning": {
			"architecture": "cawfe_latte",
			"trial_early_stopping_patience": 3,
		},
	}


def test_random_search_generates_requested_trial_configs(tmp_path: Path) -> None:
	space = {
		"training.learning_rate": [1e-4, 3e-4],
		"cawfe_latte.backbone_dim": [64, 96],
	}
	trials = generate_trial_params(space, method="random", num_trials=5, seed=7)
	configs = [
		make_trial_config(
			_base_config(tmp_path),
			params,
			trial_id=trial_id,
			output_dir=tmp_path / "hparam",
			trial_max_epochs=2,
			max_train_batches_per_epoch=10,
			max_val_batches_per_epoch=4,
		)
		for trial_id, params in enumerate(trials)
	]

	assert len(configs) == 5
	assert all(config["model"]["architecture"] == "cawfe_latte" for config in configs)
	assert configs[0]["training"]["epochs"] == 2
	assert configs[0]["training"]["run_name"] == "cawfe_latte_hparam_trial_000"
	assert configs[0]["training"]["performance"]["max_train_batches_per_epoch"] == 10
	assert configs[0]["training"]["validation"]["mode"] == "fixed_subset_every_epoch"
	assert configs[0]["training"]["validation"]["max_val_batches_per_epoch"] == 4
	assert configs[0]["cache"]["allow_config_hash_mismatch"] is True


def test_apply_tuned_params_modifies_nested_and_alias_keys(tmp_path: Path) -> None:
	config = apply_params(
		_base_config(tmp_path),
		{
			"training.learning_rate": 3e-4,
			"training.loss.mask_weight": 2.0,
			"training.loss.energy_weight": 2.0,
			"cawfe_latte.backbone_dim": 64,
		},
	)

	assert config["model"]["architecture"] == "cawfe_latte"
	assert config["training"]["learning_rate"] == 3e-4
	assert config["multitask"]["segmentation_loss_weight"] == 2.0
	assert config["multitask"]["energy_loss_weight"] == 2.0
	assert config["cawfe_latte"]["fused_dim"] == 64
	assert config["cawfe_latte"]["bottleneck_dim"] == 128
	assert config["cawfe_latte"]["decoder_channels"] == [128, 64, 64]
	assert config["cawfe_latte"]["atm_embed_dim"] == 32
	assert config["cawfe_latte"]["fire_embed_dim"] == 32
	assert config["cawfe_latte"]["num_heads"] == [4, 8]


def test_backbone_dim_search_values_resolve_valid_attention_heads(tmp_path: Path) -> None:
	for backbone_dim, expected_heads in [(64, [4, 8]), (96, [4, 6]), (128, [4, 8])]:
		config = apply_params(_base_config(tmp_path), {"cawfe_latte.backbone_dim": backbone_dim})
		num_heads = config["cawfe_latte"]["num_heads"]
		stage_dims = [backbone_dim, 2 * backbone_dim]

		assert num_heads == expected_heads
		assert all(dim % heads == 0 for dim, heads in zip(stage_dims, num_heads))


def test_best_params_schema_builds_full_cawfe_latte_config(tmp_path: Path) -> None:
	best_params = {
		"model_architecture": "cawfe_latte",
		"selection_metric": "val_multitask_loss",
		"selection_mode": "min",
		"best_trial_id": 3,
		"best_score": 0.12345,
		"params": {
			"cawfe_latte.backbone_dim": 128,
			"training.learning_rate": 5e-4,
			"training.loss.energy_weight": 2.0,
		},
	}

	config = make_final_config_from_best_params(_base_config(tmp_path), best_params)

	assert config["model"]["architecture"] == "cawfe_latte"
	assert config["training"]["epochs"] == 20
	assert config["training"]["performance"]["max_train_batches_per_epoch"] is None
	assert "early_stopping_patience" not in config["training"]
	assert config["training"]["learning_rate"] == 5e-4
	assert config["multitask"]["energy_loss_weight"] == 2.0
	assert config["cawfe_latte"]["num_heads"] == [4, 8]
	assert config["cache"]["allow_config_hash_mismatch"] is True
	assert Path(config["checkpoint"]["best_path"]).is_absolute()
	assert Path(config["fire_dataset_index_json"]).is_absolute()
	assert Path(config["normalization"]["path"]).is_absolute()


def test_ablation_config_preserves_tuned_parameters(tmp_path: Path) -> None:
	best_params = {
		"params": {
			"cawfe_latte.backbone_dim": 128,
			"cawfe_latte.neural_operator_depth": 2,
			"training.learning_rate": 3e-4,
		}
	}
	base = make_final_config_from_best_params(_base_config(tmp_path), best_params)
	config = _ablation_config(base, "cawfe_latte_no_operator.yaml", ABLATIONS["cawfe_latte_no_operator.yaml"])

	assert config["model"]["architecture"] == "cawfe_latte"
	assert config["training"]["learning_rate"] == 3e-4
	assert config["cawfe_latte"]["backbone_dim"] == 128
	assert config["cawfe_latte"]["neural_operator_depth"] == 2
	assert config["cawfe_latte"]["use_neural_operator_bottleneck"] is False
	assert config["cawfe_latte"]["neural_operator_type"] == "none"
	assert config["training"]["run_name"] == "cawfe_latte_ablation_cawfe_latte_no_operator"
	assert config["training"]["output"]["update_architecture_latest_checkpoint"] is False
	assert "cawfe_latte_no_operator" in config["checkpoint"]["best_path"]


def test_cawfe_latte_slurm_scripts_exist_and_are_executable() -> None:
	scripts = [
		"slurm/slurm_tune_cawfe_latte_a10080.sh",
		"slurm/slurm_train_cawfe_latte_tuned_a10080.sh",
		"slurm/slurm_ablate_cawfe_latte_a10080.sh",
		"scripts/submit_cawfe_latte_pipeline.sh",
	]

	for script in scripts:
		path = Path(script)
		assert path.exists()
		assert os.access(path, os.X_OK)
