from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from src.training.train import _run_epoch, resolve_validation_policy, save_validation_subset_metadata
from src.training.run_manager import RunManager


class _TinyDataset(torch.utils.data.Dataset):
	def __init__(self, length: int = 10) -> None:
		self.length = int(length)

	def __len__(self) -> int:
		return self.length

	def __getitem__(self, index: int):
		x = torch.full((2, 1, 4, 4), float(index), dtype=torch.float32)
		y = torch.full((1, 4, 4), float(index), dtype=torch.float32)
		return x, y


class _TinyModel(torch.nn.Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x[:, -1, :1]


def _loader(length: int = 10, batch_size: int = 1):
	return torch.utils.data.DataLoader(_TinyDataset(length), batch_size=batch_size, shuffle=False)


def _config(validation: dict | None = None) -> dict:
	return {
		"task_type": "regression",
		"training": {
			"validation": validation or {
				"mode": "fixed_subset_every_epoch",
				"max_val_batches_per_epoch": 4,
				"fixed_subset_seed": 7,
				"fixed_subset_shuffle": True,
				"use_same_metric_for_checkpointing": True,
			},
			"performance": {"compute_val_metrics": False, "show_progress_bar": False},
		},
	}


def test_fixed_subset_every_epoch_selects_same_batch_indices() -> None:
	loader = _loader(length=10, batch_size=1)
	first = resolve_validation_policy(_config(), val_loader=loader)
	second = resolve_validation_policy(_config(), val_loader=loader)

	assert first["validation_mode"] == "fixed_subset_every_epoch"
	assert first["selected_batch_indices"] == second["selected_batch_indices"]
	assert len(first["selected_batch_indices"]) == 4
	assert first["validation_batches_used"] == 4


def test_fixed_subset_every_epoch_saves_validation_subset_json(tmp_path) -> None:
	config = _config()
	config["training"]["output"] = {
		"root_dir": str(tmp_path / "runs"),
		"checkpoint_root": str(tmp_path / "checkpoints"),
	}
	run_manager = RunManager(config, architecture="tiny")
	run_manager.create_run_dir()
	loader = _loader(length=8, batch_size=2)
	policy = resolve_validation_policy(config, val_loader=loader)

	path = save_validation_subset_metadata(run_manager, policy, loader)

	assert path is not None
	assert path.exists()
	assert path.name == "validation_subset.json"
	assert "selected_batch_indices" in path.read_text(encoding="utf-8")


def test_full_every_epoch_evaluates_all_validation_batches() -> None:
	loader = _loader(length=6, batch_size=2)
	config = _config({"mode": "full_every_epoch", "max_val_batches_per_epoch": None})
	policy = resolve_validation_policy(config, val_loader=loader)

	results = _run_epoch(
		model=_TinyModel(),
		loader=loader,
		criterion=torch.nn.MSELoss(),
		config=config,
		device=torch.device("cpu"),
		input_sequence_length=2,
		input_channels=1,
		output_channels=1,
		train=False,
		batch_indices=policy["selected_batch_indices"],
	)

	assert policy["is_full_validation"] is True
	assert results["val_batches"] == 3.0
	assert results["val_samples"] == 6.0


def test_training_log_validation_fields_are_row_ready() -> None:
	loader = _loader(length=5, batch_size=1)
	config = _config({"mode": "fixed_subset_every_epoch", "max_val_batches_per_epoch": 2})
	policy = resolve_validation_policy(config, val_loader=loader)
	results = _run_epoch(
		model=_TinyModel(),
		loader=loader,
		criterion=torch.nn.MSELoss(),
		config=config,
		device=torch.device("cpu"),
		input_sequence_length=2,
		input_channels=1,
		output_channels=1,
		train=False,
		batch_indices=policy["selected_batch_indices"],
	)
	row = {
		"validation_mode": policy["validation_mode"],
		"validation_scope": policy["validation_scope"],
		"validation_batches_used": int(results["val_batches"]),
		"is_full_validation": bool(policy["is_full_validation"]),
	}

	assert row == {
		"validation_mode": "fixed_subset_every_epoch",
		"validation_scope": "fixed_subset",
		"validation_batches_used": 2,
		"is_full_validation": False,
	}


def test_deprecated_full_validation_every_n_epochs_is_warned_and_ignored(caplog) -> None:
	loader = _loader(length=10, batch_size=1)
	logger = logging.getLogger("validation-policy-test")
	config = _config({"mode": "fixed_subset_every_epoch", "max_val_batches_per_epoch": 3})
	config["training"]["performance"]["full_validation_every_n_epochs"] = 5

	with caplog.at_level(logging.WARNING):
		policy = resolve_validation_policy(config, val_loader=loader, logger=logger)

	assert policy["validation_mode"] == "fixed_subset_every_epoch"
	assert policy["validation_batches_used"] == 3
	assert "full_validation_every_n_epochs is deprecated and ignored" in caplog.text


def test_unknown_validation_mode_raises_clear_error() -> None:
	with pytest.raises(ValueError, match="Unsupported training.validation.mode"):
		resolve_validation_policy(_config({"mode": "full_periodic"}), val_loader=_loader())
