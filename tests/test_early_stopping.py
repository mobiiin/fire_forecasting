from __future__ import annotations

import math

import pytest

from src.training.early_stopping import EarlyStopping, build_early_stopping


def test_early_stopping_min_mode_stops_after_patience() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_loss", mode="min", patience=2, min_delta=0.1, start_epoch=1)

	assert stopper.step(1, {"val_loss": 5.0})["early_stopping_should_stop"] is False
	assert stopper.step(2, {"val_loss": 4.95})["early_stopping_bad_epochs"] == 1
	state = stopper.step(3, {"val_loss": 4.94})

	assert state["early_stopping_should_stop"] is True
	assert stopper.stop_reason == "No improvement in val_loss for 2 validation checks."


def test_early_stopping_resets_bad_epochs_on_improvement() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_loss", mode="min", patience=2, min_delta=0.01, start_epoch=1)
	stopper.step(1, {"val_loss": 5.0})
	stopper.step(2, {"val_loss": 5.0})
	state = stopper.step(3, {"val_loss": 4.98})

	assert state["early_stopping_bad_epochs"] == 0
	assert state["early_stopping_best_score"] == pytest.approx(4.98)
	assert state["early_stopping_best_epoch"] == 3


def test_early_stopping_max_mode_tracks_larger_metric() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_mask_dice", mode="max", patience=1, min_delta=0.05, start_epoch=1)
	stopper.step(1, {"val_mask_dice": 0.5})
	state = stopper.step(2, {"val_mask_dice": 0.54})

	assert state["early_stopping_should_stop"] is True


def test_early_stopping_stop_on_nan() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_loss", stop_on_nan=True)
	state = stopper.step(1, {"val_loss": math.nan})

	assert state["early_stopping_should_stop"] is True
	assert stopper.stop_reason == "Validation metric val_loss is NaN/Inf."


def test_early_stopping_missing_monitor_raises_clear_error() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_loss")

	with pytest.raises(KeyError, match="Available metrics"):
		stopper.step(1, {"val_mae": 1.0})


def test_early_stopping_state_round_trip() -> None:
	stopper = EarlyStopping(enabled=True, monitor="val_loss", patience=3)
	stopper.step(1, {"val_loss": 2.0})
	stopper.step(2, {"val_loss": 2.1})

	restored = EarlyStopping(enabled=True, monitor="val_loss", patience=3)
	restored.load_state_dict(stopper.state_dict())

	assert restored.best_score == pytest.approx(2.0)
	assert restored.best_epoch == 1
	assert restored.num_bad_epochs == 1


def test_build_early_stopping_uses_config_defaults() -> None:
	stopper = build_early_stopping(
		{
			"training": {
				"early_stopping": {
					"enabled": True,
					"monitor": "val_mask_dice",
					"mode": "max",
					"patience": 4,
					"min_delta": 0.02,
					"start_epoch": 3,
				}
			}
		}
	)

	assert stopper.enabled is True
	assert stopper.monitor == "val_mask_dice"
	assert stopper.mode == "max"
	assert stopper.patience == 4
	assert stopper.min_delta == pytest.approx(0.02)
	assert stopper.start_epoch == 3
