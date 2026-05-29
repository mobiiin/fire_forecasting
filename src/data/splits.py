"""Dataset split helpers for chronological wildfire forecasting splits."""

from __future__ import annotations

import math
from typing import Dict, List


def _validate_nonnegative_fraction(name: str, fraction: float) -> None:
	"""Validate that one split fraction is non-negative."""

	if fraction < 0.0:
		raise ValueError(f"{name} must be non-negative, got {fraction}.")


def _validate_fractions(
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> None:
	"""Validate split fractions before computing indices."""

	for name, fraction in (
		("train_fraction", train_fraction),
		("val_fraction", val_fraction),
		("test_fraction", test_fraction),
	):
		_validate_nonnegative_fraction(name, fraction)

	total = train_fraction + val_fraction + test_fraction
	if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError(
			"Split fractions must sum to 1.0. "
			f"Got train={train_fraction}, val={val_fraction}, test={test_fraction}, total={total}."
		)


def _validate_train_val_fractions(
	train_fraction: float,
	val_fraction: float,
) -> None:
	"""Validate fractions for train/validation-only chronological splitting."""

	_validate_nonnegative_fraction("train_fraction", train_fraction)
	_validate_nonnegative_fraction("val_fraction", val_fraction)
	total = train_fraction + val_fraction
	if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError(
			"For split_mode='train_val_external_test', train_fraction + val_fraction must sum to 1.0. "
			f"Got train={train_fraction}, val={val_fraction}, total={total}."
		)


def _sample_starts_for_segment(
	segment_start: int,
	segment_end: int,
	input_sequence_length: int,
	prediction_horizon: int,
) -> List[int]:
	"""Return valid sample start indices for a half-open raw-time segment."""

	latest_start = segment_end - input_sequence_length - prediction_horizon
	if latest_start < segment_start:
		return []
	return list(range(segment_start, latest_start + 1))


def chronological_train_val_split_indices(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
) -> Dict[str, List[int]]:
	"""Split forecasting sample indices chronologically into train and validation only."""

	if num_timesteps <= 0:
		raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
	if input_sequence_length <= 0:
		raise ValueError(f"input_sequence_length must be positive, got {input_sequence_length}.")
	if prediction_horizon < 0:
		raise ValueError(f"prediction_horizon must be non-negative, got {prediction_horizon}.")

	_validate_train_val_fractions(train_fraction, val_fraction)

	max_valid_start = num_timesteps - input_sequence_length - prediction_horizon
	if max_valid_start < 0:
		raise ValueError(
			"Not enough timesteps to form a single valid sample. "
			f"Need at least input_sequence_length + prediction_horizon = "
			f"{input_sequence_length + prediction_horizon}, got {num_timesteps}."
		)

	train_length = int(math.floor(num_timesteps * train_fraction))
	val_segment_start = train_length
	train = _sample_starts_for_segment(
		segment_start=0,
		segment_end=val_segment_start,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	val = _sample_starts_for_segment(
		segment_start=val_segment_start,
		segment_end=num_timesteps,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	return {"train": train, "val": val}


def chronological_split_indices(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
	split_mode: str = "train_val_test",
) -> Dict[str, List[int]]:
	"""Split forecasting sample indices chronologically.

	When ``split_mode == "train_val_external_test"``, the main dataset is split
	into train and validation only, and the returned internal test split is empty.
	"""

	split_mode = str(split_mode).lower()
	if split_mode == "train_val_external_test":
		train_val = chronological_train_val_split_indices(
			num_timesteps=num_timesteps,
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
			train_fraction=train_fraction,
			val_fraction=val_fraction,
		)
		return {"train": train_val["train"], "val": train_val["val"], "test": []}

	if num_timesteps <= 0:
		raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
	if input_sequence_length <= 0:
		raise ValueError(f"input_sequence_length must be positive, got {input_sequence_length}.")
	if prediction_horizon < 0:
		raise ValueError(f"prediction_horizon must be non-negative, got {prediction_horizon}.")

	_validate_fractions(train_fraction, val_fraction, test_fraction)

	max_valid_start = num_timesteps - input_sequence_length - prediction_horizon
	if max_valid_start < 0:
		raise ValueError(
			"Not enough timesteps to form a single valid sample. "
			f"Need at least input_sequence_length + prediction_horizon = "
			f"{input_sequence_length + prediction_horizon}, got {num_timesteps}."
		)

	train_length = int(math.floor(num_timesteps * train_fraction))
	val_length = int(math.floor(num_timesteps * val_fraction))
	val_segment_start = train_length
	test_segment_start = train_length + val_length

	train = _sample_starts_for_segment(0, val_segment_start, input_sequence_length, prediction_horizon)
	val = _sample_starts_for_segment(val_segment_start, test_segment_start, input_sequence_length, prediction_horizon)
	test = _sample_starts_for_segment(test_segment_start, num_timesteps, input_sequence_length, prediction_horizon)
	return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
	splits = chronological_train_val_split_indices(
		num_timesteps=100,
		input_sequence_length=6,
		prediction_horizon=1,
		train_fraction=0.85,
		val_fraction=0.15,
	)
	assert splits["train"], "train split should not be empty for the demo case"
	assert splits["val"], "val split should not be empty for the demo case"
	assert splits["train"] == sorted(splits["train"])
	assert splits["val"] == sorted(splits["val"])
	assert max(splits["train"]) < min(splits["val"])
	print("chronological_train_val_split_indices demo passed")
