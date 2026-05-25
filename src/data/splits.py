"""Dataset split helpers for chronological wildfire forecasting splits."""

from __future__ import annotations

import math
from typing import Dict, List


def _validate_fractions(
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> None:
	"""Validate split fractions before computing indices."""

	total = train_fraction + val_fraction + test_fraction
	if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError(
			"Split fractions must sum to 1.0. "
			f"Got train={train_fraction}, val={val_fraction}, test={test_fraction}, total={total}."
		)

	for name, fraction in (
		("train_fraction", train_fraction),
		("val_fraction", val_fraction),
		("test_fraction", test_fraction),
	):
		if fraction < 0.0:
			raise ValueError(f"{name} must be non-negative, got {fraction}.")


def _segment_lengths(
	num_timesteps: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> tuple[int, int, int]:
	"""Convert fractions into contiguous raw time-segment lengths."""

	train_length = int(math.floor(num_timesteps * train_fraction))
	val_length = int(math.floor(num_timesteps * val_fraction))
	test_length = num_timesteps - train_length - val_length
	return train_length, val_length, test_length


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


def chronological_split_indices(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> Dict[str, List[int]]:
	"""Split forecasting sample indices chronologically without time leakage.

	A sample with start index ``i`` uses raw timestamps::

		X = [i, i + 1, ..., i + input_sequence_length - 1]
		y = i + input_sequence_length - 1 + prediction_horizon

	The split is performed on the raw timestamp axis first and then converted
	into valid sample start indices for each contiguous time block.
	"""

	if num_timesteps <= 0:
		raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
	if input_sequence_length <= 0:
		raise ValueError(
			f"input_sequence_length must be positive, got {input_sequence_length}."
		)
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

	train_length, val_length, test_length = _segment_lengths(
		num_timesteps,
		train_fraction,
		val_fraction,
		test_fraction,
	)

	train_segment_start = 0
	val_segment_start = train_length
	test_segment_start = train_length + val_length

	train = _sample_starts_for_segment(
		segment_start=train_segment_start,
		segment_end=val_segment_start,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	val = _sample_starts_for_segment(
		segment_start=val_segment_start,
		segment_end=test_segment_start,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	test = _sample_starts_for_segment(
		segment_start=test_segment_start,
		segment_end=num_timesteps,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)

	# The segment lengths are useful for debugging and can expose corner cases.
	_ = (train_length, val_length, test_length, max_valid_start)

	return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
	splits = chronological_split_indices(
		num_timesteps=100,
		input_sequence_length=6,
		prediction_horizon=1,
		train_fraction=0.7,
		val_fraction=0.15,
		test_fraction=0.15,
	)

	assert splits["train"], "train split should not be empty for the demo case"
	assert splits["val"], "val split should not be empty for the demo case"
	assert splits["test"], "test split should not be empty for the demo case"
	assert splits["train"] == sorted(splits["train"])
	assert splits["val"] == sorted(splits["val"])
	assert splits["test"] == sorted(splits["test"])
	assert max(splits["train"]) < min(splits["val"])
	assert max(splits["val"]) < min(splits["test"])
	assert splits["train"][0] == 0
	assert splits["train"][-1] + 6 + 1 <= 70
	print("chronological_split_indices demo passed")
