"""Helpers for interpreting per-fire temporal trim metadata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _coerce_int(value: Any, *, name: str) -> int:
	try:
		return int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def infer_original_num_frames(record: Mapping[str, Any]) -> int:
	"""Infer the original untrimmed frame count for one fire record."""

	for key in ("original_num_frames", "original_num_npy_files", "num_npy_files", "num_files", "num_frames"):
		value = record.get(key)
		if value not in (None, "", "null"):
			return _coerce_int(value, name=key)
	for key in ("file_paths", "frame_paths", "trimmed_frame_paths"):
		value = record.get(key)
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			return len(value)
	raise ValueError("Cannot infer original frame count from fire record.")


def resolve_temporal_trim(record: Mapping[str, Any]) -> dict[str, Any]:
	"""Return validated temporal trim metadata with defaults for untrimmed fires."""

	original_num_frames = infer_original_num_frames(record)
	if original_num_frames <= 0:
		raise ValueError(f"original_num_frames must be positive, got {original_num_frames}.")

	raw_trim = record.get("temporal_trim")
	trim = dict(raw_trim) if isinstance(raw_trim, Mapping) else {}
	enabled = bool(trim.get("enabled", False))
	if enabled:
		trim_start_index = _coerce_int(trim.get("trim_start_index", 0), name="temporal_trim.trim_start_index")
		trim_end_value = trim.get("trim_end_index", original_num_frames - 1)
		trim_end_index = original_num_frames - 1 if trim_end_value in (None, "", "null") else _coerce_int(
			trim_end_value,
			name="temporal_trim.trim_end_index",
		)
	else:
		trim_start_index = 0
		trim_end_index = original_num_frames - 1

	if trim_start_index < 0 or trim_start_index >= original_num_frames:
		raise ValueError(
			f"temporal_trim.trim_start_index must be within [0, {original_num_frames - 1}], got {trim_start_index}."
		)
	if trim_end_index < trim_start_index or trim_end_index >= original_num_frames:
		raise ValueError(
			"temporal_trim.trim_end_index must be within "
			f"[{trim_start_index}, {original_num_frames - 1}], got {trim_end_index}."
		)

	trimmed_num_frames = trim_end_index - trim_start_index + 1
	resolved = dict(trim)
	resolved.update(
		{
			"enabled": enabled,
			"trim_start_index": int(trim_start_index),
			"trim_end_index": int(trim_end_index),
			"original_start_index": int(trim.get("original_start_index", 0)),
			"original_end_index": int(trim.get("original_end_index", original_num_frames - 1)),
			"original_num_frames": int(original_num_frames),
			"trimmed_num_frames": int(trimmed_num_frames),
		}
	)
	return resolved


def effective_num_frames(record: Mapping[str, Any]) -> int:
	"""Return the number of frames available after temporal trimming."""

	return int(resolve_temporal_trim(record)["trimmed_num_frames"])


def max_valid_local_start(record: Mapping[str, Any], input_sequence_length: int, prediction_horizon: int) -> int:
	"""Return the largest local sample start allowed by the trim window."""

	return effective_num_frames(record) - int(input_sequence_length) - int(prediction_horizon)


def original_index_for_local(record: Mapping[str, Any], local_index: int) -> int:
	"""Map a local trimmed-window frame index to the original frame index."""

	trim = resolve_temporal_trim(record)
	original_index = int(trim["trim_start_index"]) + int(local_index)
	if original_index > int(trim["trim_end_index"]):
		raise IndexError(
			f"local_index={local_index} maps to original_index={original_index}, "
			f"outside trim_end_index={trim['trim_end_index']}."
		)
	return int(original_index)


def temporal_sample_metadata(
	record: Mapping[str, Any],
	local_start_idx: int,
	input_sequence_length: int,
	prediction_horizon: int,
) -> dict[str, Any]:
	"""Build local and original index metadata for one temporal sample."""

	trim = resolve_temporal_trim(record)
	local_start_idx = int(local_start_idx)
	input_sequence_length = int(input_sequence_length)
	prediction_horizon = int(prediction_horizon)
	local_input_indices = list(range(local_start_idx, local_start_idx + input_sequence_length))
	original_input_indices = [int(trim["trim_start_index"]) + index for index in local_input_indices]
	original_last_input_idx = original_input_indices[-1]
	original_target_idx = original_last_input_idx + prediction_horizon
	if original_target_idx > int(trim["trim_end_index"]):
		raise IndexError(
			f"sample local_start_idx={local_start_idx} targets original frame {original_target_idx}, "
			f"outside trim_end_index={trim['trim_end_index']}."
		)
	return {
		"local_start_idx": int(local_start_idx),
		"local_input_indices": local_input_indices,
		"local_last_input_idx": int(local_input_indices[-1]),
		"local_target_idx": int(local_input_indices[-1] + prediction_horizon),
		"original_start_idx": int(original_input_indices[0]),
		"original_input_indices": original_input_indices,
		"original_last_input_idx": int(original_last_input_idx),
		"original_target_idx": int(original_target_idx),
		"trim_start_index": int(trim["trim_start_index"]),
		"trim_end_index": int(trim["trim_end_index"]),
		"trimmed_num_frames": int(trim["trimmed_num_frames"]),
		"original_num_frames": int(trim["original_num_frames"]),
		"temporal_trim_enabled": bool(trim["enabled"]),
	}
