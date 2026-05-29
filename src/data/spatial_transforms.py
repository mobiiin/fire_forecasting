"""Spatial padding and cropping helpers for variable-size wildfire inference."""

from __future__ import annotations

from typing import Any, Mapping

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	F = None


def _ensure_tensor(x):
	"""Validate torch availability and input tensor type."""

	if torch is None or F is None:
		raise ImportError("PyTorch is required for spatial padding and cropping helpers.")
	if not torch.is_tensor(x):
		raise TypeError(f"Expected a torch.Tensor, got {type(x)!r}.")
	if x.ndim not in {3, 4, 5}:
		raise ValueError(f"Expected a tensor with 3, 4, or 5 dims, got shape {tuple(x.shape)}.")
	return x


def pad_to_size(x, target_height: int, target_width: int, pad_value: float = 0.0):
	"""Center-pad the last two spatial dims to a requested size."""

	x = _ensure_tensor(x)
	target_height = int(target_height)
	target_width = int(target_width)
	original_height = int(x.shape[-2])
	original_width = int(x.shape[-1])
	if target_height < original_height or target_width < original_width:
		raise ValueError(
			"pad_to_size cannot shrink spatial dimensions. "
			f"Got original={(original_height, original_width)} target={(target_height, target_width)}."
		)

	pad_height = target_height - original_height
	pad_width = target_width - original_width
	pad_top = pad_height // 2
	pad_bottom = pad_height - pad_top
	pad_left = pad_width // 2
	pad_right = pad_width - pad_left
	padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=float(pad_value))
	metadata = {
		"original_height": original_height,
		"original_width": original_width,
		"padded_height": int(padded.shape[-2]),
		"padded_width": int(padded.shape[-1]),
		"pad_top": pad_top,
		"pad_bottom": pad_bottom,
		"pad_left": pad_left,
		"pad_right": pad_right,
	}
	return padded, metadata


def pad_to_multiple(x, multiple: int, pad_value: float = 0.0):
	"""Pad the last two spatial dims up to the next multiple."""

	x = _ensure_tensor(x)
	multiple = int(multiple)
	if multiple <= 0:
		raise ValueError(f"multiple must be positive, got {multiple}.")

	height = int(x.shape[-2])
	width = int(x.shape[-1])
	target_height = ((height + multiple - 1) // multiple) * multiple
	target_width = ((width + multiple - 1) // multiple) * multiple
	return pad_to_size(x, target_height=target_height, target_width=target_width, pad_value=pad_value)


def crop_to_original(x, metadata: Mapping[str, Any]):
	"""Crop back to the original spatial region using stored padding metadata."""

	x = _ensure_tensor(x)
	pad_top = int(metadata["pad_top"])
	pad_left = int(metadata["pad_left"])
	original_height = int(metadata["original_height"])
	original_width = int(metadata["original_width"])
	return x[..., pad_top : pad_top + original_height, pad_left : pad_left + original_width]


def resolve_external_test_spatial_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve external-test spatial handling settings from config."""

	section = config.get("external_test_spatial", {}) if isinstance(config, Mapping) else {}
	if not isinstance(section, Mapping):
		section = {}
	return {
		"mode": str(section.get("mode", "auto")).lower(),
		"train_height": int(section.get("train_height", 144)),
		"train_width": int(section.get("train_width", 144)),
		"pad_multiple": int(section.get("pad_multiple", 16)),
		"pad_value_normalized": float(section.get("pad_value_normalized", 0.0)),
		"crop_output_to_original_size": bool(section.get("crop_output_to_original_size", True)),
		"allow_resize": bool(section.get("allow_resize", False)),
	}


def infer_with_external_test_spatial_handling(model, x, config: Mapping[str, Any]):
	"""Run model inference on external-test inputs using direct or padded spatial handling."""

	x = _ensure_tensor(x)
	spatial_config = resolve_external_test_spatial_config(config)
	mode = str(spatial_config["mode"]).lower()
	if mode not in {"direct", "pad_to_train_size", "pad_to_multiple", "auto"}:
		raise ValueError(
			"external_test_spatial.mode must be one of 'direct', 'pad_to_train_size', "
			f"'pad_to_multiple', or 'auto', got {mode!r}."
		)

	def _attempt(selected_mode: str):
		if selected_mode == "direct":
			y_pred = model(x)
			return {
				"y_pred": y_pred,
				"x_model_input": x,
				"mode_used": "direct",
				"padding_metadata": None,
				"used_padding": False,
			}
		if selected_mode == "pad_to_multiple":
			x_model_input, metadata = pad_to_multiple(
				x,
				multiple=int(spatial_config["pad_multiple"]),
				pad_value=float(spatial_config["pad_value_normalized"]),
			)
		elif selected_mode == "pad_to_train_size":
			x_model_input, metadata = pad_to_size(
				x,
				target_height=int(spatial_config["train_height"]),
				target_width=int(spatial_config["train_width"]),
				pad_value=float(spatial_config["pad_value_normalized"]),
			)
		else:
			raise ValueError(f"Unsupported spatial handling mode: {selected_mode!r}.")

		y_pred = model(x_model_input)
		if bool(spatial_config["crop_output_to_original_size"]):
			y_pred = crop_to_original(y_pred, metadata)
		return {
			"y_pred": y_pred,
			"x_model_input": x_model_input,
			"mode_used": selected_mode,
			"padding_metadata": metadata,
			"used_padding": True,
		}

	if mode != "auto":
		result = _attempt(mode)
		if spatial_config["allow_resize"]:
			result["warning"] = (
				"external_test_spatial.allow_resize=true is set, but resize-based inference is intentionally "
				"not used by default because it changes physical scale."
			)
		return result

	attempt_order = ("direct", "pad_to_multiple", "pad_to_train_size")
	failures: list[str] = []
	for candidate_mode in attempt_order:
		try:
			result = _attempt(candidate_mode)
			result["attempted_modes"] = list(attempt_order)
			if spatial_config["allow_resize"]:
				result["warning"] = (
					"external_test_spatial.allow_resize=true is set, but resize-based inference is intentionally "
					"not used by default because it changes physical scale."
				)
			return result
		except Exception as exc:  # pragma: no cover - exercised through caller fallbacks
			failures.append(f"{candidate_mode}: {type(exc).__name__}: {exc}")

	raise RuntimeError(
		"external_test_spatial.mode='auto' failed for all supported inference paths. "
		+ " | ".join(failures)
	)
