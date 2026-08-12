"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import warnings

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None


def _to_builtin(value: Any) -> Any:
	"""Convert Paths and nested containers into checkpoint-safe Python types."""

	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): _to_builtin(nested_value) for key, nested_value in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_builtin(item) for item in value]
	return value


def _checkpoint_dict(config: Mapping[str, Any], **state: Any) -> dict[str, Any]:
	"""Assemble a serializable checkpoint dictionary."""

	checkpoint = {key: _to_builtin(value) for key, value in state.items()}
	checkpoint["config"] = _to_builtin(dict(config))
	return checkpoint


def save_checkpoint(
	path: str | Path,
	config: Mapping[str, Any],
	model,
	optimizer,
	scheduler,
	epoch: int,
	best_val_loss: float,
	**extra_state: Any,
) -> Path:
	"""Persist training state to disk and return the resolved checkpoint path."""

	if torch is None:
		raise ImportError("PyTorch is required to save training checkpoints.")

	checkpoint_path = Path(path).expanduser().resolve()
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

	checkpoint = _checkpoint_dict(
		config,
		model_state_dict=model.state_dict(),
		optimizer_state_dict=optimizer.state_dict(),
		scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
		epoch=int(epoch),
		best_val_loss=float(best_val_loss),
		**extra_state,
	)
	torch.save(checkpoint, checkpoint_path)
	return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
	"""Load a checkpoint file and return the deserialized dictionary."""

	if torch is None:
		raise ImportError("PyTorch is required to load training checkpoints.")

	checkpoint_path = Path(path).expanduser().resolve()
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	return torch.load(checkpoint_path, map_location=map_location)


def validate_checkpoint_model_compatibility(model, checkpoint: Mapping[str, Any], checkpoint_path: str | Path | None = None) -> None:
	"""Raise a clear error when checkpoint and model output heads are incompatible."""

	model_output_channels = int(getattr(model, "output_channels", -1))
	checkpoint_config = checkpoint.get("config", {})
	if isinstance(checkpoint_config, Mapping):
		model_config = checkpoint_config.get("model", {})
		if isinstance(model_config, Mapping) and "output_channels" in model_config:
			checkpoint_output_channels = int(model_config["output_channels"])
			if model_output_channels > 0 and checkpoint_output_channels != model_output_channels:
				raise ValueError(
					"Checkpoint output channels do not match model output_channels. "
					f"checkpoint={checkpoint_output_channels}, model={model_output_channels}. "
					"You likely need to retrain after enabling the energy_release target."
				)
		architecture = None
		if isinstance(model_config, Mapping):
			architecture = model_config.get("architecture", model_config.get("name"))
		if str(architecture or "").lower() == "convlstm_unet":
			checkpoint_convlstm = checkpoint_config.get("convlstm_unet", {})
			if not isinstance(checkpoint_convlstm, Mapping):
				checkpoint_convlstm = {}
			comparisons = {
				"use_mask_gated_regression": bool(getattr(model, "use_mask_gated_regression", False)),
				"regression_activation": str(getattr(model, "regression_activation", "none")),
				"mask_gate_mode": str(getattr(model, "mask_gate_mode", "soft")),
				"detach_mask_gate": bool(getattr(model, "detach_mask_gate", False)),
			}
			for key, current_value in comparisons.items():
				if key in checkpoint_convlstm:
					checkpoint_value = checkpoint_convlstm.get(key)
				elif key == "use_mask_gated_regression":
					checkpoint_value = False
				else:
					continue
				if checkpoint_value != current_value:
					path_text = f" ({Path(checkpoint_path).expanduser().resolve()})" if checkpoint_path is not None else ""
					warnings.warn(
						"ConvLSTM checkpoint gating config differs from the current model config"
						f"{path_text}: {key} checkpoint={checkpoint_value!r}, current={current_value!r}. "
						"Evaluating an old checkpoint with new mask-gated regression settings changes outputs and is not a fair comparison.",
						RuntimeWarning,
					)

	state_dict = checkpoint.get("model_state_dict")
	if not isinstance(state_dict, Mapping) or model_output_channels <= 0:
		return
	weight = state_dict.get("spatial_decoder.outc.proj.weight")
	if hasattr(weight, "shape") and len(weight.shape) >= 1:
		checkpoint_output_channels = int(weight.shape[0])
		if checkpoint_output_channels != model_output_channels:
			path_text = f" ({Path(checkpoint_path).expanduser().resolve()})" if checkpoint_path is not None else ""
			raise ValueError(
				"Checkpoint output channels inferred from the model head do not match the current model"
				f"{path_text}. checkpoint={checkpoint_output_channels}, model={model_output_channels}. "
				"You likely need to retrain after enabling the energy_release target."
			)


def latest_and_best_checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
	"""Return the latest and best checkpoint paths derived from a base path."""

	base_path = Path(path).expanduser().resolve()
	best_path = base_path.with_name(f"{base_path.stem}_best{base_path.suffix or '.pt'}")
	return base_path, best_path
