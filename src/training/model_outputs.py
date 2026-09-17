"""Shared helpers for model outputs used by training and evaluation."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def extract_prediction(model_output: Any):
    """Extract the prediction tensor from a tensor or prediction mapping."""
    if torch is not None and torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict) and "prediction" in model_output:
        prediction = model_output["prediction"]
        if torch is None or not torch.is_tensor(prediction):
            raise TypeError("model_output['prediction'] must be a torch.Tensor")
        return prediction
    raise TypeError("Model output must be a tensor or a dict containing 'prediction'.")


def extract_aux_outputs(model_output: Any) -> dict[str, Any]:
    if isinstance(model_output, dict):
        return {key: value for key, value in model_output.items() if key != "prediction"}
    return {}


def patch_fire_presence_target(y_true: Any):
    """Build one B x 1 future-fire-presence target from channel-2 fire masks."""
    if torch is None or not torch.is_tensor(y_true):
        raise TypeError("Patch-fire targets require a torch.Tensor.")
    if y_true.ndim != 4 or int(y_true.shape[1]) < 3:
        raise ValueError(f"Patch-fire targets expect B x C x H x W with C >= 3, got {tuple(y_true.shape)}.")
    target_mask = y_true[:, 2]
    return (target_mask > 0.5).flatten(1).any(dim=1).to(dtype=torch.float32).unsqueeze(1)
