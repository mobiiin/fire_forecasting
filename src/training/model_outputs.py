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
