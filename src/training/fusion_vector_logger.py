"""Optional per-epoch post-fusion feature-vector logging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.training.batch_utils import unpack_batch

FEATURE_KEYS = ("fused_after_terrain", "fused_dynamic", "fused_grid", "post_fusion_features")


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def extract_feature_tensor(model_output: Any) -> tuple[str, torch.Tensor]:
    """Return the first available post-fusion feature using the documented priority."""
    if not isinstance(model_output, Mapping):
        raise ValueError("save_fusion_vectors.enabled=true, but no fusion feature was found in model output. Expected one of: " + ", ".join(FEATURE_KEYS) + ".")
    for key in FEATURE_KEYS:
        value = model_output.get(key)
        if torch.is_tensor(value):
            return key, value
    raise ValueError("save_fusion_vectors.enabled=true, but no fusion feature was found in model output. Expected one of: " + ", ".join(FEATURE_KEYS) + ".")


def reduce_feature_to_vector(feature: torch.Tensor, vector_reduce: str = "mean_bt_hw") -> torch.Tensor:
    """Reduce a supported post-fusion feature layout to its D-dimensional vector."""
    if vector_reduce != "mean_bt_hw":
        raise ValueError(f"Unsupported save_fusion_vectors.vector_reduce: {vector_reduce!r}; expected 'mean_bt_hw'.")
    if feature.ndim == 5:  # B,T,D,H,W
        return feature.mean(dim=(0, 1, 3, 4))
    if feature.ndim == 4:
        # B,D,H,W is the normal grid case. B,T,N,D token tensors have a
        # small time axis and a final embedding axis; equality handles common
        # square patches where D == W as a grid feature.
        if feature.shape[1] >= feature.shape[-1]:
            return feature.mean(dim=(0, 2, 3))
        return feature.mean(dim=(0, 1, 2))  # B,T,N,D
    if feature.ndim == 3:  # B,N,D
        return feature.mean(dim=(0, 1))
    raise ValueError(f"Unsupported fusion feature shape {tuple(feature.shape)}; expected B,T,D,H,W; B,T,N,D; B,D,H,W; or B,N,D.")


class FusionVectorLogger:
    """Collect exactly one representative post-fusion vector per completed epoch."""
    def __init__(self, config: Mapping[str, Any], run_dir: str | Path) -> None:
        training = _section(config, "training")
        options = _section(training, "save_fusion_vectors")
        self.is_enabled = bool(options.get("enabled", False))
        self.vector_reduce = str(options.get("vector_reduce", "mean_bt_hw"))
        self.save_metadata = bool(options.get("save_metadata", True))
        self.output_name = str(options.get("output_name", "fusion_vectors_by_epoch.npy"))
        self.architecture = str(_section(config, "model").get("architecture", "unknown"))
        self.experiment_name = str(_section(config, "experiment").get("name", "unknown"))
        self.config_path = config.get("config_path")
        self.features_dir = Path(run_dir) / "features"
        self.output_path = self.features_dir / self.output_name
        self.metadata_path = self.features_dir / f"{Path(self.output_name).stem}_metadata.json"
        self.vectors: list[np.ndarray] = []
        self.feature_key_used: str | None = None

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def collect_epoch_vector(self, model: torch.nn.Module, batch: Any, device: torch.device, epoch: int) -> np.ndarray | None:
        if not self.is_enabled:
            return None
        x, _y, extras = unpack_batch(batch)
        terrain = extras.get("terrain")
        x = x.to(device)
        terrain = terrain.to(device) if terrain is not None else None
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output = model(x, terrain=terrain, return_features=True) if terrain is not None else model(x, return_features=True)
                feature_key, feature = extract_feature_tensor(output)
                vector = reduce_feature_to_vector(feature, self.vector_reduce).detach().float().cpu().numpy()
        finally:
            model.train(was_training)
        if vector.ndim != 1:
            raise ValueError(f"Reduced fusion vector must be one-dimensional, got {tuple(vector.shape)}.")
        self.feature_key_used = feature_key
        self.vectors.append(vector)
        return vector

    def save(self) -> Path | None:
        if not self.is_enabled:
            return None
        self.features_dir.mkdir(parents=True, exist_ok=True)
        array = np.stack(self.vectors, axis=0) if self.vectors else np.empty((0, 0), dtype=np.float32)
        np.save(self.output_path, array)
        if self.save_metadata:
            payload = {"enabled": True, "feature_key_used": self.feature_key_used, "vector_reduce": self.vector_reduce, "shape": list(array.shape), "epochs_collected": len(self.vectors), "output_path": str(self.output_path), "architecture": self.architecture, "experiment_name": self.experiment_name, "config_path": self.config_path}
            self.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self.output_path
