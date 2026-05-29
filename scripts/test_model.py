"""Evaluate a trained ConvLSTM U-Net checkpoint on the external test split with diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None

from scripts.evaluate_persistence_baseline import warning_messages_for_target_channel
from src.config import load_config
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import load_checkpoint
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.train import (
    _coerce_loss_result,
    _denormalize_target_tensors_for_metrics,
    _ensure_config_path,
    _get_device,
    _infer_input_channels_from_loader,
    _resolve_training_paths,
)
from src.data.dataset import create_dataloaders


def tensor_stats(array_like) -> dict[str, float]:
    """Compute min/max/mean/std for a tensor-like value."""

    if torch is not None and torch.is_tensor(array_like):
        array = array_like.detach().cpu().to(torch.float32).numpy()
    else:
        array = np.asarray(array_like, dtype=np.float32)

    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std()),
    }


def format_stats(label: str, stats: Mapping[str, float]) -> str:
    """Format a stats dictionary consistently."""

    return (
        f"{label}: min={stats['min']:.6g} max={stats['max']:.6g} "
        f"mean={stats['mean']:.6g} std={stats['std']:.6g}"
    )


def generate_scale_diagnostics(model, test_loader, device: torch.device, diagnostics_path: Path, config) -> str:
    """Run the first three test batches and record scale diagnostics."""

    dataset = test_loader.dataset
    target_channel = int(getattr(dataset, "target_channel", -1))
    input_channel_count = int(getattr(dataset, "input_channel_count", -1))

    lines = []
    lines.append("Test Scale Diagnostics")
    lines.append(f"target_channel: {target_channel}")
    lines.append(f"input_channel_count: {input_channel_count}")
    lines.append(f"normalize_target: {bool(getattr(dataset, 'normalize_target', False))}")
    for warning_message in warning_messages_for_target_channel(target_channel, input_channel_count):
        lines.append(f"WARNING: {warning_message}")
    lines.append("")

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if batch_index >= 3:
                break
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError("Expected test loader batches to contain input and target tensors.")

            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            spatial_result = infer_with_external_test_spatial_handling(model, x_batch, config)
            y_pred = spatial_result["y_pred"]
            x_model_input = spatial_result["x_model_input"]
            y_pred_metric, y_true_metric = _denormalize_target_tensors_for_metrics(test_loader, y_pred.detach(), y_batch.detach())
            y_true_inverse = y_true_metric.detach().cpu().numpy()
            y_pred_inverse = y_pred_metric.detach().cpu().numpy()

            abs_error_inverse = np.abs(y_pred_inverse - y_true_inverse)

            lines.append(f"Batch {batch_index}")
            lines.append(f"spatial_mode_used: {spatial_result['mode_used']}")
            lines.append(f"native_input_shape: {tuple(x_batch.shape)}")
            lines.append(f"model_input_shape: {tuple(x_model_input.shape)}")
            lines.append(f"prediction_shape_after_crop: {tuple(y_pred.shape)}")
            lines.append(f"target_shape: {tuple(y_batch.shape)}")
            lines.append(format_stats("X native normalized", tensor_stats(x_batch)))
            lines.append(format_stats("X fed to model", tensor_stats(x_model_input)))
            lines.append(format_stats("y dataset", tensor_stats(y_batch)))
            lines.append(format_stats("y_pred raw model output", tensor_stats(y_pred)))
            lines.append(format_stats("y_true inverse", tensor_stats(y_true_inverse)))
            lines.append(format_stats("y_pred inverse", tensor_stats(y_pred_inverse)))
            lines.append(format_stats("abs error inverse", tensor_stats(abs_error_inverse)))
            lines.append("")

    diagnostics_text = "\n".join(lines)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(diagnostics_text + "\n", encoding="utf-8")
    print(diagnostics_text)
    return diagnostics_text


def evaluate_external_test_loader(model, test_loader, criterion, config, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate an external test loader with native-grid or padded spatial handling."""

    model.eval()
    total_samples = 0
    total_loss = 0.0
    metric_totals: dict[str, float] = defaultdict(float)
    loss_component_totals: dict[str, float] = defaultdict(float)
    mode_counts: dict[str, int] = defaultdict(int)
    first_batch_summary: dict[str, Any] | None = None
    warning_message: str | None = None

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError("Expected external test batches to contain input and target tensors.")
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            spatial_result = infer_with_external_test_spatial_handling(model, x_batch, config)
            y_pred = spatial_result["y_pred"]
            x_model_input = spatial_result["x_model_input"]
            if tuple(y_pred.shape) != tuple(y_batch.shape):
                raise ValueError(
                    "External test prediction shape does not match target shape after spatial handling. "
                    f"Prediction={tuple(y_pred.shape)} target={tuple(y_batch.shape)} mode={spatial_result['mode_used']}."
                )

            loss_result = criterion(y_pred, y_batch)
            loss, batch_loss_components = _coerce_loss_result(loss_result)

            metric_prediction, metric_target = _denormalize_target_tensors_for_metrics(
                test_loader,
                y_pred.detach(),
                y_batch.detach(),
            )
            batch_metrics = compute_metrics(metric_prediction, metric_target, config)

            batch_size = int(x_batch.shape[0])
            total_samples += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            for component_name, component_value in batch_loss_components.items():
                loss_component_totals[component_name] += float(component_value) * batch_size
            for metric_name, metric_value in batch_metrics.items():
                metric_totals[metric_name] += float(metric_value) * batch_size
            mode_counts[str(spatial_result["mode_used"])] += batch_size

            if first_batch_summary is None:
                first_batch_summary = {
                    "native_input_shape": tuple(x_batch.shape),
                    "model_input_shape": tuple(x_model_input.shape),
                    "prediction_shape": tuple(y_pred.shape),
                    "target_shape": tuple(y_batch.shape),
                    "mode_used": str(spatial_result["mode_used"]),
                }
                warning_message = spatial_result.get("warning")

    if total_samples == 0:
        raise ValueError("External test loader produced no samples.")

    results = {"test_loss": total_loss / total_samples}
    for component_name, total_value in loss_component_totals.items():
        results[f"test_{component_name}"] = total_value / total_samples
    for metric_name, total_value in metric_totals.items():
        results[f"test_{metric_name}"] = total_value / total_samples
    return results, {
        "mode_counts": dict(mode_counts),
        "first_batch": first_batch_summary or {},
        "warning": warning_message,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for standalone test evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvLSTM U-Net checkpoint on the external test split."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. If omitted, resolve from config.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Which config-derived checkpoint to use when --checkpoint is not provided.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    if torch is None:
        raise ImportError("PyTorch is required to evaluate the ConvLSTM U-Net model.")
    config = _ensure_config_path(load_config(args.config), args.config)

    train_loader, _, test_loader = create_dataloaders(config)
    if test_loader is None:
        raise ValueError(
            "No external test_data_dir configured. This project now uses data_dir only for train/val. "
            "Set test_data_dir in the config to evaluate on an external test dataset."
        )
    if len(test_loader.dataset) == 0:
        raise ValueError("External test dataset is empty; cannot evaluate the model.")

    dataset = test_loader.dataset
    target_channel = int(getattr(dataset, "target_channel", int(config["target_channel"])))
    input_channel_count = int(getattr(dataset, "input_channel_count", config.get("input_channel_count", 0)))
    for warning_message in warning_messages_for_target_channel(target_channel, input_channel_count):
        print(f"WARNING: {warning_message}")

    input_channels = _infer_input_channels_from_loader(train_loader)
    device = _get_device(config)

    model = build_model_from_config(config, input_channels=input_channels).to(device)
    criterion = get_loss_function(config)

    if args.checkpoint is None:
        latest_checkpoint_path, best_checkpoint_path = _resolve_training_paths(config)
        resolved_checkpoint_path = best_checkpoint_path if args.checkpoint_kind == "best" else latest_checkpoint_path
    else:
        resolved_checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

    checkpoint = load_checkpoint(resolved_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results, spatial_summary = evaluate_external_test_loader(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        config=config,
        device=device,
    )

    diagnostics_path = Path("artifacts/logs/test_scale_diagnostics.txt").expanduser().resolve()
    generate_scale_diagnostics(model, test_loader, device, diagnostics_path, config)

    print(f"checkpoint: {resolved_checkpoint_path}")
    print(f"external test dataset path: {config.get('test_data_dir')}")
    print(f"external test files: {len(getattr(test_loader.dataset, 'file_paths', []))}")
    print(f"external test samples: {len(test_loader.dataset)}")
    if spatial_summary.get("first_batch"):
        print(f"external X before spatial handling: {spatial_summary['first_batch']['native_input_shape']}")
        print(f"external X fed to model: {spatial_summary['first_batch']['model_input_shape']}")
        print(f"external prediction after crop: {spatial_summary['first_batch']['prediction_shape']}")
        print(f"external y shape: {spatial_summary['first_batch']['target_shape']}")
        print(f"external spatial mode used on first batch: {spatial_summary['first_batch']['mode_used']}")
    if spatial_summary.get("warning"):
        print(f"WARNING: {spatial_summary['warning']}")
    print(f"external spatial mode counts: {spatial_summary['mode_counts']}")
    print(f"target_channel: {target_channel}")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.6f}")
    print(f"diagnostics_file: {diagnostics_path}")


if __name__ == "__main__":
    main()
