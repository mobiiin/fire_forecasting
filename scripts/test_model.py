"""Evaluate a trained ConvLSTM U-Net checkpoint on the test split with diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None

from scripts.evaluate_persistence_baseline import evaluate_persistence_for_channel, warning_messages_for_target_channel
from src.config import load_config
from src.data.preprocessing import inverse_normalize_channel_map as inverse_normalize_scalar_channel_map
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import load_checkpoint
from src.training.losses import get_loss_function
from src.training.train import (
    _ensure_config_path,
    _get_device,
    _infer_input_channels_from_loader,
    _rename_result_prefix,
    _resolve_training_paths,
    _run_epoch,
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


def resolve_target_stats_for_inverse(dataset) -> tuple[float | None, float | None, str]:
    """Resolve target normalization stats and record their source."""

    normalization_stats = getattr(dataset, "normalization_stats", None)
    target_channel = int(getattr(dataset, "target_channel", -1))
    if normalization_stats is None:
        return None, None, "none"

    stats_mean = np.asarray(normalization_stats["mean"])
    stats_std = np.asarray(normalization_stats["std"])
    if 0 <= target_channel < stats_mean.shape[0] and target_channel < stats_std.shape[0]:
        return (
            float(stats_mean[target_channel]),
            float(stats_std[target_channel]),
            "normalization_stats['mean/std'][target_channel]",
        )

    if "target_mean" in normalization_stats and "target_std" in normalization_stats:
        return (
            float(np.asarray(normalization_stats["target_mean"])),
            float(np.asarray(normalization_stats["target_std"])),
            "normalization_stats['target_mean/std']",
        )

    return None, None, "missing"


def generate_scale_diagnostics(model, test_loader, device: torch.device, diagnostics_path: Path) -> str:
    """Run the first three test batches and record scale diagnostics."""

    dataset = test_loader.dataset
    target_channel = int(getattr(dataset, "target_channel", -1))
    target_mean, target_std, target_stat_source = resolve_target_stats_for_inverse(dataset)
    dataset_target_mean = getattr(dataset, "target_mean", None)
    dataset_target_std = getattr(dataset, "target_std", None)
    input_channel_count = int(getattr(dataset, "input_channel_count", -1))

    lines = []
    lines.append("Test Scale Diagnostics")
    lines.append(f"target_channel: {target_channel}")
    lines.append(f"input_channel_count: {input_channel_count}")
    lines.append(f"normalize_target: {bool(getattr(dataset, 'normalize_target', False))}")
    lines.append(f"target_stat_source: {target_stat_source}")
    lines.append(f"target_mean_used_for_inverse: {target_mean}")
    lines.append(f"target_std_used_for_inverse: {target_std}")
    lines.append(f"dataset.target_mean: {dataset_target_mean}")
    lines.append(f"dataset.target_std: {dataset_target_std}")
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
            y_pred = model(x_batch)

            if target_mean is not None and target_std is not None and bool(getattr(dataset, "normalize_target", False)):
                y_true_inverse = inverse_normalize_scalar_channel_map(
                    y_batch.detach().cpu().numpy(),
                    target_mean,
                    target_std,
                )
                y_pred_inverse = inverse_normalize_scalar_channel_map(
                    y_pred.detach().cpu().numpy(),
                    target_mean,
                    target_std,
                )
            else:
                y_true_inverse = y_batch.detach().cpu().numpy()
                y_pred_inverse = y_pred.detach().cpu().numpy()

            abs_error_inverse = np.abs(y_pred_inverse - y_true_inverse)

            lines.append(f"Batch {batch_index}")
            lines.append(format_stats("X normalized", tensor_stats(x_batch)))
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for standalone test evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvLSTM U-Net checkpoint on the test split."
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

    if torch is None:
        raise ImportError("PyTorch is required to evaluate the ConvLSTM U-Net model.")

    args = build_argument_parser().parse_args()
    config = _ensure_config_path(load_config(args.config), args.config)

    train_loader, _, test_loader = create_dataloaders(config)
    if len(test_loader.dataset) == 0:
        raise ValueError("Test split is empty; cannot evaluate the model.")

    dataset = test_loader.dataset
    target_channel = int(getattr(dataset, "target_channel", int(config["target_channel"])))
    input_channel_count = int(getattr(dataset, "input_channel_count", config.get("input_channel_count", 0)))
    for warning_message in warning_messages_for_target_channel(target_channel, input_channel_count):
        print(f"WARNING: {warning_message}")

    input_sequence_length = int(config.get("input_sequence_length", 1))
    output_channels = int(config.get("model", {}).get("output_channels", 1))
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

    raw_results = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        config=config,
        device=device,
        input_sequence_length=input_sequence_length,
        input_channels=input_channels,
        output_channels=output_channels,
        train=False,
    )
    test_results = _rename_result_prefix(raw_results, "val_", "test_")

    diagnostics_path = Path("artifacts/logs/test_scale_diagnostics.txt").expanduser().resolve()
    generate_scale_diagnostics(model, test_loader, device, diagnostics_path)

    persistence_results = evaluate_persistence_for_channel(
        config_path=args.config,
        target_channel=target_channel,
        num_visualizations=0,
        compute_threshold_metrics=False,
        output_dir=None,
        verbose=False,
    )
    persistence_mae = float(persistence_results["regression_metrics"]["test_mae"])
    model_mae = float(test_results.get("test_mae", math.nan))
    if np.isfinite(model_mae) and np.isfinite(persistence_mae) and persistence_mae > 0.0 and model_mae > 10.0 * persistence_mae:
        print("WARNING: Model MAE is more than 10x worse than persistence MAE. This is likely a data/scale/target issue, not an architecture issue.")

    print(f"checkpoint: {resolved_checkpoint_path}")
    print(f"test samples: {len(test_loader.dataset)}")
    print(f"target_channel: {target_channel}")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.6f}")
    print(f"persistence_test_mae: {persistence_mae:.6f}")
    print(f"diagnostics_file: {diagnostics_path}")


if __name__ == "__main__":
    main()
