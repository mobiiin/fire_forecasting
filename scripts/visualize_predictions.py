"""Visualize wildfire predictions and targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None
    DataLoader = None

from src.config import load_config
from src.data.dataset import FireSequenceDataset
from src.data.splits import chronological_split_indices
from src.data.preprocessing import load_normalization_stats
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.visualization.plot_maps import plot_prediction_grid


def _get_section(config: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """Return the first nested mapping found under any of the provided names."""

    for name in names:
        section = config.get(name)
        if isinstance(section, dict):
            return section
    return {}


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
    """Resolve a configured path relative to a config file when available."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_path is None:
        return path.resolve()
    return (base_path.parent / path).resolve()


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Attach the config path so downstream helpers can resolve relative paths."""

    resolved_path = Path(config_path).expanduser().resolve()
    config = dict(config)
    config["config_path"] = str(resolved_path)
    config["_config_path"] = str(resolved_path)
    return config


def _discover_files(config: Mapping[str, Any]) -> list[Path]:
    """Discover and chronologically sort dataset files."""

    config_path_value = config.get("config_path", config.get("_config_path"))
    config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
    data_dir = _resolve_path(config_path, config["data_dir"])
    file_pattern = str(config["file_pattern"])
    files = sorted(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
    return files


def _extract_numeric_suffix(name: str) -> int | None:
    """Extract a trailing numeric suffix from a filename stem if present."""

    digits = []
    for character in reversed(name):
        if character.isdigit():
            digits.append(character)
        else:
            break
    if not digits:
        return None
    return int("".join(reversed(digits)))


def _sort_chronologically(file_paths: list[Path]) -> list[Path]:
    """Sort files by numeric suffix when available, otherwise lexicographically."""

    numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in file_paths]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
    return sorted(file_paths, key=lambda path: path.name)


def _build_test_dataset(config: Mapping[str, Any], normalization_stats) -> FireSequenceDataset:
    """Build the chronological test split with metadata enabled."""

    files = _sort_chronologically(_discover_files(config))
    input_sequence_length = int(config["input_sequence_length"])
    prediction_horizon = int(config["prediction_horizon"])
    input_channel_count = int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0)))
    if input_channel_count <= 0:
        raise KeyError("Config must define a positive input_channel_count or model.input_channels.")
    splits = chronological_split_indices(
        num_timesteps=len(files),
        input_sequence_length=input_sequence_length,
        prediction_horizon=prediction_horizon,
        train_fraction=float(config.get("train_fraction", 0.7)),
        val_fraction=float(config.get("val_fraction", 0.15)),
        test_fraction=float(config.get("test_fraction", 0.15)),
    )
    return FireSequenceDataset(
        file_paths=files,
        sample_indices=splits["test"],
        input_sequence_length=input_sequence_length,
        prediction_horizon=prediction_horizon,
        target_channel=int(config["target_channel"]),
        input_channel_count=input_channel_count,
        task_type=str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))),
        fire_threshold=float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
        normalization_stats=normalization_stats,
        return_metadata=True,
    )


def _build_test_loader(dataset):
    """Create a chronological, sample-by-sample test DataLoader."""

    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to build the visualization DataLoader.")

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _select_device(config: Mapping[str, Any]):
    """Select the inference device from config, defaulting to CPU when unavailable."""

    device_setting = str(config.get("device", _get_section(config, "training").get("device", "auto"))).lower()
    if device_setting == "auto":
        device_setting = "cuda" if torch.cuda.is_available() else "cpu"
    if device_setting == "cuda" and not torch.cuda.is_available():
        device_setting = "cpu"
    return torch.device(device_setting)


def _extract_first_item(value):
    """Normalize collated metadata values to their first scalar/string item."""

    if torch is not None and torch.is_tensor(value):
        return value.reshape(-1)[0].item() if value.numel() else None
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _metadata_to_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a collated metadata batch into a simple dictionary."""

    return {key: _extract_first_item(value) for key, value in metadata.items()}


def _build_checkpoint_path(config: Mapping[str, Any]) -> Path:
    """Resolve the best checkpoint path and fall back to the latest checkpoint if needed."""

    checkpoint_config = _get_section(config, "checkpoint")
    checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
    config_path_value = config.get("config_path", config.get("_config_path"))
    config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
    latest_path, best_path = latest_and_best_checkpoint_paths(_resolve_path(config_path, checkpoint_path))
    selected = best_path if best_path.exists() else latest_path
    if not selected.exists():
        raise FileNotFoundError(
            "No checkpoint found for visualization. "
            f"Checked best='{best_path}' and latest='{latest_path}'."
        )
    return selected


def _build_model(config: Mapping[str, Any], input_channels: int):
    """Instantiate and load the trained ConvLSTM U-Net."""

    model = build_model_from_config(config, input_channels=input_channels)
    return model


def _sample_output_name(metadata: Mapping[str, Any]) -> str:
    """Build an informative filename from the sample metadata."""

    sample_index = metadata.get("sample_index")
    file_path = metadata.get("target_file_path")
    file_stem = Path(str(file_path)).stem if file_path else "sample"
    if sample_index is None:
        return f"{file_stem}.png"
    try:
        sample_index_int = int(sample_index)
    except (TypeError, ValueError):
        return f"{file_stem}.png"
    return f"sample_{sample_index_int:05d}_{file_stem}.png"


def visualize_predictions(config_path: str | Path, num_samples: int = 10) -> list[Path]:
    """Render chronological forecast visualizations for several test samples."""

    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to visualize predictions.")

    config = _ensure_config_path(load_config(config_path), config_path)
    set_seed(int(config.get("seed", _get_section(config, "training").get("seed", 42))))
    logging_config = _get_section(config, "logging")
    logger = setup_logging(str(logging_config.get("level", "INFO")))

    config_path_value = config.get("config_path", config.get("_config_path"))
    config_path_obj = Path(config_path_value).expanduser().resolve() if config_path_value else None
    normalization_config = _get_section(config, "normalization")
    normalization_path = normalization_config.get("path")
    normalization_stats = None
    if normalization_path:
        resolved_normalization_path = _resolve_path(config_path_obj, normalization_path)
        if not resolved_normalization_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {resolved_normalization_path}")
        normalization_stats = load_normalization_stats(resolved_normalization_path)

    test_dataset = _build_test_dataset(config, normalization_stats)
    if len(test_dataset) == 0:
        raise ValueError("Test split is empty; cannot visualize predictions.")
    test_loader = _build_test_loader(test_dataset)
    task_type = str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))).lower()
    fire_threshold = float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5)))
    first_batch = next(iter(test_loader))
    x_batch, y_batch = first_batch[:2]
    if x_batch.ndim != 5:
        raise ValueError(f"Expected x batch to have shape (B, T, C, H, W), got {tuple(x_batch.shape)}.")
    if y_batch.ndim != 4:
        raise ValueError(f"Expected y batch to have shape (B, C, H, W), got {tuple(y_batch.shape)}.")
    input_channels = int(x_batch.shape[2])
    output_channels = int(_get_section(config, "model").get("output_channels", 1))
    if output_channels != 1:
        logger.warning("Visualization assumes a single-channel output; using the first output channel.")

    device = _select_device(config)
    checkpoint_path = _build_checkpoint_path(config)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = _build_model(config, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    visualization_config = _get_section(config, "visualization")
    output_dir = _resolve_path(config_path_obj, visualization_config.get("output_path", "./outputs/visualizations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = str(visualization_config.get("cmap", "inferno"))
    dpi = int(visualization_config.get("dpi", 150))

    saved_paths: list[Path] = []
    max_samples = min(int(num_samples), len(test_loader.dataset))
    with torch.no_grad():
        for sample_index, batch in enumerate(test_loader):
            if sample_index >= max_samples:
                break
            x_sample, y_sample, metadata = batch
            if not isinstance(metadata, Mapping):
                raise TypeError("Expected metadata to be a mapping when return_metadata=True.")
            metadata_dict = _metadata_to_dict(metadata)

            x_sample = x_sample.to(device)
            predicted = model(x_sample)
            if predicted.ndim != 4:
                raise ValueError(f"Model output must have shape (B, C, H, W), got {tuple(predicted.shape)}.")
            if predicted.shape[0] != x_sample.shape[0]:
                raise ValueError("Model output batch size does not match the input batch size.")
            if predicted.shape[-2:] != y_sample.shape[-2:]:
                raise ValueError(
                    f"Prediction spatial size {tuple(predicted.shape[-2:])} does not match target size {tuple(y_sample.shape[-2:])}."
                )

            current_channel_index = int(x_sample.shape[2] - 1)
            current_map = x_sample[0, -1, current_channel_index].detach().cpu().numpy()
            ground_truth_map = y_sample[0, 0].detach().cpu().numpy()
            if task_type == "segmentation":
                predicted_map = torch.sigmoid(predicted[0, 0]).detach().cpu().numpy()
                panel_target_name = "fire perimeter probability"
                contour_threshold = 0.5
            else:
                predicted_map = predicted[0, 0].detach().cpu().numpy()
                panel_target_name = "fire intensity"
                contour_threshold = fire_threshold

            sample_title = f"Sample {sample_index + 1}/{max_samples} | {metadata_dict.get('target_file_path', 'unknown')}"
            output_name = _sample_output_name(metadata_dict)
            output_path = output_dir / output_name
            if output_path.exists():
                output_path = output_dir / f"{output_path.stem}_{sample_index:05d}{output_path.suffix}"

            saved_path = plot_prediction_grid(
                current_map=current_map,
                ground_truth_map=ground_truth_map,
                predicted_map=predicted_map,
                output_path=output_path,
                title=f"{sample_title} | {panel_target_name}",
                threshold=contour_threshold,
                cmap=cmap,
                dpi=dpi,
                normalization_stats=normalization_stats,
                channel_index=current_channel_index,
                draw_contours=True,
            )
            saved_paths.append(saved_path)
            logger.info("Saved visualization: %s", saved_path)

    return saved_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Visualize ConvLSTM U-Net wildfire predictions.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of chronological test samples to visualize.")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    visualize_predictions(args.config, num_samples=args.num_samples)


if __name__ == "__main__":
    main()