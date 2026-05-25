"""Compute channel-wise normalization statistics from training input timestamps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.preprocessing import compute_channel_stats
from src.data.splits import chronological_split_indices


def resolve_path(base_path: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the config file location."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def discover_files(data_dir: Path, file_pattern: str) -> list[Path]:
    """Find and chronologically sort files matching the configured pattern."""

    files = list(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in '{data_dir}' using pattern '{file_pattern}'."
        )
    return sort_chronologically(files)


def extract_numeric_suffix(name: str) -> int | None:
    """Extract a trailing numeric suffix from a filename stem."""

    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def sort_chronologically(files: list[Path]) -> list[Path]:
    """Sort by trailing numeric suffix when available, otherwise lexicographically."""

    numeric_suffixes = [extract_numeric_suffix(file_path.stem) for file_path in files]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, files), key=lambda pair: pair[0])]
    return sorted(files, key=lambda path: path.name)


def training_input_file_indices(
    num_timesteps: int,
    input_sequence_length: int,
    prediction_horizon: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> list[int]:
    """Return raw file indices that appear in the training input windows only."""

    splits = chronological_split_indices(
        num_timesteps=num_timesteps,
        input_sequence_length=input_sequence_length,
        prediction_horizon=prediction_horizon,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    training_indices: set[int] = set()
    for start_index in splits["train"]:
        training_indices.update(range(start_index, start_index + input_sequence_length))

    return sorted(training_indices)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for normalization computation."""

    parser = argparse.ArgumentParser(description="Compute normalization statistics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    """Compute and save normalization statistics for the training split only."""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    for required_key in ("data_dir", "file_pattern", "input_sequence_length", "prediction_horizon", "train_fraction", "val_fraction", "test_fraction"):
        if required_key not in config:
            raise KeyError(f"Config is missing required key '{required_key}'.")

    data_dir = resolve_path(config_path, str(config["data_dir"]))
    file_pattern = str(config["file_pattern"])

    normalization_config = config.get("normalization", {})
    if "path" not in normalization_config:
        raise KeyError("Config is missing normalization.path.")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = discover_files(data_dir, file_pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found in '{data_dir}' using pattern '{file_pattern}'."
        )

    training_indices = training_input_file_indices(
        num_timesteps=len(files),
        input_sequence_length=int(config["input_sequence_length"]),
        prediction_horizon=int(config["prediction_horizon"]),
        train_fraction=float(config["train_fraction"]),
        val_fraction=float(config["val_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )

    if not training_indices:
        raise ValueError("No valid training input timestamps were found for normalization.")

    training_files = [files[index] for index in training_indices]
    eps = float(normalization_config.get("epsilon", 1e-6))
    stats = compute_channel_stats(training_files, eps=eps)

    output_path = resolve_path(config_path, str(normalization_config["path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)

    channel_mean = stats["mean"]
    channel_std = stats["std"]
    near_zero_std = int(np.sum(channel_std <= max(eps, 1e-6) * 10.0))

    print(f"C: {channel_mean.shape[0]}")
    print(f"global channel mean range: {channel_mean.min():.6g} to {channel_mean.max():.6g}")
    print(f"global channel std range: {channel_std.min():.6g} to {channel_std.max():.6g}")
    print(f"channels with near-zero std: {near_zero_std}")
    print(f"output path: {output_path}")


if __name__ == "__main__":
    main()
