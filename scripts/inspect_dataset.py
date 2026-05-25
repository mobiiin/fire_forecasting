"""Inspect sequential NumPy tensors and report dataset sanity checks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.config import load_config
from src.data.splits import chronological_split_indices


def resolve_path(base_path: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the config file location."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def discover_files(data_dir: Path, file_pattern: str) -> List[Path]:
    """Find files matching the configured pattern."""

    files = sorted(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in '{data_dir}' using pattern '{file_pattern}'."
        )
    return sort_chronologically(files)


def sort_chronologically(files: Sequence[Path]) -> List[Path]:
    """Sort by trailing numeric suffix when available, otherwise lexicographically."""

    numeric_suffixes = [extract_numeric_suffix(file_path.stem) for file_path in files]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, files), key=lambda pair: pair[0])]
    return sorted(files, key=lambda path: path.name)


def extract_numeric_suffix(name: str) -> int | None:
    """Extract the trailing numeric suffix from a filename stem."""

    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def load_array(file_path: Path) -> np.ndarray:
    """Load a single NumPy file without materializing the whole dataset."""

    return np.load(file_path, allow_pickle=False)


def infer_dimensions(array_shape: Sequence[int]) -> Tuple[int, int, int]:
    """Infer channel-first or channel-last dimensions from a tensor shape."""

    if len(array_shape) != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(array_shape)}.")

    height, width, channels = array_shape
    return int(channels), int(height), int(width)


def select_sample_indices(file_count: int, maximum_samples: int = 5) -> List[int]:
    """Choose evenly spaced file indices for statistics reporting."""

    if file_count <= maximum_samples:
        return list(range(file_count))

    raw_indices = np.linspace(0, file_count - 1, num=maximum_samples)
    indices = sorted({int(round(value)) for value in raw_indices})
    return indices


def summarize_array(array: np.ndarray) -> str:
    """Format basic numeric summary statistics for a tensor."""

    finite_mask = np.isfinite(array)
    has_nan = bool(np.isnan(array).any())
    has_inf = bool(np.isinf(array).any())
    if not finite_mask.any():
        return "all values are non-finite"

    finite_values = array[finite_mask]
    return (
        f"min={finite_values.min():.6g}, max={finite_values.max():.6g}, "
        f"mean={finite_values.mean():.6g}, std={finite_values.std():.6g}, "
        f"has_nan={has_nan}, has_inf={has_inf}"
    )


def inspect_dataset(files: Sequence[Path]) -> None:
    """Print a compact inspection report for a sequential tensor dataset."""

    first_array = load_array(files[0])
    target_shape = tuple(first_array.shape)
    target_dtype = first_array.dtype
    channels, height, width = infer_dimensions(target_shape)

    shape_consistent = True
    inconsistent_files: List[str] = []
    any_nan = bool(np.isnan(first_array).any())
    any_inf = bool(np.isinf(first_array).any())

    sample_indices = select_sample_indices(len(files))

    print(f"number of files: {len(files)}")
    print("first 5 filenames:")
    for file_path in files[:5]:
        print(f"  {file_path.name}")
    print("last 5 filenames:")
    for file_path in files[-5:]:
        print(f"  {file_path.name}")

    print(f"first file shape: {target_shape}")
    print(f"dtype: {target_dtype}")
    print(f"inferred dimensions: C={channels}, H={height}, W={width}")

    print("sample file statistics:")
    for index in sample_indices:
        file_path = files[index]
        array = first_array if index == 0 else load_array(file_path)
        if tuple(array.shape) != target_shape:
            shape_consistent = False
            inconsistent_files.append(f"{file_path.name}: {tuple(array.shape)}")
        any_nan = any_nan or bool(np.isnan(array).any())
        any_inf = any_inf or bool(np.isinf(array).any())
        print(f"  {file_path.name}: {summarize_array(array)}")

    for file_path in files[1:]:
        array = load_array(file_path)
        if tuple(array.shape) != target_shape:
            shape_consistent = False
            inconsistent_files.append(f"{file_path.name}: {tuple(array.shape)}")
        any_nan = any_nan or bool(np.isnan(array).any())
        any_inf = any_inf or bool(np.isinf(array).any())

    print(f"all files have the same shape: {shape_consistent}")
    print(f"any NaNs present: {any_nan}")
    print(f"any infinities present: {any_inf}")

    if not shape_consistent:
        details = "\n".join(f"  {entry}" for entry in inconsistent_files[:10])
        raise ValueError(
            "Dataset contains inconsistent tensor shapes. "
            f"Expected {target_shape}, but found:\n{details}"
        )


def report_split_sample_counts(files: Sequence[Path], config: dict) -> None:
    """Print how many valid samples are available in each chronological split."""

    required_keys = (
        "input_sequence_length",
        "prediction_horizon",
        "train_fraction",
        "val_fraction",
        "test_fraction",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Config is missing required split key(s): {', '.join(missing)}")

    splits = chronological_split_indices(
        num_timesteps=len(files),
        input_sequence_length=int(config["input_sequence_length"]),
        prediction_horizon=int(config["prediction_horizon"]),
        train_fraction=float(config["train_fraction"]),
        val_fraction=float(config["val_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )

    print("valid forecasting samples by split:")
    print(f"  train: {len(splits['train'])}")
    print(f"  val:   {len(splits['val'])}")
    print(f"  test:  {len(splits['test'])}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for dataset inspection."""

    parser = argparse.ArgumentParser(description="Inspect sequential wildfire tensors.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    """Run the dataset inspection workflow."""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    if "data_dir" not in config:
        raise KeyError("Config is missing required key 'data_dir'.")
    if "file_pattern" not in config:
        raise KeyError("Config is missing required key 'file_pattern'.")

    data_dir = resolve_path(config_path, str(config["data_dir"]))
    file_pattern = str(config["file_pattern"])

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = discover_files(data_dir, file_pattern)
    inspect_dataset(files)
    report_split_sample_counts(files, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"error: {exc}", file=sys.stderr)
        raise
