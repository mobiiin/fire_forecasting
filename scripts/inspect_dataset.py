"""Inspect sequential NumPy tensors and report dataset sanity checks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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

    return np.load(file_path, mmap_mode="r", allow_pickle=False)


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


@dataclass
class FileInspectionResult:
    """Detailed validation result for a single dataset file."""

    file_path: Path
    loaded: bool
    shape: Tuple[int, ...] | None = None
    dtype: str | None = None
    load_error: str | None = None
    is_numeric: bool = False
    finite_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    finite_min: float | None = None
    finite_max: float | None = None


@dataclass
class DatasetInspectionSummary:
    """Aggregate validation results across the full dataset."""

    file_count: int
    reference_shape: Tuple[int, ...]
    reference_dtype: str
    files_scanned: int = 0
    load_failures: List[str] | None = None
    non_3d_files: List[str] | None = None
    shape_mismatches: List[str] | None = None
    dtype_mismatches: List[str] | None = None
    non_numeric_files: List[str] | None = None
    nan_files: List[str] | None = None
    inf_files: List[str] | None = None
    all_non_finite_files: List[str] | None = None
    total_nan_count: int = 0
    total_inf_count: int = 0
    global_finite_min: float | None = None
    global_finite_max: float | None = None

    def __post_init__(self) -> None:
        self.load_failures = []
        self.non_3d_files = []
        self.shape_mismatches = []
        self.dtype_mismatches = []
        self.non_numeric_files = []
        self.nan_files = []
        self.inf_files = []
        self.all_non_finite_files = []

    def has_issues(self) -> bool:
        return any(
            (
                self.load_failures,
                self.non_3d_files,
                self.shape_mismatches,
                self.dtype_mismatches,
                self.non_numeric_files,
                self.nan_files,
                self.inf_files,
                self.all_non_finite_files,
            )
        )


def summarize_array(array: np.ndarray) -> str:
    """Format basic numeric summary statistics for a tensor."""

    if not np.issubdtype(array.dtype, np.number):
        return f"dtype={array.dtype}, non-numeric"

    finite_mask = np.isfinite(array)
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    finite_count = int(finite_mask.sum())
    if finite_count == 0:
        return (
            f"dtype={array.dtype}, finite_count=0, "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )

    finite_values = array[finite_mask]
    return (
        f"dtype={array.dtype}, min={finite_values.min():.6g}, max={finite_values.max():.6g}, "
        f"mean={finite_values.mean():.6g}, std={finite_values.std():.6g}, "
        f"nan_count={nan_count}, inf_count={inf_count}"
    )


def inspect_file(file_path: Path) -> FileInspectionResult:
    """Inspect one file and capture validation details without raising."""

    try:
        array = load_array(file_path)
    except Exception as exc:
        return FileInspectionResult(
            file_path=file_path,
            loaded=False,
            load_error=f"{type(exc).__name__}: {exc}",
        )

    result = FileInspectionResult(
        file_path=file_path,
        loaded=True,
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype),
        is_numeric=bool(np.issubdtype(array.dtype, np.number)),
    )
    if not result.is_numeric:
        return result

    finite_mask = np.isfinite(array)
    result.finite_count = int(finite_mask.sum())
    result.nan_count = int(np.isnan(array).sum())
    result.inf_count = int(np.isinf(array).sum())

    if result.finite_count > 0:
        finite_values = array[finite_mask]
        result.finite_min = float(finite_values.min())
        result.finite_max = float(finite_values.max())

    return result


def format_issue_examples(entries: Sequence[str], maximum_entries: int = 10) -> str:
    """Format a short preview list for issue summaries."""

    if not entries:
        return "0"
    preview = ", ".join(entries[:maximum_entries])
    if len(entries) > maximum_entries:
        preview += f", ... (+{len(entries) - maximum_entries} more)"
    return f"{len(entries)} [{preview}]"


def inspect_dataset(files: Sequence[Path]) -> DatasetInspectionSummary:
    """Validate the whole dataset and return a full inspection summary."""

    first_result = inspect_file(files[0])
    if not first_result.loaded:
        raise ValueError(
            f"Failed to load first file {files[0].name}: {first_result.load_error}"
        )
    if first_result.shape is None:
        raise ValueError(f"Could not determine shape for first file {files[0].name}.")
    if len(first_result.shape) != 3:
        raise ValueError(
            f"Expected first tensor to be 3D, got shape {first_result.shape} in {files[0].name}."
        )

    target_shape = first_result.shape
    target_dtype = str(first_result.dtype)
    height, width, channels = target_shape
    summary = DatasetInspectionSummary(
        file_count=len(files),
        reference_shape=target_shape,
        reference_dtype=target_dtype,
    )

    sample_indices = select_sample_indices(len(files))
    sample_results = {index: inspect_file(files[index]) for index in sample_indices}

    print(f"number of files: {len(files)}")
    print("first 5 filenames:")
    for file_path in files[:5]:
        print(f"  {file_path.name}")
    print("last 5 filenames:")
    for file_path in files[-5:]:
        print(f"  {file_path.name}")

    print(f"reference shape: {target_shape}")
    print(f"reference dtype: {target_dtype}")
    print(f"inferred dimensions: C={channels}, H={height}, W={width}")

    print("sample file statistics:")
    for index in sample_indices:
        result = sample_results[index]
        if not result.loaded:
            print(f"  {result.file_path.name}: load_error={result.load_error}")
            continue

        try:
            sample_array = load_array(result.file_path)
            print(f"  {result.file_path.name}: {summarize_array(sample_array)}")
        except Exception as exc:
            print(f"  {result.file_path.name}: failed to summarize ({type(exc).__name__}: {exc})")

    for file_path in files:
        result = first_result if file_path == files[0] else inspect_file(file_path)
        summary.files_scanned += 1

        if not result.loaded:
            summary.load_failures.append(f"{file_path.name} ({result.load_error})")
            continue
        if result.shape is None:
            summary.load_failures.append(f"{file_path.name} (shape unavailable)")
            continue
        if len(result.shape) != 3:
            summary.non_3d_files.append(f"{file_path.name}: {result.shape}")
            continue
        if result.shape != target_shape:
            summary.shape_mismatches.append(f"{file_path.name}: {result.shape}")
        if result.dtype != target_dtype:
            summary.dtype_mismatches.append(f"{file_path.name}: {result.dtype}")
        if not result.is_numeric:
            summary.non_numeric_files.append(f"{file_path.name}: {result.dtype}")
            continue

        if result.nan_count > 0:
            summary.nan_files.append(f"{file_path.name}: {result.nan_count}")
            summary.total_nan_count += result.nan_count
        if result.inf_count > 0:
            summary.inf_files.append(f"{file_path.name}: {result.inf_count}")
            summary.total_inf_count += result.inf_count
        if result.finite_count == 0:
            summary.all_non_finite_files.append(file_path.name)
            continue

        if summary.global_finite_min is None or (
            result.finite_min is not None and result.finite_min < summary.global_finite_min
        ):
            summary.global_finite_min = result.finite_min
        if summary.global_finite_max is None or (
            result.finite_max is not None and result.finite_max > summary.global_finite_max
        ):
            summary.global_finite_max = result.finite_max

    print("full dataset scan:")
    print(f"  files scanned: {summary.files_scanned}")
    print(f"  load failures: {format_issue_examples(summary.load_failures)}")
    print(f"  non-3D tensors: {format_issue_examples(summary.non_3d_files)}")
    print(f"  shape mismatches: {format_issue_examples(summary.shape_mismatches)}")
    print(f"  dtype mismatches: {format_issue_examples(summary.dtype_mismatches)}")
    print(f"  non-numeric tensors: {format_issue_examples(summary.non_numeric_files)}")
    print(
        "  files with NaNs: "
        f"{format_issue_examples(summary.nan_files)} "
        f"(total_nan_count={summary.total_nan_count})"
    )
    print(
        "  files with infinities: "
        f"{format_issue_examples(summary.inf_files)} "
        f"(total_inf_count={summary.total_inf_count})"
    )
    print(
        "  files with no finite values: "
        f"{format_issue_examples(summary.all_non_finite_files)}"
    )
    print(f"  global finite min: {summary.global_finite_min}")
    print(f"  global finite max: {summary.global_finite_max}")

    if summary.has_issues():
        raise ValueError("Dataset inspection failed. See full dataset scan summary above.")

    print("dataset inspection passed with no structural or non-finite value issues")
    return summary


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
        sys.exit(1)
