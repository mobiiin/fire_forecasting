"""Diagnose how a long ASC/fuel list should be unpacked into 2D tensor channels.

"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HEADER_KEYS = ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value")
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ShapeCandidate:
    height: int
    width: int
    source_total: int
    source_label: str
    score: float


def find_file(folder: Path, filename: str) -> Path:
    """Search recursively for the requested file name inside a folder."""

    folder = folder.expanduser().resolve()
    if folder.is_file() and folder.name == filename:
        return folder

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    matches = sorted(path for path in folder.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find '{filename}' under '{folder}'.")
    if len(matches) > 1:
        print(f"[find_file] Found {len(matches)} matches; using {matches[0]}")
    return matches[0]


def _parse_numeric_line(line: str) -> list[float]:
    return [float(token) for token in NUMBER_PATTERN.findall(line)]


def _is_standard_header_line(line: str) -> bool:
    lower = line.strip().lower()
    return any(lower.startswith(key) for key in HEADER_KEYS)


def load_numeric_asc(file_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ASC/fuel text file and extract a flat numeric array.

    The loader skips a header-like first line when it clearly contains metadata,
    and it also skips standard ASCII-grid header lines when present.
    """

    text = file_path.read_text(errors="ignore")
    lines = text.splitlines()

    header_lines: list[str] = []
    data_values: list[float] = []
    parsed_header: dict[str, Any] = {}
    first_data_seen = False

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if _is_standard_header_line(line):
            header_lines.append(line)
            key, *rest = line.split()
            numeric_tokens = _parse_numeric_line(" ".join(rest))
            if numeric_tokens:
                parsed_header[key.lower()] = numeric_tokens[0]
            continue

        numeric_tokens = _parse_numeric_line(line)
        if not numeric_tokens:
            continue

        if index == 0 and len(numeric_tokens) >= 3:
            header_lines.append(line)
            parsed_header["raw_first_line"] = line
            parsed_header["raw_first_line_numbers"] = numeric_tokens
            parsed_header["ncols_guess"] = int(round(numeric_tokens[0]))
            parsed_header["nrows_guess"] = int(round(numeric_tokens[1]))
            if len(numeric_tokens) >= 3:
                parsed_header["channels_guess"] = int(round(numeric_tokens[2]))
            if len(numeric_tokens) > 3:
                parsed_header["extra_first_line_numbers"] = numeric_tokens[3:]
            continue

        first_data_seen = True
        data_values.extend(numeric_tokens)

    if not first_data_seen and not data_values:
        raise ValueError(f"No numeric values found in {file_path}")

    array = np.asarray(data_values, dtype=np.float64)
    metadata = {
        "header_lines": header_lines,
        "parsed_header": parsed_header,
        "line_count": len(lines),
        "data_value_count": int(array.size),
    }
    return array, metadata


def factor_pairs(number: int) -> list[tuple[int, int]]:
    """Return factor pairs for a positive integer."""

    if number <= 0:
        return []

    pairs: list[tuple[int, int]] = []
    root = int(math.isqrt(number))
    for divisor in range(1, root + 1):
        if number % divisor == 0:
            pairs.append((divisor, number // divisor))
    return pairs


def _aspect_score(height: int, width: int) -> float:
    ratio = max(height / width, width / height)
    return abs(math.log(ratio))


def candidate_shapes(
    total_values: int,
    header_hint: dict[str, Any] | None = None,
    max_shapes: int = 20,
) -> list[ShapeCandidate]:
    """Propose plausible HxW shapes from total values and optional header hints."""

    header_hint = header_hint or {}
    preferred_pairs: list[tuple[int, int, str]] = []

    nrows = header_hint.get("nrows_guess")
    ncols = header_hint.get("ncols_guess")
    channels = header_hint.get("channels_guess")
    if isinstance(nrows, int) and isinstance(ncols, int):
        label = "header_full"
        source_total = nrows * ncols
        if isinstance(channels, int) and channels > 0:
            source_total *= channels
        preferred_pairs.append((int(nrows), int(ncols), label))

    candidates: list[ShapeCandidate] = []
    totals_to_try = [total_values]
    if total_values % 2 == 0:
        totals_to_try.append(total_values // 2)

    for source_total in totals_to_try:
        for left, right in factor_pairs(source_total):
            for height, width in {(left, right), (right, left)}:
                if height < 16 or width < 16:
                    continue
                if max(height, width) / min(height, width) > 4.0:
                    continue
                candidates.append(
                    ShapeCandidate(
                        height=int(height),
                        width=int(width),
                        source_total=int(source_total),
                        source_label="half" if source_total * 2 == total_values else "full",
                        score=_aspect_score(int(height), int(width)),
                    )
                )

    seen: set[tuple[int, int, int, str]] = set()
    unique_candidates: list[ShapeCandidate] = []
    for shape in candidates:
        key = (shape.height, shape.width, shape.source_total, shape.source_label)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(shape)

    def sort_key(shape: ShapeCandidate) -> tuple[float, int, int, int]:
        preferred = 0
        for pref_height, pref_width, _ in preferred_pairs:
            if (shape.height, shape.width) == (pref_height, pref_width):
                preferred = -10_000
        area_diff = abs(shape.height * shape.width - total_values)
        if shape.source_total * 2 == total_values:
            area_diff = abs(shape.height * shape.width * 2 - total_values)
        return (preferred + shape.score, area_diff, abs(shape.height - shape.width), -shape.source_total)

    ordered = sorted(unique_candidates, key=sort_key)
    return ordered[:max_shapes]


def finite_values(array: np.ndarray) -> np.ndarray:
    """Return finite values as a flattened float64 array."""

    values = np.asarray(array, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return values


def stats_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        mean = float(np.mean(finite))
        std = float(np.std(finite))
    else:
        minimum = maximum = mean = std = float("nan")
    return {
        "total_values": int(values.size),
        "finite_values": int(finite.size),
        "nan_count": int(np.isnan(values).sum()),
        "zero_count": int(np.sum(values == 0)),
        "negative_count": int(np.sum(values < 0)),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "first_20": values[:20].tolist(),
    }


def robust_limits(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    values = finite_values(array)
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, lower))
    vmax = float(np.percentile(values, upper))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def robust_imshow(
    axis: plt.Axes,
    array: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    lower: float = 1.0,
    upper: float = 99.0,
) -> Any:
    """Show a 2D array with robust percentile-based color limits."""

    array = np.asarray(array, dtype=np.float64)
    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(array, lower=lower, upper=upper)
    image = axis.imshow(array, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return image


def quadrant_tile_score(array: np.ndarray) -> dict[str, float] | None:
    """Measure similarity between quadrants to detect tiled or repeated layouts."""

    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        return None

    height, width = array.shape
    if height < 2 or width < 2 or height % 2 or width % 2:
        return None

    half_h = height // 2
    half_w = width // 2
    q1 = array[:half_h, :half_w]
    q2 = array[:half_h, half_w:]
    q3 = array[half_h:, :half_w]
    q4 = array[half_h:, half_w:]

    def corr(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        mask = np.isfinite(left) & np.isfinite(right)
        if int(mask.sum()) < 2:
            return float("nan")
        left = left[mask]
        right = right[mask]
        if np.std(left) == 0 or np.std(right) == 0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "q1_q2": corr(q1, q2),
        "q1_q3": corr(q1, q3),
        "q1_q4": corr(q1, q4),
    }


def _format_scores(scores: dict[str, float] | None) -> str:
    if not scores:
        return "n/a"
    parts = []
    for key, value in scores.items():
        if math.isnan(value):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def _histogram_axis(axis: plt.Axes, arrays: dict[str, np.ndarray], title: str) -> None:
    colors = {"surface": "tab:blue", "canopy": "tab:orange", "difference": "tab:green"}
    for name, array in arrays.items():
        values = finite_values(array)
        if values.size == 0:
            continue
        axis.hist(values, bins=80, alpha=0.45, color=colors.get(name, None), label=name, density=True)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("value")
    axis.set_ylabel("density")
    if len(arrays) > 1:
        axis.legend(fontsize=8)


def visualize_single_channel(
    output_path: Path,
    array: np.ndarray,
    layout_name: str,
    shape: tuple[int, int],
    reshape_order: str,
    channel_interpretation: str,
    tile_scores: dict[str, float] | None = None,
) -> None:
    """Save a single-channel diagnostic figure."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    array = np.asarray(array, dtype=np.float64)
    image_title = f"{layout_name}\nshape={shape}, order={reshape_order}\n{channel_interpretation}"
    robust_imshow(axes[0], array, image_title, cmap="viridis")
    _histogram_axis(axes[1], {"surface": array}, "Histogram")
    score_text = _format_scores(tile_scores)
    fig.suptitle(f"{layout_name} | {shape} | tile scores: {score_text}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def visualize_two_channels(
    output_path: Path,
    surface: np.ndarray,
    canopy: np.ndarray,
    layout_name: str,
    shape: tuple[int, int],
    reshape_order: str,
    channel_interpretation: str,
    surface_tile_scores: dict[str, float] | None = None,
    canopy_tile_scores: dict[str, float] | None = None,
) -> None:
    """Save a two-channel diagnostic figure with surface, canopy, difference, and histogram."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    surface = np.asarray(surface, dtype=np.float64)
    canopy = np.asarray(canopy, dtype=np.float64)
    difference = canopy - surface

    combined = np.concatenate([finite_values(surface), finite_values(canopy)])
    if combined.size:
        vmin = float(np.percentile(combined, 1.0))
        vmax = float(np.percentile(combined, 99.0))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    title_prefix = f"{layout_name}\nshape={shape}, order={reshape_order}\n{channel_interpretation}"
    robust_imshow(axes[0, 0], surface, f"Surface\n{title_prefix}", cmap="viridis", vmin=vmin, vmax=vmax)
    robust_imshow(axes[0, 1], canopy, f"Canopy\n{title_prefix}", cmap="viridis", vmin=vmin, vmax=vmax)
    robust_imshow(axes[1, 0], difference, f"Canopy - Surface\n{title_prefix}", cmap="coolwarm")
    _histogram_axis(
        axes[1, 1],
        {"surface": surface, "canopy": canopy, "difference": difference},
        "Histogram",
    )

    score_text = (
        f"surface tile scores: {_format_scores(surface_tile_scores)} | "
        f"canopy tile scores: {_format_scores(canopy_tile_scores)}"
    )
    fig.suptitle(f"{layout_name} | {shape} | {score_text}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _reshape_safe(array: np.ndarray, shape: tuple[int, int], order: str) -> np.ndarray | None:
    try:
        return np.asarray(array, dtype=np.float64).reshape(shape, order=order)
    except ValueError:
        return None


def _save_layout_single(
    output_dir: Path,
    layout_name: str,
    source_array: np.ndarray,
    shape: tuple[int, int],
    order: str,
    channel_interpretation: str,
    notes: list[str],
) -> tuple[Path | None, dict[str, float] | None]:
    reshaped = _reshape_safe(source_array, shape, order)
    if reshaped is None:
        notes.append(f"{layout_name}: incompatible with shape {shape} and order {order}")
        return None, None

    tile_scores = quadrant_tile_score(reshaped)
    if tile_scores:
        notes.append(f"{layout_name}: tile scores {_format_scores(tile_scores)}")

    output_path = output_dir / f"{layout_name}_H{shape[0]}_W{shape[1]}_order{order}.png"
    visualize_single_channel(
        output_path,
        reshaped,
        layout_name,
        shape,
        order,
        channel_interpretation,
        tile_scores=tile_scores,
    )
    return output_path, tile_scores


def _save_layout_two_channel(
    output_dir: Path,
    layout_name: str,
    surface: np.ndarray,
    canopy: np.ndarray,
    shape: tuple[int, int],
    order: str,
    channel_interpretation: str,
    notes: list[str],
) -> tuple[Path | None, dict[str, float] | None, dict[str, float] | None]:
    surface_2d = _reshape_safe(surface, shape, order)
    canopy_2d = _reshape_safe(canopy, shape, order)
    if surface_2d is None or canopy_2d is None:
        notes.append(f"{layout_name}: incompatible with shape {shape} and order {order}")
        return None, None, None

    surface_scores = quadrant_tile_score(surface_2d)
    canopy_scores = quadrant_tile_score(canopy_2d)
    if surface_scores:
        notes.append(f"{layout_name}: surface tile scores {_format_scores(surface_scores)}")
    if canopy_scores:
        notes.append(f"{layout_name}: canopy tile scores {_format_scores(canopy_scores)}")

    output_path = output_dir / f"{layout_name}_H{shape[0]}_W{shape[1]}_order{order}.png"
    visualize_two_channels(
        output_path,
        surface_2d,
        canopy_2d,
        layout_name,
        shape,
        order,
        channel_interpretation,
        surface_tile_scores=surface_scores,
        canopy_tile_scores=canopy_scores,
    )
    return output_path, surface_scores, canopy_scores


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fuel file flattening and reshape layouts.")
    parser.add_argument("--folder", required=True, help="Folder to search recursively for the fuel file.")
    parser.add_argument("--file", default="KINGNSM04ASC.fuel.1918", help="Exact file name to locate.")
    parser.add_argument("--outdir", default="fuel_layout_diagnostics", help="Directory for figures and summary.")
    parser.add_argument("--max_shapes", type=int, default=20, help="Maximum candidate shapes to test.")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Searching for {args.file} under {folder}")
    file_path = find_file(folder, args.file)
    print(f"[main] Using file: {file_path}")

    print("[main] Loading numeric values")
    values, metadata = load_numeric_asc(file_path)
    summary = stats_summary(values)

    print(f"[main] Total numeric values: {summary['total_values']}")
    print(
        f"[main] min={_fmt(summary['min'])}, max={_fmt(summary['max'])}, mean={_fmt(summary['mean'])}, std={_fmt(summary['std'])}"
    )
    print(
        f"[main] NaNs={summary['nan_count']}, zeros={summary['zero_count']}, negatives={summary['negative_count']}"
    )
    print(f"[main] First 20 numeric values: {summary['first_20']}")

    parsed_header = metadata.get("parsed_header", {})
    print(f"[main] Parsed header: {parsed_header if parsed_header else 'none'}")

    preferred_shape = None
    nrows_guess = parsed_header.get("nrows_guess")
    ncols_guess = parsed_header.get("ncols_guess")
    channels_guess = parsed_header.get("channels_guess")
    if isinstance(nrows_guess, int) and isinstance(ncols_guess, int):
        preferred_shape = (int(nrows_guess), int(ncols_guess))
    elif isinstance(channels_guess, int) and channels_guess == 2 and values.size % 2 == 0:
        half = values.size // 2
        for left, right in factor_pairs(half):
            if left >= 16 and right >= 16 and left == right:
                preferred_shape = (left, right)
                break

    shapes = candidate_shapes(values.size, parsed_header, max_shapes=args.max_shapes)
    if preferred_shape is not None:
        shapes = sorted(
            shapes,
            key=lambda candidate: (
                0 if (candidate.height, candidate.width) == preferred_shape else 1,
                candidate.score,
                abs(candidate.height - candidate.width),
            ),
        )

    print("[main] Candidate shapes:")
    for index, shape in enumerate(shapes, start=1):
        print(
            f"  {index:02d}. {shape.height} x {shape.width} | source={shape.source_label}:{shape.source_total} | score={shape.score:.4f}"
        )

    candidate_logs: list[str] = []
    generated_files: list[Path] = []
    suspicious_notes: list[str] = []

    total_values = int(values.size)
    for shape in shapes:
        area = shape.height * shape.width
        print(f"[main] Testing shape {shape.height} x {shape.width}")

        if area == total_values:
            print("[main]   Layout A: single-channel full grid")
            for order in ("C", "F"):
                note_list: list[str] = []
                output, tile_scores = _save_layout_single(
                    output_dir,
                    f"layoutA_full_{shape.height}x{shape.width}",
                    values,
                    (shape.height, shape.width),
                    order,
                    "single-channel full grid",
                    note_list,
                )
                candidate_logs.extend(note_list)
                if output is not None:
                    generated_files.append(output)
                    if tile_scores and any(abs(score) > 0.9 for score in tile_scores.values() if not math.isnan(score)):
                        suspicious_notes.append(f"{output.name} has strong quadrant similarity: {_format_scores(tile_scores)}")

        if area * 2 == total_values:
            half = total_values // 2
            surface_channel_major = values[:half]
            canopy_channel_major = values[half:]
            surface_interleaved = values[0::2]
            canopy_interleaved = values[1::2]

            print("[main]   Layout B: channel-major two-channel")
            for order in ("C", "F"):
                note_list = []
                output, surface_scores, canopy_scores = _save_layout_two_channel(
                    output_dir,
                    f"layoutB_channel_major_{shape.height}x{shape.width}",
                    surface_channel_major,
                    canopy_channel_major,
                    (shape.height, shape.width),
                    order,
                    "surface = x[:N], canopy = x[N:2N]",
                    note_list,
                )
                candidate_logs.extend(note_list)
                if output is not None:
                    generated_files.append(output)
                    for scores in (surface_scores, canopy_scores):
                        if scores and any(abs(score) > 0.9 for score in scores.values() if not math.isnan(score)):
                            suspicious_notes.append(f"{output.name} has strong quadrant similarity: {_format_scores(scores)}")

            print("[main]   Layout C: interleaved per-pixel two-channel")
            for order in ("C", "F"):
                note_list = []
                output, surface_scores, canopy_scores = _save_layout_two_channel(
                    output_dir,
                    f"layoutC_interleaved_{shape.height}x{shape.width}",
                    surface_interleaved,
                    canopy_interleaved,
                    (shape.height, shape.width),
                    order,
                    "surface = x[0::2], canopy = x[1::2]",
                    note_list,
                )
                candidate_logs.extend(note_list)
                if output is not None:
                    generated_files.append(output)
                    for scores in (surface_scores, canopy_scores):
                        if scores and any(abs(score) > 0.9 for score in scores.values() if not math.isnan(score)):
                            suspicious_notes.append(f"{output.name} has strong quadrant similarity: {_format_scores(scores)}")

            print("[main]   Layout D: variables x height x width flattened in C order")
            try:
                arr = values.reshape(2, shape.height, shape.width, order="C")
            except ValueError:
                candidate_logs.append(f"layoutD_{shape.height}x{shape.width}: incompatible with reshape(2, H, W)")
            else:
                note_list = []
                output_path = output_dir / f"layoutD_vhw_{shape.height}x{shape.width}_orderC.png"
                surface = arr[0]
                canopy = arr[1]
                surface_scores = quadrant_tile_score(surface)
                canopy_scores = quadrant_tile_score(canopy)
                if surface_scores:
                    note_list.append(f"layoutD_{shape.height}x{shape.width}: surface tile scores {_format_scores(surface_scores)}")
                if canopy_scores:
                    note_list.append(f"layoutD_{shape.height}x{shape.width}: canopy tile scores {_format_scores(canopy_scores)}")
                visualize_two_channels(
                    output_path,
                    surface,
                    canopy,
                    f"layoutD_vhw_{shape.height}x{shape.width}",
                    (shape.height, shape.width),
                    "C",
                    "arr = x.reshape(2, H, W, order='C')",
                    surface_tile_scores=surface_scores,
                    canopy_tile_scores=canopy_scores,
                )
                generated_files.append(output_path)
                candidate_logs.extend(note_list)

            print("[main]   Layout E: height x width x variables flattened in C order")
            try:
                arr = values.reshape(shape.height, shape.width, 2, order="C")
            except ValueError:
                candidate_logs.append(f"layoutE_{shape.height}x{shape.width}: incompatible with reshape(H, W, 2)")
            else:
                surface = arr[:, :, 0]
                canopy = arr[:, :, 1]
                surface_scores = quadrant_tile_score(surface)
                canopy_scores = quadrant_tile_score(canopy)
                if surface_scores:
                    candidate_logs.append(f"layoutE_{shape.height}x{shape.width}: surface tile scores {_format_scores(surface_scores)}")
                if canopy_scores:
                    candidate_logs.append(f"layoutE_{shape.height}x{shape.width}: canopy tile scores {_format_scores(canopy_scores)}")
                output_path = output_dir / f"layoutE_hwv_{shape.height}x{shape.width}_orderC.png"
                visualize_two_channels(
                    output_path,
                    surface,
                    canopy,
                    f"layoutE_hwv_{shape.height}x{shape.width}",
                    (shape.height, shape.width),
                    "C",
                    "arr = x.reshape(H, W, 2, order='C')",
                    surface_tile_scores=surface_scores,
                    canopy_tile_scores=canopy_scores,
                )
                generated_files.append(output_path)

    if any("tile scores" in note for note in candidate_logs):
        suspicious_notes.append("Some candidate layouts show high quadrant correlation, which is consistent with tiled or repeated maps.")

    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Fuel Layout Diagnostics\n\n")
        handle.write(f"- Loaded file: {file_path}\n")
        handle.write(f"- Total values: {summary['total_values']}\n")
        handle.write(f"- Finite values: {summary['finite_values']}\n")
        handle.write(f"- NaNs: {summary['nan_count']}\n")
        handle.write(f"- Zeros: {summary['zero_count']}\n")
        handle.write(f"- Negative values: {summary['negative_count']}\n")
        handle.write(f"- Min: {_fmt(summary['min'])}\n")
        handle.write(f"- Max: {_fmt(summary['max'])}\n")
        handle.write(f"- Mean: {_fmt(summary['mean'])}\n")
        handle.write(f"- Std: {_fmt(summary['std'])}\n")
        handle.write(f"- First 20 numeric values: {summary['first_20']}\n\n")

        handle.write("## Parsed Header\n\n")
        if parsed_header:
            for key, value in parsed_header.items():
                handle.write(f"- {key}: {value}\n")
        else:
            handle.write("- none\n")

        handle.write("\n## Candidate Shapes\n\n")
        for shape in shapes:
            handle.write(
                f"- {shape.height} x {shape.width} | source={shape.source_label}:{shape.source_total} | score={shape.score:.4f}\n"
            )

        handle.write("\n## Generated Figures\n\n")
        for path in generated_files:
            handle.write(f"- {path.name}\n")

        handle.write("\n## Tile Correlation Notes\n\n")
        if candidate_logs:
            for note in candidate_logs:
                handle.write(f"- {note}\n")
        else:
            handle.write("- none\n")

        handle.write("\n## Suspicious Layouts\n\n")
        if suspicious_notes:
            for note in suspicious_notes:
                handle.write(f"- {note}\n")
        else:
            handle.write("- none\n")

        handle.write("\n## How To Read The Result\n\n")
        handle.write("- Open the output directory and compare the PNGs side by side.\n")
        handle.write("- The correct unpacking should look like one coherent top-down fuel map, not repeated quadrants, stripes, or scrambled noise.\n")
        handle.write("- If a layout shows repeated 2x2 quadrants with high correlation, it is probably not the correct reshape.\n")

    print(f"[main] Wrote summary: {summary_path}")
    print(f"[main] Generated {len(generated_files)} figures in {output_dir}")
    print("[main] Next steps:")
    print(f"[main]   1. Open {output_dir}")
    print("[main]   2. Compare the PNGs")
    print("[main]   3. Prefer the layout that looks like one coherent spatial top-down fuel map")
    print("[main]   4. Reject layouts with repeated 2x2 quadrants, stripes, or scrambled noise")


if __name__ == "__main__":
    main()