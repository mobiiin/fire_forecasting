"""Interactively inspect raw input tensors from the wildfire dataset."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from src.config import load_config


TAIL_CHANNEL_NAMES = [
    "flux_sensible_surface",
    "flux_latent_surface",
    "flux_sensible_canopy",
    "flux_latent_canopy",
    "fuel_surface",
    "fuel_canopy",
]


def resolve_path(base_path: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the config file location."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def extract_numeric_suffix(name: str) -> int | None:
    """Extract a trailing numeric suffix from a filename stem if present."""

    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def sort_chronologically(file_paths: list[Path]) -> list[Path]:
    """Sort files by trailing numeric suffix when available, otherwise lexicographically."""

    numeric_suffixes = [extract_numeric_suffix(path.stem) for path in file_paths]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
    return sorted(file_paths, key=lambda path: path.name)


def discover_files(data_dir: Path, file_pattern: str) -> list[Path]:
    """Find and sort the dataset tensors described by the config file."""

    files = list(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
    return sort_chronologically(files)


def load_tensor(file_path: Path) -> np.ndarray:
    """Load one tensor lazily from disk."""

    return np.load(file_path, mmap_mode="r", allow_pickle=False)


def percentile_limits(array: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    """Compute robust display limits for a single channel slice."""

    finite_values = np.asarray(array, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(finite_values, lower))
    vmax = float(np.percentile(finite_values, upper))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def channel_label(channel_index: int, total_channels: int) -> str:
    """Return a readable name for the tail channels when available."""

    tail_start = max(0, total_channels - len(TAIL_CHANNEL_NAMES))
    if channel_index >= tail_start:
        tail_offset = channel_index - tail_start
        if 0 <= tail_offset < len(TAIL_CHANNEL_NAMES):
            return TAIL_CHANNEL_NAMES[tail_offset]
    return f"channel_{channel_index:02d}"


class InputDatasetViewer:
    """Interactive viewer for inspecting timestamped input tensors."""

    def __init__(
        self,
        file_paths: list[Path],
        initial_file_index: int,
        window_size: int,
        channel_start: int,
        cmap: str,
        percentile_low: float,
        percentile_high: float,
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths must not be empty.")

        self.file_paths = file_paths
        self.file_index = int(initial_file_index)
        self.window_size = int(window_size)
        self.channel_start = int(channel_start)
        self.cmap = cmap
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}.")

        first_tensor = load_tensor(self.file_paths[0])
        if first_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensors shaped (H, W, C), got {first_tensor.shape} in {self.file_paths[0]}.")

        self.height, self.width, self.num_channels = map(int, first_tensor.shape)
        if self.window_size > self.num_channels:
            raise ValueError(
                f"window_size must be <= number of channels ({self.num_channels}), got {self.window_size}."
            )

        max_file_index = len(self.file_paths) - 1
        self.file_index = max(0, min(self.file_index, max_file_index))

        if self.channel_start < 0:
            self.channel_start = self.num_channels + self.channel_start
        self.channel_start = max(0, min(self.channel_start, self.num_channels - self.window_size))

        self.fig = None
        self.axes = None
        self.images: list[object | None] = []
        self._build_figure()

    def _build_figure(self) -> None:
        """Create the plot canvas and button controls."""

        rows = math.ceil(self.window_size / 3)
        cols = min(3, self.window_size)
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows))
        self.fig.subplots_adjust(bottom=0.18, top=0.9, wspace=0.18, hspace=0.25)

        axes_array = np.atleast_1d(self.axes).ravel()
        self.axes = axes_array
        self.images = [None] * len(axes_array)

        button_specs = [
            ("prev_time", [0.10, 0.04, 0.12, 0.06], "Prev time"),
            ("next_time", [0.23, 0.04, 0.12, 0.06], "Next time"),
            ("prev_chan", [0.42, 0.04, 0.14, 0.06], "Prev channels"),
            ("next_chan", [0.57, 0.04, 0.14, 0.06], "Next channels"),
            ("reset_tail", [0.77, 0.04, 0.16, 0.06], "Jump to tail"),
        ]
        self.buttons = {}
        for name, rect, label in button_specs:
            button_axis = self.fig.add_axes(rect)
            button = Button(button_axis, label)
            self.buttons[name] = button

        self.buttons["prev_time"].on_clicked(self._previous_time)
        self.buttons["next_time"].on_clicked(self._next_time)
        self.buttons["prev_chan"].on_clicked(self._previous_channels)
        self.buttons["next_chan"].on_clicked(self._next_channels)
        self.buttons["reset_tail"].on_clicked(self._reset_tail)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _current_tensor(self) -> np.ndarray:
        """Load the tensor for the currently selected timestamp."""

        return np.asarray(load_tensor(self.file_paths[self.file_index]), dtype=np.float32)

    def _clamp_channel_start(self) -> None:
        self.channel_start = max(0, min(self.channel_start, self.num_channels - self.window_size))

    def _set_title(self) -> None:
        """Update the figure title with the current timestamp and channel range."""

        file_path = self.file_paths[self.file_index]
        end_channel = self.channel_start + self.window_size - 1
        title = (
            f"{file_path.name} | timestamp {self.file_index + 1}/{len(self.file_paths)} | "
            f"channels {self.channel_start}..{end_channel} of {self.num_channels - 1}"
        )
        self.fig.suptitle(title, fontsize=13)

    def _render(self) -> None:
        """Draw the current timestamp and channel window."""

        tensor = self._current_tensor()
        if tensor.shape != (self.height, self.width, self.num_channels):
            raise ValueError(
                f"Tensor shape changed unexpectedly for {self.file_paths[self.file_index]}: {tensor.shape}"
            )

        channel_slice = tensor[:, :, self.channel_start : self.channel_start + self.window_size]
        axes_array = np.atleast_1d(self.axes).ravel()

        for panel_index, axis in enumerate(axes_array):
            if panel_index >= self.window_size:
                axis.axis("off")
                continue

            channel_index = self.channel_start + panel_index
            channel_map = np.asarray(channel_slice[:, :, panel_index], dtype=np.float32)
            vmin, vmax = percentile_limits(channel_map, self.percentile_low, self.percentile_high)

            image = self.images[panel_index]
            if image is None:
                image = axis.imshow(channel_map, origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax)
                self.images[panel_index] = image
            else:
                image.set_data(channel_map)
                image.set_clim(vmin, vmax)

            axis.set_title(channel_label(channel_index, self.num_channels), fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

            for spine in axis.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#444444")

            axis.set_xlabel(f"ch {channel_index}", fontsize=9)

        self._set_title()
        self.fig.canvas.draw_idle()

    def _previous_time(self, _event) -> None:
        self.file_index = max(0, self.file_index - 1)
        self._render()

    def _next_time(self, _event) -> None:
        self.file_index = min(len(self.file_paths) - 1, self.file_index + 1)
        self._render()

    def _previous_channels(self, _event) -> None:
        self.channel_start -= 1
        self._clamp_channel_start()
        self._render()

    def _next_channels(self, _event) -> None:
        self.channel_start += 1
        self._clamp_channel_start()
        self._render()

    def _reset_tail(self, _event) -> None:
        self.channel_start = self.num_channels - self.window_size
        self._render()

    def _on_key_press(self, event) -> None:
        if event.key in {"left", "a"}:
            self._previous_time(event)
        elif event.key in {"right", "d"}:
            self._next_time(event)
        elif event.key in {"up", "w"}:
            self._previous_channels(event)
        elif event.key in {"down", "s"}:
            self._next_channels(event)
        elif event.key == "home":
            self._reset_tail(event)

    def show(self) -> None:
        """Render the first view and open the interactive window."""

        self._render()
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the interactive viewer."""

    parser = argparse.ArgumentParser(description="Interactively inspect raw wildfire input tensors.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Override the dataset directory from the config file. "
            "Use this when the tensors live somewhere else."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="Initial timestamp index to display after chronological sorting.",
    )
    parser.add_argument(
        "--channel-start",
        type=int,
        default=-6,
        help="Initial channel index for the visible window. Negative values are counted from the end.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Number of channels to show at once.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap for the channel maps.",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=2.0,
        help="Lower percentile used to scale each channel image.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=98.0,
        help="Upper percentile used to scale each channel image.",
    )
    return parser


def main() -> None:
    """Load the configured dataset and launch the interactive viewer."""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    if "data_dir" not in config:
        raise KeyError("Config is missing required key 'data_dir'.")
    if "file_pattern" not in config:
        raise KeyError("Config is missing required key 'file_pattern'.")

    configured_data_dir = str(config["data_dir"])
    data_dir = resolve_path(config_path, args.data_dir or configured_data_dir)
    file_pattern = str(config["file_pattern"])

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    file_paths = discover_files(data_dir, file_pattern)
    viewer = InputDatasetViewer(
        file_paths=file_paths,
        initial_file_index=args.file_index,
        window_size=args.window_size,
        channel_start=args.channel_start,
        cmap=args.cmap,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    viewer.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"error: {exc}", file=sys.stderr)
        raise