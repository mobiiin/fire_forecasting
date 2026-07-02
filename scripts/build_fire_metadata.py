"""Build a cached fire metadata dictionary from a CAWFE dataset root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "fire_metadata.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Scan a CAWFE dataset directory, extract per-fire cell sizes from flux files, "
            "and save the result as a JSON dictionary."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Path to the main dataset directory. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to the output JSON dictionary. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def find_fire_directories(dataset_root: Path) -> list[Path]:
    """Return top-level fire directories under the dataset root."""

    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def find_simulation_flux_files(fire_dir: Path) -> dict[Path, Path]:
    """Find one representative flux file for each simulation directory."""

    simulation_to_flux: dict[Path, Path] = {}
    for current_dir, _, filenames in os.walk(fire_dir):
        flux_names = sorted(name for name in filenames if ".flux." in name)
        if not flux_names:
            continue
        current_path = Path(current_dir)
        simulation_to_flux[current_path] = current_path / flux_names[0]
    return dict(sorted(simulation_to_flux.items()))


def extract_cell_size(flux_file: Path) -> tuple[float, float]:
    """Read dx and dy from the first line of a flux file."""

    with flux_file.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Flux file is empty: {flux_file}")

    parts = first_line.split()
    if len(parts) < 5:
        raise ValueError(
            f"Expected at least 5 whitespace-separated values in the first line of {flux_file}, "
            f"got: {first_line!r}"
        )

    try:
        dx = float(parts[-2])
        dy = float(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse dx/dy from first line of {flux_file}: {first_line!r}") from exc

    return dx, dy


def build_fire_record(fire_dir: Path) -> dict[str, Any] | None:
    """Build metadata for one fire directory."""

    simulation_to_flux = find_simulation_flux_files(fire_dir)
    if not simulation_to_flux:
        return None

    simulations: list[dict[str, Any]] = []
    unique_cell_sizes: set[tuple[float, float]] = set()

    for simulation_dir in sorted(simulation_to_flux):
        flux_file = simulation_to_flux[simulation_dir]
        dx, dy = extract_cell_size(flux_file)
        unique_cell_sizes.add((dx, dy))
        simulations.append(
            {
                "simulation_dir": str(simulation_dir.resolve()),
                "simulation_dir_relative_to_fire": str(simulation_dir.relative_to(fire_dir)),
                "sample_flux_file": str(flux_file.resolve()),
                "dx": dx,
                "dy": dy,
            }
        )

    record: dict[str, Any] = {
        "fire_dir": str(fire_dir.resolve()),
        "simulation_count": len(simulations),
        "simulations": simulations,
    }

    if len(unique_cell_sizes) == 1:
        dx, dy = next(iter(unique_cell_sizes))
        record["dx"] = dx
        record["dy"] = dy
    else:
        record["cell_size_consistent"] = False
        record["unique_cell_sizes"] = [
            {"dx": dx, "dy": dy} for dx, dy in sorted(unique_cell_sizes)
        ]

    return record


def build_fire_metadata(dataset_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Build the full fire metadata dictionary and collect warnings."""

    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    for fire_dir in find_fire_directories(dataset_root):
        record = build_fire_record(fire_dir)
        if record is None:
            warnings.append(f"Skipped {fire_dir.name}: no flux files found.")
            continue
        metadata[fire_dir.name] = record
        if record.get("cell_size_consistent") is False:
            warnings.append(
                f"{fire_dir.name} has multiple cell sizes across simulations; see unique_cell_sizes in the output."
            )

    return metadata, warnings


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write the metadata dictionary to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Run the metadata builder CLI."""

    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    metadata, warnings = build_fire_metadata(dataset_root)
    save_metadata(metadata, output_path)

    print(f"Saved metadata for {len(metadata)} fires to {output_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
