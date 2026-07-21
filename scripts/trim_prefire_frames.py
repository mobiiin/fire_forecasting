"""Trim initial inactive/pre-ignition frames from CAWFE fire indexes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
try:
	from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for the utility.
	tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discovery import sort_chronologically
from src.data.fire_index import discover_fire_datasets, load_fire_dataset_index, save_fire_dataset_index


FLUX_CHANNELS = (80, 81, 82, 83)
SURFACE_FUEL_CHANNEL = 84
CANOPY_FUEL_CHANNEL = 85
REQUIRED_RAW_CHANNELS = 86
SUMMARY_FIELDS = [
	"fire_name",
	"original_num_frames",
	"first_active_idx",
	"trim_start_idx",
	"removed_num_frames",
	"trimmed_num_frames",
	"first_active_reason",
	"max_flux_at_first_active",
	"active_flux_pixels_at_first_active",
	"active_consumed_pixels_at_first_active",
	"warning",
]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Create a trimmed fire dataset index by removing inactive pre-ignition frames.")
	parser.add_argument("--input_index", default="fire_dataset_index.json", help="Existing fire_dataset_index.json to trim.")
	parser.add_argument("--main_data_dir", default=None, help="Main CAWFE data directory to discover when --input_index is not provided.")
	parser.add_argument("--output_index", default="fire_dataset_index_trimmed.json", help="Output trimmed index JSON.")
	parser.add_argument("--prefire_context_frames", type=int, default=0, help="Frames to keep before the first active frame.")
	parser.add_argument("--flux_threshold", type=float, default=1.0, help="Activity threshold for summed absolute heat flux channels.")
	parser.add_argument("--consumed_threshold", type=float, default=0.001, help="Activity threshold for one-step consumed fuel.")
	parser.add_argument("--min_active_pixels", type=int, default=5, help="Minimum active pixels needed to declare a frame active.")
	parser.add_argument("--mode", choices=("index_only", "symlink", "copy"), default="index_only")
	parser.add_argument("--output_data_dir", default=None, help="Required for symlink/copy mode.")
	parser.add_argument("--dry_run", action="store_true", help="Scan and write diagnostics without writing output index/data.")
	parser.add_argument("--plot_diagnostics", action="store_true", help="Save per-fire activity curve plots.")
	parser.add_argument("--diagnostics_dir", default="artifacts/prefire_trim_diagnostics")
	parser.add_argument("--overwrite", action="store_true", help="Allow replacing output index or files inside output_data_dir.")
	parser.add_argument("--file_pattern", default="*.npy", help="File pattern used with --main_data_dir discovery.")
	parser.add_argument("--fire_dir_glob", default="*", help="Top-level fire glob used with --main_data_dir discovery.")
	parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
	parser.add_argument("--no-recursive", dest="recursive", action="store_false")
	parser.add_argument("--input_sequence_length", type=int, default=5, help="Validation minimum input sequence length.")
	parser.add_argument("--prediction_horizon", type=int, default=10, help="Validation minimum prediction horizon.")
	parser.add_argument(
		"--progress",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Show a progress bar while scanning fires. Use --no-progress to disable it.",
	)
	return parser


def _to_jsonable(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(key): _to_jsonable(nested) for key, nested in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return _to_jsonable(value.tolist())
	if isinstance(value, np.generic):
		return _to_jsonable(value.item())
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


def save_json(path: str | Path, payload: Any) -> None:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
		handle.write("\n")


def _csv_cell(value: Any) -> Any:
	if isinstance(value, float) and not math.isfinite(value):
		return ""
	if isinstance(value, (dict, list, tuple)):
		return json.dumps(_to_jsonable(value), sort_keys=True)
	return value


def write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
		writer.writeheader()
		for row in rows:
			writer.writerow({field: _csv_cell(row.get(field, "")) for field in SUMMARY_FIELDS})


def frame_paths_for_record(record: Mapping[str, Any]) -> list[Path]:
	"""Return frame paths for a fire-index record, preserving explicit order when present."""

	for key in ("trimmed_frame_paths", "frame_paths", "file_paths"):
		value = record.get(key)
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			paths = [Path(str(item)).expanduser().resolve() for item in value if str(item).strip()]
			if paths:
				return paths
	data_dir = Path(str(record["data_dir"])).expanduser().resolve()
	file_pattern = str(record.get("file_pattern", "*.npy"))
	paths = sort_chronologically([path for path in data_dir.glob(file_pattern) if path.is_file()])
	if not paths:
		raise FileNotFoundError(f"No frame files found in {data_dir} using pattern {file_pattern!r}.")
	return [path.resolve() for path in paths]


def _load_frame(path: Path) -> np.ndarray:
	frame = np.load(path, mmap_mode="r", allow_pickle=False)
	if frame.ndim != 3 or int(frame.shape[-1]) < REQUIRED_RAW_CHANNELS:
		raise ValueError(f"Expected frame shape (H,W,{REQUIRED_RAW_CHANNELS}+), got {frame.shape} in {path}.")
	return frame


def _activity_reason(flux_pixels: int, consumed_pixels: int, min_active_pixels: int) -> str | None:
	flux_active = int(flux_pixels) >= int(min_active_pixels)
	consumed_active = int(consumed_pixels) >= int(min_active_pixels)
	if flux_active and consumed_active:
		return "flux+consumed"
	if flux_active:
		return "flux"
	if consumed_active:
		return "consumed"
	return None


def detect_prefire_trim(
	frame_paths: Sequence[Path],
	prefire_context_frames: int = 6,
	flux_threshold: float = 1.0,
	consumed_threshold: float = 0.001,
	min_active_pixels: int = 5,
) -> dict[str, Any]:
	"""Scan frames one at a time and find the first active fire frame."""

	if int(prefire_context_frames) < 0:
		raise ValueError("prefire_context_frames must be nonnegative.")
	if int(min_active_pixels) <= 0:
		raise ValueError("min_active_pixels must be positive.")
	if not frame_paths:
		raise ValueError("Cannot trim a fire with no frame paths.")

	flux_counts: list[int] = []
	consumed_counts: list[int] = []
	max_flux_values: list[float] = []
	first_active_idx: int | None = None
	first_active_reason: str | None = None
	active_flux_pixels_at_first_active = 0
	active_consumed_pixels_at_first_active = 0
	max_flux_at_first_active = 0.0
	previous_surface: np.ndarray | None = None
	previous_canopy: np.ndarray | None = None

	for index, path in enumerate(frame_paths):
		frame = _load_frame(Path(path))
		flux_total = np.zeros(frame.shape[:2], dtype=np.float32)
		for channel in FLUX_CHANNELS:
			flux_total += np.abs(np.asarray(frame[..., channel], dtype=np.float32))
		flux_pixels = int(np.count_nonzero(flux_total > float(flux_threshold)))
		max_flux = float(np.nanmax(flux_total)) if flux_total.size else 0.0

		current_surface = np.asarray(frame[..., SURFACE_FUEL_CHANNEL], dtype=np.float32)
		current_canopy = np.asarray(frame[..., CANOPY_FUEL_CHANNEL], dtype=np.float32)
		consumed_pixels = 0
		if previous_surface is not None and previous_canopy is not None:
			surface_consumed = np.maximum(previous_surface - current_surface, 0.0)
			canopy_consumed = np.maximum(previous_canopy - current_canopy, 0.0)
			consumed_pixels = int(
				np.count_nonzero(
					(surface_consumed > float(consumed_threshold))
					| (canopy_consumed > float(consumed_threshold))
				)
			)

		flux_counts.append(flux_pixels)
		consumed_counts.append(consumed_pixels)
		max_flux_values.append(max_flux)
		reason = _activity_reason(flux_pixels, consumed_pixels, int(min_active_pixels))
		if first_active_idx is None and reason is not None:
			first_active_idx = int(index)
			first_active_reason = reason
			active_flux_pixels_at_first_active = int(flux_pixels)
			active_consumed_pixels_at_first_active = int(consumed_pixels)
			max_flux_at_first_active = float(max_flux)

		previous_surface = np.array(current_surface, dtype=np.float32, copy=True)
		previous_canopy = np.array(current_canopy, dtype=np.float32, copy=True)

	warning = ""
	if first_active_idx is None:
		trim_start_idx = 0
		warning = "no active frame detected"
	else:
		trim_start_idx = max(int(first_active_idx) - int(prefire_context_frames), 0)

	original_num_frames = int(len(frame_paths))
	trimmed_num_frames = int(original_num_frames - trim_start_idx)
	return {
		"first_active_idx": first_active_idx,
		"trim_start_idx": int(trim_start_idx),
		"original_num_frames": original_num_frames,
		"trimmed_num_frames": trimmed_num_frames,
		"removed_num_frames": int(trim_start_idx),
		"first_active_reason": first_active_reason,
		"max_flux_at_first_active": float(max_flux_at_first_active),
		"active_flux_pixels_at_first_active": int(active_flux_pixels_at_first_active),
		"active_consumed_pixels_at_first_active": int(active_consumed_pixels_at_first_active),
		"warning": warning,
		"flux_active_pixel_counts": flux_counts,
		"consumed_active_pixel_counts": consumed_counts,
		"max_flux_values": max_flux_values,
	}


def _safe_fire_dir_name(fire_name: str) -> str:
	return str(fire_name).replace("/", "__").replace("\\", "__")


def _link_or_copy_file(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
	if destination.exists() or destination.is_symlink():
		if not overwrite:
			raise FileExistsError(f"Output file already exists: {destination}")
		destination.unlink()
	destination.parent.mkdir(parents=True, exist_ok=True)
	if mode == "symlink":
		os.symlink(source, destination)
	elif mode == "copy":
		shutil.copy2(source, destination)
	else:
		raise ValueError(f"Unsupported file output mode: {mode!r}")


def _materialize_trimmed_fire(
	fire_name: str,
	record: Mapping[str, Any],
	selected_paths: Sequence[Path],
	output_data_dir: Path,
	mode: str,
	overwrite: bool,
	dry_run: bool,
) -> list[Path]:
	fire_output_dir = output_data_dir / _safe_fire_dir_name(fire_name)
	output_paths = [fire_output_dir / path.name for path in selected_paths]
	if dry_run:
		return output_paths
	for source, destination in zip(selected_paths, output_paths):
		_link_or_copy_file(Path(source), destination, mode=mode, overwrite=overwrite)
	for metadata_key in ("geom_path", "terrain_path"):
		metadata_path = record.get(metadata_key)
		if metadata_path in (None, "", "null"):
			continue
		source = Path(str(metadata_path)).expanduser().resolve()
		if source.exists():
			_link_or_copy_file(source, fire_output_dir / source.name, mode=mode, overwrite=overwrite)
	return output_paths


def trim_fire_record(
	fire_name: str,
	record: Mapping[str, Any],
	*,
	prefire_context_frames: int = 6,
	flux_threshold: float = 1.0,
	consumed_threshold: float = 0.001,
	min_active_pixels: int = 5,
	mode: str = "index_only",
	output_data_dir: Path | None = None,
	overwrite: bool = False,
	dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
	"""Trim one fire record and return updated record, summary row, and diagnostics."""

	original_paths = frame_paths_for_record(record)
	detection = detect_prefire_trim(
		original_paths,
		prefire_context_frames=prefire_context_frames,
		flux_threshold=flux_threshold,
		consumed_threshold=consumed_threshold,
		min_active_pixels=min_active_pixels,
	)
	trim_start_idx = int(detection["trim_start_idx"])
	selected_original_paths = original_paths[trim_start_idx:]
	if mode in {"symlink", "copy"}:
		if output_data_dir is None:
			raise ValueError("--output_data_dir is required for symlink/copy mode.")
		selected_index_paths = _materialize_trimmed_fire(
			fire_name,
			record,
			selected_original_paths,
			output_data_dir=output_data_dir,
			mode=mode,
			overwrite=overwrite,
			dry_run=dry_run,
		)
	else:
		selected_index_paths = selected_original_paths

	updated_record = dict(record)
	updated_record["original_num_npy_files"] = int(record.get("num_npy_files", len(original_paths)))
	updated_record["num_npy_files"] = int(len(selected_index_paths))
	if mode in {"symlink", "copy"} and output_data_dir is not None:
		fire_output_dir = output_data_dir / _safe_fire_dir_name(fire_name)
		updated_record["data_dir"] = str(fire_output_dir.resolve())
		updated_record["file_pattern"] = "*.npy"
		for metadata_key in ("geom_path", "terrain_path"):
			metadata_path = record.get(metadata_key)
			if metadata_path in (None, "", "null"):
				continue
			source = Path(str(metadata_path)).expanduser().resolve()
			if source.exists():
				updated_record[metadata_key] = str((fire_output_dir / source.name).resolve())
	updated_record["original_frame_paths"] = [str(path) for path in original_paths]
	updated_record["trimmed_frame_paths"] = [str(path) for path in selected_index_paths]
	updated_record["prefire_trim"] = {
		"enabled": True,
		"original_num_frames": int(detection["original_num_frames"]),
		"trimmed_num_frames": int(detection["trimmed_num_frames"]),
		"removed_num_frames": int(detection["removed_num_frames"]),
		"first_active_idx": detection["first_active_idx"],
		"trim_start_idx": int(detection["trim_start_idx"]),
		"prefire_context_frames": int(prefire_context_frames),
		"flux_threshold": float(flux_threshold),
		"consumed_threshold": float(consumed_threshold),
		"min_active_pixels": int(min_active_pixels),
	}
	if detection["warning"]:
		updated_record["prefire_trim"]["warning"] = detection["warning"]

	summary_row = {
		"fire_name": str(record.get("fire_name", fire_name)),
		"original_num_frames": int(detection["original_num_frames"]),
		"first_active_idx": detection["first_active_idx"],
		"trim_start_idx": int(detection["trim_start_idx"]),
		"removed_num_frames": int(detection["removed_num_frames"]),
		"trimmed_num_frames": int(detection["trimmed_num_frames"]),
		"first_active_reason": detection["first_active_reason"],
		"max_flux_at_first_active": float(detection["max_flux_at_first_active"]),
		"active_flux_pixels_at_first_active": int(detection["active_flux_pixels_at_first_active"]),
		"active_consumed_pixels_at_first_active": int(detection["active_consumed_pixels_at_first_active"]),
		"warning": detection["warning"],
	}
	diagnostics = {
		"fire_name": str(record.get("fire_name", fire_name)),
		"frame_paths": [str(path) for path in original_paths],
		**detection,
	}
	return updated_record, summary_row, diagnostics


def _load_source_index(args: argparse.Namespace) -> dict[str, Any]:
	if args.input_index:
		return load_fire_dataset_index(Path(args.input_index))
	if args.main_data_dir:
		return discover_fire_datasets(
			main_data_dir=Path(args.main_data_dir),
			fire_dir_glob=str(args.fire_dir_glob),
			file_pattern=str(args.file_pattern),
			recursive=bool(args.recursive),
			require_npy_files=True,
			require_geom=True,
			require_terrain=False,
		)
	raise ValueError("Provide either --input_index or --main_data_dir.")


def _validate_trimmed_fire(
	fire_name: str,
	original_paths: Sequence[Path],
	selected_paths: Sequence[Path],
	detection: Mapping[str, Any],
	min_required_frames: int,
) -> list[str]:
	errors: list[str] = []
	trim_start_idx = int(detection["trim_start_idx"])
	first_active_idx = detection.get("first_active_idx")
	if int(len(selected_paths)) < int(min_required_frames):
		errors.append(
			f"{fire_name}: trimmed_num_frames={len(selected_paths)} is smaller than "
			f"input_sequence_length+prediction_horizon={min_required_frames}."
		)
	if first_active_idx is not None and trim_start_idx > int(first_active_idx):
		errors.append(f"{fire_name}: trim_start_idx={trim_start_idx} is after first_active_idx={first_active_idx}.")
	expected = [Path(path).resolve() for path in original_paths[trim_start_idx:]]
	selected_original = [Path(path).resolve() for path in selected_paths]
	if selected_original != expected:
		errors.append(f"{fire_name}: selected frame order does not match original order from trim_start_idx onward.")
	missing = [path for path in selected_paths if not Path(path).exists()]
	if missing:
		errors.append(f"{fire_name}: selected path is missing: {missing[0]}")
	return errors


def validate_trimmed_records(
	source_index: Mapping[str, Any],
	trimmed_index: Mapping[str, Any],
	diagnostics_by_fire: Mapping[str, Mapping[str, Any]],
	min_required_frames: int,
	mode: str,
	dry_run: bool = False,
) -> list[str]:
	errors: list[str] = []
	source_fires = source_index.get("fires", {})
	trimmed_fires = trimmed_index.get("fires", {})
	if not isinstance(source_fires, Mapping) or not isinstance(trimmed_fires, Mapping):
		return ["source or trimmed index is missing a valid 'fires' mapping."]
	for fire_name, record in trimmed_fires.items():
		if not isinstance(record, Mapping):
			continue
		source_record = source_fires.get(fire_name, record) if isinstance(source_fires, Mapping) else record
		if not isinstance(source_record, Mapping):
			continue
		original_paths = frame_paths_for_record(source_record)
		if mode in {"symlink", "copy"}:
			selected_paths = [Path(str(path)).expanduser().resolve() for path in record.get("trimmed_frame_paths", [])]
			if not dry_run:
				missing_outputs = [path for path in selected_paths if not path.exists()]
				if missing_outputs:
					errors.append(f"{fire_name}: materialized output path is missing: {missing_outputs[0]}")
			selected_for_order = original_paths[int(record["prefire_trim"]["trim_start_idx"]):]
		else:
			selected_paths = [Path(str(path)).expanduser().resolve() for path in record.get("trimmed_frame_paths", [])]
			selected_for_order = selected_paths
		errors.extend(
			_validate_trimmed_fire(
				str(fire_name),
				original_paths=original_paths,
				selected_paths=selected_for_order,
				detection=diagnostics_by_fire[str(fire_name)],
				min_required_frames=min_required_frames,
			)
		)
	return errors


def _save_activity_plot(path: Path, diagnostics: Mapping[str, Any]) -> None:
	os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
	Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	flux_counts = np.asarray(diagnostics["flux_active_pixel_counts"], dtype=np.float64)
	consumed_counts = np.asarray(diagnostics["consumed_active_pixel_counts"], dtype=np.float64)
	x = np.arange(len(flux_counts))
	fig, ax = plt.subplots(figsize=(9, 4))
	ax.plot(x, flux_counts, label="active flux pixels")
	ax.plot(x, consumed_counts, label="active consumed pixels")
	first_active_idx = diagnostics.get("first_active_idx")
	if first_active_idx is not None:
		ax.axvline(int(first_active_idx), color="tab:red", linestyle="--", label="first active")
	ax.axvline(int(diagnostics["trim_start_idx"]), color="tab:green", linestyle=":", label="trim start")
	ax.set_title(str(diagnostics.get("fire_name", "fire")))
	ax.set_xlabel("frame index")
	ax.set_ylabel("active pixel count")
	ax.legend()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def trim_index(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
	source_index = _load_source_index(args)
	fires = source_index.get("fires", {})
	if not isinstance(fires, Mapping) or not fires:
		raise ValueError("Input fire dataset index is missing a nonempty 'fires' mapping.")
	if args.mode in {"symlink", "copy"} and not args.output_data_dir:
		raise ValueError("--output_data_dir is required for symlink/copy mode.")
	if args.mode == "copy":
		print("WARNING: copy mode may require a large amount of disk space.")

	output_data_dir = Path(args.output_data_dir).expanduser().resolve() if args.output_data_dir else None
	trimmed_fires: dict[str, Any] = {}
	summary_rows: list[dict[str, Any]] = []
	diagnostics_by_fire: dict[str, Any] = {}

	fire_items = sorted(fires.items())
	progress_enabled = bool(getattr(args, "progress", False))
	progress_bar = None
	fire_iterator = fire_items
	if progress_enabled and tqdm is not None:
		progress_bar = tqdm(
			fire_items,
			total=len(fire_items),
			desc="Trimming fires",
			unit="fire",
			dynamic_ncols=True,
		)
		fire_iterator = progress_bar

	try:
		for fire_name, record in fire_iterator:
			if progress_bar is not None:
				progress_bar.set_postfix_str(str(fire_name)[:48], refresh=False)
			if not isinstance(record, Mapping):
				continue
			updated_record, summary_row, diagnostics = trim_fire_record(
				str(fire_name),
				record,
				prefire_context_frames=int(args.prefire_context_frames),
				flux_threshold=float(args.flux_threshold),
				consumed_threshold=float(args.consumed_threshold),
				min_active_pixels=int(args.min_active_pixels),
				mode=str(args.mode),
				output_data_dir=output_data_dir,
				overwrite=bool(args.overwrite),
				dry_run=bool(args.dry_run),
			)
			trimmed_fires[str(fire_name)] = updated_record
			summary_rows.append(summary_row)
			diagnostics_by_fire[str(fire_name)] = diagnostics
			if summary_row["warning"]:
				message = f"WARNING: {fire_name}: {summary_row['warning']}"
				if progress_bar is not None:
					progress_bar.write(message)
				else:
					print(message)
	finally:
		if progress_bar is not None:
			progress_bar.close()

	trimmed_index = dict(source_index)
	trimmed_index["fires"] = trimmed_fires
	trimmed_index["num_fires"] = int(len(trimmed_fires))
	trimmed_index["last_prefire_trim_updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	trimmed_index["prefire_trim"] = {
		"enabled": True,
		"mode": str(args.mode),
		"prefire_context_frames": int(args.prefire_context_frames),
		"flux_threshold": float(args.flux_threshold),
		"consumed_threshold": float(args.consumed_threshold),
		"min_active_pixels": int(args.min_active_pixels),
	}
	return trimmed_index, summary_rows, diagnostics_by_fire


def _print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
	total_before = sum(int(row["original_num_frames"]) for row in rows)
	total_after = sum(int(row["trimmed_num_frames"]) for row in rows)
	removed = total_before - total_after
	percent_removed = 100.0 * removed / total_before if total_before else 0.0
	no_activity = sum(1 for row in rows if row.get("warning") == "no active frame detected")
	print("")
	print(f"fires processed: {len(rows)}")
	print(f"total frames before: {total_before}")
	print(f"total frames after: {total_after}")
	print(f"removed frames: {removed}")
	print(f"percent removed: {percent_removed:.2f}")
	print(f"fires with no detected activity: {no_activity}")


def main() -> None:
	args = build_parser().parse_args()
	output_index = Path(args.output_index).expanduser().resolve()
	diagnostics_dir = Path(args.diagnostics_dir).expanduser().resolve()
	if output_index.exists() and not bool(args.overwrite) and not bool(args.dry_run):
		raise FileExistsError(f"Output index already exists: {output_index}. Pass --overwrite to replace it.")
	if int(args.input_sequence_length) <= 0:
		raise ValueError("--input_sequence_length must be positive.")
	if int(args.prediction_horizon) < 0:
		raise ValueError("--prediction_horizon must be nonnegative.")

	trimmed_index, summary_rows, diagnostics_by_fire = trim_index(args)
	diagnostics_dir.mkdir(parents=True, exist_ok=True)
	write_summary_csv(diagnostics_dir / "prefire_trim_summary.csv", summary_rows)
	save_json(diagnostics_dir / "prefire_trim_summary.json", {"summary": summary_rows, "fires": diagnostics_by_fire})
	if bool(args.plot_diagnostics):
		for fire_name, diagnostics in diagnostics_by_fire.items():
			_save_activity_plot(diagnostics_dir / f"{_safe_fire_dir_name(fire_name)}_activity_curve.png", diagnostics)

	min_required_frames = int(args.input_sequence_length) + int(args.prediction_horizon)
	source_index = _load_source_index(args)
	validation_errors = validate_trimmed_records(
		source_index,
		trimmed_index,
		diagnostics_by_fire,
		min_required_frames=min_required_frames,
		mode=str(args.mode),
		dry_run=bool(args.dry_run),
	)
	if validation_errors:
		for error in validation_errors:
			print(f"VALIDATION ERROR: {error}")
		raise RuntimeError(f"Prefire trim validation failed with {len(validation_errors)} error(s).")

	if bool(args.dry_run):
		print(f"Dry run: not writing output index {output_index}")
	else:
		save_fire_dataset_index(trimmed_index, output_index)
		print(f"Saved trimmed fire dataset index: {output_index}")
	print(f"Saved diagnostics: {diagnostics_dir}")
	_print_summary(summary_rows)


if __name__ == "__main__":
	main()
