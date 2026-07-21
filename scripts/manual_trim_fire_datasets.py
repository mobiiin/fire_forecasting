"""Interactively choose compact temporal trim metadata for fire datasets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib


def _configure_matplotlib_backend() -> None:
	if os.environ.get("MPLBACKEND"):
		return
	if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
		matplotlib.use("Agg", force=True)
		return
	for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
		try:
			matplotlib.use(backend, force=True)
			return
		except Exception:
			continue
	matplotlib.use("Agg", force=True)


_configure_matplotlib_backend()
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np


FLUX_CHANNELS = (80, 81, 82, 83)
SURFACE_FUEL_CHANNEL = 84
CANOPY_FUEL_CHANNEL = 85
SUMMARY_FIELDS = [
	"fire_name",
	"original_num_frames",
	"trim_start_index",
	"trim_end_index",
	"trimmed_num_frames",
	"removed_start_frames",
	"removed_end_frames",
	"has_manual_choice",
	"output_index",
]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Manually trim fire dataset starts into compact temporal_trim metadata.")
	parser.add_argument("--input_index", default=str(PROJECT_ROOT / "fire_dataset_index.json"), help="Input fire dataset index JSON.")
	parser.add_argument("--output_index", default=str(PROJECT_ROOT / "fire_dataset_index_trimmed.json"), help="Output trimmed index JSON.")
	parser.add_argument("--trim_config", default=str(PROJECT_ROOT / "configs" / "manual_fire_trim.json"), help="Intermediate manual choices JSON.")
	parser.add_argument("--start_fire", default=None, help="Optional fire name to begin reviewing from.")
	parser.add_argument("--only_fire", default=None, help="Optional single fire name to trim.")
	parser.add_argument("--skip_existing_choices", action="store_true", help="Skip fires that already have a trim choice.")
	parser.add_argument("--overwrite_existing_choices", action="store_true", help="Allow replacing existing trim choices.")
	parser.add_argument("--no_overwrite", action="store_true", help="Do not replace output_index if it already exists.")
	parser.add_argument("--default_start_index", type=int, default=0, help="Default trim_start_index for fires without choices.")
	parser.add_argument("--default_end_index", type=int, default=None, help="Default trim_end_index; null means last frame.")
	parser.add_argument("--jump", type=int, default=10, help="Frames to jump with f/b commands.")
	parser.add_argument("--save_every_choice", action=argparse.BooleanOptionalAction, default=True, help="Save choices immediately.")
	parser.add_argument("--plot_diagnostics", action="store_true", help="Save per-fire activity diagnostics when practical.")
	parser.add_argument("--diagnostics_dir", default=str(PROJECT_ROOT / "artifacts" / "prefire_trim_diagnostics"), help="Diagnostics output directory.")
	parser.add_argument("--mode", choices=("terminal", "matplotlib"), default="matplotlib", help="Interaction mode. Matplotlib opens a button-based frame browser.")
	parser.add_argument("--preview_window", action=argparse.BooleanOptionalAction, default=True, help="Open a matplotlib preview window when a GUI backend is available.")
	parser.add_argument(
		"--system_viewer",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="When Matplotlib is non-interactive, open the saved current_preview.png with the desktop image viewer.",
	)
	parser.add_argument("--apply_only", action="store_true", help="Write output_index from trim_config without interactive review.")
	parser.add_argument("--flux_threshold", type=float, default=1.0, help="Flux threshold used only for visual stats.")
	parser.add_argument("--consumed_threshold", type=float, default=0.001, help="Fuel-delta threshold used only for visual stats.")
	parser.add_argument("--min_active_pixels", type=int, default=5, help="Minimum active pixels shown in visual stats.")
	return parser


def _extract_numeric_suffix(name: str) -> int | None:
	match = re.search(r"(\d+)$", name)
	return int(match.group(1)) if match else None


def sort_chronologically(paths: Sequence[Path]) -> list[Path]:
	numeric = [_extract_numeric_suffix(path.stem) for path in paths]
	if paths and all(value is not None for value in numeric):
		return [path for _, path in sorted(zip(numeric, paths), key=lambda item: item[0])]
	return sorted(paths, key=lambda path: path.name)


def _load_json(path: str | Path) -> Any:
	with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _atomic_write_json(path: str | Path, payload: Any) -> None:
	output_path = Path(path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	temp_path = output_path.with_name(f".{output_path.name}.tmp")
	with temp_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")
	os.replace(temp_path, output_path)


def _fire_name(record: Mapping[str, Any], fallback: str | int | None = None) -> str:
	for key in ("name", "fire_name", "dataset_name"):
		value = record.get(key)
		if value not in (None, "", "null"):
			return str(value)
	for key in ("path", "data_dir", "fire_root_dir"):
		value = record.get(key)
		if value not in (None, "", "null"):
			return Path(str(value)).expanduser().name
	if fallback is not None:
		return str(fallback)
	raise ValueError(f"Cannot infer fire name from record: {record!r}")


def load_fire_index(path: str | Path) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
	"""Load common fire-index schemas into a list of entries plus root schema metadata."""

	index_path = Path(path).expanduser().resolve()
	root = _load_json(index_path)
	if isinstance(root, list):
		entries = [
			{"name": _fire_name(record, offset), "record": dict(record), "key": offset}
			for offset, record in enumerate(root)
			if isinstance(record, Mapping)
		]
		return entries, root, {"kind": "root_list", "index_path": str(index_path)}
	if not isinstance(root, Mapping):
		raise ValueError(f"Expected list or object in fire index: {index_path}")
	for container_key in ("fires", "datasets"):
		container = root.get(container_key)
		if isinstance(container, Mapping):
			entries = [
				{"name": _fire_name(record, key), "record": dict(record), "key": str(key)}
				for key, record in container.items()
				if isinstance(record, Mapping)
			]
			return entries, dict(root), {"kind": "dict_mapping", "container_key": container_key, "index_path": str(index_path)}
		if isinstance(container, list):
			entries = [
				{"name": _fire_name(record, offset), "record": dict(record), "key": offset}
				for offset, record in enumerate(container)
				if isinstance(record, Mapping)
			]
			return entries, dict(root), {"kind": "dict_list", "container_key": container_key, "index_path": str(index_path)}
	raise ValueError(f"Could not find a supported fire list in {index_path}; expected a list, 'fires', or 'datasets'.")


def _path_from_record(record: Mapping[str, Any]) -> Path:
	for key in ("path", "data_dir", "directory", "dataset_path"):
		value = record.get(key)
		if value not in (None, "", "null"):
			return Path(str(value)).expanduser().resolve()
	raise ValueError(f"Fire record has no path/data_dir field: {_fire_name(record, None)!r}")


def frame_paths_for_record(record: Mapping[str, Any]) -> list[Path]:
	for key in ("frame_paths", "file_paths", "trimmed_frame_paths"):
		value = record.get(key)
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			paths = [Path(str(item)).expanduser().resolve() for item in value if str(item).strip()]
			if paths:
				return paths
	data_dir = _path_from_record(record)
	pattern = str(record.get("file_pattern", record.get("frame_pattern", "*.npy")))
	paths = sort_chronologically([path for path in data_dir.glob(pattern) if path.is_file()])
	if not paths:
		paths = sort_chronologically([path for path in data_dir.rglob(pattern) if path.is_file()])
	if not paths:
		raise FileNotFoundError(f"No frame files found for {_fire_name(record, None)!r} under {data_dir} using {pattern!r}.")
	return paths


def infer_original_num_frames(record: Mapping[str, Any], frame_paths: Sequence[Path] | None = None) -> int:
	trim = record.get("temporal_trim")
	if isinstance(trim, Mapping) and trim.get("original_num_frames") not in (None, "", "null"):
		return int(trim["original_num_frames"])
	for key in ("original_num_frames", "original_num_npy_files", "num_npy_files", "num_files", "num_frames"):
		value = record.get(key)
		if value not in (None, "", "null"):
			return int(value)
	if frame_paths is not None:
		return len(frame_paths)
	return len(frame_paths_for_record(record))


def load_trim_config(path: str | Path, source_index: str | Path) -> dict[str, Any]:
	trim_path = Path(path).expanduser().resolve()
	if trim_path.exists():
		payload = _load_json(trim_path)
		if not isinstance(payload, dict):
			raise ValueError(f"Trim config must be a JSON object: {trim_path}")
		payload.setdefault("version", "manual_fire_trim_v1")
		payload.setdefault("source_index", str(Path(source_index).expanduser().resolve()))
		payload.setdefault("fires", {})
		return payload
	return {
		"version": "manual_fire_trim_v1",
		"source_index": str(Path(source_index).expanduser().resolve()),
		"updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"fires": {},
	}


def save_trim_config(path: str | Path, trim_config: Mapping[str, Any]) -> None:
	payload = dict(trim_config)
	payload["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	_atomic_write_json(path, payload)


def set_trim_choice(
	trim_config: Mapping[str, Any],
	fire_name: str,
	trim_start_index: int,
	trim_end_index: int | None,
) -> dict[str, Any]:
	payload = dict(trim_config)
	fires = dict(payload.get("fires", {})) if isinstance(payload.get("fires"), Mapping) else {}
	existing = fires.get(str(fire_name), {})
	notes = str(existing.get("notes", "")) if isinstance(existing, Mapping) else ""
	fires[str(fire_name)] = {
		"trim_start_index": int(trim_start_index),
		"trim_end_index": None if trim_end_index is None else int(trim_end_index),
		"notes": notes,
		"selected_with": "manual_trim_fire_datasets.py",
		"updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
	}
	payload["fires"] = fires
	payload["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	return payload


def _load_frame(path: Path) -> np.ndarray:
	frame = np.load(path, mmap_mode="r", allow_pickle=False)
	if frame.ndim != 3 or int(frame.shape[-1]) <= CANOPY_FUEL_CHANNEL:
		raise ValueError(f"Expected frame shape (H,W,C>=86), got {frame.shape} in {path}.")
	return np.asarray(frame, dtype=np.float32)


def _diagnostic_maps(frame: np.ndarray, previous_frame: np.ndarray | None, flux_threshold: float, consumed_threshold: float) -> dict[str, np.ndarray]:
	surface = np.asarray(frame[:, :, SURFACE_FUEL_CHANNEL], dtype=np.float32)
	canopy = np.asarray(frame[:, :, CANOPY_FUEL_CHANNEL], dtype=np.float32)
	total_flux = np.zeros(frame.shape[:2], dtype=np.float32)
	for channel in FLUX_CHANNELS:
		total_flux += np.abs(np.asarray(frame[:, :, channel], dtype=np.float32))
	if previous_frame is None:
		fuel_delta = np.zeros(frame.shape[:2], dtype=np.float32)
	else:
		prev_surface = np.asarray(previous_frame[:, :, SURFACE_FUEL_CHANNEL], dtype=np.float32)
		prev_canopy = np.asarray(previous_frame[:, :, CANOPY_FUEL_CHANNEL], dtype=np.float32)
		fuel_delta = np.maximum(prev_surface - surface, 0.0) + np.maximum(prev_canopy - canopy, 0.0)
	active_mask = (total_flux > float(flux_threshold)) | (fuel_delta > float(consumed_threshold))
	return {
		"surface fuel": surface,
		"canopy fuel": canopy,
		"total flux": total_flux,
		"fuel delta": fuel_delta,
		"active mask": active_mask.astype(np.float32),
	}


def _limits(array: np.ndarray) -> tuple[float, float]:
	finite = np.asarray(array, dtype=np.float32)
	finite = finite[np.isfinite(finite)]
	if finite.size == 0:
		return 0.0, 1.0
	vmin = float(np.percentile(finite, 2.0))
	vmax = float(np.percentile(finite, 98.0))
	if math.isclose(vmin, vmax):
		vmax = vmin + 1.0
	return vmin, vmax


def frame_stats(maps: Mapping[str, np.ndarray], frame: np.ndarray, threshold_flux: float, threshold_consumed: float) -> dict[str, Any]:
	surface = maps["surface fuel"]
	canopy = maps["canopy fuel"]
	total_flux = maps["total flux"]
	fuel_delta = maps["fuel delta"]
	return {
		"height": int(frame.shape[0]),
		"width": int(frame.shape[1]),
		"surface_min": float(np.nanmin(surface)),
		"surface_mean": float(np.nanmean(surface)),
		"surface_max": float(np.nanmax(surface)),
		"canopy_min": float(np.nanmin(canopy)),
		"canopy_mean": float(np.nanmean(canopy)),
		"canopy_max": float(np.nanmax(canopy)),
		"total_flux_max": float(np.nanmax(total_flux)),
		"total_flux_active_pixels": int(np.count_nonzero(total_flux > float(threshold_flux))),
		"fuel_delta_max": float(np.nanmax(fuel_delta)),
		"fuel_delta_active_pixels": int(np.count_nonzero(fuel_delta > float(threshold_consumed))),
	}


def print_frame_stats(fire_name: str, frame_index: int, total_frames: int, stats: Mapping[str, Any]) -> None:
	print("")
	print(f"fire: {fire_name}")
	print(f"frame: {frame_index}/{total_frames - 1} | HxW={stats['height']}x{stats['width']} | total_frames={total_frames}")
	print(f"surface fuel min/mean/max: {stats['surface_min']:.6g} / {stats['surface_mean']:.6g} / {stats['surface_max']:.6g}")
	print(f"canopy fuel  min/mean/max: {stats['canopy_min']:.6g} / {stats['canopy_mean']:.6g} / {stats['canopy_max']:.6g}")
	print(f"total flux max={stats['total_flux_max']:.6g} active_pixels={stats['total_flux_active_pixels']}")
	print(f"fuel delta max={stats['fuel_delta_max']:.6g} active_pixels={stats['fuel_delta_active_pixels']}")


def _open_with_system_viewer(path: Path) -> str:
	candidates: list[tuple[str, list[str]]] = []
	xdg_open = shutil.which("xdg-open")
	if xdg_open:
		candidates.append(("xdg-open", [xdg_open, str(path)]))
	gio = shutil.which("gio")
	if gio:
		candidates.append(("gio open", [gio, "open", str(path)]))
	if not candidates:
		return "system viewer: no xdg-open/gio command found"
	for label, command in candidates:
		try:
			subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			return f"system viewer: requested open via {label}"
		except OSError as exc:
			last_error = f"{label}: {exc}"
	return f"system viewer: failed to open preview ({last_error})"


def render_preview(
	fire_name: str,
	frame_index: int,
	total_frames: int,
	selected_start: int | None,
	selected_end: int | None,
	maps: Mapping[str, np.ndarray],
	output_path: Path,
	show_window: bool = True,
	open_system_viewer: bool = True,
) -> tuple[Any, str | None]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	current_preview_path = output_path.parent / "current_preview.png"
	fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
	for axis, (title, array) in zip(axes.ravel(), maps.items()):
		vmin, vmax = _limits(array)
		image = axis.imshow(array, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
		axis.set_title(title)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	for axis in axes.ravel()[len(maps):]:
		axis.axis("off")
	fig.suptitle(
		f"{fire_name} | frame {frame_index}/{total_frames - 1} | "
		f"trim_start={selected_start} trim_end={selected_end}",
		fontsize=10,
	)
	fig.tight_layout()
	fig.savefig(output_path, dpi=130, bbox_inches="tight")
	if current_preview_path != output_path:
		fig.savefig(current_preview_path, dpi=130, bbox_inches="tight")
	noninteractive_backend = _is_noninteractive_backend()
	try:
		if show_window and not noninteractive_backend:
			plt.ion()
			plt.show(block=False)
			plt.pause(0.5)
			return fig, None
	except Exception:
		pass
	plt.close(fig)
	viewer_message = _open_with_system_viewer(current_preview_path) if open_system_viewer else None
	return None, viewer_message


def _matplotlib_display_summary() -> str:
	backend = str(plt.get_backend())
	display = os.environ.get("DISPLAY") or ""
	wayland = os.environ.get("WAYLAND_DISPLAY") or ""
	backend_lower = backend.lower()
	noninteractive = _is_noninteractive_backend()
	if noninteractive:
		return (
			f"Matplotlib backend={backend}; DISPLAY={display or 'unset'}; WAYLAND_DISPLAY={wayland or 'unset'}. "
			"Matplotlib cannot open a GUI window with this backend, so PNG previews will be saved."
		)
	return (
		f"Matplotlib backend={backend}; DISPLAY={display or 'unset'}; WAYLAND_DISPLAY={wayland or 'unset'}. "
		"Preview windows should open and stay visible."
	)


def _is_noninteractive_backend() -> bool:
	backend = str(plt.get_backend()).lower()
	backend_name = backend.rsplit(".", maxsplit=1)[-1].replace("backend_", "")
	return backend_name in {"agg", "pdf", "ps", "svg", "template", "inline"} or "matplotlib_inline" in backend


def print_help() -> None:
	print(
		"commands: n next | p previous | j <idx> absolute jump | f forward jump | b backward jump | "
		"s set start | e set end | save continue | keep whole fire | skip unchanged | q quit | h help"
	)


def _existing_choice(trim_config: Mapping[str, Any], fire_name: str) -> Mapping[str, Any] | None:
	fires = trim_config.get("fires", {})
	if not isinstance(fires, Mapping):
		return None
	choice = fires.get(str(fire_name))
	return choice if isinstance(choice, Mapping) else None


def review_fire_terminal(
	entry: Mapping[str, Any],
	args: argparse.Namespace,
	trim_config: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
	fire_name = str(entry["name"])
	record = entry["record"]
	frame_paths = frame_paths_for_record(record)
	total_frames = len(frame_paths)
	last_index = total_frames - 1
	existing = _existing_choice(trim_config, fire_name)
	if existing and bool(args.skip_existing_choices):
		print(f"{fire_name}: existing choice found; skipping.")
		return dict(trim_config), True
	replace_allowed = bool(args.overwrite_existing_choices) or existing is None
	selected_start = int(existing["trim_start_index"]) if existing and existing.get("trim_start_index") is not None else int(args.default_start_index)
	selected_end = int(existing["trim_end_index"]) if existing and existing.get("trim_end_index") is not None else None
	frame_index = max(0, min(int(selected_start), last_index))
	print(f"\nReviewing {fire_name} ({total_frames} frames)")
	print(_matplotlib_display_summary())
	if existing and not replace_allowed:
		print("Existing choice will be preserved. Pass --overwrite_existing_choices to replace it.")
	print_help()
	current_fig = None

	while True:
		if current_fig is not None:
			plt.close(current_fig)
			current_fig = None
		frame = _load_frame(frame_paths[frame_index])
		previous = _load_frame(frame_paths[frame_index - 1]) if frame_index > 0 else None
		maps = _diagnostic_maps(frame, previous, float(args.flux_threshold), float(args.consumed_threshold))
		stats = frame_stats(maps, frame, float(args.flux_threshold), float(args.consumed_threshold))
		preview_path = Path(args.diagnostics_dir).expanduser().resolve() / f"{_safe_name(fire_name)}_frame_{frame_index:06d}.png"
		current_fig, viewer_message = render_preview(
			fire_name,
			frame_index,
			total_frames,
			selected_start,
			selected_end,
			maps,
			preview_path,
			show_window=bool(args.preview_window),
			open_system_viewer=bool(args.system_viewer),
		)
		print_frame_stats(fire_name, frame_index, total_frames, stats)
		print(f"preview saved: {preview_path}")
		print(f"latest preview: {preview_path.parent / 'current_preview.png'}")
		if current_fig is None:
			print("window: not opened by this backend/session")
			if viewer_message:
				print(viewer_message)
		else:
			print("window: opened; leave it open while entering commands here")
		command = input(f"{fire_name} command [n/p/j/f/b/s/e/save/keep/skip/q/h]: ").strip().lower()
		if command in {"h", "help"}:
			print_help()
		elif command in {"n", "next"}:
			frame_index = min(last_index, frame_index + 1)
		elif command in {"p", "prev", "previous"}:
			frame_index = max(0, frame_index - 1)
		elif command.startswith("j "):
			frame_index = max(0, min(last_index, int(command.split(maxsplit=1)[1])))
		elif command in {"f", "forward"}:
			frame_index = min(last_index, frame_index + max(1, int(args.jump)))
		elif command in {"b", "back"}:
			frame_index = max(0, frame_index - max(1, int(args.jump)))
		elif command == "s":
			if replace_allowed:
				selected_start = frame_index
				if args.save_every_choice:
					trim_config = set_trim_choice(trim_config, fire_name, selected_start, selected_end)
					save_trim_config(args.trim_config, trim_config)
					print(f"saved start={selected_start}")
			else:
				print("Existing choice preserved; pass --overwrite_existing_choices to replace it.")
		elif command == "e":
			if replace_allowed:
				selected_end = frame_index
				if args.save_every_choice:
					trim_config = set_trim_choice(trim_config, fire_name, selected_start, selected_end)
					save_trim_config(args.trim_config, trim_config)
					print(f"saved end={selected_end}")
			else:
				print("Existing choice preserved; pass --overwrite_existing_choices to replace it.")
		elif command == "keep":
			if replace_allowed:
				selected_start = 0
				selected_end = last_index
				trim_config = set_trim_choice(trim_config, fire_name, selected_start, selected_end)
				save_trim_config(args.trim_config, trim_config)
			if current_fig is not None:
				plt.close(current_fig)
			return dict(trim_config), True
		elif command == "save":
			if replace_allowed:
				trim_config = set_trim_choice(trim_config, fire_name, selected_start, selected_end)
				save_trim_config(args.trim_config, trim_config)
			if current_fig is not None:
				plt.close(current_fig)
			return dict(trim_config), True
		elif command == "skip":
			if current_fig is not None:
				plt.close(current_fig)
			return dict(trim_config), True
		elif command in {"q", "quit"}:
			if current_fig is not None:
				plt.close(current_fig)
			return dict(trim_config), False
		else:
			print(f"unknown command: {command!r}")


class MatplotlibFireTrimBrowser:
	def __init__(
		self,
		entry: Mapping[str, Any],
		args: argparse.Namespace,
		trim_config: Mapping[str, Any],
	) -> None:
		self.entry = entry
		self.args = args
		self.trim_config = dict(trim_config)
		self.fire_name = str(entry["name"])
		self.record = entry["record"]
		self.frame_paths = frame_paths_for_record(self.record)
		self.total_frames = len(self.frame_paths)
		self.last_index = self.total_frames - 1
		self.existing = _existing_choice(trim_config, self.fire_name)
		self.replace_allowed = bool(args.overwrite_existing_choices) or self.existing is None
		self.selected_start = (
			int(self.existing["trim_start_index"])
			if self.existing and self.existing.get("trim_start_index") is not None
			else int(args.default_start_index)
		)
		self.selected_end = (
			int(self.existing["trim_end_index"])
			if self.existing and self.existing.get("trim_end_index") is not None
			else None
		)
		self.frame_index = max(0, min(int(self.selected_start), self.last_index))
		self.should_continue = True
		self.finished = False
		self.images: dict[str, Any] = {}
		self.buttons: list[Button] = []
		self.text_boxes: list[TextBox] = []
		self.status_message = "Browse to the first useful fire frame, click Set Trim, then Next Fire."

		self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8.8), squeeze=False)
		self.fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.24, hspace=0.25, wspace=0.18)
		self.help_text = self.fig.text(
			0.04,
			0.885,
			"",
			fontsize=8.5,
			va="top",
			ha="left",
		)
		self.status_text = self.fig.text(0.04, 0.015, "", fontsize=9, va="bottom")
		self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
		self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
		self.fig.canvas.mpl_connect("close_event", self._on_close)
		self._create_controls()
		self._initialize_images()
		if self.existing and not self.replace_allowed:
			self.status_message = "Existing choice is locked. Pass --overwrite_existing_choices to replace it."
		self.update_display()

	def _create_button(self, bounds: Sequence[float], label: str, callback: Any) -> None:
		button = Button(self.fig.add_axes(bounds), label)
		button.on_clicked(callback)
		self.buttons.append(button)

	def _create_controls(self) -> None:
		y_nav = 0.135
		y_action = 0.065
		height = 0.045
		self._create_button([0.04, y_nav, 0.075, height], "Prev", lambda _event: self.move(-1))
		self._create_button([0.125, y_nav, 0.075, height], "Next", lambda _event: self.move(1))
		self._create_button([0.21, y_nav, 0.075, height], "-Jump", lambda _event: self.move(-max(1, int(self.args.jump))))
		self._create_button([0.295, y_nav, 0.075, height], "+Jump", lambda _event: self.move(max(1, int(self.args.jump))))
		text_box = TextBox(self.fig.add_axes([0.395, y_nav, 0.10, height]), "Frame", initial=str(self.frame_index))
		text_box.on_submit(self.jump_to_text)
		self.text_boxes.append(text_box)
		self.frame_text_box = text_box
		self._create_button([0.51, y_nav, 0.07, height], "Go", lambda _event: self.jump_to_text(self.frame_text_box.text))

		self._create_button([0.04, y_action, 0.10, height], "Trim+Next", lambda _event: self.trim_here_and_next())
		self._create_button([0.15, y_action, 0.085, height], "Set Trim", lambda _event: self.set_start())
		self._create_button([0.245, y_action, 0.08, height], "Set End", lambda _event: self.set_end())
		self._create_button([0.335, y_action, 0.09, height], "Next Fire", lambda _event: self.next_fire())
		self._create_button([0.435, y_action, 0.105, height], "Keep Whole", lambda _event: self.keep_whole())
		self._create_button([0.55, y_action, 0.075, height], "Skip", lambda _event: self.skip())
		self._create_button([0.635, y_action, 0.075, height], "Quit", lambda _event: self.quit())

	def _load_current_maps(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
		frame = _load_frame(self.frame_paths[self.frame_index])
		previous = _load_frame(self.frame_paths[self.frame_index - 1]) if self.frame_index > 0 else None
		maps = _diagnostic_maps(frame, previous, float(self.args.flux_threshold), float(self.args.consumed_threshold))
		stats = frame_stats(maps, frame, float(self.args.flux_threshold), float(self.args.consumed_threshold))
		return maps, stats

	def _initialize_images(self) -> None:
		maps, _stats = self._load_current_maps()
		for axis, (title, array) in zip(self.axes.ravel(), maps.items()):
			vmin, vmax = _limits(array)
			image = axis.imshow(array, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
			axis.set_title(title)
			axis.set_xticks([])
			axis.set_yticks([])
			self.fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
			self.images[title] = image
		for axis in self.axes.ravel()[len(maps):]:
			axis.axis("off")

	def update_display(self) -> None:
		maps, stats = self._load_current_maps()
		for title, array in maps.items():
			image = self.images[title]
			vmin, vmax = _limits(array)
			image.set_data(array)
			image.set_clim(vmin, vmax)
		text_box_events = self.frame_text_box.eventson
		self.frame_text_box.eventson = False
		self.frame_text_box.set_val(str(self.frame_index))
		self.frame_text_box.eventson = text_box_events
		self.fig.suptitle(
			f"{self.fire_name} | frame {self.frame_index}/{self.last_index} | "
			f"trim_start={self.selected_start} trim_end={self.selected_end}",
			fontsize=11,
		)
		self.help_text.set_text(
			"Workflow: browse frames until the first useful/fire frame is visible, click Set Trim, then click Next Fire to save this fire and open the next one.\n"
			f"Buttons: Prev/Next move 1 frame; -Jump/+Jump move {max(1, int(self.args.jump))} frames; Frame+Go jumps directly; "
			"Trim+Next sets the current frame as trim_start and immediately advances; Set End is optional; Keep Whole saves all frames; Skip leaves this fire unchanged; Quit stops review."
		)
		lock_state = "locked" if not self.replace_allowed else "editable"
		self.status_text.set_text(
			f"{self.status_message}\n"
			f"surface mean={stats['surface_mean']:.6g} max={stats['surface_max']:.6g} | "
			f"canopy mean={stats['canopy_mean']:.6g} max={stats['canopy_max']:.6g} | "
			f"flux max={stats['total_flux_max']:.6g} active={stats['total_flux_active_pixels']} | "
			f"fuel_delta max={stats['fuel_delta_max']:.6g} active={stats['fuel_delta_active_pixels']} | {lock_state}"
		)
		if bool(self.args.plot_diagnostics):
			preview_path = Path(self.args.diagnostics_dir).expanduser().resolve() / f"{_safe_name(self.fire_name)}_frame_{self.frame_index:06d}.png"
			preview_path.parent.mkdir(parents=True, exist_ok=True)
			self.fig.savefig(preview_path, dpi=130, bbox_inches="tight")
			self.fig.savefig(preview_path.parent / "current_preview.png", dpi=130, bbox_inches="tight")
		self.fig.canvas.draw_idle()

	def move(self, delta: int) -> None:
		self.frame_index = max(0, min(self.last_index, self.frame_index + int(delta)))
		self.status_message = f"Moved to frame {self.frame_index}."
		self.update_display()

	def jump_to_text(self, value: str) -> None:
		try:
			frame_index = int(str(value).strip())
		except ValueError:
			self.status_message = f"Invalid frame index: {value!r}"
			self.update_display()
			return
		self.frame_index = max(0, min(self.last_index, frame_index))
		self.status_message = f"Jumped to frame {self.frame_index}."
		self.update_display()

	def set_start(self) -> None:
		if not self.replace_allowed:
			self.status_message = "Existing choice is locked. Pass --overwrite_existing_choices to replace it."
			self.update_display()
			return
		self.selected_start = int(self.frame_index)
		if self.selected_end is not None and self.selected_end < self.selected_start:
			self.selected_end = None
		if bool(self.args.save_every_choice):
			self.trim_config = set_trim_choice(self.trim_config, self.fire_name, self.selected_start, self.selected_end)
			save_trim_config(self.args.trim_config, self.trim_config)
		self.status_message = f"Set trim_start_index={self.selected_start}."
		self.update_display()

	def set_end(self) -> None:
		if not self.replace_allowed:
			self.status_message = "Existing choice is locked. Pass --overwrite_existing_choices to replace it."
			self.update_display()
			return
		if self.frame_index < self.selected_start:
			self.status_message = "End frame cannot be before the start frame."
			self.update_display()
			return
		self.selected_end = int(self.frame_index)
		if bool(self.args.save_every_choice):
			self.trim_config = set_trim_choice(self.trim_config, self.fire_name, self.selected_start, self.selected_end)
			save_trim_config(self.args.trim_config, self.trim_config)
		self.status_message = f"Set trim_end_index={self.selected_end}."
		self.update_display()

	def _save_choice(self, start: int, end: int | None) -> None:
		if not self.replace_allowed:
			self.status_message = "Existing choice is locked. Pass --overwrite_existing_choices to replace it."
			self.update_display()
			return
		self.selected_start = int(start)
		self.selected_end = None if end is None else int(end)
		self.trim_config = set_trim_choice(self.trim_config, self.fire_name, self.selected_start, self.selected_end)
		save_trim_config(self.args.trim_config, self.trim_config)
		self.finished = True
		self.should_continue = True
		plt.close(self.fig)

	def trim_here_and_next(self) -> None:
		self._save_choice(self.frame_index, self.selected_end if self.selected_end is None or self.selected_end >= self.frame_index else None)

	def next_fire(self) -> None:
		self._save_choice(self.selected_start, self.selected_end)

	def save_and_next(self) -> None:
		self._save_choice(self.selected_start, self.selected_end)

	def keep_whole(self) -> None:
		self._save_choice(0, self.last_index)

	def skip(self) -> None:
		self.finished = True
		self.should_continue = True
		plt.close(self.fig)

	def quit(self) -> None:
		self.finished = True
		self.should_continue = False
		plt.close(self.fig)

	def _on_key_press(self, event: Any) -> None:
		key = str(getattr(event, "key", "") or "").lower()
		if key in {"right", "n"}:
			self.move(1)
		elif key in {"left", "p"}:
			self.move(-1)
		elif key in {"pageup", "f"}:
			self.move(max(1, int(self.args.jump)))
		elif key in {"pagedown", "b"}:
			self.move(-max(1, int(self.args.jump)))
		elif key == "s":
			self.set_start()
		elif key == "e":
			self.set_end()
		elif key in {"enter", "return"}:
			self.trim_here_and_next()
		elif key == "k":
			self.keep_whole()
		elif key == "q":
			self.quit()

	def _on_scroll(self, event: Any) -> None:
		step = int(getattr(event, "step", 0) or 0)
		if step:
			self.move(1 if step > 0 else -1)

	def _on_close(self, _event: Any) -> None:
		if not self.finished:
			self.should_continue = False

	def run(self) -> tuple[dict[str, Any], bool]:
		plt.show(block=True)
		return dict(self.trim_config), bool(self.should_continue)


def review_fire_matplotlib(
	entry: Mapping[str, Any],
	args: argparse.Namespace,
	trim_config: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
	fire_name = str(entry["name"])
	existing = _existing_choice(trim_config, fire_name)
	if existing and bool(args.skip_existing_choices):
		print(f"{fire_name}: existing choice found; skipping.")
		return dict(trim_config), True
	print(f"\nReviewing {fire_name} with Matplotlib GUI")
	print(_matplotlib_display_summary())
	if _is_noninteractive_backend():
		print("Matplotlib GUI mode is unavailable in this session; falling back to terminal workflow.")
		return review_fire_terminal(entry, args, trim_config)
	browser = MatplotlibFireTrimBrowser(entry, args, trim_config)
	return browser.run()


def _safe_name(value: str) -> str:
	return str(value).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _resolve_end(choice_end: Any, default_end: int | None, last_index: int) -> int:
	if choice_end in (None, "", "null"):
		return last_index if default_end is None else int(default_end)
	return int(choice_end)


def build_trimmed_index(
	entries: Sequence[Mapping[str, Any]],
	root: Any,
	schema: Mapping[str, Any],
	trim_config: Mapping[str, Any],
	args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
	updated_records: dict[Any, dict[str, Any]] = {}
	rows: list[dict[str, Any]] = []
	fires = trim_config.get("fires", {}) if isinstance(trim_config.get("fires"), Mapping) else {}
	now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	output_path = str(Path(args.output_index).expanduser().resolve())
	total_before = 0
	total_after = 0
	manual_count = 0
	total_removed_start = 0

	for entry in entries:
		fire_name = str(entry["name"])
		record = dict(entry["record"])
		frame_paths = frame_paths_for_record(record)
		original_num_frames = infer_original_num_frames(record, frame_paths)
		last_index = original_num_frames - 1
		choice = fires.get(fire_name) if isinstance(fires, Mapping) else None
		has_manual = isinstance(choice, Mapping) and choice.get("trim_start_index") not in (None, "", "null")
		manual_count += int(has_manual)
		choice_map = dict(choice) if isinstance(choice, Mapping) else {}
		trim_start = int(choice_map.get("trim_start_index", args.default_start_index))
		trim_end = _resolve_end(choice_map.get("trim_end_index"), args.default_end_index, last_index)
		if trim_start < 0 or trim_start >= original_num_frames:
			raise ValueError(f"{fire_name}: trim_start_index must be within [0, {last_index}], got {trim_start}.")
		if trim_end < trim_start or trim_end >= original_num_frames:
			raise ValueError(f"{fire_name}: trim_end_index must be within [{trim_start}, {last_index}], got {trim_end}.")
		trimmed_num_frames = trim_end - trim_start + 1
		for legacy_key in ("trimmed_frame_paths", "original_frame_paths"):
			record.pop(legacy_key, None)
		record["temporal_trim"] = {
			"enabled": True,
			"mode": "manual",
			"trim_start_index": int(trim_start),
			"trim_end_index": int(trim_end),
			"original_num_frames": int(original_num_frames),
			"trimmed_num_frames": int(trimmed_num_frames),
			"removed_start_frames": int(trim_start),
			"removed_end_frames": int(last_index - trim_end),
			"selected_by": "manual_trim_fire_datasets.py",
			"trim_config": str(Path(args.trim_config).expanduser().resolve()),
			"updated_at": now,
		}
		updated_records[entry["key"]] = record
		rows.append(
			{
				"fire_name": fire_name,
				"original_num_frames": int(original_num_frames),
				"trim_start_index": int(trim_start),
				"trim_end_index": int(trim_end),
				"trimmed_num_frames": int(trimmed_num_frames),
				"removed_start_frames": int(trim_start),
				"removed_end_frames": int(last_index - trim_end),
				"has_manual_choice": bool(has_manual),
				"output_index": output_path,
			}
		)
		total_before += original_num_frames
		total_after += trimmed_num_frames
		total_removed_start += trim_start

	if schema["kind"] == "root_list":
		output_root = [updated_records.get(offset, item) for offset, item in enumerate(root)]
	elif schema["kind"] == "dict_mapping":
		output_root = dict(root)
		container_key = str(schema["container_key"])
		original = root[container_key]
		output_root[container_key] = {key: updated_records.get(str(key), dict(value)) for key, value in original.items()}
		output_root["num_fires"] = len(output_root[container_key])
	elif schema["kind"] == "dict_list":
		output_root = dict(root)
		container_key = str(schema["container_key"])
		original = root[container_key]
		output_root[container_key] = [updated_records.get(offset, item) for offset, item in enumerate(original)]
		output_root["num_fires"] = len(output_root[container_key])
	else:
		raise ValueError(f"Unsupported schema kind: {schema['kind']!r}")
	if isinstance(output_root, dict):
		output_root["temporal_trim"] = {
			"enabled": True,
			"mode": "manual",
			"source_index": str(Path(args.input_index).expanduser().resolve()),
			"trim_config": str(Path(args.trim_config).expanduser().resolve()),
			"updated_at": now,
			"stores_frame_paths": False,
		}
		output_root["last_manual_fire_trim_updated"] = now
	summary = {
		"fires_processed": len(rows),
		"manually_chosen_fires": manual_count,
		"default_kept_fires": len(rows) - manual_count,
		"total_frames_before": int(total_before),
		"total_frames_after": int(total_after),
		"total_removed_start_frames": int(total_removed_start),
		"output_index": output_path,
	}
	return output_root, rows, summary


def write_summary(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], diagnostics_dir: str | Path) -> None:
	output_dir = Path(diagnostics_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	csv_path = output_dir / "manual_trim_summary.csv"
	json_path = output_dir / "manual_trim_summary.json"
	with csv_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
		writer.writeheader()
		writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)
	_atomic_write_json(json_path, {"summary": dict(summary), "fires": list(rows)})


def _selected_entries(entries: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[Mapping[str, Any]]:
	if args.only_fire:
		selected = [entry for entry in entries if str(entry["name"]) == str(args.only_fire)]
		if not selected:
			raise KeyError(f"--only_fire {args.only_fire!r} was not found in the input index.")
		return selected
	if args.start_fire:
		names = [str(entry["name"]) for entry in entries]
		if str(args.start_fire) not in names:
			raise KeyError(f"--start_fire {args.start_fire!r} was not found in the input index.")
		start = names.index(str(args.start_fire))
		return list(entries[start:])
	return list(entries)


def write_output_index(output_index: str | Path, payload: Any, no_overwrite: bool) -> None:
	path = Path(output_index).expanduser().resolve()
	if path.exists() and bool(no_overwrite):
		raise FileExistsError(f"Output index already exists: {path}")
	_atomic_write_json(path, payload)


def run(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
	entries, root, schema = load_fire_index(args.input_index)
	if not entries:
		raise ValueError(f"No fire records found in {args.input_index}.")
	trim_config = load_trim_config(args.trim_config, args.input_index)
	if not args.apply_only:
		for entry in _selected_entries(entries, args):
			if args.mode == "matplotlib":
				trim_config, should_continue = review_fire_matplotlib(entry, args, trim_config)
			else:
				trim_config, should_continue = review_fire_terminal(entry, args, trim_config)
			if not should_continue:
				break
	output_root, rows, summary = build_trimmed_index(entries, root, schema, trim_config, args)
	write_output_index(args.output_index, output_root, bool(args.no_overwrite))
	write_summary(rows, summary, args.diagnostics_dir)
	return output_root, rows, summary


def print_summary(summary: Mapping[str, Any]) -> None:
	print("")
	print(f"fires processed: {summary['fires_processed']}")
	print(f"manually chosen fires: {summary['manually_chosen_fires']}")
	print(f"default-kept fires: {summary['default_kept_fires']}")
	print(f"total frames before: {summary['total_frames_before']}")
	print(f"total frames after: {summary['total_frames_after']}")
	print(f"total removed start frames: {summary['total_removed_start_frames']}")
	print(f"output index path: {summary['output_index']}")


def main() -> None:
	args = build_arg_parser().parse_args()
	_, _, summary = run(args)
	print_summary(summary)


if __name__ == "__main__":
	main()
