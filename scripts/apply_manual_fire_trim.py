"""Apply manual fire-start trim decisions to a fire dataset index."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fire_index import load_fire_dataset_index, save_fire_dataset_index
from src.data.temporal_trim import infer_original_num_frames


SUMMARY_FIELDS = [
	"fire_name",
	"original_num_frames",
	"trim_start_index",
	"trim_end_index",
	"trimmed_num_frames",
	"removed_start_frames",
	"removed_end_frames",
	"trim_mode",
	"warning",
]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Apply manual temporal trim choices to fire_dataset_index.json.")
	parser.add_argument("--input_index", default="fire_dataset_index.json", help="Input fire dataset index JSON.")
	parser.add_argument("--trim_config", default="configs/manual_fire_trim.json", help="Manual trim decision JSON.")
	parser.add_argument("--output_index", default="fire_dataset_index_trimmed.json", help="Output compact trimmed index JSON.")
	parser.add_argument("--default_start_index", type=int, default=0, help="Start index for fires missing from trim_config.")
	parser.add_argument("--default_end_index", type=int, default=None, help="End index for fires missing trim_end_index; default is last frame.")
	parser.add_argument("--input_sequence_length", type=int, default=5, help="Minimum input sequence length used for warnings.")
	parser.add_argument("--prediction_horizon", type=int, default=10, help="Minimum prediction horizon used for warnings.")
	parser.add_argument("--require_all_fires", action="store_true", help="Error if any input fire lacks a manual trim entry.")
	parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output index.")
	parser.add_argument("--dry_run", action="store_true", help="Print/write diagnostics without writing output index.")
	parser.add_argument(
		"--summary_csv",
		default="artifacts/prefire_trim_diagnostics/manual_trim_summary.csv",
		help="CSV summary output path.",
	)
	return parser


def _load_json(path: str | Path) -> dict[str, Any]:
	resolved = Path(path).expanduser().resolve()
	with resolved.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, dict):
		raise ValueError(f"Expected JSON object in {resolved}.")
	return payload


def _save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
	output_path = Path(path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")


def _write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
	output_path = Path(path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
		writer.writeheader()
		writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def _manual_entries(trim_config: Mapping[str, Any]) -> Mapping[str, Any]:
	fires = trim_config.get("fires", {})
	if not isinstance(fires, Mapping):
		raise ValueError("trim_config must contain a 'fires' mapping.")
	return fires


def _configured_int(value: Any, fallback: int, *, name: str) -> int:
	if value in (None, "", "null"):
		return int(fallback)
	try:
		return int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def apply_manual_fire_trim(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
	input_index_path = Path(args.input_index).expanduser().resolve()
	trim_config_path = Path(args.trim_config).expanduser().resolve()
	index = load_fire_dataset_index(input_index_path)
	trim_config = _load_json(trim_config_path)
	manual_fires = _manual_entries(trim_config)
	fires = index.get("fires", {})
	if not isinstance(fires, Mapping) or not fires:
		raise ValueError("Input index is missing a nonempty 'fires' mapping.")

	min_required_frames = int(args.input_sequence_length) + int(args.prediction_horizon)
	if min_required_frames <= 0:
		raise ValueError("input_sequence_length + prediction_horizon must be positive.")

	trimmed_fires: dict[str, Any] = {}
	rows: list[dict[str, Any]] = []
	manual_count = 0
	default_count = 0
	total_original = 0
	total_trimmed = 0
	now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

	for fire_name, record in sorted(fires.items()):
		if not isinstance(record, Mapping):
			continue
		entry = manual_fires.get(str(fire_name))
		if args.require_all_fires and not isinstance(entry, Mapping):
			raise ValueError(f"Missing manual trim entry for fire {fire_name!r}.")
		using_manual = isinstance(entry, Mapping)
		entry_map = dict(entry) if isinstance(entry, Mapping) else {}
		manual_count += int(using_manual)
		default_count += int(not using_manual)

		original_num_frames = infer_original_num_frames(record)
		last_index = original_num_frames - 1
		trim_start_index = _configured_int(
			entry_map.get("trim_start_index"),
			int(args.default_start_index),
			name=f"{fire_name}.trim_start_index",
		)
		default_end = last_index if args.default_end_index is None else int(args.default_end_index)
		trim_end_index = _configured_int(entry_map.get("trim_end_index"), default_end, name=f"{fire_name}.trim_end_index")

		if trim_start_index < 0 or trim_start_index >= original_num_frames:
			raise ValueError(
				f"{fire_name}: trim_start_index must be within [0, {last_index}], got {trim_start_index}."
			)
		if trim_end_index < trim_start_index or trim_end_index >= original_num_frames:
			raise ValueError(
				f"{fire_name}: trim_end_index must be within [{trim_start_index}, {last_index}], got {trim_end_index}."
			)

		trimmed_num_frames = trim_end_index - trim_start_index + 1
		warning = ""
		if trimmed_num_frames < min_required_frames:
			warning = (
				f"trimmed_num_frames={trimmed_num_frames} is smaller than "
				f"input_sequence_length+prediction_horizon={min_required_frames}"
			)

		updated = dict(record)
		for legacy_key in ("trimmed_frame_paths", "original_frame_paths"):
			updated.pop(legacy_key, None)
		updated["temporal_trim"] = {
			"enabled": True,
			"mode": "manual" if using_manual else "default",
			"original_start_index": 0,
			"original_end_index": int(last_index),
			"trim_start_index": int(trim_start_index),
			"trim_end_index": int(trim_end_index),
			"original_num_frames": int(original_num_frames),
			"trimmed_num_frames": int(trimmed_num_frames),
			"selected_by": str(entry_map.get("selected_with", "manual_visualizer" if using_manual else "apply_manual_fire_trim.py")),
			"notes": str(entry_map.get("notes", "")),
			"created_at": now,
		}
		if warning:
			updated["temporal_trim"]["warning"] = warning
		trimmed_fires[str(fire_name)] = updated

		row = {
			"fire_name": str(fire_name),
			"original_num_frames": int(original_num_frames),
			"trim_start_index": int(trim_start_index),
			"trim_end_index": int(trim_end_index),
			"trimmed_num_frames": int(trimmed_num_frames),
			"removed_start_frames": int(trim_start_index),
			"removed_end_frames": int(last_index - trim_end_index),
			"trim_mode": updated["temporal_trim"]["mode"],
			"warning": warning,
		}
		rows.append(row)
		total_original += original_num_frames
		total_trimmed += trimmed_num_frames

	output_index = dict(index)
	output_index["fires"] = trimmed_fires
	output_index["num_fires"] = int(len(trimmed_fires))
	output_index["temporal_trim"] = {
		"enabled": True,
		"mode": "manual",
		"source_index": str(input_index_path),
		"trim_config": str(trim_config_path),
		"created_at": now,
		"stores_frame_paths": False,
	}
	output_index["last_manual_fire_trim_updated"] = now
	summary = {
		"fires_processed": int(len(rows)),
		"fires_manually_trimmed": int(manual_count),
		"fires_using_default_start": int(default_count),
		"total_original_frames": int(total_original),
		"total_trimmed_frames": int(total_trimmed),
		"total_removed_frames": int(total_original - total_trimmed),
	}
	return output_index, rows, summary


def main() -> None:
	args = build_arg_parser().parse_args()
	output_index = Path(args.output_index).expanduser().resolve()
	if output_index.exists() and not bool(args.overwrite) and not bool(args.dry_run):
		raise FileExistsError(f"Output index already exists: {output_index}. Pass --overwrite to replace it.")

	trimmed_index, rows, summary = apply_manual_fire_trim(args)
	summary_csv = Path(args.summary_csv).expanduser().resolve()
	summary_json = summary_csv.with_suffix(".json")
	_write_summary_csv(summary_csv, rows)
	_save_json(summary_json, {"summary": summary, "fires": rows})
	if bool(args.dry_run):
		print(f"Dry run: not writing output index {output_index}")
	else:
		save_fire_dataset_index(trimmed_index, output_index)
		print(f"Saved trimmed fire dataset index: {output_index}")
	print(f"Saved summary CSV: {summary_csv}")
	print(f"Saved summary JSON: {summary_json}")
	for key, value in summary.items():
		print(f"{key}: {value}")


if __name__ == "__main__":
	main()
