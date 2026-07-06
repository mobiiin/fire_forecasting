"""Discover fire datasets beneath a main dataset directory and save an index JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.fire_index import (
	DEFAULT_MAIN_DATA_DIR,
	discover_fire_datasets,
	load_fire_dataset_index,
	save_fire_dataset_index,
	update_fire_dataset_index,
)


DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parents[1] / "fire_dataset_index.json"


def build_parser() -> argparse.ArgumentParser:
	"""Create the CLI argument parser."""

	parser = argparse.ArgumentParser(description="Discover fire datasets and save an updateable index JSON.")
	parser.add_argument("--main_data_dir", default=str(DEFAULT_MAIN_DATA_DIR), help="Main dataset directory to scan.")
	parser.add_argument("--output_json", default=None, help="Path to output fire_dataset_index.json.")
	parser.add_argument("--fire_dir_glob", default="*", help="Glob for top-level fire directories.")
	parser.add_argument("--file_pattern", default="*.npy", help="Tensor file glob inside dataset folders.")
	parser.add_argument("--recursive", action="store_true", help="Search recursively below each fire root for tensor dirs.")
	parser.add_argument("--require_npy_files", dest="require_npy_files", action="store_true", default=True)
	parser.add_argument("--no-require_npy_files", dest="require_npy_files", action="store_false")
	parser.add_argument("--require_geom", dest="require_geom", action="store_true", default=True)
	parser.add_argument("--no-require_geom", dest="require_geom", action="store_false")
	parser.add_argument("--require_terrain", dest="require_terrain", action="store_true", default=False)
	parser.add_argument("--no-require_terrain", dest="require_terrain", action="store_false")
	parser.add_argument("--update_existing", dest="update_existing", action="store_true", default=True)
	parser.add_argument("--no-update_existing", dest="update_existing", action="store_false")
	return parser


def main() -> None:
	"""CLI entry point."""

	args = build_parser().parse_args()
	main_data_dir = Path(args.main_data_dir).expanduser().resolve()
	output_json = Path(args.output_json).expanduser().resolve() if args.output_json else DEFAULT_OUTPUT_JSON

	discovered = discover_fire_datasets(
		main_data_dir=main_data_dir,
		fire_dir_glob=args.fire_dir_glob,
		file_pattern=args.file_pattern,
		recursive=bool(args.recursive),
		require_npy_files=bool(args.require_npy_files),
		require_geom=bool(args.require_geom),
		require_terrain=bool(args.require_terrain),
	)
	index = discovered
	if output_json.exists() and args.update_existing:
		index = update_fire_dataset_index(load_fire_dataset_index(output_json), discovered)
	save_fire_dataset_index(index, output_json)

	fires = index.get("fires", {})
	print("Found {} fire datasets under:".format(index.get("num_fires", len(fires))))
	print(f"  {main_data_dir}")
	print("")
	for index_number, fire_name in enumerate(sorted(fires), start=1):
		record = fires[fire_name]
		print(f"[{index_number:03d}] {fire_name}")
		print(f"      path: {record['data_dir']}")
		print(f"      npy files: {record['num_npy_files']}")
		print(f"      geom: {'yes' if record.get('has_geom') else 'no'}")
		print(f"      terrain: {'yes' if record.get('has_terrain') else 'no'}")
	print("")
	print(f"Saved fire dataset index: {output_json}")


if __name__ == "__main__":
	main()
