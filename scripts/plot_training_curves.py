"""Regenerate standard training-curve figures for saved runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.run_plots import save_training_run_figures


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Regenerate standard figures for training runs.")
	parser.add_argument("--run_dir", default=None, help="Specific artifacts/runs/<architecture>/<run_name> directory.")
	parser.add_argument("--root", default="artifacts/runs", help="Run root directory.")
	parser.add_argument("--architecture", default=None, help="Optional architecture filter used with --all.")
	parser.add_argument("--all", action="store_true", help="Regenerate figures for all runs under --root.")
	return parser


def _candidate_run_dirs(args: argparse.Namespace) -> list[Path]:
	if args.run_dir:
		return [Path(args.run_dir).expanduser().resolve()]
	root = Path(args.root).expanduser().resolve()
	if args.all and args.architecture:
		return sorted(path for path in (root / args.architecture).glob("*") if path.is_dir())
	if args.all:
		return sorted(path for path in root.glob("*/*") if path.is_dir())
	raise ValueError("Pass --run_dir or --all.")


def _test_results(run_dir: Path) -> dict:
	summary_path = run_dir / "metadata" / "run_summary.json"
	if not summary_path.exists():
		return {}
	try:
		with summary_path.open("r", encoding="utf-8") as handle:
			summary = json.load(handle)
	except json.JSONDecodeError:
		return {}
	test_results = summary.get("test_results", {})
	return test_results if isinstance(test_results, dict) else {}


def main() -> None:
	args = build_argument_parser().parse_args()
	run_dirs = _candidate_run_dirs(args)
	if not run_dirs:
		raise SystemExit("No run directories found.")
	for run_dir in run_dirs:
		figures = save_training_run_figures(
			run_dir,
			architecture=run_dir.parent.name,
			run_name=run_dir.name,
			test_results=_test_results(run_dir),
		)
		count = sum(len(paths) for paths in figures.values())
		print(f"{run_dir}: wrote {count} figure file(s)")


if __name__ == "__main__":
	main()
