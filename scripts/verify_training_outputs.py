"""Verify expected files for one or more training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.training.checkpoints import load_checkpoint


REQUIRED_FILES = [
	"checkpoints/best_model.pt",
	"checkpoints/latest_model.pt",
	"logs/training_log.csv",
	"figures/loss_curves.png",
	"configs/resolved_config.yaml",
	"metadata/run_summary.json",
]


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Verify run-scoped wildfire training outputs.")
	parser.add_argument("--run_dir", default=None, help="Specific artifacts/runs/<architecture>/<run_name> directory.")
	parser.add_argument("--root", default="artifacts/runs", help="Run root directory.")
	parser.add_argument("--architecture", default=None, help="Verify runs for one architecture under --root.")
	parser.add_argument("--all", action="store_true", help="Verify all runs under --root.")
	return parser


def _candidate_run_dirs(args: argparse.Namespace) -> list[Path]:
	if args.run_dir:
		return [Path(args.run_dir).expanduser().resolve()]
	root = Path(args.root).expanduser().resolve()
	if args.all:
		return sorted(path for path in root.glob("*/*") if path.is_dir())
	if args.architecture:
		return sorted(path for path in (root / args.architecture).glob("*") if path.is_dir())
	raise ValueError("Pass --run_dir, --architecture, or --all.")


def _read_training_rows(path: Path) -> list[dict[str, str]]:
	with path.open("r", newline="", encoding="utf-8") as handle:
		return list(csv.DictReader(handle))


def _finite(value: Any) -> bool:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return False
	return math.isfinite(number)


def verify_run(run_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
	architecture = run_dir.parent.name
	run_name = run_dir.name
	messages: list[str] = []
	ok = True
	for relative_path in REQUIRED_FILES:
		path = run_dir / relative_path
		if not path.exists():
			ok = False
			messages.append(f"missing: {relative_path}")

	checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
	if checkpoint_path.exists():
		checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
		if checkpoint.get("architecture") != architecture:
			ok = False
			messages.append(f"checkpoint architecture mismatch: {checkpoint.get('architecture')} != {architecture}")
		if checkpoint.get("run_name") != run_name:
			ok = False
			messages.append(f"checkpoint run_name mismatch: {checkpoint.get('run_name')} != {run_name}")

	training_log_path = run_dir / "logs" / "training_log.csv"
	rows: list[dict[str, str]] = []
	if training_log_path.exists():
		rows = _read_training_rows(training_log_path)
		if not rows:
			ok = False
			messages.append("training_log.csv has no epoch rows")
		elif "train_loss" not in rows[0]:
			ok = False
			messages.append("training_log.csv missing train_loss")
		elif not any(_finite(row.get("train_loss")) for row in rows):
			ok = False
			messages.append("training_log.csv train_loss is all NaN/non-finite")

	summary_path = run_dir / "metadata" / "run_summary.json"
	summary: dict[str, Any] = {}
	if summary_path.exists():
		with summary_path.open("r", encoding="utf-8") as handle:
			summary = json.load(handle)

	info = {
		"architecture": architecture,
		"run_name": run_name,
		"status": summary.get("status", "unknown"),
		"best_epoch": summary.get("best_epoch"),
		"best_val_loss": summary.get("best_metric_value"),
		"checkpoint_path": str(checkpoint_path),
		"loss_curve_path": str(run_dir / "figures" / "loss_curves.png"),
		"epochs_logged": len(rows),
	}
	return ok, messages, info


def main() -> None:
	args = build_argument_parser().parse_args()
	run_dirs = _candidate_run_dirs(args)
	if not run_dirs:
		raise SystemExit("No run directories found.")

	all_ok = True
	for run_dir in run_dirs:
		ok, messages, info = verify_run(run_dir)
		all_ok = all_ok and ok
		status = "OK" if ok else "FAILED"
		print(
			f"{status} | {info['architecture']} | {info['run_name']} | "
			f"status={info['status']} | best_epoch={info['best_epoch']} | best_val_loss={info['best_val_loss']}"
		)
		print(f"  checkpoint: {info['checkpoint_path']}")
		print(f"  loss_curve: {info['loss_curve_path']}")
		for message in messages:
			print(f"  - {message}")
	if not all_ok:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
