"""List training runs saved under artifacts/runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="List run-scoped wildfire training outputs.")
	parser.add_argument("--root", default="artifacts/runs", help="Run root directory.")
	parser.add_argument("--architecture", default=None, help="Optional architecture filter.")
	return parser


def _load_summary(run_dir: Path) -> dict[str, Any]:
	path = run_dir / "metadata" / "run_summary.json"
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	return payload if isinstance(payload, dict) else {}


def _rows(root: Path, architecture: str | None) -> list[dict[str, Any]]:
	search_root = root / architecture if architecture else root
	pattern = "*" if architecture else "*/*"
	rows: list[dict[str, Any]] = []
	for run_dir in sorted(path for path in search_root.glob(pattern) if path.is_dir()):
		summary = _load_summary(run_dir)
		rows.append(
			{
				"architecture": run_dir.parent.name,
				"run_name": run_dir.name,
				"status": summary.get("status", "unknown"),
				"best_epoch": summary.get("best_epoch", ""),
				"best_metric": summary.get("best_metric_value", ""),
				"final_val_loss": summary.get("final_val_loss", ""),
				"checkpoint": summary.get("best_checkpoint_path", str(run_dir / "checkpoints" / "best_model.pt")),
				"start_time": summary.get("start_time", ""),
				"duration": summary.get("duration_sec", ""),
			}
		)
	return rows


def main() -> None:
	args = build_argument_parser().parse_args()
	root = Path(args.root).expanduser().resolve()
	rows = _rows(root, args.architecture)
	if not rows:
		raise SystemExit(f"No runs found under {root}.")
	headers = ["architecture", "run_name", "status", "best_epoch", "best_metric", "final_val_loss", "checkpoint", "start_time", "duration"]
	widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
	print(" | ".join(header.ljust(widths[header]) for header in headers))
	print("-+-".join("-" * widths[header] for header in headers))
	for row in rows:
		print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


if __name__ == "__main__":
	main()
