"""Construct full-frame target tensors for the processed dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from src.config import load_config
from src.data.processed_dataset import SPLITS, load_dataset_manifest, load_fire_manifest, manifest_split_fires
from src.data.processed_targets import build_processed_target
from src.data.fire_mask_thresholds import resolve_frozen_thresholds


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
	config = load_config(args.config)
	pc = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
	root = Path(args.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
	if not root.is_absolute(): root = (Path(args.config).resolve().parent / root).resolve()
	tc = config.get("target_construction", {}) if isinstance(config.get("target_construction"), dict) else {}
	horizon = int(args.horizon or tc.get("horizon", config.get("prediction_horizon", 10)))
	base = root / "targets" / f"h{horizon}"
	thresholds, threshold_meta = resolve_frozen_thresholds(config, args.config, require=bool(tc.get("fire_mask", {}).get("require_frozen_thresholds", True)))
	manifest = load_dataset_manifest(root)
	splits = manifest.get("splits", {})
	rows: list[dict[str, Any]] = []
	stats: dict[str, list[float]] = {key: [] for key in ("surface_consumed", "canopy_consumed", "energy_release_mw", "energy_log", "mask_fraction")}
	target_fires = [(split, fire_name) for split in args.splits for fire_name in manifest_split_fires(manifest, split)]
	for split, fire_name in tqdm(target_fires, desc="Fires", unit="fire"):
		fire = load_fire_manifest(root, fire_name)
		nframes = int(fire["num_processed_frames"])
		if nframes <= horizon: continue
		area_path = root / "fires" / fire_name / fire["geometry"]["area_2d_path"]
		area = np.load(area_path, allow_pickle=False)
		fire_dir = base / fire_name; fire_dir.mkdir(parents=True, exist_ok=True)
		indices = list(range(0, nframes - horizon))
		if args.max_targets_per_fire: indices = indices[:args.max_targets_per_fire]
		for current_index in tqdm(indices, desc=f"{split}/{fire_name} targets", unit="target", leave=False):
			future_index = current_index + horizon
			current_path = root / "fires" / fire_name / "frames" / f"frame_{current_index:06d}.npz"
			future_path = root / "fires" / fire_name / "frames" / f"frame_{future_index:06d}.npz"
			with np.load(current_path, allow_pickle=False) as current_archive, np.load(future_path, allow_pickle=False) as future_archive:
				if "x_raw" not in current_archive or "x_raw" not in future_archive:
					raise KeyError("Target construction requires x_raw in processed frame files; rebuild with save_raw_channels=true")
				target = build_processed_target(current_archive["x_raw"], future_archive["x_raw"], area, config, thresholds=thresholds)
			out = fire_dir / f"target_current_{current_index:06d}_future_{future_index:06d}.npz"
			np.savez_compressed(out, **target)
			row = {"fire_name": fire_name, "split": split, "current_index": current_index, "future_index": future_index, "horizon": horizon, "path": str(out.relative_to(root)), "source_current_frame_path": str(current_path.relative_to(root)), "source_future_frame_path": str(future_path.relative_to(root)), "mask_thresholds": thresholds, "threshold_source": threshold_meta.get("source"), "threshold_file": threshold_meta.get("threshold_file"), "threshold_version": threshold_meta.get("threshold_version")}
			rows.append(row)
			for key in ("surface_consumed", "canopy_consumed", "energy_release_mw", "energy_log"): stats[key].extend([float(np.mean(target[key])), float(np.median(target[key])), float(np.max(target[key]))])
			stats["mask_fraction"].append(float(np.mean(target["fire_mask"])))
	manifest_path = base / "target_manifest.jsonl"; manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
	by_split = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
	summary = {"horizon": horizon, "num_targets": len(rows), "targets_by_split": by_split, "thresholds": thresholds, "threshold_source": threshold_meta, "stats": {key: {"mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None, "max": float(np.max(values)) if values else None} for key, values in stats.items()}}
	_write_json(base / "target_manifest.json", {"horizon": horizon, "created_at": datetime.now(timezone.utc).isoformat(), "thresholds": thresholds, "threshold_source": threshold_meta, "targets": rows})
	_write_json(base / "target_summary.json", summary)
	return summary


def main() -> None:
	p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--horizon", type=int); p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS)); p.add_argument("--overwrite", action="store_true"); p.add_argument("--max_targets_per_fire", type=int); print(json.dumps(build(p.parse_args()), indent=2))


if __name__ == "__main__": main()
