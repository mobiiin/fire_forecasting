"""Build deterministic spatial patch metadata for processed full-frame data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data.processed_dataset import SPLITS, load_dataset_manifest, load_fire_manifest, make_patch_id, manifest_split_fires, patch_starts


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--patch_size", type=int); p.add_argument("--stride", type=int); p.add_argument("--include_border_patches", action="store_true", default=None); p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS)); p.add_argument("--overwrite", action="store_true")
	a = p.parse_args(); config = load_config(a.config); pc = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
	root = Path(a.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
	if not root.is_absolute(): root = (Path(a.config).resolve().parent / root).resolve()
	patch_cfg = pc.get("patch_index", {}) if isinstance(pc.get("patch_index"), dict) else {}
	patch_size = int(a.patch_size or patch_cfg.get("patch_size", config.get("patching", {}).get("patch_size", 64)))
	stride = int(a.stride or patch_cfg.get("stride", config.get("patching", {}).get("train_stride", 60)))
	include_border = bool(patch_cfg.get("include_border_patches", True) if a.include_border_patches is None else a.include_border_patches)
	out = root / "indices" / "patches" / f"patches_{patch_size}_stride{stride}_{'border' if include_border else 'noborder'}.jsonl"
	if out.exists() and not a.overwrite: raise FileExistsError(f"Patch index exists; pass --overwrite: {out}")
	out.parent.mkdir(parents=True, exist_ok=True); log_path = root / "logs" / "build_patch_index.log"; log_path.parent.mkdir(parents=True, exist_ok=True); logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
	manifest = load_dataset_manifest(root); rows: list[dict[str, Any]] = []; by_split = {s: 0 for s in SPLITS}
	for split in a.splits:
		for fire_name in manifest_split_fires(manifest, split):
			fire = load_fire_manifest(root, fire_name); _, h, w = [int(x) for x in fire["frame_shape"]]
			for y0 in patch_starts(h, patch_size, stride, include_border):
				for x0 in patch_starts(w, patch_size, stride, include_border):
					row = {"patch_id": make_patch_id(fire_name, y0, x0, patch_size, patch_size), "fire_name": fire_name, "split": split, "y0": y0, "x0": x0, "height": patch_size, "width": patch_size, "frame_height": h, "frame_width": w, "include_border": include_border, "patch_size": patch_size, "stride": stride, "min_valid_fraction": float(patch_cfg.get("min_valid_fraction", 0.0))}
					rows.append(row); by_split[split] += 1
	with out.open("w", encoding="utf-8") as handle:
		for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
	summary = {"path": str(out.relative_to(root)), "patch_size": patch_size, "stride": stride, "include_border_patches": include_border, "num_patches": len(rows), "patches_by_split": by_split, "fires": sorted({r["fire_name"] for r in rows})}
	(out.with_name(out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
