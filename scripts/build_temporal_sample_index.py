"""Build metadata-only temporal sample indices for processed full frames."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data.processed_dataset import SPLITS, load_dataset_manifest, manifest_split_fires


DEFAULT_PATTERNS = {
	"consecutive5_h10": {"input_offsets": [-4, -3, -2, -1, 0], "horizon": 10},
	"single1_h10": {"input_offsets": [0], "horizon": 10},
	"sparse5_h10": {"input_offsets": [-40, -30, -20, -10, 0], "horizon": 10},
}


def build(args: argparse.Namespace) -> dict[str, Any]:
	config = load_config(args.config); pc = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}; root=Path(args.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser(); root=root if root.is_absolute() else (Path(args.config).resolve().parent/root).resolve()
	temporal = config.get("temporal_sampling", {}) if isinstance(config.get("temporal_sampling"), dict) else {}; configured = temporal.get("sample_patterns", {}) if isinstance(temporal.get("sample_patterns"), dict) else {}
	patterns = {}
	for name, default in DEFAULT_PATTERNS.items():
		value = configured.get(name, {}) if isinstance(configured.get(name, {}), dict) else {}
		patterns[name] = {"input_offsets": [int(x) for x in value.get("input_offsets", default["input_offsets"])], "horizon": int(value.get("horizon", default["horizon"])), "enabled": bool(value.get("enabled", True))}
	selected = list(patterns) if args.pattern == "all" else [args.pattern]
	patch_files = sorted((root/"indices"/"patches").glob("patches_*.jsonl"));
	if not patch_files: raise FileNotFoundError("No patch index found; run build_patch_index.py first")
	patches = [json.loads(line) for line in patch_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
	root_manifest=load_dataset_manifest(root); rows_by_pattern={}; summary={}
	for pattern in selected:
		if pattern not in patterns: raise ValueError(f"Unknown pattern {pattern!r}")
		info=patterns[pattern]
		rows=[]
		for split in args.splits:
			fires=manifest_split_fires(root_manifest, split)
			for patch in patches:
				if patch["split"] != split or patch["fire_name"] not in fires: continue
				fire_manifest=json.loads((root/"fires"/patch["fire_name"] / "fire_manifest.json").read_text())
				nframes=int(fire_manifest["num_processed_frames"]); offsets=info["input_offsets"]; lo=max(0,-min(offsets)); hi=nframes-1-max(offsets)-info["horizon"]
				for current in range(lo, hi+1):
					inputs=[current+offset for offset in offsets]; target=current+info["horizon"]
					target_rel=Path("targets")/f"h{info['horizon']}"/patch["fire_name"]/f"target_current_{current:06d}_future_{target:06d}.npz"
					if not all((root/"fires"/patch["fire_name"]/"frames"/f"frame_{idx:06d}.npz").exists() for idx in inputs) or not (root/target_rel).exists(): continue
					row={"sample_id":f"{patch['fire_name']}_patch_y{patch['y0']:03d}_x{patch['x0']:03d}_t{current:06d}_pattern_{pattern}","fire_name":patch["fire_name"],"split":split,"pattern":pattern,"patch_id":patch["patch_id"],"input_indices":inputs,"current_index":current,"target_index":target,"horizon":info["horizon"],"target_path":str(target_rel),"patch":{key:patch[key] for key in ("y0","x0","height","width")}}
					rows.append(row)
		out=root/"indices"/"temporal"/f"samples_{pattern}.jsonl"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(json.dumps(row,sort_keys=True) for row in rows)+("\n" if rows else "")); rows_by_pattern[pattern]=rows
		counter=Counter((r["split"],r["fire_name"]) for r in rows); summary[pattern]={"num_samples":len(rows),"by_split":dict(Counter(r["split"] for r in rows)),"by_fire":{f"{split}/{fire}":n for (split,fire),n in counter.items()}}
	(root/"indices"/"temporal"/"sample_index_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
	return summary


def main():
	p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--pattern",default="all"); p.add_argument("--splits",nargs="+",choices=SPLITS,default=list(SPLITS)); print(json.dumps(build(p.parse_args()),indent=2))


if __name__=="__main__": main()
