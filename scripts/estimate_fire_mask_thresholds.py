"""Estimate frozen fire-mask thresholds from train fires only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.config import compute_file_sha256, load_config
from src.data.processed_dataset import SPLITS, load_dataset_manifest, load_fire_manifest, manifest_split_fires


def _percentile_summary(values: np.ndarray, percentile: float) -> dict[str, float | int]:
	if values.size == 0: raise ValueError("No positive values were observed; cannot estimate a threshold.")
	percentiles = [0.1, 1, 5, 10, 50, 90, 95, 99]
	result: dict[str, float | int] = {"positive_count_total": int(values.size), "positive_count_used": int(values.size), "min": float(values.min()), "max": float(values.max())}
	for p in percentiles: result[f"p{str(p).replace('.', '_')}"] = float(np.percentile(values, p))
	result["selected_percentile"] = float(percentile); result["selected_value"] = float(np.percentile(values, percentile)); return result


def _derived_config(original: Path, destination: Path, thresholds: dict[str, float], threshold_path: Path) -> None:
	relative_base = Path(__import__("os").path.relpath(original, destination.parent))
	payload = {"base_config": str(relative_base), "target_construction": {"fire_mask": {"method": "threshold_union", **thresholds, "threshold_file": str(threshold_path), "require_frozen_thresholds": True}}}
	destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
	config_path = Path(args.config).expanduser().resolve(); config = load_config(config_path); pc=config.get("processed_dataset",{}) if isinstance(config.get("processed_dataset"),dict) else {}; root=Path(args.dataset_root or pc.get("root","/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser(); root=root if root.is_absolute() else (config_path.parent/root).resolve(); tc=config.get("target_construction",{}) if isinstance(config.get("target_construction"),dict) else {}; estimation=tc.get("threshold_estimation",{}) if isinstance(tc.get("threshold_estimation"),dict) else {}; fit_split=args.fit_split or estimation.get("fit_split","train")
	if fit_split != "train": raise ValueError("Threshold estimation is restricted to fit_split=train for split safety.")
	horizon=int(args.horizon or tc.get("horizon",config.get("prediction_horizon",10))); percentile=float(args.percentile if args.percentile is not None else estimation.get("percentile",1.0));
	if not 0 < percentile <= 100: raise ValueError("percentile must be in (0, 100]")
	manifest=load_dataset_manifest(root); train_fires=manifest_split_fires(manifest, "train");
	if not train_fires: raise ValueError("Processed dataset has no train fires")
	minimum=estimation.get("min_positive_value",{}) if isinstance(estimation.get("min_positive_value",{}),dict) else {}; names=("energy_release_mw","surface_consumed","canopy_consumed"); values={name:[] for name in names}; total_positive={name:0 for name in names}; clip=tc.get("consumed_fuel",{}) if isinstance(tc.get("consumed_fuel"),dict) else {}; rng=np.random.default_rng(args.seed); pair_cap=args.max_pairs_per_fire or estimation.get("max_frame_pairs_per_fire"); pair_counts={}
	for fire_name in train_fires:
		fire=load_fire_manifest(root,fire_name); nframes=int(fire["num_processed_frames"]); area=np.load(root/"fires"/fire_name/fire["geometry"]["area_2d_path"],allow_pickle=False); valid_indices=np.arange(max(0,nframes-horizon),dtype=np.int64); total_pairs=int(valid_indices.size)
		if pair_cap is not None and int(pair_cap) < total_pairs:
			valid_indices=rng.choice(valid_indices,size=int(pair_cap),replace=False)
		pair_counts[fire_name]={"available":total_pairs,"used":int(valid_indices.size),"shuffled_subsample":int(valid_indices.size)<total_pairs}
		for current_index in valid_indices:
			future_index=current_index+horizon; current_path=root/"fires"/fire_name/"frames"/f"frame_{current_index:06d}.npz"; future_path=root/"fires"/fire_name/"frames"/f"frame_{future_index:06d}.npz"
			with np.load(current_path,allow_pickle=False) as ca, np.load(future_path,allow_pickle=False) as fa:
				current=np.asarray(ca["x_engineered"],dtype=np.float32); future=np.asarray(fa["x_engineered"],dtype=np.float32)
			surface=current[84]-future[84]; canopy=current[85]-future[85]
			if bool(clip.get("clip_negative",True)): surface=np.maximum(surface,0); canopy=np.maximum(canopy,0)
			if bool(clip.get("clip_to_available_fuel",True)): surface=np.minimum(surface,np.maximum(current[84],0)); canopy=np.minimum(canopy,np.maximum(current[85],0))
			energy=np.maximum(area*(future[80]+future[81]+future[82]+future[83])/1e6,0)
			for name,array in (("energy_release_mw",energy),("surface_consumed",surface),("canopy_consumed",canopy)):
				positive=np.asarray(array)[np.asarray(array)>float(minimum.get(name,0.0))].ravel(); total_positive[name]+=int(positive.size); values[name].append(positive)
	max_values=args.max_values_per_quantity or estimation.get("max_samples_per_quantity")
	distributions={}; thresholds={};
	for name,chunks in values.items():
		array=np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float32); downsampled=False
		if max_values is not None and array.size>int(max_values): array=rng.choice(array,size=int(max_values),replace=False); downsampled=True
		stats=_percentile_summary(array,percentile); stats["positive_count_total"]=total_positive[name]; stats["positive_count_used"]=int(array.size); stats["downsampled"]=downsampled; distributions[name]=stats
		thresholds["energy_threshold_mw" if name=="energy_release_mw" else ("surface_fuel_threshold" if name=="surface_consumed" else "canopy_fuel_threshold")]=float(stats["selected_value"])
	output=Path(args.output_dir or estimation.get("output_dir",root/"thresholds")); output=output if output.is_absolute() else (config_path.parent/output).resolve(); output.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"); name=config_path.stem; timestamped=output/f"fire_mask_thresholds_{name}_h{horizon}_p{str(percentile).replace('.','p')}_{stamp}.json"; latest=output/"fire_mask_thresholds_latest.json"
	payload={"threshold_version":"v1_train_split_percentile","created_at":datetime.now(timezone.utc).isoformat(),"timestamp":stamp,"config":{"config_name":name,"config_path":str(config_path),"config_path_absolute":str(config_path),"config_sha256":compute_file_sha256(config_path),"resolved_config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()},"dataset":{"dataset_root":str(root),"dataset_version":manifest.get("dataset_version"),"dataset_manifest_path":str(root/"dataset_manifest.json"),"dataset_manifest_hash":compute_file_sha256(root/"dataset_manifest.json"),"channel_manifest_hash":compute_file_sha256(root/"channel_manifest.json") if (root/"channel_manifest.json").exists() else None,"array_layout":"C,H,W"},"fit":{"fit_split":"train","horizon":horizon,"percentile":percentile,"positive_only":True,"clip_negative":bool(clip.get("clip_negative",True)),"clip_to_available_fuel":bool(clip.get("clip_to_available_fuel",True)),"num_train_fires":len(train_fires),"train_fires":train_fires,"frame_pair_sampling":{"max_pairs_per_fire":int(pair_cap) if pair_cap is not None else None,"seed":int(args.seed),"per_fire":pair_counts}},"thresholds":thresholds,"distributions":distributions,"split_safety":{"used_splits":["train"],"excluded_splits":["val","test"],"val_test_used":False},"files":{"timestamped_path":str(timestamped),"latest_path":str(latest)}}
	timestamped.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); latest.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
	if args.update_config: _derived_config(config_path,Path(args.derived_config_path or estimation.get("derived_config_path",f"configs/derived/{name}_with_fire_mask_thresholds.yaml")),thresholds,latest)
	print(json.dumps(payload,indent=2)); return payload


def main():
	p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--dataset_root'); p.add_argument('--horizon',type=int); p.add_argument('--percentile',type=float, default=5.0); p.add_argument('--fit_split',default='train'); p.add_argument('--output_dir'); p.add_argument('--update_config',action='store_true'); p.add_argument('--derived_config_path'); p.add_argument('--max_values_per_quantity',type=int); p.add_argument('--max_pairs_per_fire',type=int,help='Randomly process at most this many current/future pairs per train fire.', default=500); p.add_argument('--seed',type=int,default=42); p.add_argument('--overwrite',action='store_true'); build(p.parse_args())
if __name__=='__main__': main()
