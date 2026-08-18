"""Compute train-only normalization over dynamic processed patches."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.config import compute_file_sha256, load_config
from src.data.processed_sample_dataset import ProcessedTemporalPatchDataset


def main():
	p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--pattern"); p.add_argument("--splits",nargs="+",default=["train"]); p.add_argument("--max_samples",type=int); p.add_argument("--output_dir"); p.add_argument("--config_name"); p.add_argument("--no_latest_alias",action="store_true"); a=p.parse_args()
	c=load_config(a.config); pc=c.get("processed_dataset",{}) if isinstance(c.get("processed_dataset"),dict) else {}; root=Path(a.dataset_root or pc.get("root","/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser(); root=root if root.is_absolute() else (Path(a.config).resolve().parent/root).resolve(); norm=c.get("normalization",{}) if isinstance(c.get("normalization"),dict) else {}; pattern=a.pattern or norm.get("sample_pattern","consecutive5_h10"); sample_path=root/"indices"/"temporal"/f"samples_{pattern}.jsonl"; output=Path(a.output_dir or norm.get("output_dir",root/"normalization")); output=output if output.is_absolute() else root/output; output.mkdir(parents=True,exist_ok=True)
	records=[json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("split") in a.splits];
	if a.max_samples: records=records[:a.max_samples]
	if not records: raise ValueError("No samples available for normalization")
	first=records[0]; with_archive=np.load(root/"fires"/first["fire_name"]/"frames"/f"frame_{first['input_indices'][0]:06d}.npz",allow_pickle=False); channels=int(with_archive["x_engineered"].shape[0]); with_archive.close(); total=0; sum_=np.zeros(channels,dtype=np.float64); sumsq=np.zeros(channels,dtype=np.float64); min_=np.full(channels,np.inf); max_=np.full(channels,-np.inf)
	for record in tqdm(records, desc=f"Normalization samples ({pattern})", unit="sample"):
		patch=record["patch"]
		for frame_index in tqdm(record["input_indices"], desc="Input frames", unit="frame", leave=False):
			path=root/"fires"/record["fire_name"]/"frames"/f"frame_{frame_index:06d}.npz"
			with np.load(path,allow_pickle=False) as archive: x=np.asarray(archive["x_engineered"],dtype=np.float64)[:,patch["y0"]:patch["y0"]+patch["height"],patch["x0"]:patch["x0"]+patch["width"]]
			if not np.isfinite(x).all(): raise ValueError(f"Non-finite input in {path}")
			flat=x.reshape(channels,-1); total+=flat.shape[1]; sum_+=flat.sum(axis=1); sumsq+=(flat*flat).sum(axis=1); min_=np.minimum(min_,flat.min(axis=1)); max_=np.maximum(max_,flat.max(axis=1))
	mean=sum_/total; std=np.sqrt(np.maximum(sumsq/total-mean*mean,0.0)); std=np.maximum(std,float(norm.get("eps",1e-6))); timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); name=a.config_name or Path(a.config).stem; stem=f"normalization_{name}_{pattern}_{timestamp}"; npz=output/(stem+".npz"); js=output/(stem+".json"); channel_manifest=root/"channel_manifest.json"; channel_names=[]
	if channel_manifest.exists(): channel_names=[entry["name"] for entry in json.loads(channel_manifest.read_text())["channels"]]
	np.savez_compressed(npz,mean=mean.astype(np.float32),std=std.astype(np.float32),min=min_.astype(np.float32),max=max_.astype(np.float32),count=np.asarray(total,dtype=np.int64),channel_indices=np.arange(channels,dtype=np.int64),channel_names=np.asarray(channel_names,dtype="U"))
	payload={"normalization_version":"processed_full_frames_v1","timestamp":timestamp,"config_name":name,"config_path":str(Path(a.config).resolve()),"config_sha256":compute_file_sha256(a.config),"dataset_root":str(root),"dataset_manifest_hash":compute_file_sha256(root/"dataset_manifest.json"),"channel_manifest_hash":compute_file_sha256(channel_manifest) if channel_manifest.exists() else None,"temporal_sample_index_path":str(sample_path),"sample_pattern":pattern,"fit_split":"train","num_samples_used":len(records),"input_channels":channels,"pixel_count":int(total),"npz_path":str(npz)}; js.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
	if not a.no_latest_alias:
		arrays={"mean":mean.astype(np.float32),"std":std.astype(np.float32),"min":min_.astype(np.float32),"max":max_.astype(np.float32),"count":np.asarray(total,dtype=np.int64),"channel_indices":np.arange(channels,dtype=np.int64),"channel_names":np.asarray(channel_names,dtype="U")}
		latest_npz=output/"latest_normalization.npz"; latest_json=output/"latest_normalization.json"
		np.savez_compressed(latest_npz,**arrays); latest_json.write_text(json.dumps({**payload,"npz_path":str(latest_npz)},indent=2,sort_keys=True)+"\n")
		pattern_alias=pattern.replace("/","_").replace("\\","_")
		pattern_npz=output/f"latest_normalization_{pattern_alias}.npz"; pattern_json=output/f"latest_normalization_{pattern_alias}.json"
		np.savez_compressed(pattern_npz,**arrays); pattern_json.write_text(json.dumps({**payload,"npz_path":str(pattern_npz)},indent=2,sort_keys=True)+"\n")
	saved_paths={"output_dir":str(output.resolve()),"json":str(js.resolve()),"npz":str(npz.resolve()),"latest_json":str((output/"latest_normalization.json").resolve()) if not a.no_latest_alias else None,"latest_npz":str((output/"latest_normalization.npz").resolve()) if not a.no_latest_alias else None,"num_samples":len(records),"channels":channels}
	for key in ("json","npz"):
		if not Path(saved_paths[key]).exists(): raise RuntimeError(f"Normalization output was not created: {saved_paths[key]}")
	print(json.dumps(saved_paths,indent=2))


if __name__=="__main__": main()
