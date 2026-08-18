"""Dynamic temporal patch Dataset backed by processed full-frame files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
	import torch
	from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
	torch = None
	class Dataset:  # type: ignore
		pass


class ProcessedTemporalPatchDataset(Dataset):
	def __init__(self, dataset_root: str | Path, sample_index_path: str | Path, split: str | None = None, normalization_stats_path: str | Path | None = None, normalize_inputs: bool = True, return_metadata: bool = False, input_key: str = "x_engineered", target_keys: Sequence[str] = ("surface_consumed", "canopy_consumed", "fire_mask", "energy_log"), force_sequence_length: int | None = None, single_frame_mode: str = "as_is", repeat_to_length: int | None = None, return_terrain: bool = False, terrain_key: str = "terrain_features") -> None:
		if torch is None: raise ImportError("PyTorch is required for ProcessedTemporalPatchDataset")
		self.root=Path(dataset_root).expanduser().resolve(); self.sample_index_path=Path(sample_index_path).expanduser().resolve(); self.split=split; self.normalize_inputs=bool(normalize_inputs); self.return_metadata=bool(return_metadata); self.input_key=input_key; self.target_keys=tuple(target_keys); self.force_sequence_length=force_sequence_length; self.single_frame_mode=single_frame_mode; self.repeat_to_length=repeat_to_length; self.return_terrain=bool(return_terrain); self.terrain_key=str(terrain_key); self.input_normalization_on_device=False; self.inputs_are_normalized=False; self.normalization_stats=None
		self.records=[json.loads(line) for line in self.sample_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
		if split is not None: self.records=[r for r in self.records if r.get("split")==split]
		if not self.records: raise ValueError(f"No processed samples found in {self.sample_index_path} for split={split!r}")
		self.stats=self._load_stats(normalization_stats_path) if normalization_stats_path and self.normalize_inputs else None; self.normalization_stats=self.stats; self.inputs_are_normalized=self.stats is not None
		if self.single_frame_mode not in {"as_is","repeat_to_5"}: raise ValueError("single_frame_mode must be as_is or repeat_to_5")

	def _load_stats(self, path):
		path=Path(path)
		if not path.exists(): raise FileNotFoundError(f"Normalization statistics file not found: {path}")
		if path.suffix==".json":
			payload=json.loads(path.read_text(encoding="utf-8")); npz_value=payload.get("npz_path") or (payload.get("paths",{}).get("npz_path") if isinstance(payload.get("paths"),Mapping) else None); npz=Path(npz_value) if npz_value else path.with_suffix(".npz"); npz=npz if npz.is_absolute() else path.parent/npz
		else: npz=path
		if not npz.exists(): raise FileNotFoundError(f"Normalization NPZ referenced by {path} not found: {npz}")
		with np.load(npz,allow_pickle=False) as z: return {key:z[key] for key in z.files}

	def __len__(self): return len(self.records)

	def __getitem__(self, index):
		record=self.records[index]; fire=record["fire_name"]; patch=record["patch"]; frames=[]
		for frame_index in record["input_indices"]:
			path=self.root/"fires"/fire/"frames"/f"frame_{int(frame_index):06d}.npz"
			with np.load(path,allow_pickle=False) as archive: frame=np.asarray(archive[self.input_key],dtype=np.float32)
			frames.append(frame[:,int(patch["y0"]):int(patch["y0"])+int(patch["height"]),int(patch["x0"]):int(patch["x0"])+int(patch["width"])])
		x=np.stack(frames).astype(np.float32)
		if self.single_frame_mode=="repeat_to_5" and x.shape[0]==1: x=np.repeat(x,self.repeat_to_length or 5,axis=0)
		if self.repeat_to_length is not None and x.shape[0] == 1 and int(self.repeat_to_length) > 1: x=np.repeat(x,int(self.repeat_to_length),axis=0)
		if self.force_sequence_length is not None:
			if x.shape[0]==1 and self.force_sequence_length>1: x=np.repeat(x,int(self.force_sequence_length),axis=0)
			elif x.shape[0]!=int(self.force_sequence_length): raise ValueError(f"Sample T={x.shape[0]} does not match force_sequence_length={self.force_sequence_length}")
		if self.stats is not None:
			mean=np.asarray(self.stats["mean"],dtype=np.float32)[:,None,None]; std=np.maximum(np.asarray(self.stats["std"],dtype=np.float32),1e-6)[:,None,None]; x=(x-mean)/std
		target_path=self.root/record["target_path"]
		if not target_path.exists(): raise FileNotFoundError(f"Missing target for sample_id={record.get('sample_id', index)}: {target_path}")
		with np.load(target_path,allow_pickle=False) as archive:
			y=np.stack([
				np.asarray(archive[key], dtype=np.float32)[
					int(patch["y0"]):int(patch["y0"])+int(patch["height"]),
					int(patch["x0"]):int(patch["x0"])+int(patch["width"]),
				]
				for key in self.target_keys
			]).astype(np.float32)
		if y.shape[1:] != (int(patch["height"]), int(patch["width"])):
			raise ValueError(f"Target patch shape {y.shape} does not match sample patch {patch}")
		if not np.isfinite(y).all(): raise ValueError(f"Target contains NaN/Inf; sample_id={record.get('sample_id', index)}")
		y[2]=(y[2]>0.5).astype(np.float32)
		y=np.asarray(y,dtype=np.float32)
		terrain = None
		if self.return_terrain:
			y0,x0,h,w=(int(patch[key]) for key in ("y0","x0","height","width"))
			terrain_path=self.root/"fires"/str(record["fire_name"])/"terrain"/f"{self.terrain_key}.npy"
			if not terrain_path.exists(): raise FileNotFoundError(f"Terrain features are required but missing for sample_id={record.get('sample_id', index)}: {terrain_path}")
			terrain_array=np.asarray(np.load(terrain_path,allow_pickle=False),dtype=np.float32)
			if terrain_array.ndim!=3 or y0<0 or x0<0 or y0+h>terrain_array.shape[-2] or x0+w>terrain_array.shape[-1]: raise ValueError(f"Terrain crop is invalid for sample_id={record.get('sample_id', index)}: shape={terrain_array.shape} patch={patch}")
			terrain=terrain_array[:,y0:y0+h,x0:x0+w]
			if not np.isfinite(terrain).all(): raise ValueError(f"Terrain contains NaN/Inf; sample_id={record.get('sample_id', index)}")
		if self.return_terrain:
			result={"x":torch.from_numpy(x).float(),"y":torch.from_numpy(y).float(),"terrain":torch.from_numpy(terrain).float()}
			if self.return_metadata: result["metadata"]=dict(record)
			return result
		if self.return_metadata: return torch.from_numpy(x),torch.from_numpy(y),dict(record)
		return torch.from_numpy(x),torch.from_numpy(y)
