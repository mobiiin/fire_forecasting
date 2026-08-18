"""Visualize patch boxes over a processed full-frame tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import load_config
from src.data.processed_dataset import SPLITS, load_frame, load_split_fires


def main():
	p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--patch_index"); p.add_argument("--fire"); p.add_argument("--split", choices=SPLITS, default="train"); p.add_argument("--frame_index", type=int, default=0); p.add_argument("--max_patches_display", type=int, default=200); p.add_argument("--output_dir", default="artifacts/patch_index"); p.add_argument("--mode", choices=("interactive", "save"), default="interactive")
	a=p.parse_args(); c=load_config(a.config); pc=c.get("processed_dataset", {}) if isinstance(c.get("processed_dataset"), dict) else {}; root=Path(a.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser(); fire=a.fire or load_split_fires(root,a.split)[0]
	if a.patch_index: index=Path(a.patch_index)
	else:
		files=sorted((root/"indices"/"patches").glob("patches_*.jsonl"));
		if not files: raise FileNotFoundError("No patch JSONL found; run build_patch_index.py first")
		index=files[0]
	rows=[json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("fire_name")==fire][:a.max_patches_display]
	x=load_frame(root,fire,a.frame_index); background=x[84 if x.shape[0]>84 else 0]; fig,ax=plt.subplots(figsize=(10,8)); ax.imshow(background,cmap="gray")
	for i,row in enumerate(rows):
		r=patches.Rectangle((row["x0"],row["y0"]),row["width"],row["height"],fill=False,edgecolor="red",linewidth=1); ax.add_patch(r); ax.text(row["x0"],row["y0"],str(i),color="yellow",fontsize=7)
	ax.set_title(f"{fire} | {a.split} | frame {a.frame_index} | patches={len(rows)}"); ax.set_axis_off(); out=Path(a.output_dir)/f"{fire}_frame_{a.frame_index:06d}.png"
	if a.mode=="save": out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=130)
	else: plt.show()


if __name__ == "__main__": main()
