"""Visualize static terrain maps in the rebuilt processed dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.config import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--dataset_root"); p.add_argument("--fire"); p.add_argument("--split",default="train"); p.add_argument("--mode",choices=("interactive","save"),default="interactive"); p.add_argument("--output_dir",default="artifacts/terrain"); a=p.parse_args()
    c=load_config(a.config); root=Path(a.dataset_root or c.get("processed_dataset",{}).get("root","/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")); root=root.expanduser().resolve()
    split=json.loads((root/"split_manifest.json").read_text())["splits"]; fires=list(split.get(a.split,[]));
    if a.fire: fires=[a.fire]
    if not fires: raise ValueError(f"No fires found for split={a.split}")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    state={"index":0}
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    def draw():
        fire=fires[state["index"]]; fdir=root/"fires"/fire/"terrain"; height=np.load(fdir/"terrain_height.npy"); features=np.load(fdir/"terrain_features.npy")
        images=[(height,"raw elevation",None,None),(features[0],"relative elevation [0,1]",0.0,1.0),(features[1],"slope magnitude [0,1]",0.0,1.0),(features[2],"slope_x [-1,1]",-1.0,1.0),(features[3],"slope_y [-1,1]",-1.0,1.0)]
        for ax,(image,label,vmin,vmax) in zip(axes.flat,images): ax.clear(); ax.imshow(image,cmap="viridis",vmin=vmin,vmax=vmax); ax.set_title(label); ax.set_xticks([]); ax.set_yticks([])
        axes.flat[-1].axis("off"); fig.suptitle(f"{fire} | {state['index'] + 1}/{len(fires)} | terrain={height.shape}"); fig.canvas.draw_idle()
    def on_key(event):
        if event.key.lower()=="q": plt.close(fig)
        elif event.key.lower()=="f": state["index"]=(state["index"]+1)%len(fires); draw()
        elif event.key.lower()=="b": state["index"]=(state["index"]-1)%len(fires); draw()
        elif event.key.lower()=="w": fig.savefig(out / f"{fires[state['index']]}_terrain.png", dpi=150)
    fig.canvas.mpl_connect("key_press_event",on_key); draw()
    if a.mode=="save":
        for i,fire in enumerate(fires): state["index"]=i; draw(); fig.savefig(out/f"{fire}_terrain.png",dpi=150)
        plt.close(fig)
    else: plt.show()
if __name__=="__main__": main()
