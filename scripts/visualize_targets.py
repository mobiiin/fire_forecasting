"""Interactive viewer for full-frame target tensors.

Controls:
  Left/Right  previous/next target timestamp
  n/p         next/previous fire
  w           save current figure
  q           quit
  h           print controls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.fire_mask_thresholds import resolve_frozen_thresholds
from src.data.processed_dataset import SPLITS, load_dataset_manifest, load_fire_manifest, manifest_split_fires


def _load_view(root: Path, fire: str, current_index: int, horizon: int):
	frame_dir = root / "fires" / fire / "frames"
	current_path = frame_dir / f"frame_{current_index:06d}.npz"
	future_path = frame_dir / f"frame_{current_index + horizon:06d}.npz"
	target_path = root / "targets" / f"h{horizon}" / fire / f"target_current_{current_index:06d}_future_{current_index + horizon:06d}.npz"
	with np.load(current_path, allow_pickle=False) as archive:
		current = np.asarray(archive["x_raw"])
	with np.load(future_path, allow_pickle=False) as archive:
		future = np.asarray(archive["x_raw"])
	with np.load(target_path, allow_pickle=False) as archive:
		arrays = [current[84], future[84], archive["surface_consumed"], current[85], future[85], archive["canopy_consumed"], archive["energy_release_mw"], archive["energy_log"], archive["fire_mask"]]
	return arrays


def _draw(fig, arrays, labels, title):
	fig.clear()
	fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.10, hspace=0.25)
	axes = fig.subplots(3, 3)
	for ax, image, label in zip(axes.flat, arrays, labels):
		ax.imshow(image, cmap="magma")
		ax.set_title(label, fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
	fig.suptitle(title)
	fig.canvas.draw_idle()
	fig.canvas.flush_events()


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/default.yaml")
	p.add_argument("--dataset_root")
	p.add_argument("--split", choices=SPLITS, default="train")
	p.add_argument("--fire")
	p.add_argument("--current_index", type=int, default=0)
	p.add_argument("--horizon", type=int)
	p.add_argument("--mode", choices=("interactive", "save"), default="interactive")
	p.add_argument("--output_dir", default="artifacts/targets")
	a = p.parse_args()

	config = load_config(a.config)
	thresholds, threshold_meta = resolve_frozen_thresholds(config, a.config, require=False)
	processed = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
	root = Path(a.dataset_root or processed.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
	horizon = int(a.horizon or config.get("target_construction", {}).get("horizon", 10))
	manifest = load_dataset_manifest(root)
	fires = manifest_split_fires(manifest, a.split)
	if not fires:
		raise ValueError(f"No fires found for split {a.split!r}")
	fire_index = fires.index(a.fire) if a.fire in fires else 0
	state = {"fire_index": fire_index, "current_index": max(0, int(a.current_index))}
	labels = ["current surface fuel", "future surface fuel", "surface consumed", "current canopy fuel", "future canopy fuel", "canopy consumed", "energy MW", "energy log", "fire mask"]
	fig = plt.figure(figsize=(12, 11))

	def fire_name():
		return fires[state["fire_index"]]

	def max_current_index():
		fire = load_fire_manifest(root, fire_name())
		return max(0, int(fire["num_processed_frames"]) - horizon - 1)

	def save_current():
		output = Path(a.output_dir) / f"{fire_name()}_current_{state['current_index']:06d}.png"
		output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output, dpi=130)
		print(f"Saved: {output}")

	def redraw():
		state["current_index"] = max(0, min(state["current_index"], max_current_index()))
		arrays = _load_view(root, fire_name(), state["current_index"], horizon)
		threshold_text = str(thresholds) if thresholds else "not available"
		title = f"{fire_name()} | {a.split} | current={state['current_index']} future={state['current_index'] + horizon} | thresholds={threshold_text}"
		_draw(fig, arrays, labels, title)
		print(f"Fire {state['fire_index'] + 1}/{len(fires)}: {fire_name()} | timestamp {state['current_index'] + 1}/{max_current_index() + 1}")
		if thresholds:
			print("Threshold source:", threshold_meta.get("threshold_file") or threshold_meta.get("source"))
			print("Active pixels:", {"union": int(np.sum(arrays[8]))})

	def on_key(event):
		if event.key == "right":
			state["current_index"] += 1
		elif event.key == "left":
			state["current_index"] -= 1
		elif event.key == "n":
			state["fire_index"] = (state["fire_index"] + 1) % len(fires)
			state["current_index"] = 0
		elif event.key == "p":
			state["fire_index"] = (state["fire_index"] - 1) % len(fires)
			state["current_index"] = 0
		elif event.key == "w":
			save_current()
			return
		elif event.key == "h":
			print("Controls: Left/Right=previous/next timestamp | n/p=next/previous fire | w=save | q=quit")
			return
		elif event.key == "q":
			plt.close(fig)
			return
		else:
			return
		redraw()

	if a.mode == "save":
		redraw()
		fig.savefig(Path(a.output_dir) / f"{fire_name()}_current_{state['current_index']:06d}.png", dpi=130)
		plt.close(fig)
	else:
		fig.canvas.mpl_connect("key_press_event", on_key)
		print("Controls: Left/Right=previous/next timestamp | n/p=next/previous fire | w=save | q=quit | h=help")
		redraw()
		plt.show()


if __name__ == "__main__":
	main()
