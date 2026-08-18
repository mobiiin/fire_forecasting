"""Interactive viewer for samples assembled from the processed dataset.

Controls:
  Left/Right  previous/next timestamp for the current fire and patch
  Up/Down      previous/next spatial patch at the current timestamp
  n/p         next/previous fire
  w           save current visualization
  q           quit
  h           print controls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.processed_sample_dataset import ProcessedTemporalPatchDataset


LABELS = [
	"last input: surface fuel",
	"last input: canopy fuel",
	"last input: total flux (80+81+82+83)",
	"target: surface consumed",
	"target: canopy consumed",
	"target: fire mask",
	"target: energy log",
	"derived future surface estimate",
	"derived future canopy estimate",
]


def _arrays(x, y):
	last = x[-1].numpy()
	target = y.numpy()
	return [
		last[84], last[85], last[80] + last[81] + last[82] + last[83],
		target[0], target[1], target[2], target[3],
		last[84] - target[0], last[85] - target[1],
	]


def _draw(fig, arrays, meta):
	fig.clear()
	fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.10, hspace=0.28)
	axes = fig.subplots(3, 3)
	for ax, image, label in zip(axes.flat, arrays, LABELS):
		ax.imshow(image, cmap="magma")
		ax.set_title(label, fontsize=9)
		ax.set_xticks([])
		ax.set_yticks([])
	fig.suptitle(
		f"{meta['fire_name']} | {meta['split']} | pattern={meta['pattern']} | "
		f"inputs={meta['input_indices']} | current={meta['current_index']} | target={meta['target_index']}\n"
		f"patch={meta['patch_id']} | X={tuple(meta['_x_shape'])} | y={tuple(meta['_y_shape'])}"
	)
	fig.canvas.draw_idle()
	fig.canvas.flush_events()


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/default.yaml")
	p.add_argument("--dataset_root")
	p.add_argument("--pattern", default="consecutive5_h10")
	p.add_argument("--split", default="train")
	p.add_argument("--index", type=int, default=0)
	p.add_argument("--mode", choices=("interactive", "save"), default="interactive")
	p.add_argument("--output_dir", default="artifacts/processed_samples")
	a = p.parse_args()
	c = load_config(a.config)
	pc = c.get("processed_dataset", {}) if isinstance(c.get("processed_dataset"), dict) else {}
	root = Path(a.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1"))
	norm = c.get("normalization", {}) if isinstance(c.get("normalization"), dict) else {}
	stats = norm.get("stats_path") if bool(c.get("dataloader", {}).get("normalize_inputs", True)) else None
	ds = ProcessedTemporalPatchDataset(root, root / "indices" / "temporal" / f"samples_{a.pattern}.jsonl", split=a.split, normalization_stats_path=stats, normalize_inputs=False, return_metadata=True)

	# Organize records by fire and patch. Left/Right stays on one patch while
	# Up/Down changes patch at the same timestamp.
	record_indices_by_fire_patch = {}
	for dataset_index, record in enumerate(ds.records):
		record_indices_by_fire_patch.setdefault(record["fire_name"], {}).setdefault(record["patch_id"], []).append((dataset_index, record))
	for patch_records in record_indices_by_fire_patch.values():
		for records in patch_records.values():
			records.sort(key=lambda item: int(item[1]["current_index"]))
	fires = list(record_indices_by_fire_patch)
	if not fires:
		raise ValueError(f"No samples found for split {a.split!r}")
	fire_index = 0
	initial_dataset_index = max(0, min(a.index, len(ds) - 1))
	initial_fire = ds.records[initial_dataset_index]["fire_name"]
	fire_index = fires.index(initial_fire)
	initial_patch = ds.records[initial_dataset_index]["patch_id"]
	state = {"fire_index": fire_index, "patch_id": initial_patch, "sample_position": 0}
	fig = plt.figure(figsize=(12, 11))

	def current_pairs():
		patches = record_indices_by_fire_patch[fires[state["fire_index"]]]
		return patches[state["patch_id"]]

	def patch_ids():
		return list(record_indices_by_fire_patch[fires[state["fire_index"]]])

	def current_dataset_index():
		state["sample_position"] = max(0, min(state["sample_position"], len(current_pairs()) - 1))
		return current_pairs()[state["sample_position"]][0]

	def select_patch(offset):
		ids = patch_ids()
		position = ids.index(state["patch_id"])
		new_position = max(0, min(position + offset, len(ids) - 1))
		if new_position == position:
			return
		current_index = current_pairs()[state["sample_position"]][1]["current_index"]
		state["patch_id"] = ids[new_position]
		new_records = current_pairs()
		state["sample_position"] = min(range(len(new_records)), key=lambda i: abs(int(new_records[i][1]["current_index"]) - int(current_index)))

	def show_current():
		x, y, meta = ds[current_dataset_index()]
		meta = dict(meta)
		meta["_x_shape"] = tuple(x.shape)
		meta["_y_shape"] = tuple(y.shape)
		print(meta)
		print(f"X shape {tuple(x.shape)} y shape {tuple(y.shape)} mask fraction {float(y[2].mean())}")
		_draw(fig, _arrays(x, y), meta)
		pairs = current_pairs()
		print(f"Fire {state['fire_index'] + 1}/{len(fires)}: {fires[state['fire_index']]} | patch {patch_ids().index(state['patch_id']) + 1}/{len(patch_ids())} | timestamp {state['sample_position'] + 1}/{len(pairs)}")

	def save_current():
		meta = ds[current_dataset_index()][2]
		output = Path(a.output_dir) / f"{meta['sample_id']}.png"
		output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output, dpi=130)
		print(f"Saved: {output}")

	def on_key(event):
		if event.key == "right":
			state["sample_position"] = min(state["sample_position"] + 1, len(current_pairs()) - 1)
		elif event.key == "left":
			state["sample_position"] = max(state["sample_position"] - 1, 0)
		elif event.key == "up":
			select_patch(-1)
		elif event.key == "down":
			select_patch(1)
		elif event.key == "n":
			state["fire_index"] = (state["fire_index"] + 1) % len(fires)
			state["patch_id"] = patch_ids()[0]
			state["sample_position"] = 0
		elif event.key == "p":
			state["fire_index"] = (state["fire_index"] - 1) % len(fires)
			state["patch_id"] = patch_ids()[0]
			state["sample_position"] = 0
		elif event.key == "w":
			save_current(); return
		elif event.key == "h":
			print("Controls: Left/Right=previous/next timestamp | Up/Down=previous/next patch | n/p=next/previous fire | w=save | q=quit")
			return
		elif event.key == "q":
			plt.close(fig); return
		else:
			return
		show_current()

	if a.mode == "save":
		show_current(); save_current(); plt.close(fig)
	else:
		fig.canvas.mpl_connect("key_press_event", on_key)
		print("Controls: Left/Right=previous/next timestamp | Up/Down=previous/next patch | n/p=next/previous fire | w=save | q=quit | h=help")
		show_current()
		plt.show()


if __name__ == "__main__":
	main()
