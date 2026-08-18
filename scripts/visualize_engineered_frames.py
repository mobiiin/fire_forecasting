"""Interactive viewer for processed engineered frames.

Controls in interactive mode:
  Left/Right  previous/next timestamp
  Up/Down     previous/next 9-channel page
  n/p         next/previous fire
  w           save the current view
  h           print controls
  q           quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.processed_dataset import SPLITS, load_channel_manifest, load_fire_manifest, load_frame, load_split_fires


def _available_channels(manifest: dict, group: str) -> list[int]:
	entries = manifest["channels"]
	if group == "core":
		indices = [84, 85, 80, 81, 82, 83]
		for name in ("horizontal_wind_speed_86", "low_level_mean_wind_speed_94", "updraft_95"):
			indices.extend(entry["index"] for entry in entries if entry["name"] == name)
		return [index for index in indices if index < len(entries)]
	if group == "engineered":
		return list(range(int(manifest["num_raw_channels"]), len(entries)))
	if group.startswith("atmosphere_level_"):
		level = int(group.rsplit("_", 1)[1])
		return list(range(level * 10, min(level * 10 + 10, len(entries))))
	return list(range(len(entries)))


def _draw(fig, root: Path, fire: str, frame: int, manifest: dict, group: str, channel_position: int, page_start: int | None = None) -> None:
	"""Redraw one persistent figure; do not close/recreate it on key presses."""
	fig.clear()
	fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.16, hspace=0.28)
	axes = fig.subplots(3, 3)
	x = load_frame(root, fire, frame)
	available = _available_channels(manifest, group)
	if not available:
		raise ValueError(f"No channels available for group {group!r}")
	channel_position = max(0, min(channel_position, len(available) - 1))
	max_page_start = max(0, len(available) - 9)
	if page_start is None:
		page_start = (channel_position // 9) * 9
	page_start = max(0, min(page_start, max_page_start))
	if channel_position < page_start:
		page_start = (channel_position // 9) * 9
	elif channel_position >= page_start + 9:
		page_start = min((channel_position // 9) * 9, max_page_start)
	page = available[page_start:page_start + 9]
	for slot, (ax, channel) in enumerate(zip(axes.flat, page)):
		image = x[channel]
		lo, hi = np.nanpercentile(image, [2, 98])
		im = ax.imshow(image, cmap="viridis", vmin=lo if hi > lo else None, vmax=hi if hi > lo else None)
		entry = manifest["channels"][channel]
		ax.set_title(f"ch {channel}: {entry['name']}\nmin={np.nanmin(image):.4g} max={np.nanmax(image):.4g} mean={np.nanmean(image):.4g}", fontsize=9)
		ax.set_xticks([])
		ax.set_yticks([])
		if page_start + slot == channel_position:
			for spine in ax.spines.values():
				spine.set_color("red")
				spine.set_linewidth(3)
		# Keep a fixed nine-panel layout while browsing. Per-panel colorbars
		# create new layout axes on every redraw and make the image drift.
	for ax in axes.flat[len(page):]:
		ax.axis("off")
	fig.suptitle(f"{fire} | timestamp/frame {frame} | selected channel {available[channel_position]} | group={group} | shape={x.shape}")
	fig.canvas.draw_idle()
	fig.canvas.flush_events()


def render(root: Path, fire: str, frame: int, group: str, channel_position: int, output: Path | None = None):
	manifest = load_channel_manifest(root)
	fig = plt.figure(figsize=(14, 11))
	_draw(fig, root, fire, frame, manifest, group, channel_position)
	if output:
		output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output, dpi=130)
	return fig


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/default.yaml")
	p.add_argument("--dataset_root")
	p.add_argument("--split", choices=SPLITS, default="train")
	p.add_argument("--fire")
	p.add_argument("--frame_index", type=int, default=0)
	p.add_argument("--channel_group", default="all", help="all, core, engineered, or atmosphere_level_N")
	p.add_argument("--channel", type=int, help="Initial channel number")
	p.add_argument("--mode", choices=("interactive", "save"), default="interactive")
	p.add_argument("--output_dir", default="artifacts/engineered_quicklooks")
	p.add_argument("--random", action="store_true")
	a = p.parse_args()

	config = load_config(a.config)
	pc = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
	root = Path(a.dataset_root or pc.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
	fires = load_split_fires(root, a.split)
	if not fires:
		raise ValueError(f"No fires found for split {a.split!r}")
	fire_index = fires.index(a.fire) if a.fire in fires else 0
	manifest = load_channel_manifest(root)
	available = _available_channels(manifest, a.channel_group)
	if not available:
		raise ValueError(f"No channels available for group {a.channel_group!r}")
	channel_position = available.index(a.channel) if a.channel in available else len(available) - 1
	frame = 0 if a.random else max(0, a.frame_index)
	initial_page = (channel_position // 9) * 9 if a.channel in available else max(0, len(available) - 9)
	state = {"fire_index": fire_index, "frame": frame, "channel_position": channel_position, "page_start": initial_page}
	fig = plt.figure(figsize=(14, 11))

	def fire_name() -> str:
		return fires[state["fire_index"]]

	def frame_count() -> int:
		return int(load_fire_manifest(root, fire_name())["num_processed_frames"])

	def available_now() -> list[int]:
		return _available_channels(manifest, a.channel_group)

	def save_current() -> None:
		output = Path(a.output_dir) / f"{fire_name()}_frame_{state['frame']:06d}_channel_{available_now()[state['channel_position']]}.png"
		output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output, dpi=130)
		print(f"Saved: {output}")

	def redraw() -> None:
		state["frame"] = max(0, min(state["frame"], frame_count() - 1))
		state["channel_position"] = max(0, min(state["channel_position"], len(available_now()) - 1))
		max_page_start = max(0, len(available_now()) - 9)
		if state["channel_position"] < state["page_start"]:
			state["page_start"] = (state["channel_position"] // 9) * 9
		elif state["channel_position"] >= state["page_start"] + 9:
			state["page_start"] = min((state["channel_position"] // 9) * 9, max_page_start)
		_draw(fig, root, fire_name(), state["frame"], manifest, a.channel_group, state["channel_position"], state["page_start"])
		print(f"Fire {state['fire_index'] + 1}/{len(fires)}: {fire_name()} | frame {state['frame'] + 1}/{frame_count()} | channel {available_now()[state['channel_position']]}")

	def on_key(event) -> None:
		key = event.key
		if key == "right":
			state["frame"] += 1
		elif key == "left":
			state["frame"] -= 1
		elif key == "up":
			state["channel_position"] -= 9
		elif key == "down":
			state["channel_position"] += 9
		elif key == "n":
			state["fire_index"] = (state["fire_index"] + 1) % len(fires)
			state["frame"] = 0
		elif key == "p":
			state["fire_index"] = (state["fire_index"] - 1) % len(fires)
			state["frame"] = 0
		elif key == "w":
			save_current()
			return
		elif key == "h":
			print("Controls: Left/Right=previous/next timestamp | Up/Down=previous/next 9-channel page | n/p=next/previous fire | w=save | q=quit")
			return
		elif key == "q":
			plt.close(fig)
			return
		else:
			return
		redraw()

	if a.mode == "save":
		redraw()
		output = Path(a.output_dir) / f"{fire_name()}_frame_{state['frame']:06d}_channel_{available_now()[state['channel_position']]}.png"
		fig.savefig(output, dpi=130)
		plt.close(fig)
	else:
		fig.canvas.mpl_connect("key_press_event", on_key)
		print("Controls: Left/Right=previous/next timestamp | Up/Down=previous/next 9-channel page | n/p=next/previous fire | w=save | q=quit | h=help")
		redraw()
		plt.show()


if __name__ == "__main__":
	main()
