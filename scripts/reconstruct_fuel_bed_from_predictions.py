"""Reconstruct future fuel-bed maps from multitask model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.visualization.plot_maps import plot_reconstructed_fuel_beds_grid
from scripts.visualize_predictions import (
	_build_dataset_for_split,
	_build_checkpoint_path,
	_build_model,
	_build_multitask_visualization_maps,
	_build_test_loader,
	_ensure_config_path,
	_get_section,
	_metadata_to_dict,
	_resolve_path,
	_select_device,
	_sample_output_name,
)
from src.config import load_config
from src.data.preprocessing import load_normalization_stats
from src.training.checkpoints import load_checkpoint
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the CLI argument parser."""

	parser = argparse.ArgumentParser(description="Reconstruct future fuel-bed maps from multitask predictions.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--num_samples", type=int, default=10, help="Number of chronological test samples to reconstruct.")
	return parser


def main() -> None:
	"""CLI entry point."""

	args = build_argument_parser().parse_args()
	if torch is None:
		raise ImportError("PyTorch is required to reconstruct future fuel beds.")
	config = _ensure_config_path(load_config(args.config), args.config)
	if str(config.get("task_type", "regression")).lower() != "multitask":
		raise ValueError("reconstruct_fuel_bed_from_predictions.py requires task_type='multitask'.")

	set_seed(int(config.get("seed", _get_section(config, "training").get("seed", 42))))
	logger = setup_logging(str(_get_section(config, "logging").get("level", "INFO")))
	config_path_obj = Path(config["config_path"]).expanduser().resolve()
	normalization_path = _get_section(config, "normalization").get("path")
	normalization_stats = None
	if normalization_path:
		resolved_normalization_path = _resolve_path(config_path_obj, normalization_path)
		if resolved_normalization_path.exists():
			normalization_stats = load_normalization_stats(resolved_normalization_path)

	reconstruction_split = "test" if config.get("test_data_dir") not in (None, "", "null") else "val"
	if reconstruction_split == "val":
		print("No external test_data_dir configured; reconstructing from validation samples.")
	test_dataset = _build_dataset_for_split(config, normalization_stats, split=reconstruction_split)
	test_loader = _build_test_loader(test_dataset)
	input_channels = int(next(iter(test_loader))[0].shape[2])
	device = _select_device(config)
	checkpoint_path = _build_checkpoint_path(config)
	checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
	model = _build_model(config, input_channels=input_channels).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	output_dir = _resolve_path(config_path_obj, "outputs/reconstructed_fuel_beds")
	output_dir.mkdir(parents=True, exist_ok=True)
	cmap = str(_get_section(config, "visualization").get("cmap", "inferno"))
	dpi = int(_get_section(config, "visualization").get("dpi", 150))

	print("output channel 0 is predicted surface consumed fuel")
	print("output channel 1 is predicted canopy consumed fuel")
	print("future fuel bed is reconstructed by subtracting predicted consumed fuel from the last input fuel bed")

	max_samples = min(int(args.num_samples), len(test_loader.dataset))
	with torch.no_grad():
		for sample_index, batch in enumerate(test_loader):
			if sample_index >= max_samples:
				break
			x_sample, y_sample, metadata = batch
			metadata_dict = _metadata_to_dict(metadata)
			predicted = model(x_sample.to(device))
			plot_inputs = _build_multitask_visualization_maps(predicted, y_sample, metadata_dict, config, test_dataset)
			output_path = output_dir / _sample_output_name(metadata_dict)
			saved_path = plot_reconstructed_fuel_beds_grid(
				current_surface_fuel=plot_inputs["current_surface_fuel"],
				true_future_surface_fuel=plot_inputs["true_future_surface_fuel"],
				pred_future_surface_fuel=plot_inputs["pred_future_surface_fuel"],
				current_canopy_fuel=plot_inputs["current_canopy_fuel"],
				true_future_canopy_fuel=plot_inputs["true_future_canopy_fuel"],
				pred_future_canopy_fuel=plot_inputs["pred_future_canopy_fuel"],
				output_path=output_path,
				title=f"{reconstruction_split.capitalize()} reconstructed fuel beds | sample {sample_index + 1}/{max_samples}",
				cmap=cmap,
				dpi=dpi,
			)
			logger.info("Saved reconstructed fuel-bed figure: %s", saved_path)


if __name__ == "__main__":
	main()
