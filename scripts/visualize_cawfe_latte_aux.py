"""Visualize auxiliary maps from a full CAWFE-Latte checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, validate_checkpoint_model_compatibility
from src.training.train import _ensure_config_path, _get_device, _infer_input_channels_from_loader


def _config_override(config_path: str) -> dict:
	config = _ensure_config_path(load_config(config_path), config_path)
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = "cawfe_latte"
	model_config["name"] = "cawfe_latte"
	config["model"] = model_config
	config["return_metadata"] = True
	return config


def _select_loader(train_loader, val_loader, test_loader, split: str):
	if split == "train":
		return train_loader
	if split == "val":
		return val_loader
	if split == "test":
		if test_loader is None:
			raise ValueError("No test loader configured.")
		return test_loader
	raise ValueError(f"Unsupported split: {split!r}.")


def _image(tensor: torch.Tensor):
	return tensor.detach().float().cpu().squeeze().numpy()


def _plot_sample(output_dir: Path, index: int, aux_index: int, x: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, aux: dict) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
	latest = x[0, -1]
	axes[0, 0].imshow(_image(latest[84]), cmap="viridis")
	axes[0, 0].set_title("latest surface fuel")
	axes[0, 1].imshow(_image(target[0, 2]), cmap="gray")
	axes[0, 1].set_title("target mask")
	axes[0, 2].imshow(_image(torch.sigmoid(pred[0, 2])), cmap="magma")
	axes[0, 2].set_title("pred mask probability")
	fire_gate = aux.get("fire_gate_map")
	if fire_gate is not None:
		axes[0, 3].imshow(_image(fire_gate[aux_index, -1, 0]), cmap="inferno")
	axes[0, 3].set_title("fire gate")
	wind = aux.get("wind_direction_summary", {})
	if wind.get("wind_speed") is not None:
		axes[1, 0].imshow(_image(wind["wind_speed"][aux_index, -1, 0]), cmap="viridis")
	axes[1, 0].set_title("wind speed")
	if wind.get("wind_cos") is not None:
		axes[1, 1].imshow(_image(wind["wind_cos"][aux_index, -1, 0]), cmap="coolwarm", vmin=-1, vmax=1)
	axes[1, 1].set_title("wind cos")
	if wind.get("wind_sin") is not None:
		axes[1, 2].imshow(_image(wind["wind_sin"][aux_index, -1, 0]), cmap="coolwarm", vmin=-1, vmax=1)
	axes[1, 2].set_title("wind sin")
	axes[1, 3].imshow(_image(pred[0, 3]), cmap="inferno")
	axes[1, 3].set_title("energy log1p pred")
	for axis in axes.ravel():
		axis.set_xticks([])
		axis.set_yticks([])
	operator_stats = aux.get("neural_operator_energy", {})
	if operator_stats:
		fig.suptitle(
			"operator residual norm="
			f"{float(operator_stats.get('operator_residual_norm', torch.tensor(0.0)).detach().cpu()):.4f}"
		)
	fig.savefig(output_dir / f"cawfe_latte_aux_{index:04d}.png", dpi=150)
	plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visualize full CAWFE-Latte auxiliary maps.")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--checkpoint", default="artifacts/checkpoints/cawfe_latte/best_model.pt")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test")
	parser.add_argument("--num_samples", type=int, default=5)
	parser.add_argument("--output_dir", default="outputs/cawfe_latte_aux/")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	config = _config_override(args.config)
	train_loader, val_loader, test_loader = create_dataloaders(config)
	loader = _select_loader(train_loader, val_loader, test_loader, args.split)
	input_channels = _infer_input_channels_from_loader(train_loader)
	device = _get_device(config)
	model = build_model_from_config(config, input_channels=input_channels).to(device)
	checkpoint = load_checkpoint(args.checkpoint, map_location=device)
	validate_checkpoint_model_compatibility(model, checkpoint, args.checkpoint)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	output_dir = Path(args.output_dir)
	saved = 0
	with torch.no_grad():
		for batch in loader:
			x_batch = batch[0].to(device)
			y_batch = batch[1].to(device)
			pred, aux = model(x_batch, return_aux=True)
			for sample_index in range(int(x_batch.shape[0])):
				_plot_sample(
					output_dir,
					saved,
					sample_index,
					x_batch[sample_index : sample_index + 1],
					y_batch[sample_index : sample_index + 1],
					pred[sample_index : sample_index + 1],
					aux,
				)
				saved += 1
				if saved >= int(args.num_samples):
					print(f"saved_aux_figures: {output_dir.resolve()}")
					return
	print(f"saved_aux_figures: {output_dir.resolve()}")


if __name__ == "__main__":
	main()
