"""Test ConvLSTM U-Net spatial-size compatibility without loading dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.training.train import _ensure_config_path, _get_device


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the command-line parser."""

	parser = argparse.ArgumentParser(description="Test ConvLSTM U-Net compatibility with different spatial sizes.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument(
		"--checkpoint",
		default=None,
		help="Optional explicit checkpoint path. If omitted, the script will try the config-derived best checkpoint and continue if it is missing.",
	)
	return parser


def _resolve_checkpoint_path(config, explicit_checkpoint: str | None) -> Path | None:
	if explicit_checkpoint:
		path = Path(explicit_checkpoint).expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {path}")
		return path

	config_path = Path(config["config_path"]).expanduser().resolve()
	checkpoint_config = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	latest_path, best_path = latest_and_best_checkpoint_paths((config_path.parent / checkpoint_path).resolve())
	if best_path.exists():
		return best_path
	if latest_path.exists():
		return latest_path
	return None


def main() -> None:
	args = build_argument_parser().parse_args()
	if torch is None:
		raise ImportError("PyTorch is required for test_spatial_size_compatibility.py.")

	config = _ensure_config_path(load_config(args.config), args.config)
	device = _get_device(config)
	input_channels = int(config.get("model", {}).get("input_channels", config.get("input_channel_count", 0)))
	if input_channels <= 0:
		raise ValueError("model.input_channels must be positive for spatial compatibility testing.")

	model = build_model_from_config(config, input_channels=input_channels).to(device)
	checkpoint_path = _resolve_checkpoint_path(config, args.checkpoint)
	if checkpoint_path is not None:
		checkpoint = load_checkpoint(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	batch_size = 1
	sequence_length = int(config["input_sequence_length"])
	output_lines = [
		"ConvLSTM U-Net spatial-size compatibility report",
		f"device: {device}",
		f"checkpoint: {checkpoint_path if checkpoint_path is not None else 'none'}",
		f"input_sequence_length: {sequence_length}",
		f"input_channels: {input_channels}",
		"",
	]

	for size in (72, 80, 96, 128, 144):
		dummy_input = torch.zeros((batch_size, sequence_length, input_channels, size, size), dtype=torch.float32, device=device)
		try:
			with torch.no_grad():
				output = model(dummy_input)
			line = f"{size}x{size}: PASS -> output shape {tuple(output.shape)}"
		except Exception as exc:  # pragma: no cover - exercised in real model compatibility failures
			line = f"{size}x{size}: FAIL -> {type(exc).__name__}: {exc}"
		print(line)
		output_lines.append(line)

	report_path = Path("artifacts/logs/spatial_size_compatibility.txt").expanduser().resolve()
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
	print(f"report: {report_path}")


if __name__ == "__main__":
	main()
