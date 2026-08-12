"""Run a bounded train-like diagnostic pass through the shared pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dependency
	torch = None

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.model_factory import build_model_from_config
from src.training.hardware import choose_amp_dtype, configure_torch_backend, get_cuda_device_info, get_performance_config
from src.training.losses import get_loss_function
from src.training.train import (
	_apply_auto_hardware_tuning,
	_apply_dataloader_worker_tuning,
	_as_batch,
	_build_optimizer,
	_ensure_config_path,
	_get_device,
	_infer_input_channels_from_loader,
	_loader_summary,
	_maybe_probe_auto_batch_size,
	_run_epoch,
	resolve_validation_policy,
)
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _override_architecture(config: dict, architecture: str | None) -> None:
	if not architecture:
		return
	model_config = dict(config.get("model", {}))
	model_config["architecture"] = architecture
	model_config["name"] = architecture
	config["model"] = model_config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Diagnose the shared wildfire training pipeline.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
	parser.add_argument("--model_architecture", default=None, help="Optional architecture override.")
	parser.add_argument("--num_batches", type=int, default=50, help="Number of train-like batches to run.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	if torch is None:
		raise SystemExit("PyTorch is not installed in this environment; activate the training environment first.")
	config = _ensure_config_path(load_config(args.config), args.config)
	_override_architecture(config, args.model_architecture)
	training_config = config.setdefault("training", {})
	performance_config = get_performance_config(config)
	performance_section = training_config.setdefault("performance", {})
	performance_section["log_timing"] = True
	performance_section["timing_log_every_n_batches"] = max(1, min(10, int(args.num_batches)))
	set_seed(int(training_config.get("seed", config.get("seed", 42))))

	log_dir = Path(config.get("logging", {}).get("log_dir", "./artifacts/logs")).expanduser().resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging("INFO", str(log_dir / "diagnose_training_pipeline.log"))
	configure_torch_backend(config)
	_apply_dataloader_worker_tuning(config, logger)
	_apply_auto_hardware_tuning(config, logger)
	performance_config = get_performance_config(config)
	if bool(performance_config.get("auto_batch_size", False)) and torch.cuda.is_available():
		current_batch_size = int(training_config.get("batch_size", config.get("batch_size", 1)))
		probe_batch_size = max(1, int(performance_config.get("auto_batch_probe_batch_size", min(current_batch_size, 8))))
		if probe_batch_size < current_batch_size:
			training_config["_auto_batch_original_batch_size"] = current_batch_size
			training_config["batch_size"] = probe_batch_size
			config["batch_size"] = probe_batch_size

	train_loader, val_loader, test_loader = create_dataloaders(config)
	input_channels = _infer_input_channels_from_loader(train_loader)
	device = _get_device(config)
	if _maybe_probe_auto_batch_size(config, train_loader, input_channels, device, logger):
		train_loader, val_loader, test_loader = create_dataloaders(config)
		input_channels = _infer_input_channels_from_loader(train_loader)

	model = build_model_from_config(config, input_channels=input_channels).to(device)
	criterion = get_loss_function(config)
	optimizer = _build_optimizer(model, config)
	amp_dtype = choose_amp_dtype(config, device)
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)

	first_batch = next(iter(train_loader))
	x_batch, y_batch = _as_batch(first_batch)
	output_channels = int(y_batch.shape[1])
	results = _run_epoch(
		model=model,
		loader=train_loader,
		criterion=criterion,
		config=config,
		device=device,
		input_sequence_length=int(x_batch.shape[1]),
		input_channels=input_channels,
		output_channels=output_channels,
		train=True,
		optimizer=optimizer,
		scaler=None,
		amp_dtype=amp_dtype,
		gradient_accumulation_steps=max(1, int(training_config.get("gradient_accumulation_steps", 1))),
		max_batches=max(1, int(args.num_batches)),
		logger=logger,
		epoch_number=0,
		timing_csv_path=log_dir / "diagnose_training_timing.csv",
	)

	peak_memory_gb = 0.0
	if device.type == "cuda":
		peak_memory_gb = float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
	effective_batch = int(training_config.get("batch_size", config.get("batch_size", 1))) * max(
		1,
		int(training_config.get("gradient_accumulation_steps", 1)),
	)
	print(f"gpu: {get_cuda_device_info()}")
	print(f"precision: {amp_dtype or 'fp32'}")
	print(f"input_sequence_length: {int(config['input_sequence_length'])}")
	print(f"prediction_horizon: {int(config['prediction_horizon'])}")
	print(f"batch_size: {training_config.get('batch_size', config.get('batch_size'))}")
	print(f"effective_batch_size: {effective_batch}")
	print(f"train_loader: {_loader_summary(train_loader)}")
	print(f"val_loader: {_loader_summary(val_loader)}")
	if test_loader is not None:
		print(f"test_loader: {_loader_summary(test_loader)}")
	validation_policy = resolve_validation_policy(config, val_loader=val_loader, logger=logger)
	print("validation:")
	print(f"  mode: {validation_policy['validation_mode']}")
	print(f"  scope: {validation_policy['validation_scope']}")
	print(f"  batches_used: {validation_policy['validation_batches_used']}")
	print(f"  is_full_validation: {validation_policy['is_full_validation']}")
	print(f"train_loss: {results['train_loss']:.6f}")
	print(f"avg_data_wait_s: {results.get('train_data_wait_avg', 0.0):.6f}")
	print(f"avg_h2d_s: {results.get('train_h2d_avg', 0.0):.6f}")
	print(f"avg_forward_s: {results.get('train_forward_avg', 0.0):.6f}")
	print(f"avg_backward_s: {results.get('train_backward_avg', 0.0):.6f}")
	print(f"samples_per_second: {results.get('train_samples_per_second', 0.0):.2f}")
	print(f"peak_gpu_memory_gb: {peak_memory_gb:.2f}")


if __name__ == "__main__":
	main()
