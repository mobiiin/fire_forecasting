#!/usr/bin/env python3
"""Small validation-only smoke tests for Table 2 baselines; never iterate test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor
from src.config import load_config
from src.data.dataset import metadata_batch_to_list
from src.data.processed_sample_dataset import ProcessedTemporalPatchDataset
from src.models.model_factory import build_model_from_config
from src.training.batch_utils import unpack_batch
from src.training.losses import get_loss_function
from src.training.model_outputs import extract_prediction


REGISTRY = Path("configs/table2_baselines/table2_baselines.yaml")


def _smoke_loader(config: dict[str, Any], split: str) -> DataLoader:
	"""Build only the requested train/val split; never construct a test dataset."""

	dataloader = config["dataloader"]
	root = Path(str(dataloader.get("dataset_root", config["processed_dataset"]["root"]))).expanduser().resolve()
	pattern = str(dataloader["sample_pattern"])
	dataset = ProcessedTemporalPatchDataset(
		dataset_root=root,
		sample_index_path=root / "indices" / "temporal" / f"samples_{pattern}.jsonl",
		split=split,
		normalization_stats_path=config["normalization"]["stats_path"],
		normalize_inputs=bool(dataloader.get("normalize_inputs", True)),
		input_key=str(dataloader.get("input_key", "x_engineered")),
		return_metadata=True,
		return_terrain=False,
	)
	return DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)


def _smoke_config(path: str | Path, baseline: str) -> dict[str, Any]:
	config = load_config(path)
	config["return_metadata"] = True
	dataloader = dict(config.get("dataloader", {}))
	dataloader["return_metadata"] = True
	config["dataloader"] = dataloader
	training = dict(config.get("training", {}))
	training.update({"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False})
	data_loader = dict(config.get("data_loader", {}))
	data_loader.update({"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "drop_last": False})
	for split in ("train", "val"):
		split_options = dict(data_loader.get(split, {}))
		split_options.update({"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "drop_last": False})
		data_loader[split] = split_options
	config["data_loader"] = data_loader
	training["max_train_batches"] = 2
	training["max_epochs"] = 1
	config["training"] = training
	config["seed"] = 42
	return config


def smoke_deterministic(baseline: str, config: dict[str, Any]) -> None:
	val_loader = _smoke_loader(config, "val")
	predictor = ProcessedHistoryBaselinePredictor(
		baseline,
		config["processed_dataset"]["root"],
		config,
	)
	visited = 0
	for batch in val_loader:
		_x, target, extra = unpack_batch(batch)
		metadata = metadata_batch_to_list(extra["metadata"], batch_size=int(target.shape[0]))
		prediction = predictor(metadata)
		assert prediction.shape == target.shape
		assert torch.isfinite(prediction).all()
		visited += int(target.shape[0])
		if visited >= 2:
			break
	if visited < 2:
		raise RuntimeError(f"{baseline} smoke test did not visit two validation samples.")
	print(f"PASS {baseline}: {visited} validation samples; test dataset was not constructed or iterated")


def smoke_learned(baseline: str, config: dict[str, Any]) -> None:
	if baseline == "cawfe_st_mamba" and not torch.cuda.is_available():
		raise RuntimeError(
			"The publishable CAWFE-ST-Mamba smoke test requires a CUDA GPU for the official "
			"mamba_ssm kernels. Submit scripts/slurm_smoke_table2_baselines_a10080.sh; "
			"the debug fallback is intentionally not used."
		)
	train_loader = _smoke_loader(config, "train")
	val_loader = _smoke_loader(config, "val")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_model_from_config(config, input_channels=129).to(device)
	if baseline == "cawfe_st_mamba" and getattr(model, "mamba_backend_used", None) != "mamba_ssm":
		raise RuntimeError("CAWFE-ST-Mamba smoke test is not using the official mamba_ssm backend.")
	criterion = get_loss_function(config)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
	model.train()
	train_batches = 0
	for batch in train_loader:
		x, target, _extra = unpack_batch(batch)
		x = x.to(device).float()
		target = target.to(device).float()
		optimizer.zero_grad(set_to_none=True)
		prediction = extract_prediction(model(x))
		if prediction.shape != (int(x.shape[0]), 4, 64, 64) or not torch.isfinite(prediction).all():
			raise RuntimeError(f"{baseline} produced invalid shape/values: {tuple(prediction.shape)}")
		loss_result = criterion(prediction, target)
		loss = loss_result["total_loss"] if isinstance(loss_result, dict) else loss_result
		if not torch.isfinite(loss):
			raise RuntimeError(f"{baseline} produced non-finite training loss.")
		loss.backward()
		if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
			raise RuntimeError(f"{baseline} backward pass produced no gradients.")
		optimizer.step()
		train_batches += 1
		if train_batches == 2:
			break
	model.eval()
	val_batches = 0
	with torch.inference_mode():
		for batch in val_loader:
			x, target, _extra = unpack_batch(batch)
			prediction = extract_prediction(model(x.to(device).float()))
			if prediction.shape != (int(x.shape[0]), 4, 64, 64) or not torch.isfinite(prediction).all():
				raise RuntimeError(f"{baseline} validation produced invalid output.")
			_ = target
			val_batches += 1
			if val_batches == 2:
				break
	if (train_batches, val_batches) != (2, 2):
		raise RuntimeError(f"{baseline} smoke counts were train={train_batches}, val={val_batches}.")
	print(
		f"PASS {baseline}: 2 train + 2 validation batches, parameters="
		f"{sum(parameter.numel() for parameter in model.parameters())}, device={device}; test dataset was not constructed or iterated"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--baseline", action="append", default=[])
	args = parser.parse_args()
	registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
	selected = args.baseline or list(registry["baselines"])
	for baseline in selected:
		if baseline not in registry["baselines"]:
			raise ValueError(f"Unknown baseline {baseline!r}.")
		entry = registry["baselines"][baseline]
		config = _smoke_config(entry["config_path"], baseline)
		if bool(entry["learned"]):
			smoke_learned(baseline, config)
		else:
			smoke_deterministic(baseline, config)


if __name__ == "__main__":
	main()

