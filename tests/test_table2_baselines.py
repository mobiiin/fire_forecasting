"""Regression tests for the locked Table 2 baseline pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from scripts.run_table2_baseline import _complete_test_artifact, _write_run_metadata
from scripts.summarize_table2_results import METRIC_COLUMNS, _aggregate_model
from src.baselines.table2_deterministic import (
	ProcessedHistoryBaselinePredictor,
	linear_extrapolation_from_observed_history,
	load_observed_raw_history,
	persistence_from_observed_history,
)
from src.config import load_config
from src.data.dataset import metadata_batch_to_list
from src.evaluation.held_out_test import evaluate_locked_test
from src.evaluation.table2_protocol import validate_table2_config
from src.models.earthformer_blocks import AxialCuboidAttentionBlock
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function


REGISTRY = Path("configs/table2_baselines/table2_baselines.yaml")


def _minimal_target_config() -> dict:
	return {
		"target_construction": {
			"consumed_fuel": {"clip_negative": True, "clip_to_available_fuel": True},
			"fire_mask": {
				"energy_threshold_mw": 0.1,
				"surface_fuel_threshold": 0.01,
				"canopy_fuel_threshold": 0.01,
			},
			"energy": {"save_mw": True, "save_log1p": True, "use_area_2d": True},
		}
	}


def _raw_frame(surface: float, canopy: float, flux: float) -> np.ndarray:
	frame = np.zeros((86, 4, 4), dtype=np.float32)
	frame[80] = flux
	frame[84] = surface
	frame[85] = canopy
	return frame


def test_table2_configs_match_frozen_protocol() -> None:
	registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
	for baseline, entry in registry["baselines"].items():
		config = load_config(entry["config_path"])
		if entry["learned"]:
			for seed in registry["seeds"]:
				resolved = dict(config)
				training = dict(config["training"])
				training["seed"] = seed
				resolved["training"] = training
				validate_table2_config(resolved, baseline, seed)
		else:
			validate_table2_config(config, baseline, None)


def test_deterministic_formulas_use_observed_intervals_and_return_logits() -> None:
	frames = [
		_raw_frame(10.0, 5.0, 0.0),
		_raw_frame(9.0, 4.8, 1.0e6),
		_raw_frame(7.0, 4.3, 2.0e6),
	]
	area = np.ones((4, 4), dtype=np.float32)
	config = _minimal_target_config()
	persistence = persistence_from_observed_history(frames, area, config)
	linear = linear_extrapolation_from_observed_history(frames, area, config)
	assert persistence.shape == linear.shape == (4, 4, 4)
	assert np.allclose(persistence[0], 2.0)
	assert np.allclose(persistence[1], 0.5)
	assert np.allclose(linear[0], 3.0)
	assert np.allclose(linear[1], 0.8)
	assert np.all(np.abs(persistence[2]) > 1.0)
	assert np.isfinite(persistence).all() and np.isfinite(linear).all()


def test_deterministic_predictors_never_load_future_frame_or_target(tmp_path: Path) -> None:
	root = tmp_path / "dataset"
	frame_dir = root / "fires" / "FIRE" / "frames"
	geometry_dir = root / "fires" / "FIRE" / "geometry"
	frame_dir.mkdir(parents=True)
	geometry_dir.mkdir(parents=True)
	indices = [0, 10, 20, 30, 40]
	for offset, index in enumerate(indices):
		np.savez_compressed(frame_dir / f"frame_{index:06d}.npz", x_raw=_raw_frame(10.0 - offset, 5.0 - 0.2 * offset, float(offset) * 1.0e6))
	# A deliberately extreme forbidden future frame exists but must never be opened.
	np.savez_compressed(frame_dir / "frame_000050.npz", x_raw=_raw_frame(-999.0, -999.0, 9.9e12))
	np.save(geometry_dir / "area_2d.npy", np.ones((4, 4), dtype=np.float32))
	target_path = root / "future_target.npz"
	np.savez_compressed(target_path, fire_mask=np.ones((4, 4), dtype=np.float32))
	record = {
		"fire_name": "FIRE",
		"input_indices": indices,
		"current_index": 40,
		"target_index": 50,
		"target_path": str(target_path),
		"patch": {"y0": 0, "x0": 0, "height": 4, "width": 4},
	}
	opened: list[int] = []
	def tracked_loader(path: Path) -> np.ndarray:
		index = int(path.stem.split("_")[-1])
		opened.append(index)
		if index > 40:
			raise AssertionError("A deterministic predictor attempted to read a future frame.")
		with np.load(path, allow_pickle=False) as archive:
			return np.asarray(archive["x_raw"], dtype=np.float32)

	for method in ("persistence", "linear_extrapolation"):
		predictor = ProcessedHistoryBaselinePredictor(method, root, _minimal_target_config(), frame_loader=tracked_loader)
		before = predictor.predict_one(record)
		np.savez_compressed(frame_dir / "frame_000050.npz", x_raw=_raw_frame(999.0, 999.0, 0.0))
		np.savez_compressed(target_path, fire_mask=np.zeros((4, 4), dtype=np.float32))
		predictor_after = ProcessedHistoryBaselinePredictor(method, root, _minimal_target_config(), frame_loader=tracked_loader)
		after = predictor_after.predict_one(record)
		np.testing.assert_allclose(before, after)
	assert opened and max(opened) == 40
	assert set(opened).issubset(set(indices))


def test_observed_history_loader_rejects_noncausal_indexing(tmp_path: Path) -> None:
	with pytest.raises(ValueError, match="current_index"):
		load_observed_raw_history(
			tmp_path,
			{"fire_name": "F", "input_indices": [0, 10], "current_index": 20},
		)


def test_metadata_batch_reconstructs_collated_fixed_length_histories() -> None:
	metadata = {
		"sample_id": ["left", "right"],
		"input_indices": [torch.tensor([0, 1]), torch.tensor([10, 11]), torch.tensor([20, 21])],
		"patch": {"y0": torch.tensor([4, 8]), "height": torch.tensor([64, 64])},
	}
	items = metadata_batch_to_list(metadata, batch_size=2)
	assert items[0]["input_indices"] == [0, 10, 20]
	assert items[1]["input_indices"] == [1, 11, 21]
	assert items[0]["patch"] == {"y0": 4, "height": 64}


@pytest.mark.parametrize("baseline", ["convlstm_unet", "earthformer_lite", "cawfe_st_mamba"])
def test_learned_architecture_contract_and_backward(baseline: str) -> None:
	registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
	config = load_config(registry["baselines"][baseline]["config_path"])
	# Keep the unit test lightweight; the explicit smoke driver uses 64x64.
	if baseline == "earthformer_lite":
		config["earthformer_lite"] = {**config["earthformer_lite"], "patch_size": 16, "depths": [1, 1], "use_global_vectors": False}
	if baseline == "cawfe_st_mamba":
		config["st_mamba_lite"] = {**config["st_mamba_lite"], "patch_size": 16, "depths": [1, 1], "mamba_backend": "fallback"}
	model = build_model_from_config(config, input_channels=129)
	executed = {"attention": 0}
	hooks = []
	if baseline == "earthformer_lite":
		for module in model.modules():
			if isinstance(module, AxialCuboidAttentionBlock):
				hooks.append(module.register_forward_hook(lambda *_: executed.__setitem__("attention", executed["attention"] + 1)))
	x = torch.randn(1, 5, 129, 16, 16)
	target = torch.zeros(1, 4, 16, 16)
	target[:, 2, 4:8, 4:8] = 1.0
	prediction = model(x)
	assert prediction.shape == (1, 4, 16, 16)
	assert torch.isfinite(prediction).all()
	loss_result = get_loss_function(config)(prediction, target)
	loss = loss_result["total_loss"] if isinstance(loss_result, dict) else loss_result
	assert torch.isfinite(loss)
	loss.backward()
	assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
	if baseline == "earthformer_lite":
		assert executed["attention"] > 0
	if baseline == "cawfe_st_mamba":
		assert getattr(model, "mamba_backend_used") == "fallback"
	for hook in hooks:
		hook.remove()


def test_convlstm_output_depends_on_earlier_sequence_frames() -> None:
	config = load_config("configs/table2_baselines/convlstm_unet.yaml")
	model = build_model_from_config(config, input_channels=129).eval()
	first = torch.zeros(1, 5, 129, 16, 16)
	second = first.clone()
	second[:, 0] = 1.0
	with torch.inference_mode():
		left = model(first)
		right = model(second)
	assert not torch.equal(left, right)


def test_complete_test_artifact_rejects_partial_counts(tmp_path: Path) -> None:
	evaluation = tmp_path / "evaluation"
	evaluation.mkdir()
	(evaluation / "test_per_fire_metrics.csv").write_text("fire_name,sample_count\nF,10\n", encoding="utf-8")
	(evaluation / "test_sample_metrics.csv").write_text("sample_id\nS\n", encoding="utf-8")
	np.savez_compressed(evaluation / "qualitative_test_predictions.npz", sample_id=np.asarray(["S"]))
	payload = {
		"metric_scope": "complete_locked_held_out_test",
		"split": "test",
		"test_used_for_model_selection": False,
		"dataset_sample_count": 10,
		"evaluated_test_samples": 9,
		"unique_sample_id_count": 9,
		"fire_count": 5,
		"no_fire_count": 4,
	}
	(evaluation / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
	assert not _complete_test_artifact(tmp_path)
	payload.update({"evaluated_test_samples": 10, "unique_sample_id_count": 10, "fire_count": 6, "no_fire_count": 4})
	(evaluation / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
	assert _complete_test_artifact(tmp_path)


def test_table2_metadata_never_overwrites_existing_cawfe_metadata(tmp_path: Path) -> None:
	metadata = tmp_path / "metadata"
	metadata.mkdir()
	original = b'{"original": true}\n'
	(metadata / "run_metadata.json").write_bytes(original)
	_write_run_metadata(
		run_dir=tmp_path,
		baseline="cawfe_latte_final",
		seed=42,
		identity={"identity_sha256": "frozen"},
		model=None,
		checkpoint=tmp_path / "checkpoints" / "best_model.pt",
	)
	assert (metadata / "run_metadata.json").read_bytes() == original
	assert (metadata / "table2_test_metadata.json").is_file()


def test_locked_evaluator_writes_only_complete_test_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	class TinyTestDataset(Dataset):
		split = "test"
		root = tmp_path
		sample_index_path = tmp_path / "samples.jsonl"
		records = [
			{"sample_id": f"sample-{index}", "fire_name": "HELD_OUT", "split": "test"}
			for index in range(3)
		]

		def __len__(self) -> int:
			return len(self.records)

		def __getitem__(self, index: int):
			target = torch.zeros(4, 4, 4)
			if index:
				target[2, index, index] = 1.0
			return torch.empty(0), target, self.records[index]

	def qualitative(*_args, **_kwargs):
		return {
			"selected_samples": [
				{"sample_id": "sample-0", "activity_bin": "no_fire"},
				{"sample_id": "sample-1", "activity_bin": "tiny_fire"},
			]
		}

	monkeypatch.setattr("src.evaluation.held_out_test.ensure_qualitative_validation_samples", qualitative)
	loader = DataLoader(TinyTestDataset(), batch_size=2, shuffle=False, drop_last=False)
	result = evaluate_locked_test(
		test_loader=loader,
		config={"table2": {"qualitative_test_index_path": str(tmp_path / "qualitative.json")}},
		run_dir=tmp_path / "run",
		model_name="synthetic",
		seed=None,
		deterministic_predictor=lambda metadata: torch.zeros(len(metadata), 4, 4, 4),
	)
	payload = result["artifact"]
	assert payload["dataset_sample_count"] == payload["evaluated_test_samples"] == 3
	assert payload["unique_sample_id_count"] == 3
	assert payload["fire_count"] + payload["no_fire_count"] == 3
	assert payload["test_used_for_model_selection"] is False
	for name in ("test_metrics.json", "test_per_fire_metrics.csv", "test_sample_metrics.csv", "qualitative_test_predictions.npz"):
		assert (tmp_path / "run" / "evaluation" / name).is_file()


def test_table2_aggregation_uses_mean_and_sample_std(tmp_path: Path) -> None:
	runs = []
	for seed, offset in zip((42, 123, 2026), (0.0, 0.1, 0.2)):
		run = tmp_path / str(seed)
		(run / "evaluation").mkdir(parents=True)
		metrics = {key: 0.5 + offset for key in METRIC_COLUMNS.values()}
		payload = {
			"metric_scope": "complete_locked_held_out_test",
			"split": "test",
			"test_used_for_model_selection": False,
			"seed": seed,
			"dataset_sample_count": 10,
			"evaluated_test_samples": 10,
			"unique_sample_id_count": 10,
			"fire_count": 6,
			"no_fire_count": 4,
			"metrics": metrics,
		}
		evaluation = run / "evaluation"
		(evaluation / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
		(evaluation / "test_per_fire_metrics.csv").write_text("fire_name,sample_count\nF,10\n", encoding="utf-8")
		(evaluation / "test_sample_metrics.csv").write_text("sample_id\nS\n", encoding="utf-8")
		np.savez_compressed(evaluation / "qualitative_test_predictions.npz", sample_id=np.asarray(["S"]))
		runs.append(run)
	row = _aggregate_model("model", "Model", runs)
	assert row["metrics"]["Dice"]["mean"] == pytest.approx(0.6)
	assert row["metrics"]["Dice"]["std"] == pytest.approx(0.1)


def test_default_submission_dry_run_has_exactly_eleven_jobs() -> None:
	environment = dict(os.environ)
	environment.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": "sbatch-must-not-run"})
	result = subprocess.run(
		["bash", "scripts/submit_table2_baselines.sh", "--dry-run"],
		check=True,
		capture_output=True,
		text=True,
		env=environment,
	)
	assert "Selected Table 2 jobs (11):" in result.stdout
	assert result.stdout.count("sbatch-must-not-run --parsable") == 11
	assert "no validation preparation or Slurm submission was performed" in result.stdout
