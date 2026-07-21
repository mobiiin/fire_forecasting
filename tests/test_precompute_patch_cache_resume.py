from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import precompute_patch_cache as precompute


class _ArrayTensor:
	def __init__(self, array: np.ndarray) -> None:
		self._array = np.asarray(array, dtype=np.float32)

	def detach(self) -> "_ArrayTensor":
		return self

	def cpu(self) -> "_ArrayTensor":
		return self

	def numpy(self) -> np.ndarray:
		return self._array


class _FakePatchDataset:
	total_input_channels = 3
	base_input_channel_count = 3
	fuel_flux_engineered_channel_count = 0
	atmospheric_engineered_channel_count = 0
	energy_history_channel_count = 0
	engineered_channel_count = 0

	def __init__(self, length: int) -> None:
		self.length = int(length)

	def __len__(self) -> int:
		return self.length

	def __getitem__(self, index: int):
		x = np.full((2, 3, 4, 4), float(index), dtype=np.float32)
		y = np.zeros((4, 4, 4), dtype=np.float32)
		y[0].fill(float(index))
		metadata = {
			"dataset_id": 0,
			"dataset_name": "synthetic_fire",
			"sample_index": int(index),
			"current_idx": int(index + 1),
			"future_idx": int(index + 2),
			"patch": {"y0": 0, "y1": 4, "x0": 0, "x1": 4},
		}
		return _ArrayTensor(x), _ArrayTensor(y), metadata


def _config(cache_dir: Path) -> dict:
	return {
		"config_path": str((cache_dir / "config.yaml").resolve()),
		"input_sequence_length": 2,
		"prediction_horizon": 1,
		"target_channel": 0,
		"input_channel_count": 3,
		"task_type": "multitask",
		"patching": {
			"enabled": True,
			"patch_size": 4,
			"patch_height": 4,
			"patch_width": 4,
		},
		"cache": {
			"cache_dir": str(cache_dir),
			"shard_format": "npz",
			"samples_per_shard": 2,
			"compressed": False,
			"save_metadata": True,
			"save_preview_images": False,
		},
		"model": {
			"input_channels": 3,
			"output_channels": 4,
		},
	}


def _metadata_row(sample_index: int, local_index: int) -> dict:
	return {
		"split": "train",
		"fire_name": "synthetic_fire",
		"dataset_id": 0,
		"sample_index": int(sample_index),
		"current_idx": int(sample_index + 1),
		"future_idx": int(sample_index + 2),
		"patch": {"y0": 0, "y1": 4, "x0": 0, "x1": 4},
		"patch_type": "random",
		"x_shape": [2, 3, 4, 4],
		"y_shape": [4, 4, 4],
		"sample_id": f"train:synthetic_fire:{sample_index}:0:0",
		"shard": "train/shard_000000.npz",
		"local_index": int(local_index),
	}


def _write_completed_shard(
	split_dir: Path,
	sample_indices: list[int],
	*,
	write_per_shard_metadata: bool = True,
	write_aggregate_metadata: bool = False,
	aggregate_rows: int | None = None,
) -> None:
	split_dir.mkdir(parents=True, exist_ok=True)
	x = np.stack([np.full((2, 3, 4, 4), float(index), dtype=np.float32) for index in sample_indices], axis=0)
	y = np.zeros((len(sample_indices), 4, 4, 4), dtype=np.float32)
	for local_index, sample_index in enumerate(sample_indices):
		y[local_index, 0].fill(float(sample_index))
	np.savez(
		split_dir / "shard_000000.npz",
		X=x,
		y=y,
		sample_ids=np.asarray([f"train:synthetic_fire:{sample_index}:0:0" for sample_index in sample_indices]),
		dataset_ids=np.zeros(len(sample_indices), dtype=np.int64),
		patch_y0=np.zeros(len(sample_indices), dtype=np.int64),
		patch_x0=np.zeros(len(sample_indices), dtype=np.int64),
		sample_indices=np.asarray(sample_indices, dtype=np.int64),
	)
	rows = [_metadata_row(sample_index, local_index) for local_index, sample_index in enumerate(sample_indices)]
	content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
	if write_per_shard_metadata:
		(split_dir / "shard_000000.metadata.jsonl").write_text(content, encoding="utf-8")
	if write_aggregate_metadata:
		selected_rows = rows if aggregate_rows is None else rows[:aggregate_rows]
		(split_dir / "metadata.jsonl").write_text(
			"".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
			encoding="utf-8",
		)


def test_resume_uses_complete_shards_when_checkpoint_is_ahead(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	split_dir = cache_dir / "train"
	_write_completed_shard(split_dir, [0, 1])
	precompute._save_resume_checkpoint(split_dir, "train", next_episode=3, next_shard_index=1, total_samples=3)

	manifest = {"shards": {"train": [], "val": [], "test": []}}
	precompute._precompute_split(_config(cache_dir), _FakePatchDataset(length=5), "train", cache_dir, manifest)

	assert [entry["num_samples"] for entry in manifest["shards"]["train"]] == [2, 2, 1]
	assert manifest["num_train_patches"] == 5

	metadata_rows = [
		json.loads(line)
		for line in (split_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	assert [row["sample_index"] for row in metadata_rows] == [0, 1, 2, 3, 4]
	assert metadata_rows[0]["input_indices"] == [0, 1]
	assert metadata_rows[0]["last_input_idx"] == 1
	assert metadata_rows[0]["target_idx"] == 2
	assert metadata_rows[0]["prediction_horizon"] == 1
	assert metadata_rows[0]["target_offset_from_start"] == 2

	shard_sample_indices: list[int] = []
	for shard_path in sorted(split_dir.glob("shard_*.npz")):
		with np.load(shard_path, allow_pickle=False) as shard:
			shard_sample_indices.extend(int(value) for value in shard["sample_indices"])
	assert shard_sample_indices == [0, 1, 2, 3, 4]

	checkpoint = json.loads((split_dir / "resume_checkpoint.json").read_text(encoding="utf-8"))
	assert checkpoint["next_episode"] == 5
	assert checkpoint["next_shard_index"] == 3


def test_resume_materializes_legacy_split_metadata(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	split_dir = cache_dir / "train"
	_write_completed_shard(
		split_dir,
		[0, 1],
		write_per_shard_metadata=False,
		write_aggregate_metadata=True,
	)

	manifest = {"shards": {"train": [], "val": [], "test": []}}
	precompute._precompute_split(_config(cache_dir), _FakePatchDataset(length=5), "train", cache_dir, manifest)

	assert (split_dir / "shard_000000.metadata.jsonl").exists()
	assert [entry["num_samples"] for entry in manifest["shards"]["train"]] == [2, 2, 1]

	metadata_rows = [
		json.loads(line)
		for line in (split_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	assert [row["sample_index"] for row in metadata_rows] == [0, 1, 2, 3, 4]
	assert all(row["input_sequence_length"] == 2 for row in metadata_rows)
	assert all(row["prediction_horizon"] == 1 for row in metadata_rows)


def test_resume_reconstructs_partial_legacy_shard_metadata(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	split_dir = cache_dir / "train"
	_write_completed_shard(
		split_dir,
		[0, 1],
		write_per_shard_metadata=False,
		write_aggregate_metadata=True,
		aggregate_rows=1,
	)

	manifest = {"shards": {"train": [], "val": [], "test": []}}
	precompute._precompute_split(_config(cache_dir), _FakePatchDataset(length=5), "train", cache_dir, manifest)

	metadata_rows = [
		json.loads(line)
		for line in (split_dir / "shard_000000.metadata.jsonl").read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	assert [row["sample_index"] for row in metadata_rows] == [0, 1]
	assert metadata_rows[1]["patch_type"] == "reconstructed"
	assert metadata_rows[1]["input_indices"] == [1, 2]
	assert metadata_rows[1]["last_input_idx"] == 2
	assert metadata_rows[1]["target_idx"] == 3
	assert [entry["num_samples"] for entry in manifest["shards"]["train"]] == [2, 2, 1]


def test_resume_from_episode_cannot_skip_past_complete_shards(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	split_dir = cache_dir / "train"
	_write_completed_shard(split_dir, [0, 1])

	manifest = {"shards": {"train": [], "val": [], "test": []}}
	try:
		precompute._precompute_split(
			_config(cache_dir),
			_FakePatchDataset(length=5),
			"train",
			cache_dir,
			manifest,
			resume_from_episode=3,
		)
	except RuntimeError as exc:
		assert "ahead of the last complete cached patch index" in str(exc)
	else:
		raise AssertionError("Expected resume_from_episode ahead of complete shards to fail.")
