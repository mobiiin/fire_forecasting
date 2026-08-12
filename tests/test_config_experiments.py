from __future__ import annotations

from pathlib import Path

import pytest

from src.config import compute_file_sha256, load_config
from scripts.precompute_patch_cache import _acquire_cache_lock, _config_provenance, _release_cache_lock


def test_load_config_supports_base_config_recursive_merge_and_interpolation(tmp_path: Path) -> None:
	base = tmp_path / "base.yaml"
	child = tmp_path / "experiments" / "child.yaml"
	child.parent.mkdir()
	base.write_text(
		"""
paths:
  scratch_root: /scratch/example
  artifacts_root: artifacts/base
  runs_root: ${paths.artifacts_root}/runs
cache:
  cache_dir: ${paths.scratch_root}/cache/base
  cache_version: base
model:
  architecture: convlstm_unet
input_sequence_length: 5
prediction_horizon: 10
""".strip()
		+ "\n",
		encoding="utf-8",
	)
	child.write_text(
		"""
base_config: ../base.yaml
experiment:
  name: demo_child
paths:
  artifacts_root: artifacts/demo_child
cache:
  cache_dir: ${paths.scratch_root}/cache/demo_child
""".strip()
		+ "\n",
		encoding="utf-8",
	)

	config = load_config(child)

	assert config["experiment"]["name"] == "demo_child"
	assert config["paths"]["runs_root"] == "artifacts/demo_child/runs"
	assert config["cache"]["cache_dir"] == "/scratch/example/cache/demo_child"
	assert config["cache"]["cache_version"] == "base"
	assert config["base_config"] == str(base.resolve())
	assert config["_config_sha256"] == compute_file_sha256(child)
	assert config["_base_config_sha256"] == compute_file_sha256(base)


def test_load_config_unknown_interpolation_key_raises_clear_error(tmp_path: Path) -> None:
	config_path = tmp_path / "bad.yaml"
	config_path.write_text(
		"""
paths:
  cache: ${paths.missing}/cache
input_sequence_length: 5
prediction_horizon: 10
""".strip()
		+ "\n",
		encoding="utf-8",
	)

	with pytest.raises(KeyError, match="unknown key"):
		load_config(config_path)


def test_precompute_cache_lock_blocks_second_owner(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	cache_dir.mkdir()
	config_path = tmp_path / "experiment.yaml"
	config_path.write_text("input_sequence_length: 5\nprediction_horizon: 10\n", encoding="utf-8")
	config = {"config_path": str(config_path), "input_sequence_length": 5, "prediction_horizon": 10}

	lock_path = _acquire_cache_lock(cache_dir, config)
	try:
		with pytest.raises(RuntimeError, match="Patch cache lock already exists"):
			_acquire_cache_lock(cache_dir, config)
	finally:
		_release_cache_lock(lock_path)


def test_precompute_config_provenance_hashes_original_config(tmp_path: Path) -> None:
	config_path = tmp_path / "experiment.yaml"
	config_path.write_text("input_sequence_length: 5\nprediction_horizon: 10\n", encoding="utf-8")
	config = {"config_path": str(config_path), "input_sequence_length": 5, "prediction_horizon": 10}

	provenance = _config_provenance(config)

	assert provenance["config_path"] == str(config_path.resolve())
	assert provenance["config_sha256"] == compute_file_sha256(config_path)
	assert provenance["resolved_config_sha256"]
