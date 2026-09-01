from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_training_outputs import verify_run
from src.training.run_manager import RunManager
from src.training.train import apply_training_cli_overrides


def _config(tmp_path: Path, run_name: str = "demo") -> dict:
	return {
		"config_path": str(tmp_path / "configs" / "default.yaml"),
		"training": {
			"run_name": run_name,
			"overwrite_run": False,
			"output": {
				"root_dir": str(tmp_path / "runs"),
				"checkpoint_root": str(tmp_path / "checkpoints"),
			},
		},
	}


def test_run_manager_creates_unique_run_directory_suffix(tmp_path: Path) -> None:
	first = RunManager(_config(tmp_path), "convlstm_unet")
	second = RunManager(_config(tmp_path), "convlstm_unet")

	assert first.create_run_dir().name == "demo"
	assert second.create_run_dir().name == "demo_v2"
	assert first.checkpoint_path("best") == tmp_path / "runs" / "convlstm_unet" / "demo" / "checkpoints" / "best_model.pt"
	assert second.log_path("training").name == "training_log.csv"


def test_run_manager_resolves_relative_output_root_from_repo_root(tmp_path: Path) -> None:
	manager = RunManager(
		{
			"config_path": str(tmp_path / "configs" / "default.yaml"),
			"training": {
				"run_name": "relative_paths",
				"output": {
					"root_dir": "artifacts/test_runs_relative",
					"checkpoint_root": "artifacts/test_checkpoints_relative",
				},
			},
		},
		"convlstm_unet",
	)

	assert manager.root_dir == Path.cwd() / "artifacts" / "test_runs_relative"
	assert manager.checkpoint_root == Path.cwd() / "artifacts" / "test_checkpoints_relative"


def test_run_manager_saves_exact_and_resolved_config_artifacts(tmp_path: Path) -> None:
	config_file = tmp_path / "experiment.yaml"
	config_text = "experiment:\n  name: artifact_demo\ninput_sequence_length: 5\nprediction_horizon: 10\n"
	config_file.write_text(config_text, encoding="utf-8")
	config = {
		"config_path": str(config_file),
		"_config_path": str(config_file),
		"experiment": {"name": "artifact_demo"},
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"training": {
			"run_name": "artifact_demo",
			"output": {
				"root_dir": str(tmp_path / "runs"),
				"checkpoint_root": str(tmp_path / "checkpoints"),
				"save_original_config": True,
				"save_resolved_config": True,
			},
		},
	}
	manager = RunManager(config, "convlstm_unet")

	paths = manager.save_configs(original_config={"old": True}, resolved_config=config)

	original_path = Path(paths["original_config_path"])
	resolved_path = Path(paths["resolved_config_path"])
	config_used_path = Path(paths["config_used_path"])
	metadata_path = manager.metadata_path("config_metadata.json")
	metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

	assert original_path.read_text(encoding="utf-8") == config_text
	assert resolved_path.exists()
	assert config_used_path.exists()
	assert metadata["config_path_passed"] == str(config_file.resolve())
	assert metadata["config_sha256"]
	assert metadata["resolved_config_sha256"]
	assert metadata["experiment_name"] == "artifact_demo"


def test_run_manager_does_not_write_compatibility_checkpoints(tmp_path: Path) -> None:
	manager = RunManager(_config(tmp_path, run_name="copy_test"), "weatherformer_lite")
	source = manager.checkpoint_path("best")
	source.write_bytes(b"checkpoint")

	written = manager.copy_checkpoint_to_compatibility(source, "best")

	assert written == []
	assert not (tmp_path / "checkpoints" / "weatherformer_lite" / "copy_test" / "best_model.pt").exists()
	assert not (tmp_path / "checkpoints" / "weatherformer_lite" / "best_model.pt").exists()


def test_apply_training_cli_overrides_sets_nested_output() -> None:
	config = apply_training_cli_overrides(
		{"training": {"output": {"root_dir": "old"}}},
		run_name="manual",
		output_root="new_runs",
		overwrite_run=True,
	)

	assert config["training"]["run_name"] == "manual"
	assert config["training"]["output"]["root_dir"] == "new_runs"
	assert config["training"]["overwrite_run"] is True


def test_verify_training_outputs_accepts_valid_synthetic_run(tmp_path: Path) -> None:
	torch = pytest.importorskip("torch")
	run_dir = tmp_path / "runs" / "convlstm_unet" / "convlstm_unet_test"
	for relative in ("checkpoints", "logs", "figures", "configs", "metadata"):
		(run_dir / relative).mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"architecture": "convlstm_unet",
			"run_name": "convlstm_unet_test",
			"model_state_dict": {},
		},
		run_dir / "checkpoints" / "best_model.pt",
	)
	torch.save(
		{
			"architecture": "convlstm_unet",
			"run_name": "convlstm_unet_test",
			"model_state_dict": {},
		},
		run_dir / "checkpoints" / "latest_model.pt",
	)
	(run_dir / "logs" / "training_log.csv").write_text("epoch,train_loss,val_loss\n1,1.0,1.2\n", encoding="utf-8")
	(run_dir / "figures" / "loss_curves.png").write_bytes(b"png")
	(run_dir / "configs" / "resolved_config.yaml").write_text("ok: true\n", encoding="utf-8")
	(run_dir / "metadata" / "run_summary.json").write_text(
		'{"status": "completed", "best_epoch": 1, "best_metric_value": 1.2}\n',
		encoding="utf-8",
	)

	ok, messages, info = verify_run(run_dir)

	assert ok, messages
	assert info["run_name"] == "convlstm_unet_test"
