from __future__ import annotations

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


def test_run_manager_copies_compatibility_checkpoints_atomically(tmp_path: Path) -> None:
	manager = RunManager(_config(tmp_path, run_name="copy_test"), "weatherformer_lite")
	source = manager.checkpoint_path("best")
	source.write_bytes(b"checkpoint")

	written = manager.copy_checkpoint_to_compatibility(source, "best")

	assert tmp_path / "checkpoints" / "weatherformer_lite" / "copy_test" / "best_model.pt" in written
	assert tmp_path / "checkpoints" / "weatherformer_lite" / "best_model.pt" in written
	assert all(path.read_bytes() == b"checkpoint" for path in written)


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
	run_dir = tmp_path / "runs" / "cawfe_latte" / "cawfe_latte_test"
	for relative in ("checkpoints", "logs", "figures", "configs", "metadata"):
		(run_dir / relative).mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"architecture": "cawfe_latte",
			"run_name": "cawfe_latte_test",
			"model_state_dict": {},
		},
		run_dir / "checkpoints" / "best_model.pt",
	)
	torch.save(
		{
			"architecture": "cawfe_latte",
			"run_name": "cawfe_latte_test",
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
	assert info["run_name"] == "cawfe_latte_test"
