from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_trained_models import (
	LATEX_HEADER,
	PAPER_COLUMNS,
	build_argument_parser,
	build_paper_row,
	render_latex_table,
	run_qualitative_mode,
	run_quantitative,
	_write_csv,
)
from src.data.dataset import metadata_batch_to_list
from src.evaluation.run_discovery import discover_runs, find_best_run
from src.models.evaluation import _validate_checkpoint_architecture


def _fake_run(
	root: Path,
	architecture: str,
	run_name: str,
	*,
	status: str = "completed",
	best_metric_value: float = 1.0,
	final_val_loss: float = 1.0,
	val_mask_dice: float | None = None,
) -> Path:
	run_dir = root / architecture / run_name
	(run_dir / "checkpoints").mkdir(parents=True)
	(run_dir / "metadata").mkdir()
	(run_dir / "configs").mkdir()
	(run_dir / "logs").mkdir()
	(run_dir / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint")
	(run_dir / "configs" / "resolved_config.yaml").write_text("model:\n  architecture: convlstm_unet\n", encoding="utf-8")
	summary = {
		"architecture": architecture,
		"run_name": run_name,
		"run_dir": str(run_dir),
		"status": status,
		"best_metric_name": "val_loss",
		"best_metric_value": best_metric_value,
		"best_epoch": 3,
		"best_checkpoint_path": str(run_dir / "checkpoints" / "best_model.pt"),
		"final_val_loss": final_val_loss,
		"resolved_config_path": str(run_dir / "configs" / "resolved_config.yaml"),
	}
	(run_dir / "metadata" / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
	if val_mask_dice is not None:
		with (run_dir / "logs" / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=["epoch", "val_mask_dice", "val_loss"])
			writer.writeheader()
			writer.writerow({"epoch": 1, "val_mask_dice": val_mask_dice, "val_loss": final_val_loss})
	return run_dir


def test_discover_runs_finds_fake_architecture_run_directories(tmp_path: Path) -> None:
	_fake_run(tmp_path, "convlstm_unet", "run_a")

	runs = discover_runs(tmp_path, architecture="convlstm_unet")

	assert len(runs) == 1
	assert runs[0].architecture == "convlstm_unet"
	assert runs[0].run_name == "run_a"
	assert runs[0].checkpoint_path.endswith("best_model.pt")


def test_find_best_run_selects_lower_val_loss_when_min(tmp_path: Path) -> None:
	_fake_run(tmp_path, "convlstm_unet", "worse", final_val_loss=0.7)
	_fake_run(tmp_path, "convlstm_unet", "better", final_val_loss=0.2)
	runs = discover_runs(tmp_path, architecture="convlstm_unet")

	best = find_best_run(runs, "convlstm_unet", selection_metric="val_loss", selection_mode="min")

	assert best is not None
	assert best.run_name == "better"


def test_find_best_run_selects_higher_dice_when_max(tmp_path: Path) -> None:
	_fake_run(tmp_path, "convlstm_unet", "low_dice", val_mask_dice=0.4)
	_fake_run(tmp_path, "convlstm_unet", "high_dice", val_mask_dice=0.9)
	runs = discover_runs(tmp_path, architecture="convlstm_unet")

	best = find_best_run(runs, "convlstm_unet", selection_metric="val_dice", selection_mode="max")

	assert best is not None
	assert best.run_name == "high_dice"


def test_checkpoint_architecture_mismatch_raises() -> None:
	with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
		_validate_checkpoint_architecture({"architecture": "earthformer_lite"}, "convlstm_unet", Path("best_model.pt"))


def test_paper_metrics_csv_has_exact_required_columns(tmp_path: Path) -> None:
	row = build_paper_row(
		"Demo",
		{
			"Surf. MAE ↓": 1.0,
			"Canopy MAE ↓": 2.0,
			"Dice ↑": 0.8,
			"IoU ↑": 0.7,
			"Energy MAE ↓": 3.0,
			"Active Energy MAE ↓": 4.0,
			"Skill ↑": 0.2,
		},
	)
	path = tmp_path / "paper_metrics.csv"
	_write_csv(path, [row], PAPER_COLUMNS)

	with path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		header = next(reader)

	assert header == PAPER_COLUMNS


def test_paper_table_contains_latex_header() -> None:
	table = render_latex_table([build_paper_row("Demo", {"Dice ↑": 0.8})])

	assert LATEX_HEADER in table


def test_qualitative_mode_raises_not_implemented() -> None:
	with pytest.raises(NotImplementedError, match="Qualitative prediction and rollout visualization"):
		run_qualitative_mode()


def test_all_model_mode_skips_missing_architecture_runs_without_failing(tmp_path: Path) -> None:
	args = build_argument_parser().parse_args(
		[
			"--runs_root",
			str(tmp_path / "missing_runs"),
			"--output_dir",
			str(tmp_path / "results"),
			"--no-include-baselines",
		]
	)

	output_dir = run_quantitative(args)

	summary = json.loads((output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
	assert "convlstm_unet" in summary["skipped_models"]
	assert summary["failed_models"] == []


def test_single_model_mode_fails_if_requested_architecture_has_no_valid_run(tmp_path: Path) -> None:
	args = build_argument_parser().parse_args(
		[
			"--runs_root",
			str(tmp_path / "missing_runs"),
			"--output_dir",
			str(tmp_path / "results"),
			"--model_architecture",
			"convlstm_unet",
			"--no-include-baselines",
		]
	)

	with pytest.raises(FileNotFoundError, match="No trained runs found"):
		run_quantitative(args)


def test_metadata_batch_to_list_preserves_short_batch_level_lists() -> None:
	torch = pytest.importorskip("torch")
	metadata_batch = {
		"dataset_id": torch.tensor([1, 2, 3]),
		"dataset_name": ["a", "b", "c"],
		"raw_shape": [216, 168],
		"patch": {
			"y0": torch.tensor([0, 1, 2]),
			"y1": torch.tensor([64, 65, 66]),
			"x0": torch.tensor([5, 6, 7]),
			"x1": torch.tensor([69, 70, 71]),
		},
	}

	items = metadata_batch_to_list(metadata_batch, batch_size=3)

	assert len(items) == 3
	assert items[1]["dataset_name"] == "b"
	assert items[1]["raw_shape"] == [216, 168]
	assert items[1]["patch"] == {"y0": 1, "y1": 65, "x0": 6, "x1": 70}
