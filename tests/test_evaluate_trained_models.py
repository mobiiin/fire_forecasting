from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_trained_models import (
	LATEX_HEADER,
	PAPER_COLUMNS,
	apply_skill,
	build_argument_parser,
	build_paper_row,
	paper_columns_for_metric,
	paper_values_from_metrics,
	render_latex_table,
	run_all_modes,
	run_qualitative_mode,
	run_quantitative,
	_build_evaluated_model_entries,
	_resolve_requested_targets,
	_validate_checkpoint_arg,
	_write_csv,
)
from src.data.dataset import metadata_batch_to_list
from src.evaluation.run_discovery import discover_runs, find_best_run
from src.models.evaluation import _validate_checkpoint_architecture, _validate_checkpoint_sequence


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


def test_checkpoint_sequence_mismatch_raises() -> None:
	config = {
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"cache": {"target_definition_version": "interval_consumed_current_to_horizon_target_v1"},
	}
	checkpoint = {
		"input_sequence_length": 6,
		"prediction_horizon": 1,
		"target_offset_from_start": 6,
		"target_offset_from_last_input": 1,
		"target_definition_version": "interval_consumed_current_to_horizon_target_v1",
	}
	with pytest.raises(ValueError, match="Checkpoint sequence metadata mismatch"):
		_validate_checkpoint_sequence(checkpoint, config, Path("best_model.pt"))


def test_paper_metrics_csv_has_exact_required_columns(tmp_path: Path) -> None:
	row = build_paper_row(
		"Demo",
		{
			"Surf. MAE ↓": 1.0,
			"Canopy MAE ↓": 2.0,
			"Dice ↑": 0.8,
			"IoU ↑": 0.7,
			"Energy Log MAE ↓": 3.0,
			"Active Energy Log MAE ↓": 4.0,
			"Skill ↑": 0.2,
		},
	)
	path = tmp_path / "paper_metrics.csv"
	_write_csv(path, [row], PAPER_COLUMNS)

	with path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		header = next(reader)

	assert header == PAPER_COLUMNS
	assert "Energy Log MAE ↓" in header
	assert "Active Energy Log MAE ↓" in header
	assert "Energy MAE ↓" not in header
	assert "Active Energy MAE ↓" not in header


def test_paper_table_contains_latex_header() -> None:
	table = render_latex_table([build_paper_row("Demo", {"Dice ↑": 0.8})])

	assert LATEX_HEADER in table
	assert "Energy Log MAE" in table
	assert "Active Energy Log MAE" in table


def test_paper_energy_metric_mw_changes_columns_and_latex_labels() -> None:
	values, _sources = paper_values_from_metrics(
		{
			"test_energy_log_mae": 1.0,
			"test_active_energy_log_mae": 2.0,
			"test_energy_mw_mae": 10.0,
			"test_active_energy_mw_mae": 20.0,
		},
		paper_energy_metric="mw",
	)
	row = build_paper_row("Demo", values, paper_energy_metric="mw")
	table = render_latex_table([row], paper_energy_metric="mw")

	assert paper_columns_for_metric("mw") == [
		"Model",
		"Surf. MAE ↓",
		"Canopy MAE ↓",
		"Dice ↑",
		"IoU ↑",
		"Energy MW MAE ↓",
		"Active Energy MW MAE ↓",
		"Skill ↑",
	]
	assert row["Energy MW MAE ↓"] == 10.0
	assert row["Active Energy MW MAE ↓"] == 20.0
	assert "Energy MW MAE" in table


def test_skill_uses_active_energy_log_mae_by_default() -> None:
	values = {"Energy Log MAE ↓": 10.0, "Active Energy Log MAE ↓": 2.0}
	persistence_values = {"Energy Log MAE ↓": 10.0, "Active Energy Log MAE ↓": 4.0}

	apply_skill(values, persistence_values)

	assert values["Skill ↑"] == 0.5


def test_evaluated_model_entries_keep_secondary_mw_metrics() -> None:
	entries = _build_evaluated_model_entries(
		[
			{
				"identity": {"architecture": "convlstm_unet", "model_name": "ConvLSTM U-Net", "run_name": "run1", "checkpoint_path": "best.pt"},
				"paper_values": {"Energy Log MAE ↓": 1.0, "Active Energy Log MAE ↓": 2.0},
				"secondary_energy_metrics": {"energy_mw_mae": 10.0, "active_energy_mw_mae": 20.0},
			}
		],
		per_fire_rows=[],
		type_by_architecture={"convlstm_unet": "learned"},
	)

	assert entries[0]["metrics"]["energy_log_mae"] == 1.0
	assert entries[0]["metrics"]["active_energy_log_mae"] == 2.0
	assert entries[0]["metrics"]["secondary"]["energy_mw_mae"] == 10.0
	assert entries[0]["metrics"]["secondary"]["active_energy_mw_mae"] == 20.0


def test_qualitative_parser_defaults_are_config_only_outputs() -> None:
	args = build_argument_parser().parse_args(["--mode", "qualitative"])

	assert args.num_samples == 10
	assert args.qualitative_seed == 42
	assert args.qualitative_output_format == "png"
	assert args.qualitative_dpi == 180
	assert args.save_individual_model_panels is False


def test_all_mode_parser_and_dispatch_runs_both_modes(monkeypatch, tmp_path: Path) -> None:
	import scripts.evaluate_trained_models as evaluator

	args = build_argument_parser().parse_args(["--mode", "all", "--output_dir", str(tmp_path)])
	calls: list[str] = []

	def fake_quantitative(parsed_args):
		assert parsed_args is args
		calls.append("quantitative")
		return tmp_path / "quantitative"

	def fake_qualitative(parsed_args):
		assert parsed_args is args
		calls.append("qualitative")
		return tmp_path / "qualitative"

	monkeypatch.setattr(evaluator, "run_quantitative", fake_quantitative)
	monkeypatch.setattr(evaluator, "run_qualitative_mode", fake_qualitative)

	result = run_all_modes(args)

	assert args.mode == "all"
	assert calls == ["quantitative", "qualitative"]
	assert result == {
		"quantitative": tmp_path / "quantitative",
		"qualitative": tmp_path / "qualitative",
	}


def test_model_architecture_resolution_is_exact() -> None:
	learned, baselines, all_mode = _resolve_requested_targets("convlstm_unet")
	assert learned == ["convlstm_unet"]
	assert baselines == []
	assert all_mode is False

	learned, baselines, all_mode = _resolve_requested_targets("baseline")
	assert learned == []
	assert baselines == ["persistence", "linear_extrapolation"]
	assert all_mode is False

	learned, baselines, all_mode = _resolve_requested_targets("all")
	assert "convlstm_unet" in learned
	assert baselines == ["persistence", "linear_extrapolation"]
	assert all_mode is True


def test_checkpoint_arg_only_valid_for_single_learned_model() -> None:
	with pytest.raises(ValueError, match="single learned model"):
		_validate_checkpoint_arg("best_model.pt", ["convlstm_unet", "earthformer_lite"], [], "all")
	with pytest.raises(ValueError, match="Baselines do not use"):
		_validate_checkpoint_arg("best_model.pt", [], ["persistence"], "baseline")


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

	report_text = (output_dir / "evaluation_report.md").read_text(encoding="utf-8")
	assert output_dir.parent == tmp_path / "results"
	assert "convlstm_unet" in report_text
	assert not (output_dir / "evaluation_summary.json").exists()
	assert not (output_dir / "evaluation_report.json").exists()
	assert (output_dir / "evaluation_report.md").exists()

	json_args = build_argument_parser().parse_args(
		[
			"--runs_root",
			str(tmp_path / "missing_runs"),
			"--output_dir",
			str(tmp_path / "results"),
			"--eval_name",
			"json_output",
			"--no-include-baselines",
			"--include_json_outputs",
		]
	)
	json_output_dir = run_quantitative(json_args)

	summary = json.loads((json_output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
	assert "convlstm_unet" in summary["skipped_models"]
	assert summary["failed_models"] == []
	assert (json_output_dir / "evaluation_report.json").exists()


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
