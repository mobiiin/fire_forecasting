"""Regression tests for the four-finalist, three-seed full-training pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

pytest.importorskip("torch")

from scripts.check_cawfe_latte_finalist_configs import check_configs
from scripts.run_cawfe_latte_finalist import history_with_aliases, output_parent
from scripts.summarize_cawfe_latte_finalists import summarize
from src.config import load_config
from src.data.dataset import _resolve_dataloader_options
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator


REGISTRY = Path("configs/final_training/cawfe_latte_finalists.yaml")
ABLATIONS = Path("configs/ablations/cawfe_latte_ablations.yaml")


def _registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def test_all_finalist_configs_resolve_and_exactly_reuse_source_architectures() -> None:
    registry = _registry()
    ablations = yaml.safe_load(ABLATIONS.read_text(encoding="utf-8"))["ablations"]
    assert list(registry["finalists"]) == ["baseline", "CGP", "GA_Q2", "GPK"]
    assert registry["seeds"] == [42, 123, 2026]
    for name, entry in registry["finalists"].items():
        final = load_config(entry["config_path"])
        source = load_config(ablations[entry["source_ablation"]]["config_path"])
        assert final["cawfe_latte"] == source["cawfe_latte"], name
        assert final["model"] == source["model"], name
    assert check_configs(REGISTRY) == []


def test_full_training_policy_and_seed_output_paths_are_safe() -> None:
    registry = _registry()
    paths = set()
    for name, entry in registry["finalists"].items():
        config = load_config(entry["config_path"])
        training = config["training"]
        assert training["max_epochs"] == training["epochs"] == 60
        assert training["max_train_batches_per_epoch"] is None
        assert training["performance"]["max_train_batches_per_epoch"] is None
        assert training["max_train_batches"] == 7500
        assert training["batch_size"] == 8
        assert training["gradient_accumulation_steps"] == 1
        assert training["auto_hardware_tuning"]["enabled"] is False
        assert training["performance"]["auto_batch_size"] is False
        assert training["performance"]["cudnn_benchmark"] is False
        expected_early = {
            "enabled": True, "monitor": "val_loss", "mode": "min", "patience": 8,
            "min_delta": 0.001, "start_epoch": 10,
        }
        assert {key: training["early_stopping"][key] for key in expected_early} == expected_early
        assert training["run_test_after_training"] is False
        assert training["run_external_test_after_training"] is False
        assert training["validation"]["screening"]["seed"] == 12345
        assert training["validation"]["screening"]["sampling"] == "stratified_fixed"
        assert training["validation"]["full"]["enabled"] is True
        for seed in registry["seeds"]:
            path = output_parent(name, seed)
            assert str(path).endswith(f"/{name}/seed_{seed}")
            assert path not in paths
            paths.add(path)
    assert len(paths) == 12


def test_dataloader_generator_uses_training_seed() -> None:
    options = _resolve_dataloader_options(
        {"training": {"seed": 2026, "batch_size": 2}, "data_loader": {"num_workers": 0}},
        "train",
    )
    assert options["generator"].initial_seed() == 2026


def test_required_training_history_aliases_preserve_raw_diagnostics() -> None:
    rows = history_with_aliases([
        {
            "epoch": 1, "train_loss": 2.0, "val_loss": 3.0,
            "train_loss_surface": 0.1, "val_loss_surface": 0.2,
            "train_loss_mask_total": 0.3, "val_loss_mask_total": 0.4,
            "train_mask_dice": 0.5, "val_mask_dice": 0.6,
            "epoch_time_sec": 7.0, "train_mmd_loss": 0.8,
            "train_mmd_valid_batch_fraction": 0.25,
        }
    ])
    row = rows[0]
    assert row["train_total_loss"] == 2.0
    assert row["screening_val_total_loss"] == 3.0
    assert row["train_surface_loss"] == 0.1
    assert row["screening_val_mask_loss"] == 0.4
    assert row["screening_val_dice"] == 0.6
    assert row["epoch_time_seconds"] == 7.0
    assert row["train_mmd_loss"] == 0.8
    assert row["mmd_valid_batch_fraction"] == 0.25


def test_no_fire_absolute_prediction_and_mae_metrics_do_not_cancel() -> None:
    import torch

    target = torch.zeros(1, 4, 1, 2)
    target[:, 0] = 1.0
    target[:, 1] = 2.0
    target[:, 3] = 0.5
    prediction = target.clone()
    prediction[:, 0] = torch.tensor([[-1.0, 1.0]])
    prediction[:, 1] = torch.tensor([[-2.0, 2.0]])
    prediction[:, 3] = torch.tensor([[-0.5, 0.5]])
    accumulator = FullValidationNoFireAccumulator()
    accumulator.update(prediction, target)
    metrics = accumulator.finalize()
    assert metrics["full_val_no_fire_surface_pred_mean"] == pytest.approx(0.0)
    assert metrics["full_val_no_fire_surface_abs_pred_mean"] == pytest.approx(1.0)
    assert metrics["full_val_no_fire_surface_mae"] == pytest.approx(1.0)
    assert metrics["full_val_no_fire_canopy_abs_pred_mean"] == pytest.approx(2.0)
    assert metrics["full_val_no_fire_canopy_mae"] == pytest.approx(2.0)
    assert metrics["full_val_no_fire_energy_log_abs_pred_mean"] == pytest.approx(0.5)
    assert metrics["full_val_no_fire_energy_log_mae"] == pytest.approx(0.5)


def _completed_run(root: Path, architecture: str, seed: int, dice: float) -> None:
    run = root / architecture / f"seed_{seed}" / "run1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint")
    metrics = {
        "architecture": architecture,
        "seed": seed,
        "best_epoch": 3,
        "efficiency": {
            "total_parameter_count": 100,
            "total_training_time_seconds": 10.0,
            "peak_gpu_memory_gb": 2.0,
            "mean_epoch_time_seconds": 3.0,
        },
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    full = {
        "metrics": {
            "full_val_dice": dice,
            "full_val_iou": dice / 2,
            "full_val_energy_log_mae": 0.4,
            "full_val_surface_mae": 0.3,
            "full_val_canopy_mae": 0.2,
            "full_val_active_canopy_mae": 0.1,
            "full_val_active_energy_log_mae": 0.5,
            "full_val_no_fire_mask_prob_mean": 0.1,
            "full_val_no_fire_mask_false_positive_rate": 0.01,
            "full_val_no_fire_patch_false_positive_rate": 0.02,
            "full_val_no_fire_surface_mae": 0.03,
            "full_val_no_fire_canopy_mae": 0.04,
            "full_val_no_fire_energy_log_mae": 0.05,
            "full_val_samples_per_second": 20.0,
        }
    }
    (run / "full_validation_metrics.json").write_text(json.dumps(full), encoding="utf-8")


def test_summarizer_handles_incomplete_runs_and_computes_paired_delta(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "seeds: [42, 123, 2026]\nfinalists:\n  baseline: {}\n  CGP: {}\n",
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _completed_run(root, "baseline", 42, 0.5)
    _completed_run(root, "CGP", 42, 0.6)
    payload = summarize(root, registry)
    assert payload["completed_run_count"] == 2
    assert payload["expected_run_count"] == 6
    assert len(payload["missing_or_incomplete_runs"]) == 4
    paired = payload["paired_seed_comparisons_vs_baseline"]["CGP"]["metrics"]["Dice"]
    assert paired["by_seed"] == [{"seed": 42, "delta": pytest.approx(0.1)}]
    assert (root / "finalist_results_by_seed.csv").is_file()
    assert (root / "finalist_results_summary.csv").is_file()
    assert (root / "finalist_results.json").is_file()
    assert (root / "finalist_results.md").is_file()
    assert (root / "finalist_results.txt").is_file()


def test_submission_script_selects_exact_failed_runs_without_submitting() -> None:
    command = [
        "bash",
        "scripts/submit_cawfe_latte_finalists.sh",
        "--dry-run",
        "--run",
        "baseline:123",
        "--run",
        "CGP:42",
        "--run",
        "baseline:123",
    ]
    environment = dict(os.environ)
    environment.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": "sbatch-must-not-run"})
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    assert "Selected finalist jobs (2):" in result.stdout
    assert result.stdout.count("baseline seed=123") == 1
    assert result.stdout.count("CGP seed=42") == 1
    assert "baseline seed=42" not in result.stdout
    assert "no validation preparation or Slurm submission was performed" in result.stdout
    assert "sbatch-must-not-run --parsable" in result.stdout


def test_submission_script_repeatable_filters_form_only_requested_product() -> None:
    command = [
        "bash",
        "scripts/submit_cawfe_latte_finalists.sh",
        "--dry-run",
        "--finalist",
        "baseline",
        "--finalist",
        "GA_Q2",
        "--seed",
        "123",
        "--seed",
        "2026",
    ]
    environment = dict(os.environ)
    environment.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": "sbatch-must-not-run"})
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    assert "Selected finalist jobs (4):" in result.stdout
    for selected in (
        "baseline seed=123",
        "baseline seed=2026",
        "GA_Q2 seed=123",
        "GA_Q2 seed=2026",
    ):
        assert selected in result.stdout
    assert "CGP seed=" not in result.stdout
    assert "GPK seed=" not in result.stdout


def test_submission_script_has_independent_jobs_and_worker_matches_resources() -> None:
    submit = Path("scripts/submit_cawfe_latte_finalists.sh").read_text(encoding="utf-8")
    worker = Path("scripts/slurm_train_cawfe_latte_finalist_a10080.sh").read_text(encoding="utf-8")
    ablation_worker = Path("scripts/slurm_train_cawfe_latte_ablation_a10080.sh").read_text(encoding="utf-8")
    assert "--dependency" not in submit
    for directive in (
        "#SBATCH --account=", "#SBATCH --partition=", "#SBATCH --cpus-per-task=",
        "#SBATCH --mem=", "#SBATCH --gpus=", "#SBATCH --constraint=",
    ):
        expected = next(line for line in ablation_worker.splitlines() if line.startswith(directive))
        assert expected in worker
