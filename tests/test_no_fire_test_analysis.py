"""Regression tests for the frozen no-fire held-out test analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.evaluate_no_fire_test import MODELS, SEEDS, aggregate
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator


def test_no_fire_accumulator_distinguishes_abs_prediction_from_mae_and_reports_rmse() -> None:
    target = torch.zeros(1, 4, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[0, 0] = torch.tensor([[-2.0, -1.0], [1.0, 2.0]])
    prediction[0, 1] = 3.0
    prediction[0, 3] = -0.5
    accumulator = FullValidationNoFireAccumulator()
    accumulator.update(prediction, target)
    metrics = accumulator.finalize()
    assert metrics["full_val_no_fire_surface_pred_mean"] == pytest.approx(0.0)
    assert metrics["full_val_no_fire_surface_abs_pred_mean"] == pytest.approx(1.5)
    assert metrics["full_val_no_fire_surface_mae"] == pytest.approx(1.5)
    assert metrics["full_val_no_fire_surface_pred_rmse"] == pytest.approx((2.5) ** 0.5)
    assert metrics["full_val_no_fire_canopy_abs_pred_mean"] == pytest.approx(3.0)
    assert metrics["full_val_no_fire_energy_log_abs_pred_mean"] == pytest.approx(0.5)
    assert metrics["full_val_no_fire_surface_target_abs_mean"] == 0.0


def _write_complete_run(root: Path, model: str, seed: int, offset: int) -> None:
    run = root / model / f"seed_{seed}" / "completed"
    evaluation = run / "evaluation"
    checkpoints = run / "checkpoints"
    evaluation.mkdir(parents=True)
    checkpoints.mkdir()
    checkpoint = checkpoints / "best_model.pt"
    checkpoint.write_bytes(b"frozen")
    (run / "full_validation_metrics.json").write_text("{}\n", encoding="utf-8")
    (run / "resolved_config.yaml").write_text(
        "final_training:\n"
        f"  finalist: {model}\n"
        "training:\n"
        f"  seed: {seed}\n"
        "  sampling_protocol:\n"
        "    protocol_id: epoch_random_subset_without_replacement_v1\n"
        "    batches_per_epoch: 7500\n"
        "    batch_size: 8\n",
        encoding="utf-8",
    )

    mask_fp = 0.25 * (offset + 1)
    surface = float(offset + 1)
    canopy = float(offset + 2)
    energy = float(offset + 3)
    rows = []
    for index in range(4):
        no_fire = index < 2
        row = {
            "model_name": model,
            "seed": seed,
            "sample_id": f"sample-{index}",
            "fire_name": "FIRE_A" if index % 2 == 0 else "FIRE_B",
            "is_no_fire": int(no_fire),
            "target_active_fraction": 0.0 if no_fire else 0.25,
            "pixel_count": 4,
            "predicted_fire_fraction": mask_fp if no_fire else 0.0,
            "mean_mask_probability": 0.4 if no_fire else 0.6,
            "mean_surface_prediction": -surface if no_fire else 0.0,
            "mean_canopy_prediction": canopy if no_fire else 0.0,
            "mean_energy_log_prediction": -energy if no_fire else 0.0,
            "no_fire_mask_false_positive_rate": mask_fp if no_fire else "",
            "no_fire_patch_false_positive": 1 if no_fire else "",
            "no_fire_surface_abs_pred_mean": surface if no_fire else "",
            "no_fire_canopy_abs_pred_mean": canopy if no_fire else "",
            "no_fire_energy_log_abs_pred_mean": energy if no_fire else "",
        }
        rows.append(row)
    with (evaluation / "test_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (evaluation / "test_per_fire_metrics.csv").write_text(
        "fire_name,sample_count\nFIRE_A,2\nFIRE_B,2\n", encoding="utf-8"
    )
    np.savez_compressed(evaluation / "qualitative_test_predictions.npz", sample_id=np.asarray(["sample-0"]))
    metrics = {
        "test_total_patch_count": 4,
        "test_fire_patch_count": 2,
        "test_no_fire_patch_count": 2,
        "test_no_fire_pixel_count": 8,
        "test_no_fire_mask_prob_mean": 0.4,
        "test_no_fire_mask_false_positive_rate": mask_fp,
        "test_no_fire_patch_false_positive_rate": 1.0,
        "test_no_fire_surface_pred_mean": -surface,
        "test_no_fire_surface_abs_pred_mean": surface,
        "test_no_fire_surface_mae": surface,
        "test_no_fire_canopy_pred_mean": canopy,
        "test_no_fire_canopy_abs_pred_mean": canopy,
        "test_no_fire_canopy_mae": canopy,
        "test_no_fire_energy_log_pred_mean": -energy,
        "test_no_fire_energy_log_abs_pred_mean": energy,
        "test_no_fire_energy_log_mae": energy,
    }
    payload = {
        "schema_version": 1,
        "metric_scope": "complete_locked_held_out_test",
        "split": "test",
        "test_used_for_model_selection": False,
        "model": model,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": offset + 1,
        "dataset_sample_count": 4,
        "evaluated_test_samples": 4,
        "unique_sample_id_count": 4,
        "fire_count": 2,
        "no_fire_count": 2,
        "sample_metrics_path": str((evaluation / "test_sample_metrics.csv").resolve()),
        "metrics": metrics,
    }
    (evaluation / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_only_produces_auditable_exact_outputs(tmp_path: Path) -> None:
    root = tmp_path / "finalists"
    for model in MODELS:
        for offset, seed in enumerate(SEEDS):
            _write_complete_run(root, model, seed, offset)
    output = tmp_path / "analysis"
    manifest = aggregate(cawfe_root=root, output_root=output)
    assert manifest["complete"] is True
    assert manifest["no_retraining_performed"] is True
    assert manifest["full_test_sample_count"] == 4
    assert manifest["no_fire_sample_count"] == 2

    with (output / "no_fire_test_mean_std.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["baseline", "GA_Q2"]
    assert float(rows[0]["mask_fp_mean"]) == pytest.approx(0.5)
    assert float(rows[0]["mask_fp_std"]) == pytest.approx(0.25)
    assert float(rows[0]["surface_abs_mean_mean"]) == pytest.approx(2.0)
    assert float(rows[0]["surface_abs_mean_std"]) == pytest.approx(1.0)

    ids = json.loads((output / "no_fire_test_sample_ids.json").read_text(encoding="utf-8"))
    assert ids["sample_ids"] == ["sample-0", "sample-1"]
    assert len({item["no_fire_subset_hash"] for item in ids["evaluations"]}) == 1
    assert len(ids["evaluations"]) == 6

    per_fire = list(csv.DictReader((output / "no_fire_test_per_fire.csv").open(newline="", encoding="utf-8")))
    assert len(per_fire) == 12
    assert sum(int(row["no_fire_patch_count"]) for row in per_fire) == 12
    latex = (output / "no_fire_table.tex").read_text(encoding="utf-8")
    assert "\\label{tab:no_fire}" in latex
    assert "FLARE baseline &" in latex
    assert "FLARE final &" in latex
    assert "\\pm" in latex
    assert latex.count("\\\\\n") == 3
    summary = (output / "no_fire_summary.txt").read_text(encoding="utf-8")
    assert "No-fire subset identical across models/seeds:\n  YES" in summary
    assert "NO RETRAINING PERFORMED:\n  YES" in summary


def test_no_fire_submission_default_and_exact_pair_dry_runs() -> None:
    environment = dict(os.environ)
    environment.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": "sbatch-must-not-run"})
    full = subprocess.run(
        ["bash", "scripts/submit_no_fire_test.sh", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "Selected no-fire test jobs (6):" in full.stdout
    assert full.stdout.count("sbatch-must-not-run --parsable") == 6
    exact = subprocess.run(
        [
            "bash", "scripts/submit_no_fire_test.sh",
            "--run", "baseline:42", "--run", "GA_Q2:2026", "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "Selected no-fire test jobs (2):" in exact.stdout
    assert exact.stdout.count("sbatch-must-not-run --parsable") == 2
