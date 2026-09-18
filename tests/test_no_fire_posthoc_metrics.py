"""Synthetic regression tests for canonical post-hoc no-fire metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.check_patch_fire_balance import classify_patch
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator, classify_target_masks
from src.training.checkpoints import load_model_state_dict_compatible
from src.training.metrics import compute_metrics


def targets_from_masks(*masks: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(len(masks), 4, *masks[0].shape, dtype=torch.float32)
    target[:, 2] = torch.stack(masks)
    return target


def test_all_zero_mask_is_no_fire_and_matches_canonical_checker() -> None:
    mask = torch.zeros(3, 4)
    result = classify_target_masks(targets_from_masks(mask))
    assert result["no_fire"].tolist() == [True]
    canonical = classify_patch(mask.numpy(), {"y0": 0, "x0": 0, "height": 3, "width": 4})
    assert canonical["has_fire"] is False
    assert result["active_pixels"].tolist() == [canonical["active_pixels"]]


def test_one_pixel_strictly_above_half_is_fire_at_zero_fraction_threshold() -> None:
    mask = torch.zeros(3, 4)
    mask[1, 2] = 0.50001
    result = classify_target_masks(targets_from_masks(mask))
    assert result["has_fire"].tolist() == [True]
    assert result["active_pixels"].tolist() == [1]
    canonical = classify_patch(
        mask.numpy(),
        {"y0": 0, "x0": 0, "height": 3, "width": 4},
        threshold=0.5,
        active_fraction_threshold=0.0,
    )
    assert canonical["has_fire"] is True


def test_values_at_or_below_half_are_no_fire() -> None:
    mask = torch.tensor([[0.0, 0.25], [0.5, -1.0]])
    result = classify_target_masks(targets_from_masks(mask))
    assert result["no_fire"].tolist() == [True]
    assert result["active_pixels"].tolist() == [0]


def test_mixed_batch_has_correct_fire_and_no_fire_counts() -> None:
    zero = torch.zeros(2, 2)
    one = zero.clone()
    one[0, 0] = 1.0
    half = torch.full((2, 2), 0.5)
    two = zero.clone()
    two[0, 1] = 0.9
    result = classify_target_masks(targets_from_masks(zero, one, half, two))
    assert result["has_fire"].tolist() == [False, True, False, True]
    assert int(result["has_fire"].sum()) == 2
    assert int(result["no_fire"].sum()) == 2


def test_no_fire_all_zero_logits_have_zero_false_positive_rate() -> None:
    target = targets_from_masks(torch.zeros(2, 2))
    prediction = torch.zeros_like(target)
    accumulator = FullValidationNoFireAccumulator()
    accumulator.update(prediction, target)
    metrics = accumulator.finalize()
    assert metrics["full_val_no_fire_patch_count"] == 1
    assert metrics["full_val_no_fire_mask_false_positive_rate"] == 0.0
    assert metrics["full_val_no_fire_patch_false_positive_rate"] == 0.0
    assert metrics["full_val_no_fire_mask_prob_mean"] == pytest.approx(0.5)


def test_no_fire_known_positive_pixels_have_exact_pixel_and_patch_fp_rates() -> None:
    target = targets_from_masks(torch.zeros(2, 2), torch.zeros(2, 2))
    prediction = torch.zeros_like(target)
    prediction[1, 2, 0, 1] = 2.0
    accumulator = FullValidationNoFireAccumulator()
    accumulator.update(prediction, target)
    metrics = accumulator.finalize()
    assert metrics["full_val_no_fire_patch_count"] == 2
    assert metrics["full_val_no_fire_mask_false_positive_rate"] == pytest.approx(1.0 / 8.0)
    assert metrics["full_val_no_fire_patch_false_positive_rate"] == pytest.approx(1.0 / 2.0)
    assert metrics["full_val_no_fire_false_positive_pixel_count"] == 1
    assert metrics["full_val_no_fire_false_positive_patch_count"] == 1



def test_training_screening_patch_false_positive_rate_is_patch_level() -> None:
    target = targets_from_masks(torch.zeros(2, 2), torch.zeros(2, 2))
    prediction = torch.zeros_like(target)
    prediction[1, 2, 0, 1] = 2.0
    metrics = compute_metrics(
        prediction,
        target,
        {
            "task_type": "multitask",
            "dataloader": {"source": "processed_full_frames"},
            "energy_release": {"enabled": True, "target_transform": "log1p", "predict_total": True},
        },
    )
    assert metrics["no_fire_patch_count"] == 2.0
    assert metrics["no_fire_mask_false_positive_rate"] == pytest.approx(1.0 / 8.0)
    assert metrics["no_fire_patch_false_positive_rate"] == pytest.approx(1.0 / 2.0)

def test_target_mask_alone_determines_subset_membership() -> None:
    target = targets_from_masks(torch.zeros(2, 2))
    prediction = torch.full_like(target, 100.0)
    result = classify_target_masks(target)
    assert result["no_fire"].tolist() == [True]
    accumulator = FullValidationNoFireAccumulator()
    accumulator.update(prediction, target)
    assert accumulator.finalize()["full_val_no_fire_patch_count"] == 1


def test_training_screening_metrics_use_the_same_any_target_pixel_rule() -> None:
    target = torch.zeros(1, 4, 64, 64)
    target[0, 2, 0, 0] = 1.0
    prediction = torch.zeros_like(target)
    metrics = compute_metrics(
        prediction,
        target,
        {
            "task_type": "multitask",
            "dataloader": {"source": "processed_full_frames"},
            "energy_release": {"enabled": True, "target_transform": "log1p", "predict_total": True},
        },
    )
    assert metrics["fire_patch_count"] == 1.0
    assert metrics["active_patch_count"] == 1.0
    assert metrics["no_fire_patch_count"] == 0.0


def test_existing_summarizer_reads_full_validation_sidecar(tmp_path) -> None:
    from scripts.summarize_cawfe_latte_ablations import discover_rows

    root = tmp_path / "ablations"
    run = root / "baseline" / "run1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"ablation": "baseline", "best_epoch": 1, "best_epoch_metrics": {"validation": {}}}),
        encoding="utf-8",
    )
    (run / "no_fire_metrics.json").write_text(
        json.dumps({"metrics": {
            "full_val_total_patch_count": 10,
            "full_val_fire_patch_count": 6,
            "full_val_no_fire_patch_count": 4,
            "full_val_no_fire_percent": 40.0,
            "full_val_no_fire_mask_prob_mean": 0.2,
            "full_val_no_fire_mask_false_positive_rate": 0.125,
            "full_val_no_fire_patch_false_positive_rate": 0.5,
        }}),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text("ablations:\n  baseline:\n    name: baseline\n", encoding="utf-8")
    row = discover_rows(root, registry)[0]
    assert row["Full Val Fire Patches"] == 6
    assert row["Full Val No-Fire Patches"] == 4
    assert row["Full Val No-Fire %"] == pytest.approx(40.0)
    assert row["No-Fire Pixel FP Rate"] == pytest.approx(0.125)
    assert row["No-Fire Patch FP Rate"] == pytest.approx(0.5)


def test_checkpoint_loader_restores_lazy_cawfe_spatial_position_exactly() -> None:
    class Alignment(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dim = 3
            self.register_parameter("spatial_pos", None)
            self.temporal_pos = torch.nn.Parameter(torch.zeros(1, 2, 1, 3))
            self._spatial_shape = None

    class CAWFELatte(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alignment = Alignment()

    model = CAWFELatte()
    saved_spatial = torch.arange(36, dtype=torch.float32).reshape(1, 1, 12, 3)
    checkpoint = {
        "config": {"model": {"architecture": "cawfe_latte"}},
        "model_state_dict": {
            "alignment.spatial_pos": saved_spatial,
            "alignment.temporal_pos": torch.ones(1, 2, 1, 3),
        },
    }
    with pytest.raises(ValueError, match="Cannot infer square"):
        load_model_state_dict_compatible(model, checkpoint)

    saved_spatial = torch.arange(27, dtype=torch.float32).reshape(1, 1, 9, 3)
    checkpoint["model_state_dict"]["alignment.spatial_pos"] = saved_spatial
    result = load_model_state_dict_compatible(model, checkpoint)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert model.alignment._spatial_shape == (3, 3)
    torch.testing.assert_close(model.alignment.spatial_pos, saved_spatial)

def _completed_run(root: Path, ablation: str, run_name: str, *, mtime: float) -> Path:
    run_dir = root / ablation / run_name
    (run_dir / "checkpoints").mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"ablation": ablation}), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("model: {}\n", encoding="utf-8")
    (run_dir / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint")
    os.utime(metrics_path, (mtime, mtime))
    return run_dir


def test_slurm_all_selects_only_latest_pending_run_per_ablation(tmp_path) -> None:
    from scripts.recompute_ablation_no_fire_metrics import latest_runs_by_ablation, pending_submission_runs

    old_baseline = _completed_run(tmp_path, "baseline", "old", mtime=1.0)
    new_baseline = _completed_run(tmp_path, "baseline", "new", mtime=2.0)
    latest_a = _completed_run(tmp_path, "A_resblocks", "only", mtime=3.0)
    selected = latest_runs_by_ablation([old_baseline, new_baseline, latest_a])
    assert selected == [latest_a, new_baseline]

    (latest_a / "no_fire_metrics.json").write_text("{}", encoding="utf-8")
    assert pending_submission_runs(selected, overwrite=False) == [new_baseline]
    assert pending_submission_runs(selected, overwrite=True) == selected


def test_slurm_command_targets_one_exact_run_without_shell_expansion(tmp_path) -> None:
    from scripts.recompute_ablation_no_fire_metrics import build_sbatch_command

    run_dir = _completed_run(tmp_path, "baseline", "run with spaces", mtime=1.0)
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = SimpleNamespace(
        slurm_script=str(worker), device="auto", batch_size=None,
        num_workers=8, overwrite=False,
    )
    command = build_sbatch_command(args, run_dir, tmp_path)
    assert command[:2] == ["sbatch", "--parsable"]
    assert command[3] == str(run_dir.resolve())
    assert command[5:] == ["auto", "default", "8", "0"]

