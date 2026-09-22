"""Sanity checks for publication-quality full-training data exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.run_cawfe_latte_finalist import paper_epoch_history
from scripts.summarize_full_training import summarize_full_training
from src.evaluation.full_validation import evaluate_full_validation


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_paper_epoch_history_has_one_unique_row_and_exact_best_marker() -> None:
    raw = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "val_loss": 1.5,
            "train_loss_surface": 0.2,
            "val_mask_dice": 0.4,
            "train_mmd_loss": 0.03,
            "train_mmd_valid_batch_fraction": 0.5,
            "train_epoch_seconds": 10.0,
            "val_epoch_seconds": 2.0,
            "epoch_time_sec": 12.5,
            "train_batches_this_epoch": 7500,
            "train_samples_this_epoch": 60000,
            "train_fraction_of_dataset": 0.1919,
            "cumulative_train_samples_seen": 60000,
            "equivalent_full_dataset_epochs": 0.1919,
        },
        {"epoch": 2, "train_loss": 1.0, "val_loss": 0.8, "val_mask_dice": 0.6},
    ]
    rows = paper_epoch_history(
        raw, model_name="GA_Q2", architecture_name="GA_Q2_fire_domain_mmd", seed=123, best_epoch=2
    )
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row["model_name"] == "GA_Q2" and row["seed"] == 123 for row in rows)
    assert [row["is_best_epoch"] for row in rows] == [0, 1]
    assert rows[0]["train_total_loss"] == 2.0
    assert rows[0]["val_dice"] == 0.4
    assert rows[0]["train_mmd_loss"] == 0.03
    assert rows[0]["mmd_valid_batch_fraction"] == 0.5
    assert rows[0]["val_energy_log_rmse"] is None
    assert rows[0]["checkpoint_selection_metric"] == 1.5
    assert rows[0]["train_batches_this_epoch"] == 7500
    assert rows[0]["train_samples_this_epoch"] == 60000
    assert rows[0]["equivalent_full_dataset_epochs"] == pytest.approx(0.1919)


class _PaperDataset(torch.utils.data.Dataset):
    input_normalization_on_device = False

    def __init__(self, root: Path) -> None:
        self.root = root
        self.split = "val"
        self.sample_index_path = root / "samples.jsonl"
        full_mask = np.zeros((200, 200), dtype=np.float32)
        full_mask[0, 100] = 1.0  # tiny: 1 / 10,000
        full_mask[100, 0:100] = 1.0  # medium: 100 / 10,000
        full_mask[100:110, 100:200] = 1.0  # large: 1,000 / 10,000
        np.savez(root / "target.npz", fire_mask=full_mask)
        patches = [(0, 0), (0, 100), (100, 0), (100, 100)]
        self.records = []
        self.targets = []
        for index, (y0, x0) in enumerate(patches):
            self.records.append(
                {
                    "sample_id": f"sample_{index}",
                    "fire_name": f"fire_{index % 2}",
                    "target_path": "target.npz",
                    "patch": {"y0": y0, "x0": x0, "height": 100, "width": 100},
                }
            )
            target = torch.zeros(4, 100, 100)
            target[0] = 0.25
            target[1] = 0.5
            target[2] = torch.from_numpy(full_mask[y0 : y0 + 100, x0 : x0 + 100])
            target[3] = 0.75
            self.targets.append(target)
        self.sample_index_path.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        target = self.targets[index]
        prediction = target.clone()
        prediction[2] = torch.where(target[2] > 0.5, 10.0, -10.0)
        record = self.records[index]
        return torch.stack([prediction]), target, {
            "sample_id": record["sample_id"],
            "fire_name": record["fire_name"],
        }


class _IdentityForecast(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0]


def test_full_validation_writes_one_sample_row_and_shared_qualitative_archive(tmp_path: Path) -> None:
    dataset = _PaperDataset(tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    run_dir = tmp_path / "run"
    shared_path = tmp_path / "shared" / "qualitative_validation_samples.json"
    result = evaluate_full_validation(
        model=_IdentityForecast(),
        val_loader=loader,
        config={
            "model": {"input_channels": 4},
            "final_training": {"finalist": "baseline"},
            "training": {
                "seed": 42,
                "validation": {
                    "full": {
                        "save_analysis_data": True,
                        "qualitative_index_path": str(shared_path),
                        "qualitative_samples_per_group": 1,
                        "qualitative_seed": 7,
                    }
                },
            },
        },
        device=torch.device("cpu"),
        amp_dtype=None,
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
        checkpoint_epoch=2,
        expected_counts={"total": 4, "fire": 3, "no_fire": 1},
    )
    sample_rows = _read_csv(run_dir / "evaluation" / "validation_sample_metrics.csv")
    assert len(sample_rows) == len(dataset)
    assert len({row["sample_id"] for row in sample_rows}) == len(dataset)
    assert {row["fire_activity_bin"] for row in sample_rows} == {"no_fire", "tiny_fire", "medium_fire", "large_fire"}
    assert all(row["model_name"] == "baseline" and row["seed"] == "42" for row in sample_rows)
    assert sum(int(row["sample_count"]) for row in _read_csv(run_dir / "evaluation" / "per_fire_validation_metrics.csv")) == len(dataset)
    selected = json.loads(shared_path.read_text(encoding="utf-8"))["selected_sample_ids"]
    with np.load(run_dir / "evaluation" / "qualitative_predictions.npz", allow_pickle=False) as archive:
        assert archive["sample_id"].tolist() == selected
        assert set(("target_surface", "target_canopy", "target_mask", "target_energy_log", "pred_surface", "pred_canopy", "pred_mask_probability", "pred_energy_log")) <= set(archive.files)
    assert result["metrics"]["full_val_inference_time_per_batch_ms"] > 0.0
    assert result["metrics"]["full_val_inference_time_per_sample_ms"] > 0.0


def _complete_analysis_run(root: Path) -> Path:
    run = root / "baseline" / "seed_42" / "run1"
    epoch_rows = [
        {"model_name": "baseline", "architecture_name": "baseline", "seed": 42, "epoch": 1, "global_step": 7500, "equivalent_full_dataset_epochs": 0.2, "train_total_loss": 2.0, "val_total_loss": 1.0, "val_dice": 0.5, "is_best_epoch": 0},
        {"model_name": "baseline", "architecture_name": "baseline", "seed": 42, "epoch": 2, "global_step": 15000, "equivalent_full_dataset_epochs": 0.4, "train_total_loss": 1.0, "val_total_loss": 0.8, "val_dice": 0.6, "is_best_epoch": 1},
    ]
    _write_csv(run / "history" / "epoch_history.csv", epoch_rows)
    _write_csv(run / "history" / "step_history.csv", [{"global_step": 50, "epoch": 1, "train_total_loss": 1.2}])
    _write_csv(run / "evaluation" / "per_fire_validation_metrics.csv", [{"model_name": "baseline", "seed": 42, "fire_name": "fire_a", "sample_count": 1, "dice": 0.6}])
    _write_csv(run / "evaluation" / "validation_sample_metrics.csv", [{"model_name": "baseline", "seed": 42, "sample_id": "sample_a", "fire_name": "fire_a", "surface_mae": 0.1}])
    efficiency = {"model_name": "baseline", "seed": 42, "parameter_count": 10, "inference_time_per_sample_ms": 2.0}
    (run / "evaluation" / "efficiency_metrics.json").write_text(json.dumps(efficiency), encoding="utf-8")
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "metadata" / "training_sampling_protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": "epoch_random_subset_without_replacement_v1",
                "batch_size": 8,
                "batches_per_epoch": 7500,
                "sampling_within_epoch": "without_replacement",
                "subset_changes_each_epoch": True,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run / "evaluation" / "qualitative_predictions.npz", sample_id=np.asarray(["sample_a"]))
    metrics = {
        "source_ablation": "baseline",
        "best_epoch": 2,
        "epochs_trained": 2,
        "stopped_early": True,
        "best_screening_validation": {"total_loss": 0.8},
        "full_validation": {"full_val_dice": 0.6, "full_val_surface_mae": 0.1},
        "efficiency": efficiency,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


def test_full_training_aggregator_preserves_seed_and_single_seed_statistics(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"seeds": [42], "finalists": {"baseline": {}}}), encoding="utf-8")
    run_root = tmp_path / "runs"
    _complete_analysis_run(run_root)
    legacy = run_root / "baseline" / "seed_42" / "legacy_run"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(json.dumps({"best_epoch": 1}), encoding="utf-8")
    analysis = tmp_path / "analysis"
    manifest = summarize_full_training(run_root=run_root, registry_path=registry, analysis_root=analysis)
    assert manifest["complete_run_count"] == 1
    assert manifest["missing_registered_runs"] == []
    assert manifest["sample_metrics_format"] == "csv"
    assert manifest["excluded_legacy_runs"][0]["status"] == "legacy_full_split_training"
    for name in (
        "all_epoch_history.csv", "all_step_history.csv", "final_metrics_by_run.csv",
        "final_metrics_mean_std.csv", "per_fire_metrics_all_runs.csv",
        "sample_metrics_all_runs.csv", "efficiency_all_runs.csv", "best_epochs.csv",
        "excluded_legacy_runs.csv", "learning_curves_mean_std.csv",
    ):
        assert (analysis / name).is_file(), name
    raw = _read_csv(analysis / "all_epoch_history.csv")
    assert [row["epoch"] for row in raw] == ["1", "2"]
    assert all(row["model_name"] == "baseline" and row["seed"] == "42" for row in raw)
    curve = next(
        row for row in _read_csv(analysis / "learning_curves_mean_std.csv")
        if row["model_name"] == "baseline" and row["epoch"] == "1" and row["metric"] == "val_total_loss"
    )
    assert float(curve["mean"]) == pytest.approx(1.0)
    assert float(curve["std"]) == pytest.approx(0.0)
    assert curve["n_seeds_at_epoch"] == "1"
    final = next(
        row for row in _read_csv(analysis / "final_metrics_mean_std.csv")
        if row["metric"] == "full_val_dice"
    )
    assert float(final["mean"]) == pytest.approx(0.6)
    assert float(final["std"]) == pytest.approx(0.0)
    assert final["number_of_seeds"] == "1"
