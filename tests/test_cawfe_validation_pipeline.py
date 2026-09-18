"""Tests for one-job CAWFE-Latte screening and full validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.summarize_cawfe_latte_ablations import discover_rows
from src.evaluation.full_validation import FullValidationAccumulator, evaluate_full_validation
from src.evaluation.validation_subset import ensure_screening_validation_indices, select_stratified_indices
from src.training.train import resolve_validation_policy, validation_loader_for_epoch


def test_stratified_selection_is_deterministic_natural_ratio_and_covers_fires() -> None:
    classified = [
        {"index": index, "class": "fire" if index < 65 else "no_fire", "fire_name": f"fire_{index % 4}"}
        for index in range(100)
    ]
    first, first_counts = select_stratified_indices(classified, 40, seed=12345)
    second, second_counts = select_stratified_indices(classified, 40, seed=12345)
    assert first == second
    assert first_counts == second_counts
    assert first_counts["fire"] == 26
    assert first_counts["no_fire"] == 14
    assert len(first_counts["per_class_per_fire"]["fire"]) == 4
    assert len(first_counts["per_class_per_fire"]["no_fire"]) == 4


class _RecordDataset:
    def __init__(self, root: Path, records: list[dict], index_path: Path) -> None:
        self.root = root
        self.records = records
        self.sample_index_path = index_path
        self.split = "val"

    def __len__(self) -> int:
        return len(self.records)


def test_screening_index_artifact_is_reproducible_and_reused(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    mask = np.zeros((4, 5), dtype=np.float32)
    mask.reshape(-1)[:12] = 1.0
    np.savez(target_dir / "target.npz", fire_mask=mask)
    records = []
    for index in range(20):
        records.append(
            {
                "sample_id": f"sample_{index}",
                "fire_name": f"fire_{index % 3}",
                "split": "val",
                "target_path": "targets/target.npz",
                "patch": {"y0": index // 5, "x0": index % 5, "height": 1, "width": 1},
            }
        )
    index_path = tmp_path / "samples.jsonl"
    index_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    dataset = _RecordDataset(tmp_path, records, index_path)
    output = tmp_path / "screening_validation_indices.json"
    config = {"dataloader": {"sample_pattern": "synthetic_h1", "target_horizon": 1}, "patch_size": 1}

    first = ensure_screening_validation_indices(dataset, config, requested_samples=10, seed=12345, output_path=output)
    first_text = output.read_text(encoding="utf-8")
    second = ensure_screening_validation_indices(dataset, config, requested_samples=10, seed=12345, output_path=output)
    assert first == second
    assert output.read_text(encoding="utf-8") == first_text
    assert first["screening_counts"]["fire"] == 6
    assert first["screening_counts"]["no_fire"] == 4
    assert first["full_validation_counts"] == {
        "total": 20,
        "fire": 12,
        "no_fire": 8,
        "fire_percent": 60.0,
        "no_fire_percent": 40.0,
    }


def test_training_policy_reuses_same_stratified_sample_loader_every_epoch(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    mask = np.zeros((2, 4), dtype=np.float32)
    mask.reshape(-1)[:4] = 1.0
    np.savez(target_dir / "target.npz", fire_mask=mask)
    records = [
        {
            "sample_id": f"sample_{index}", "fire_name": f"fire_{index % 2}", "split": "val",
            "target_path": "targets/target.npz",
            "patch": {"y0": index // 4, "x0": index % 4, "height": 1, "width": 1},
        }
        for index in range(8)
    ]
    index_path = tmp_path / "samples.jsonl"
    index_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    dataset = _RecordDataset(tmp_path, records, index_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    config = {
        "patch_size": 1,
        "dataloader": {"sample_pattern": "synthetic_h1", "target_horizon": 1},
        "training": {
            "validation": {
                "screening": {
                    "enabled": True, "max_batches": 2, "sampling": "stratified_fixed", "seed": 12345,
                    "index_path": str(tmp_path / "screening.json"),
                }
            },
            "performance": {},
        },
    }
    policy = resolve_validation_policy(config, val_loader=loader)
    epoch_one, indices_one = validation_loader_for_epoch(loader, policy, 1)
    epoch_two, indices_two = validation_loader_for_epoch(loader, policy, 2)
    assert policy["validation_mode"] == "stratified_fixed_every_epoch"
    assert policy["screening_counts"]["fire"] == 2
    assert policy["screening_counts"]["no_fire"] == 2
    assert indices_one == indices_two == policy["selected_sample_indices"]
    assert epoch_one is epoch_two
    assert len(epoch_one.dataset) == 4


def _targets() -> torch.Tensor:
    target = torch.zeros(4, 4, 2, 2)
    target[1, 2, 0, 0] = 1.0
    target[3, 2, 1, 1] = 1.0
    target[:, 0] = 0.25
    target[:, 1] = 0.5
    target[:, 3] = 0.75
    return target


def test_full_validation_accumulator_counts_and_exact_metrics() -> None:
    target = _targets()
    prediction = target.clone()
    prediction[:, 2] = -10.0
    prediction[1, 2, 0, 0] = 10.0
    prediction[3, 2, 1, 1] = 10.0
    prediction[0, 2, 0, 1] = 10.0  # one FP pixel in one of two no-fire patches
    accumulator = FullValidationAccumulator()
    accumulator.update(prediction[:2], target[:2])
    accumulator.update(prediction[2:], target[2:])
    metrics = accumulator.finalize()
    assert metrics["full_val_total_patch_count"] == 4
    assert metrics["full_val_fire_patch_count"] == 2
    assert metrics["full_val_no_fire_patch_count"] == 2
    assert metrics["full_val_no_fire_mask_false_positive_rate"] == pytest.approx(1.0 / 8.0)
    assert metrics["full_val_no_fire_patch_false_positive_rate"] == pytest.approx(1.0 / 2.0)
    assert metrics["full_val_surface_mae"] == pytest.approx(0.0)
    assert metrics["full_val_canopy_mae"] == pytest.approx(0.0)
    assert metrics["full_val_energy_log_mae"] == pytest.approx(0.0)


class _FullDataset(torch.utils.data.Dataset):
    input_normalization_on_device = False

    def __init__(self) -> None:
        self.target = _targets()

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        prediction = self.target[index].clone()
        prediction[2] = -10.0
        if self.target[index, 2].any():
            prediction[2][self.target[index, 2] > 0.5] = 10.0
        return torch.stack([prediction]), self.target[index], {
            "sample_id": f"sample_{index}",
            "fire_name": "alpha" if index < 2 else "beta",
        }


class _IdentityForecast(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0]


def test_full_evaluator_visits_every_sample_once_without_shuffle_or_drop_last(tmp_path: Path) -> None:
    loader = torch.utils.data.DataLoader(_FullDataset(), batch_size=3, shuffle=False, drop_last=False)
    result = evaluate_full_validation(
        model=_IdentityForecast(),
        val_loader=loader,
        config={"model": {"input_channels": 4}},
        device=torch.device("cpu"),
        amp_dtype=None,
        run_dir=tmp_path,
        checkpoint_path=tmp_path / "best_model.pt",
        checkpoint_epoch=2,
        expected_counts={"total": 4, "fire": 2, "no_fire": 2},
    )
    assert result["metrics"]["full_val_total_patch_count"] == len(loader.dataset)
    assert sum(item["sample_count"] for item in result["per_fire"].values()) == len(loader.dataset)
    payload = json.loads((tmp_path / "full_validation_metrics.json").read_text(encoding="utf-8"))
    assert payload["evaluated_sample_count"] == 4
    assert payload["unique_sample_id_count"] == 4
    assert (tmp_path / "full_validation_per_fire.json").is_file()

    bad_loader = torch.utils.data.DataLoader(_FullDataset(), batch_size=3, shuffle=False, drop_last=True)
    with pytest.raises(RuntimeError, match="drop_last=false"):
        evaluate_full_validation(
            model=_IdentityForecast(), val_loader=bad_loader,
            config={"model": {"input_channels": 4}}, device=torch.device("cpu"), amp_dtype=None,
            run_dir=tmp_path / "bad", checkpoint_path=tmp_path / "best_model.pt", checkpoint_epoch=2,
        )


def test_summarizer_prefers_full_validation_file_over_screening(tmp_path: Path) -> None:
    root = tmp_path / "ablations"
    run = root / "baseline" / "run1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "ablation": "baseline",
                "best_epoch": 1,
                "best_screening_validation": {"mask_dice": 0.9, "mask_iou": 0.8},
                "full_validation": {"full_val_dice": 0.3, "full_val_iou": 0.2},
            }
        ),
        encoding="utf-8",
    )
    (run / "full_validation_metrics.json").write_text(
        json.dumps({"metrics": {"full_val_dice": 0.4, "full_val_iou": 0.25}}), encoding="utf-8"
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text("ablations:\n  baseline:\n    name: baseline\n", encoding="utf-8")
    row = discover_rows(root, registry)[0]
    assert row["Screening Val Dice"] == pytest.approx(0.9)
    assert row["Full Val Dice"] == pytest.approx(0.4)
    assert row["Full Val IoU"] == pytest.approx(0.25)
