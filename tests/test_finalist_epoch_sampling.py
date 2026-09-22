"""Tests for deterministic partial-epoch finalist training sampling."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import EpochRandomSubsetSampler
from src.training.train import (
    _build_training_sampling_protocol,
    _set_training_sampler_epoch,
    _validate_resume_training_sampling_protocol,
)


FULL_TRAINING_SAMPLES = 312_675
BATCH_SIZE = 8
BATCHES_PER_EPOCH = 7_500
SAMPLES_PER_EPOCH = BATCH_SIZE * BATCHES_PER_EPOCH


def test_exactly_7500_batches_and_60000_unique_samples_are_consumed() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(FULL_TRAINING_SAMPLES))
    sampler = EpochRandomSubsetSampler(
        dataset_size=len(dataset), num_samples=SAMPLES_PER_EPOCH, seed=42
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    _set_training_sampler_epoch(loader, 1)
    actual_batches = 0
    actual_samples = 0
    for (indices,) in loader:
        actual_batches += 1
        actual_samples += int(indices.numel())
    assert actual_batches == BATCHES_PER_EPOCH
    assert actual_samples == SAMPLES_PER_EPOCH
    assert len(sampler.last_indices) == SAMPLES_PER_EPOCH
    assert len(set(sampler.last_indices)) == SAMPLES_PER_EPOCH


def test_epoch_schedule_is_deterministic_paired_and_epoch_dependent() -> None:
    # Architecture is deliberately not an input to the sampler. Two model runs
    # with the same training seed therefore get the same temporal-index schedule.
    baseline = EpochRandomSubsetSampler(dataset_size=1000, num_samples=200, seed=123)
    cgp = EpochRandomSubsetSampler(dataset_size=1000, num_samples=200, seed=123)
    different_seed = EpochRandomSubsetSampler(dataset_size=1000, num_samples=200, seed=2026)

    baseline_epoch_1 = baseline.indices_for_epoch(1)
    baseline_epoch_2 = baseline.indices_for_epoch(2)
    assert baseline_epoch_1 != baseline_epoch_2
    assert baseline_epoch_1 == baseline.indices_for_epoch(1)
    assert baseline_epoch_1 == cgp.indices_for_epoch(1)
    assert baseline_epoch_1 != different_seed.indices_for_epoch(1)
    assert len(baseline_epoch_1) == len(set(baseline_epoch_1))
    assert len(baseline_epoch_2) == len(set(baseline_epoch_2))


def test_sampling_protocol_reports_partial_epoch_exposure_without_touching_validation() -> None:
    train_dataset = torch.utils.data.TensorDataset(torch.arange(FULL_TRAINING_SAMPLES))
    sampler = EpochRandomSubsetSampler(
        dataset_size=len(train_dataset), num_samples=SAMPLES_PER_EPOCH, seed=42
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False
    )
    validation_dataset = torch.utils.data.TensorDataset(torch.arange(101))
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    config = {
        "seed": 42,
        "training": {
            "seed": 42,
            "batch_size": BATCH_SIZE,
            "max_train_batches": BATCHES_PER_EPOCH,
        },
    }
    protocol = _build_training_sampling_protocol(config, train_loader, max_epochs=60)
    assert protocol is not None
    assert protocol["batches_per_epoch"] == BATCHES_PER_EPOCH
    assert protocol["samples_per_epoch"] == SAMPLES_PER_EPOCH
    assert protocol["approximate_fraction_per_epoch"] == pytest.approx(
        SAMPLES_PER_EPOCH / FULL_TRAINING_SAMPLES
    )
    assert protocol["estimated_full_dataset_equivalent_epochs_at_max"] == pytest.approx(
        60 * SAMPLES_PER_EPOCH / FULL_TRAINING_SAMPLES
    )
    assert protocol["sampling_within_epoch"] == "without_replacement"
    assert protocol["subset_changes_each_epoch"] is True
    assert protocol["epoch_seed_rule"] == "base_seed + one_based_epoch"
    # The new control belongs only to the training loader; validation remains complete/sequential.
    assert len(validation_loader.dataset) == 101
    assert len(validation_loader) == 13
    assert isinstance(validation_loader.sampler, torch.utils.data.SequentialSampler)



def test_legacy_or_different_budget_checkpoint_is_rejected_before_resume() -> None:
    current = {
        "protocol_id": "epoch_random_subset_without_replacement_v1",
        "full_training_samples": FULL_TRAINING_SAMPLES,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": BATCHES_PER_EPOCH,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "sampling_within_epoch": "without_replacement",
        "subset_changes_each_epoch": True,
        "base_seed": 42,
        "epoch_seed_rule": "base_seed + one_based_epoch",
    }
    with pytest.raises(RuntimeError, match="Refusing to resume a legacy"):
        _validate_resume_training_sampling_protocol(current, None)
    incompatible = dict(current, batches_per_epoch=39_085)
    with pytest.raises(RuntimeError, match="Refusing to resume a legacy"):
        _validate_resume_training_sampling_protocol(current, incompatible)
    _validate_resume_training_sampling_protocol(current, dict(current))
