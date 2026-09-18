#!/usr/bin/env python3
"""Create or verify the shared CAWFE-Latte screening-validation indices."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.validation_subset import DEFAULT_SCREENING_INDEX_PATH, ensure_screening_validation_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ablations/cawfe_latte_baseline.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    _train_loader, val_loader, _test_loader = create_dataloaders(config)
    training = config.get("training", {})
    validation = training.get("validation", {}) if isinstance(training, Mapping) else {}
    screening = validation.get("screening", {}) if isinstance(validation, Mapping) else {}
    if not isinstance(screening, Mapping) or not bool(screening.get("enabled", False)):
        raise RuntimeError("The CAWFE-Latte config does not enable training.validation.screening.")
    if str(screening.get("sampling", "")).lower() != "stratified_fixed":
        raise RuntimeError("The CAWFE-Latte screening sampler is not stratified_fixed.")
    batch_size = int(getattr(val_loader, "batch_size", 0) or 0)
    if batch_size <= 0:
        raise RuntimeError("Validation loader has no positive batch size.")
    requested_samples = min(len(val_loader.dataset), int(screening.get("max_batches", 50)) * batch_size)
    payload = ensure_screening_validation_indices(
        val_loader.dataset,
        config,
        requested_samples=requested_samples,
        seed=int(screening.get("seed", 12345)),
        output_path=screening.get("index_path", DEFAULT_SCREENING_INDEX_PATH),
    )
    screening_counts = payload["screening_counts"]
    full_counts = payload["full_validation_counts"]
    print(f"Shared screening index: {Path(screening.get('index_path', DEFAULT_SCREENING_INDEX_PATH)).resolve()}")
    print(
        "SCREENING "
        f"total={screening_counts['total']} fire={screening_counts['fire']} no_fire={screening_counts['no_fire']}"
    )
    print(
        "FULL VALIDATION CANONICAL "
        f"total={full_counts['total']} fire={full_counts['fire']} no_fire={full_counts['no_fire']} "
        f"no_fire_percent={full_counts['no_fire_percent']:.6f}"
    )


if __name__ == "__main__":
    main()
