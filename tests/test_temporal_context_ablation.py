"""Temporal-context sample-index and job-inventory regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_temporal_context_ablation import (
    MODE_ORDER,
    MODE_T,
    SEEDS,
    build_common_records,
    mode_record,
)


def _write_timeline(root: Path, fire: str, split: str) -> None:
    fire_root = root / "fires" / fire
    frame_root = fire_root / "frames"
    target_root = root / "targets" / "h10" / fire
    frame_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for minute in range(61):
        rows.append({"local_index": minute, "original_index": minute, "source_raw_file": f"tensor{minute:04d}.npy"})
        (frame_root / f"frame_{minute:06d}.npz").touch()
    (fire_root / "frame_manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    for current in range(40, 51):
        (target_root / f"target_current_{current:06d}_future_{current + 10:06d}.npz").touch()


def test_common_index_enforces_exact_minutes_and_equal_cases(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    patch_root = root / "indices" / "patches"
    patch_root.mkdir(parents=True)
    fires = {"train": ["TRAIN_FIRE"], "val": ["VAL_FIRE"]}
    patches = []
    for split, names in fires.items():
        del split
        for fire in names:
            _write_timeline(root, fire, split="unused")
            patches.append(
                {
                    "fire_name": fire,
                    "patch_id": f"{fire}_p0",
                    "y0": 0,
                    "x0": 0,
                    "height": 64,
                    "width": 64,
                }
            )
    (patch_root / "patches_64_stride60_border.jsonl").write_text(
        "\n".join(json.dumps(row) for row in patches) + "\n", encoding="utf-8"
    )
    common, audit = build_common_records(root, fires)
    assert len(common["train"]) == len(common["val"]) == 11
    assert audit["dense_span_minutes"]["train"] == {"4": 11}
    for split in ("train", "val"):
        for shared in common[split]:
            records = {mode: mode_record(shared, mode) for mode in MODE_ORDER}
            assert len({record["sample_id"] for record in records.values()}) == 1
            assert records["single"]["temporal_offsets_minutes"] == [0]
            assert records["sparse5"]["temporal_offsets_minutes"] == [-40, -30, -20, -10, 0]
            assert records["dense5"]["temporal_offsets_minutes"] == [-4, -3, -2, -1, 0]
            assert records["single"]["target_time_minutes"] == records["single"]["current_time_minutes"] + 10
            assert {len(records[mode]["input_indices"]) for mode in MODE_ORDER} == {1, 5}
            assert [len(records[mode]["input_indices"]) for mode in MODE_ORDER] == [MODE_T[mode] for mode in MODE_ORDER]


def test_exact_slurm_inventory_and_fixed_mode_seed_assignments() -> None:
    root = Path(__file__).resolve().parents[1] / "slurm" / "temporal_ablation"
    expected = ["00_prepare_temporal_ablation.slurm"]
    expected.extend(
        [
            "01_single_seed42.slurm",
            "02_single_seed123.slurm",
            "03_single_seed2026.slurm",
            "04_sparse5_seed42.slurm",
            "05_sparse5_seed123.slurm",
            "06_sparse5_seed2026.slurm",
            "07_dense5_seed42.slurm",
            "08_dense5_seed123.slurm",
            "09_dense5_seed2026.slurm",
        ]
    )
    assert sorted(path.name for path in root.glob("*.slurm")) == sorted(expected)
    training = sorted(root.glob("0[1-9]_*.slurm"))
    observed = []
    for path in training:
        text = path.read_text(encoding="utf-8")
        mode = next(mode for mode in MODE_ORDER if f'MODE="{mode}"' in text)
        seed = next(seed for seed in SEEDS if f'SEED="{seed}"' in text)
        assert "Budget: exactly 10 epochs" in text
        observed.append((mode, seed))
    assert observed == [(mode, seed) for mode in MODE_ORDER for seed in SEEDS]
