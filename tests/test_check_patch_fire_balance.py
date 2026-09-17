import json

import numpy as np

from scripts.check_patch_fire_balance import (
    check_balance,
    classify_patch,
    get_patch,
    get_target_path,
    load_fire_mask,
)


def test_classify_patch_uses_cropped_fire_mask():
    mask = np.zeros((6, 8), dtype=np.float32)
    mask[2, 3] = 1.0
    patch = {"y0": 1, "x0": 2, "height": 3, "width": 4}
    result = classify_patch(mask, patch, threshold=0.5, active_fraction_threshold=0.0)
    assert result["has_fire"] is True
    assert result["active_pixels"] == 1
    assert result["total_pixels"] == 12
    assert result["bin"] == "large_fire"


def test_patch_fallbacks_and_target_path_fallback(tmp_path):
    sample = {
        "fire": "FIRE_A",
        "patch_y": 2,
        "patch_x": 3,
        "patch_h": 4,
        "patch_w": 5,
        "horizon": 10,
        "current_index": 7,
        "future_index": 17,
    }
    assert get_patch(sample) == {"y0": 2, "x0": 3, "height": 4, "width": 5}
    assert get_target_path(sample, tmp_path) == tmp_path / "targets" / "h10" / "FIRE_A" / "target_current_000007_future_000017.npz"


def test_load_fire_mask_prefers_named_key_and_supports_stacked_target(tmp_path):
    direct = tmp_path / "direct.npz"
    np.savez_compressed(direct, fire_mask=np.ones((2, 3), dtype=bool))
    np.testing.assert_array_equal(load_fire_mask(direct), np.ones((2, 3), dtype=bool))

    stacked = tmp_path / "stacked.npz"
    y = np.zeros((4, 2, 3), dtype=np.float32)
    y[2, 1, 2] = 1.0
    np.savez_compressed(stacked, y=y)
    np.testing.assert_array_equal(load_fire_mask(stacked), y[2])


def test_check_balance_counts_fire_no_fire_bins_and_per_fire(tmp_path):
    root = tmp_path / "dataset"
    sample_dir = root / "indices" / "temporal"
    target_dir = root / "targets" / "h10" / "FIRE_A"
    sample_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    mask0 = np.zeros((4, 4), dtype=np.float32)
    mask1 = np.zeros((4, 4), dtype=np.float32)
    mask1[0, 0] = 1.0
    np.savez_compressed(target_dir / "target_current_000000_future_000010.npz", fire_mask=mask0)
    np.savez_compressed(target_dir / "target_current_000001_future_000011.npz", fire_mask=mask1)

    records = [
        {
            "split": "train",
            "fire_name": "FIRE_A",
            "patch": {"y0": 0, "x0": 0, "height": 2, "width": 2},
            "target_path": "targets/h10/FIRE_A/target_current_000000_future_000010.npz",
        },
        {
            "split": "train",
            "fire_name": "FIRE_A",
            "patch": {"y0": 0, "x0": 0, "height": 2, "width": 2},
            "target_path": "targets/h10/FIRE_A/target_current_000001_future_000011.npz",
        },
    ]
    (sample_dir / "samples_consecutive5_h10.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    summary = check_balance(root, sample_pattern="consecutive5_h10", split="train")
    assert summary["counts"]["total"] == 2
    assert summary["counts"]["fire"] == 1
    assert summary["counts"]["no_fire"] == 1
    assert summary["counts"]["fire_percent"] == 50.0
    assert summary["counts"]["bins"]["no_fire"] == 1
    assert summary["counts"]["bins"]["large_fire"] == 1
    assert summary["per_fire"]["FIRE_A"]["total"] == 2
