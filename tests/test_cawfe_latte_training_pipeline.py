"""Focused CAWFE-Latte processed-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.data.dataset import create_dataloaders  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.batch_utils import unpack_batch  # noqa: E402
from src.training.losses import get_loss_function  # noqa: E402
from src.training.model_outputs import extract_aux_outputs, extract_prediction  # noqa: E402


C = 129
T = 5
H = 8
W = 8


def _write_fake_processed_dataset(root: Path) -> Path:
    records = []
    root.mkdir(parents=True, exist_ok=True)
    (root / "indices" / "temporal").mkdir(parents=True, exist_ok=True)
    (root / "normalization").mkdir(parents=True, exist_ok=True)
    split_fires = {"train": ["TRAIN_FIRE"], "val": ["VAL_FIRE"], "test": ["TEST_FIRE"]}
    for split, fires in split_fires.items():
        for fire_index, fire in enumerate(fires):
            frame_dir = root / "fires" / fire / "frames"
            terrain_dir = root / "fires" / fire / "terrain"
            target_dir = root / "targets" / "h10" / fire
            frame_dir.mkdir(parents=True, exist_ok=True)
            terrain_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            terrain = np.stack(
                [
                    np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W),
                    np.full((H, W), 0.25, dtype=np.float32),
                    np.linspace(-1, 1, H * W, dtype=np.float32).reshape(H, W),
                    np.linspace(1, -1, H * W, dtype=np.float32).reshape(H, W),
                ],
                axis=0,
            )
            np.save(terrain_dir / "terrain_features.npy", terrain.astype(np.float32))
            for idx in range(T + 10 + 1):
                frame = np.full((C, H, W), fill_value=0.1 * (idx + 1 + fire_index), dtype=np.float32)
                np.savez_compressed(frame_dir / f"frame_{idx:06d}.npz", x_engineered=frame)
            surface = np.full((H, W), 0.1, dtype=np.float32)
            canopy = np.full((H, W), 0.05, dtype=np.float32)
            fire_mask = np.zeros((H, W), dtype=np.float32)
            fire_mask[2:4, 3:5] = 1.0
            energy_log = np.full((H, W), 0.2, dtype=np.float32)
            target_rel = f"targets/h10/{fire}/target_current_000004_future_000014.npz"
            np.savez_compressed(root / target_rel, surface_consumed=surface, canopy_consumed=canopy, fire_mask=fire_mask, energy_log=energy_log)
            records.append(
                {
                    "sample_id": f"{fire}_sample",
                    "split": split,
                    "fire_name": fire,
                    "dataset_name": fire,
                    "current_index": 4,
                    "target_index": 14,
                    "horizon": 10,
                    "input_indices": list(range(T)),
                    "target_path": target_rel,
                    "patch": {"y0": 0, "x0": 0, "height": H, "width": W},
                }
            )
    sample_path = root / "indices" / "temporal" / "samples_consecutive5_h10.jsonl"
    sample_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    (root / "dataset_manifest.json").write_text(json.dumps({"splits": {f"{split}_fires": fires for split, fires in split_fires.items()}}), encoding="utf-8")
    (root / "channel_manifest.json").write_text(json.dumps({"num_channels": C, "channels": [f"ch_{i}" for i in range(C)]}), encoding="utf-8")
    mean = np.zeros(C, dtype=np.float32)
    std = np.ones(C, dtype=np.float32)
    np.savez_compressed(root / "normalization" / "latest_normalization.npz", mean=mean, std=std, fit_split=np.asarray(["train"]))
    (root / "normalization" / "latest_normalization.json").write_text(
        json.dumps({"npz_path": str(root / "normalization" / "latest_normalization.npz"), "fit_split": "train"}),
        encoding="utf-8",
    )
    return sample_path


def _config(root: Path) -> dict:
    return {
        "task_type": "multitask",
        "input_sequence_length": T,
        "prediction_horizon": 10,
        "model": {"architecture": "cawfe_latte", "input_channels": C, "output_channels": 4},
        "energy_release": {"enabled": True, "output_mode": "total", "target_transform": "log1p"},
        "processed_dataset": {"root": str(root)},
        "dataloader": {
            "source": "processed_full_frames",
            "sample_pattern": "consecutive5_h10",
            "normalize_inputs": True,
            "return_terrain": "auto",
        },
        "normalization": {
            "enabled": True,
            "require_stats": True,
            "stats_path": str(root / "normalization" / "latest_normalization.json"),
        },
        "data_loader": {
            "batch_size": 1,
            "num_workers": 0,
            "train": {"batch_size": 1, "num_workers": 0},
            "val": {"batch_size": 1, "num_workers": 0},
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "training": {
            "task_type": "multitask",
            "batch_size": 1,
            "num_workers": 0,
            "input_normalization_device": "none",
            "loss": {
                "surface": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "canopy": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "mask": {"type": "bce_dice", "weight": 5.0, "bce_weight": 1.0, "dice_weight": 1.0},
                "energy": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "auxiliary_fire_support": {"enabled": True, "weight": 0.2},
            },
        },
        "cawfe_latte": {
            "input_channels": C,
            "input_sequence_length": T,
            "output_channels": 4,
            "output_dim": 16,
            "use_terrain_conditioning": True,
            "atmosphere": {"out_dim": 16},
            "wind": {"out_dim": 16},
            "fire_fuel": {"out_dim": 16},
            "flux_energy": {"out_dim": 16},
            "fusion": {"input_format": "tokens", "dim": 16, "num_heads": 4},
            "terrain_encoder": {"in_channels": 4, "hidden_dim": 8, "out_dim": 16},
            "terrain_film": {"dim": 16},
            "backbone": {"dim": 16, "num_blocks": 1},
            "decoder": {"in_dim": 16, "hidden_dim": 16, "num_blocks": 1},
            "auxiliary": {"fire_support_head": {"enabled": True, "weight": 0.2}},
        },
    }


def test_processed_dataloader_auto_returns_aligned_terrain(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    _write_fake_processed_dataset(root)
    train_loader, val_loader, test_loader = create_dataloaders(_config(root))
    assert len(train_loader.dataset) == len(val_loader.dataset) == len(test_loader.dataset) == 1
    x, y, extra = unpack_batch(next(iter(train_loader)))
    terrain = extra["terrain"]
    assert tuple(x.shape) == (1, T, C, H, W)
    assert tuple(y.shape) == (1, 4, H, W)
    assert tuple(terrain.shape) == (1, 4, H, W)
    assert bool(torch.isfinite(terrain).all())
    assert float(terrain[:, 0].min()) >= 0.0 and float(terrain[:, 0].max()) <= 1.0
    assert float(terrain[:, 2].min()) >= -1.0 and float(terrain[:, 2].max()) <= 1.0


def test_cawfe_latte_processed_batch_loss_and_optimizer_step(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    _write_fake_processed_dataset(root)
    config = _config(root)
    train_loader, _val_loader, _test_loader = create_dataloaders(config)
    x, y, extra = unpack_batch(next(iter(train_loader)))
    terrain = extra["terrain"]
    model = build_model_from_config(config, input_channels=C)
    output = model(x, terrain=terrain)
    pred = extract_prediction(output)
    aux = extract_aux_outputs(output)
    assert tuple(pred.shape) == (1, 4, H, W)
    assert "aux_fire_support_logits" in aux
    criterion = get_loss_function(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    loss_result = criterion(output, y)
    assert "loss_aux_fire_support_total" in loss_result
    loss = loss_result["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_cawfe_latte_requires_terrain_when_enabled(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    _write_fake_processed_dataset(root)
    config = _config(root)
    model = build_model_from_config(config, input_channels=C)
    x = torch.randn(1, T, C, H, W)
    with pytest.raises(ValueError, match="terrain conditioning is enabled"):
        model(x)
