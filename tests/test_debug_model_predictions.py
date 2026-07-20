from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
debug = pytest.importorskip("scripts.debug_model_predictions")


def test_tensor_stats_returns_required_fields_for_normal_tensor() -> None:
	stats = debug.tensor_stats("normal", torch.tensor([0.0, 1.0, 2.0, 3.0]))

	assert stats["name"] == "normal"
	assert stats["numel"] == 4
	for field in debug.STATS_FIELDS:
		assert field in stats
	assert stats["mean"] == pytest.approx(1.5)
	assert stats["frac_nan"] == 0.0
	assert stats["frac_inf"] == 0.0


def test_tensor_stats_handles_nan_and_inf() -> None:
	stats = debug.tensor_stats("bad_values", torch.tensor([1.0, float("nan"), float("inf"), -float("inf")]))

	assert stats["frac_nan"] == pytest.approx(0.25)
	assert stats["frac_inf"] == pytest.approx(0.5)
	assert math.isfinite(stats["mean"])
	assert stats["mean"] == pytest.approx(1.0)


def test_safe_expm1_log_energy_clamps_large_values_and_avoids_inf() -> None:
	values = debug.safe_expm1_log_energy(torch.tensor([-1.0, 0.0, 2.0, 1000.0]))

	assert torch.isfinite(values).all()
	assert values[0].item() == pytest.approx(0.0)
	assert values[1].item() == pytest.approx(0.0)
	assert values[2].item() == pytest.approx(math.expm1(2.0))
	assert values[3].item() == pytest.approx(math.expm1(20.0))


def test_compute_mask_metrics_handles_empty_prediction_and_target_masks() -> None:
	metrics = debug.compute_mask_metrics(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), threshold=0.5)

	assert metrics["mask_dice"] == pytest.approx(1.0)
	assert metrics["mask_iou"] == pytest.approx(1.0)
	assert metrics["mask_precision"] == pytest.approx(1.0)
	assert metrics["mask_recall"] == pytest.approx(1.0)


def test_plot_sample_debug_handles_all_zero_arrays(tmp_path: Path) -> None:
	output_path = tmp_path / "sample.png"

	debug.plot_sample_debug(
		output_path,
		pred_sample=torch.zeros(4, 8, 8),
		y_sample=torch.zeros(4, 8, 8),
		metadata={},
		title_prefix="unit-test",
	)

	assert output_path.exists()
	assert output_path.stat().st_size > 0


def test_checkpoint_metadata_payload_fails_clearly_on_architecture_mismatch(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / "best_model.pt"
	checkpoint_path.write_bytes(b"placeholder")

	with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
		debug.checkpoint_metadata_payload(
			{"architecture": "cawfe_latte", "model_state_dict": {}},
			checkpoint_path,
			requested_architecture="convlstm_unet",
			allow_architecture_mismatch=False,
		)
