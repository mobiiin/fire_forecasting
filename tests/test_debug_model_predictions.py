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
			{"architecture": "earthformer_lite", "model_state_dict": {}},
			checkpoint_path,
			requested_architecture="convlstm_unet",
			allow_architecture_mismatch=False,
		)


def _multitask_config() -> dict:
	return {
		"model": {"task_type": "multitask"},
		"training": {"task_type": "multitask"},
		"metrics": {"task_type": "multitask"},
		"multitask": {"energy_active_threshold_MW": 0.001, "consumed_active_threshold": 0.001},
	}


def test_background_diagnostic_computes_expected_inactive_active_means() -> None:
	y = torch.zeros(1, 4, 2, 2)
	y[:, 2, 0, 0] = 1.0
	y[:, 0, 0, 0] = 2.0
	pred = torch.zeros_like(y)
	pred[:, 0] = torch.tensor([[4.0, 1.0], [1.0, 1.0]])
	pred[:, 1] = 2.0
	pred[:, 2] = 0.0
	pred[:, 3] = 3.0

	rows, summary = debug.compute_background_diagnostics(pred, y, active_definition="mask_only", inactive_threshold=1.0e-6)

	assert rows[0]["active_pixel_count"] == 1
	assert rows[0]["inactive_pixel_count"] == 3
	assert rows[0]["active_pred_surface_mean"] == pytest.approx(4.0)
	assert rows[0]["inactive_pred_surface_mean"] == pytest.approx(1.0)
	assert summary["inactive_pred_canopy"]["mean"] == pytest.approx(2.0)


def test_mask_gating_threshold_zero_keeps_positive_mask_predictions_active() -> None:
	pred = torch.zeros(1, 4, 2, 2)
	pred[:, 0] = 5.0
	pred[:, 1] = 6.0
	pred[:, 2] = 0.0
	pred[:, 3] = 7.0

	gated = debug.apply_predicted_mask_gating(pred, threshold=0.0)

	assert torch.equal(gated[:, 0], pred[:, 0])
	assert torch.equal(gated[:, 1], pred[:, 1])
	assert torch.equal(gated[:, 3], pred[:, 3])
	assert torch.equal(gated[:, 2], pred[:, 2])


def test_oracle_gating_zeros_predictions_outside_target_active_mask() -> None:
	y = torch.zeros(1, 4, 2, 2)
	y[:, 2, 0, 0] = 1.0
	pred = torch.ones(1, 4, 2, 2)

	gated = debug.apply_oracle_gating(pred, y, active_definition="mask_only")

	assert gated[0, 0, 0, 0].item() == pytest.approx(1.0)
	assert gated[0, 0, 0, 1].item() == pytest.approx(0.0)
	assert gated[0, 1, 1, 0].item() == pytest.approx(0.0)
	assert gated[0, 3, 1, 1].item() == pytest.approx(0.0)
	assert torch.equal(gated[:, 2], pred[:, 2])


def test_checkpoint_comparison_sample_keys_use_same_metadata_indices() -> None:
	metadata = [
		{"fire_name": "fire_a", "sample_index": 10, "target_idx": 12, "patch_top": 0, "patch_left": 0},
		{"fire_name": "fire_a", "sample_index": 11, "target_idx": 13, "patch_top": 0, "patch_left": 64},
	]

	keys_a = [debug._metadata_sample_key(item, index) for index, item in enumerate(metadata)]
	keys_b = [debug._metadata_sample_key(item, index) for index, item in enumerate(metadata)]

	assert keys_a == keys_b


def test_diagnostics_summary_text_includes_main_conclusions() -> None:
	text = debug._diagnostics_summary_text(
		{
			"background_summary": {
				"inactive_pred_surface": {"mean": 0.1},
				"inactive_pred_canopy": {"mean": 0.0},
				"inactive_pred_energy_log": {"mean": 0.0},
				"inactive_pred_mask_prob": {"mean": 0.2},
			}
		}
	)

	assert "Main Diagnostic Conclusions" in text
	assert "Background overprediction" in text


def test_mask_and_oracle_gating_diagnostics_return_metric_rows() -> None:
	y = torch.zeros(1, 4, 2, 2)
	y[:, 2, 0, 0] = 1.0
	pred = torch.zeros_like(y)
	pred[:, 0] = 1.0
	pred[:, 2] = 0.0

	mask_rows, mask_summary = debug.compute_mask_gating_diagnostics(pred, y, _multitask_config(), thresholds=[0.5])
	oracle_rows, oracle_summary = debug.compute_oracle_gating_diagnostics(pred, y, _multitask_config(), active_definition="mask_only")

	assert mask_rows
	assert "raw_metrics" in mask_summary
	assert oracle_rows
	assert "relative_improvements" in oracle_summary
