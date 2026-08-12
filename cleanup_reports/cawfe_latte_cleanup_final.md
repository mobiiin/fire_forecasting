# CAWFE-Latte Cleanup Final Report

Created: 2026-08-12

## Files Deleted

- Old model modules under `src/models/cawfe_latte*.py`
- Latte-only tuning utility `src/training/hyperparameter_tuning.py`
- Latte train/test/smoke/visualization/ablation/tuning scripts under `scripts/`
- Latte Slurm launchers under `slurm/`
- Latte-only tests: `tests/test_cawfe_latte.py`, `tests/test_cawfe_latte_lite.py`, `tests/test_hyperparameter_tuning.py`

No files under `artifacts/`, data, scratch, patch cache, normalization stats, checkpoints, or run outputs were removed.

## Files Archived

- `cawfe_latte.md` moved to `docs/archive/old_cawfe_latte/cawfe_latte.md`

## Files Edited

- `configs/default.yaml`
- `src/models/architecture_registry.py`
- `src/models/model_factory.py`
- `src/models/__init__.py`
- `src/models/input_adapters.py`
- `src/config.py`
- `scripts/evaluate_trained_models.py`
- `scripts/evaluate_all_baselines.py`
- `scripts/diagnose_model_architectures.py`
- `tests/test_debug_model_predictions.py`
- `tests/test_convlstm_mask_gated_regression.py`
- `tests/test_run_manager.py`
- `README.md`
- `run.md`

## Files Added

- `cleanup_reports/cawfe_latte_cleanup_audit.md`
- `cleanup_reports/cawfe_latte_cleanup_final.md`
- `cleanup_reports/script_inventory.md`
- `tests/test_removed_architectures.py`

## Remaining Active Learned Architectures

- `convlstm_unet`
- `earthformer_lite`
- `cawfe_st_mamba` aliasing `st_mamba_lite`
- `weatherformer_lite`

Baselines remain:

- `persistence`
- `linear_extrapolation`

## Removed Architecture Behavior

Requests for `cawfe_latte` or `cawfe_latte_lite` now fail clearly with:

`The old CAWFE-Latte implementation has been removed. A new design will be added later.`

## Validation Run

- `python -m py_compile src/models/architecture_registry.py src/models/model_factory.py src/models/__init__.py src/models/input_adapters.py src/config.py scripts/train_forecasting_model.py scripts/evaluate_trained_models.py scripts/debug_model_predictions.py scripts/precompute_patch_cache.py scripts/compute_normalization.py scripts/evaluate_all_baselines.py scripts/diagnose_model_architectures.py`
- Config load smoke check for `configs/default.yaml`
- Model registry/factory smoke check for active and removed architectures
- `pytest tests/test_removed_architectures.py tests/test_debug_model_predictions.py tests/test_convlstm_mask_gated_regression.py tests/test_run_manager.py -q`: 8 passed, 3 skipped
- `pytest tests -q`: 100 passed, 19 skipped

## Remaining Latte References

Intentional remaining references are limited to:

- removal notes in `README.md` and `run.md`
- removed-architecture error messages/tests
- cleanup reports
- archived historical doc under `docs/archive/old_cawfe_latte/`

No active model, training, evaluation, config, Slurm, or script path remains for the old CAWFE-Latte implementation.
