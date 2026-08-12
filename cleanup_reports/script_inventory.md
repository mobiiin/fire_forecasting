# Script Inventory

Created: 2026-08-12

## Core Pipeline

- `scripts/apply_manual_fire_trim.py`
- `scripts/benchmark_patch_cache_io.py`
- `scripts/cache_engineered_dataset.py`
- `scripts/compute_normalization.py`
- `scripts/debug_model_predictions.py`
- `scripts/discover_fire_datasets.py`
- `scripts/evaluate_trained_models.py`
- `scripts/inspect_normalization_pipeline.py`
- `scripts/inspect_patch_cache.py`
- `scripts/manual_trim_fire_datasets.py`
- `scripts/precompute_patch_cache.py`
- `scripts/slurm_precompute_cache_with_config.sh`
- `scripts/train_forecasting_model.py`
- `scripts/trim_prefire_frames.py`
- `scripts/visualize_patch_cache.py`

## Model-Specific Wrappers

- `scripts/train_convlstm_unet.py`
- `scripts/train_earthformer_lite.py`
- `scripts/train_forecasting_model.py`
- `scripts/train_st_mamba_lite.py`
- `scripts/train_weatherformer_lite.py`

## Diagnostics

- `scripts/debug_model_predictions.py`
- `scripts/diagnose_mask_generalization.py`
- `scripts/diagnose_model_architectures.py`
- `scripts/diagnose_training_pipeline.py`
- `scripts/inspect_atmospheric_features.py`
- `scripts/inspect_dataset.py`
- `scripts/inspect_energy_release.py`
- `scripts/inspect_engineered_features.py`
- `scripts/inspect_multi_fire_dataset.py`
- `scripts/inspect_normalization_pipeline.py`
- `scripts/inspect_patch_cache.py`
- `scripts/inspect_target_channels.py`
- `scripts/sanity_check_project.py`
- `scripts/verify_training_outputs.py`

## Candidate For Future Consolidation

- `scripts/evaluate_all_baselines.py`
- `scripts/evaluate_consumed_fuel_derived_mask.py`
- `scripts/evaluate_persistence_all_candidate_targets.py`
- `scripts/evaluate_persistence_baseline.py`
- `scripts/plot_training_curves.py`
- `scripts/run_full_pipeline.sh`
- `scripts/run_linear_extrapolation_baseline.py`
- `scripts/run_persistence_baseline.py`
- `scripts/run_quick_smoke_test.sh`
- `scripts/run_training_only.sh`
- `scripts/run_visualization_only.sh`
- `scripts/smoke_test_earthformer_lite.py`
- `scripts/smoke_test_st_mamba_lite.py`
- `scripts/smoke_test_weatherformer_lite.py`
- `scripts/test_earthformer_lite.py`
- `scripts/test_model.py`
- `scripts/test_spatial_size_compatibility.py`
- `scripts/test_st_mamba_lite.py`
- `scripts/test_weatherformer_lite.py`
- `scripts/visualize_input_dataset.py`
- `scripts/visualize_model_vs_persistence.py`
- `scripts/visualize_predictions.py`
- `scripts/visualize_st_mamba_lite_predictions.py`

Note: this inventory intentionally does not recommend deleting non-Latte scripts in this cleanup pass.
