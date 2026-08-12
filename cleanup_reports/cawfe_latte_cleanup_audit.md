# CAWFE-Latte Cleanup Audit

Created: 2026-08-12

## Scope

The old `cawfe_latte` and `cawfe_latte_lite` implementations are removed from the active project. Data, patch caches, checkpoints, run outputs, and artifacts are left untouched.

## Files To Delete

These files are Latte-specific source, script, Slurm, or test files. They are safe to remove because they only import or instantiate the old CAWFE-Latte / CAWFE-Latte-Lite implementation.

- `src/models/cawfe_latte.py`
- `src/models/cawfe_latte_lite.py`
- `src/models/cawfe_latte_blocks.py`
- `src/models/cawfe_latte_vertical.py`
- `src/models/cawfe_latte_fire.py`
- `src/models/cawfe_latte_backbone.py`
- `src/models/cawfe_latte_decoder.py`
- `src/models/cawfe_latte_constraints.py`
- `src/models/cawfe_latte_wind.py`
- `src/models/cawfe_latte_operator.py`
- `src/training/hyperparameter_tuning.py`
- `scripts/train_cawfe_latte.py`
- `scripts/train_cawfe_latte_lite.py`
- `scripts/test_cawfe_latte.py`
- `scripts/test_cawfe_latte_lite.py`
- `scripts/smoke_test_cawfe_latte.py`
- `scripts/smoke_test_cawfe_latte_lite.py`
- `scripts/visualize_cawfe_latte_aux.py`
- `scripts/ablate_cawfe_latte.py`
- `scripts/ablate_cawfe_latte_lite.py`
- `scripts/tune_cawfe_latte.py`
- `scripts/apply_cawfe_latte_tuned_params.py`
- `scripts/submit_cawfe_latte_pipeline.sh`
- `slurm/slurm_tune_cawfe_latte_a10080.sh`
- `slurm/slurm_train_cawfe_latte_tuned_a10080.sh`
- `slurm/slurm_train_cawfe_latte_lite_a10080.sh`
- `slurm/slurm_train_cawfe_latte_h100.sh`
- `slurm/slurm_ablate_cawfe_latte_a10080.sh`
- `tests/test_cawfe_latte.py`
- `tests/test_cawfe_latte_lite.py`
- `tests/test_hyperparameter_tuning.py`

## Files To Archive

- `cawfe_latte.md` -> `docs/archive/old_cawfe_latte/cawfe_latte.md`

The design doc may be useful as historical context, but it should not be linked from active README/run instructions.

## Files To Edit

- `src/models/model_factory.py`: remove old imports/build branches; raise a clear removed-architecture error.
- `src/models/architecture_registry.py`: remove active registry entries; keep aliases that point only to active models.
- `src/models/__init__.py`: remove old package exports.
- `src/models/input_adapters.py`: remove Latte names from active sequence architecture handling.
- `src/config.py`: remove old architecture sections from sequence normalization.
- `configs/default.yaml`: remove old architecture choices, Latte config sections, hardware tuning entries, and Latte hparam tuning defaults.
- Evaluation/diagnostic scripts: remove Latte from active architecture lists and display names.
- `README.md`, `run.md`: remove old Latte commands/sections and add a short note that the previous implementation has been removed.
- Tests: update old Latte references or replace them with removed-architecture assertions.

## References Found

The audit searched for `cawfe_latte`, `cawfe-latte`, `cawfe latte`, `CAWFE-Latte`, `latte_lite`, `latte-lite`, `LatteLite`, `CAWFELatte`, `CAWFELatteLite`, `cawfe_latte_lite`, `use_wind_guided_directional_module`, `neural_operator_bottleneck`, `fire_front_gate`, `vertical_atmosphere_encoder`, `WindGuidedDirectional`, `AFNO`, and `cawfe_latte_operator`.

Matches were found in the files listed above plus active docs/config/factory/evaluation references. No Latte-specific config files or config directories were found under `configs/`.

## Manual Review

No ambiguous non-Latte files were selected for deletion. References to `CAWFE` in dataset paths and the active `cawfe_st_mamba` architecture are intentionally kept.
