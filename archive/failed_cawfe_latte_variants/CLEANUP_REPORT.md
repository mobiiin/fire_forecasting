# CAWFE-Latte failed-variant cleanup report

## Summary

Archived active support for failed CAWFE-Latte variants:

- `cawfe_latte_v1_1`
- `cawfe_latte_v1_2`
- `cawfe_latte_v1_3`

Original `cawfe_latte` remains active and trainable.

## Files archived

Configs:

- `configs/cawfe_latte_v1_1.yaml`
- `configs/cawfe_latte_v1_2.yaml`
- `configs/cawfe_latte_v1_3.yaml`

Scripts:

- `scripts/slurm_train_cawfe_latte_v1_1_a10080.sh`
- `scripts/slurm_train_cawfe_latte_v1_2_a10080.sh`
- `scripts/slurm_train_cawfe_latte_v1_3_a10080.sh`

Tests:

- `tests/test_cawfe_latte_v1_1.py`
- `tests/test_cawfe_latte_v1_2.py`
- `tests/test_cawfe_latte_v1_3.py`

Model snapshot:

- `src_models/cawfe_latte_with_failed_variants.py`

## Active registrations removed

Removed active model-factory and registry support for:

- `cawfe_latte_v1_1`
- `cawfe_latte_v1_2`
- `cawfe_latte_v1_3`

These names now raise a clear archived-variant error instead of silently mapping to `cawfe_latte`.

## Preserved behavior

Kept original CAWFE-Latte v1 behavior:

- `model.architecture: cawfe_latte`
- 4-channel output convention
- original auxiliary fire-support head
- original multitask losses

No training artifacts, checkpoints, logs, qualitative results, or processed datasets were deleted.

## Validation commands run

```bash
conda run -n fire_forecasting python -m py_compile \
  src/models/cawfe_latte.py \
  src/models/model_factory.py \
  src/models/__init__.py \
  src/models/architecture_registry.py \
  src/training/losses.py \
  src/training/train.py \
  src/data/dataset.py \
  scripts/sanity_check_project.py \
  scripts/smoke_test_cawfe_latte_training.py \
  scripts/evaluate_trained_models.py \
  scripts/debug_model_predictions.py \
  tests/test_cawfe_latte_v1.py \
  tests/test_cawfe_latte_active_architectures.py
```

Result: passed.

```bash
conda run -n fire_forecasting python -m pytest \
  tests/test_cawfe_latte_v1.py \
  tests/test_cawfe_latte_active_architectures.py \
  tests/test_cawfe_latte_training_pipeline.py \
  tests/test_removed_architectures.py \
  tests/test_validation_policy.py \
  -q
```

Result: `26 passed`.

```bash
conda run -n fire_forecasting python scripts/sanity_check_project.py \
  --config configs/experiments/cawfe_latte_v1.yaml \
  --deep
```

Result: `Status: OK`.

## Remaining references

Remaining active-code references to `cawfe_latte_v1_1`, `cawfe_latte_v1_2`, and `cawfe_latte_v1_3` are explicit archived-architecture guards/tests only:

- `src/models/architecture_registry.py` removed-architecture set/message
- `src/models/model_factory.py` archived-name error
- `tests/test_cawfe_latte_active_architectures.py`

References to `aux_fire_support_logits` remain because original CAWFE-Latte v1 still uses its auxiliary fire-support head.

Historical references remain under this archive and may remain in old run artifacts/logs.
