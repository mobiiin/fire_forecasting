# Run Guide

Project base directory: `/home/mhabibp/fire_forecasting`

Before running anything else, verify that `configs/default.yaml` is configured for your manual fire split:

- `split_mode: manual_fire_holdout`
- `manual_fire_split.train_fires`, `val_fires`, and `test_fires` contain your hand-picked fires
- every listed fire exists in `fire_dataset_index.json`
- `main_data_dir` points to `/media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/`

If you want a fresh dataset index, regenerate it first:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
```

That command should rewrite `fire_dataset_index.json` in the project root. If any fire path is stale or missing, update the manual fire lists in `configs/default.yaml` before continuing.

Then run the intermediate validation step:

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

This should:

- load only the training fires
- save normalization stats to `artifacts/normalization_stats.npz`
- save the resolved split file to `artifacts/splits/manual_fire_split_resolved.json`

Optional inspection steps before training:

```bash
python scripts/inspect_dataset.py --config configs/default.yaml
python scripts/inspect_target_channels.py --config configs/default.yaml
python scripts/inspect_energy_release.py --config configs/default.yaml --split train
```

Use these checks to confirm:

- train, validation, and test fire names and sample counts look right
- target channels are correct
- energy-release geometry is sane

Non-neural baselines:
```bash
python scripts/run_persistence_baseline.py --config configs/default.yaml --split test
python scripts/run_linear_extrapolation_baseline.py --config configs/default.yaml --split test
python scripts/evaluate_all_baselines.py --config configs/default.yaml --splits train val test
```

## Optimized Training Workflow On Palmetto
Use this sequence when checking training speed or after rebuilding the patch cache:

```bash
python scripts/inspect_patch_cache.py \
  --config configs/default.yaml

python scripts/benchmark_patch_cache_io.py \
  --config configs/default.yaml \
  --split train \
  --num_batches 100

python scripts/diagnose_training_pipeline.py \
  --config configs/default.yaml \
  --model_architecture cawfe_latte \
  --num_batches 50

python scripts/train_cawfe_latte.py \
  --config configs/default.yaml
```

Watch these timing fields in `artifacts/logs/training_timing_<run_name>.csv` and the training log:
- high `data_wait`: DataLoader/cache bottleneck
- high `h2d`: host-to-device transfer bottleneck; use pinned memory, non-blocking transfer, or CUDA prefetching
- low VRAM: increase batch size or enable auto batch size
- high `metrics`: reduce train metric cadence
- high `forward`/`backward`: model compute bottleneck

Running Earthformer-lite:
```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_earthformer_lite.py --config configs/default.yaml
python scripts/train_earthformer_lite.py --config configs/default.yaml
python scripts/test_earthformer_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/earthformer_lite/best_model.pt --split test
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/earthformer_lite/best_model.pt --model_architecture earthformer_lite
```

Running ST-Mamba-Lite:
```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
python scripts/train_st_mamba_lite.py --config configs/default.yaml
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --split test
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --model_architecture st_mamba_lite
```

Train the model:

```bash
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Expected outputs:

- checkpoints under `artifacts/checkpoints/`
- logs under `artifacts/logs/`
- training and validation loss printed each epoch

Test the trained model:

```bash
python scripts/test_model.py --config configs/default.yaml
```

This evaluates the model on the held-out test fires from your manual split.

Visualize results:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml
```

If your branch supports a split flag, you can also run:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --split test
```

Shortest end-to-end command sequence:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
python scripts/compute_normalization.py --config configs/default.yaml
python scripts/train_forecasting_model.py --config configs/default.yaml
python scripts/test_model.py --config configs/default.yaml
python scripts/visualize_predictions.py --config configs/default.yaml
```

After the run, inspect these outputs:

- `fire_dataset_index.json`
- `artifacts/splits/manual_fire_split_resolved.json`
- `artifacts/normalization_stats.npz`
- `artifacts/checkpoints/`
- `artifacts/logs/`
- the visualization output directory configured for your run

If training fails with a data-directory error, rerun dataset discovery so `fire_dataset_index.json` matches the mounted `New_CAWFE` tree, then fix the manual fire lists if needed.

## Running ST-Mamba-Lite
`st_mamba_lite` is the CAWFE-tailored spatial-temporal Mamba architecture in this repo. It is inspired by MetMamba and ST-Mamba ideas, but it is not an official reproduction of either paper.

Recommended workflow:

```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
python scripts/train_st_mamba_lite.py --config configs/default.yaml
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --split test
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --model_architecture st_mamba_lite
```

Notes:
- canonical ST-Mamba-Lite patch input is `(B, 6, 129, 64, 64)`
- canonical output is `(B, 4, 64, 64)`
- channel `2` remains mask logits; the model does not apply sigmoid internally
- for real comparisons, install `mamba-ssm` and set `st_mamba_lite.mamba_backend: mamba_ssm`
- if `st_mamba_lite.mamba_backend: auto` and `mamba-ssm` is missing, the fallback backend is intended for smoke/debug use only

## Running WeatherFormer-lite
`weatherformer_lite` is the CAWFE-tailored factorized transformer in this repo. It is inspired by WeatherFormer, but it is not an official WeatherFormer implementation.

Recommended workflow:

```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_weatherformer_lite.py --config configs/default.yaml
python scripts/train_weatherformer_lite.py --config configs/default.yaml
python scripts/test_weatherformer_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/weatherformer_lite/best_model.pt --split test
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/weatherformer_lite/best_model.pt --model_architecture weatherformer_lite
```

Notes:
- canonical WeatherFormer-lite patch input is `(B, 6, 129, 64, 64)`
- canonical output is `(B, 4, 64, 64)`
- it uses factorized temporal attention and local spatial window attention
- the shifted-window path currently uses cyclic shifts without a masking scheme
- channel `2` remains mask logits; the model does not apply sigmoid internally

## Running CAWFE-Latte-Lite
`cawfe_latte_lite` is the custom paper architecture. It explicitly encodes CAWFE vertical atmospheric levels, fire/fuel state variables, fire-front attention, and a hybrid Transformer + Mamba backbone.

Recommended workflow:

```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_cawfe_latte_lite.py --config configs/default.yaml
python scripts/train_cawfe_latte_lite.py --config configs/default.yaml
python scripts/test_cawfe_latte_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte_lite/best_model.pt --split test
python scripts/ablate_cawfe_latte_lite.py --base_config configs/default.yaml --output_dir configs/ablations/cawfe_latte_lite/
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/cawfe_latte_lite/best_model.pt --model_architecture cawfe_latte_lite
```

Notes:
- canonical CAWFE-Latte-Lite patch input is `(B, 6, 129, 64, 64)`
- canonical output is `(B, 4, 64, 64)`
- channel `2` remains mask logits; the model does not apply sigmoid internally
- detailed architecture documentation is in `cawfe_latte.md`

## Running Full CAWFE-Latte
`cawfe_latte` is the main custom paper model. It extends CAWFE-Latte-Lite with wind-guided directional modulation and an AFNO-style neural-operator bottleneck.

Recommended workflow:

```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/smoke_test_cawfe_latte.py --config configs/default.yaml
python scripts/train_cawfe_latte.py --config configs/default.yaml
python scripts/test_cawfe_latte.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test
python scripts/visualize_cawfe_latte_aux.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test --num_samples 5
python scripts/ablate_cawfe_latte.py --base_config configs/default.yaml --output_dir configs/ablations/cawfe_latte/
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --model_architecture cawfe_latte
```

Notes:
- canonical full CAWFE-Latte patch input is `(B, 6, 129, 64, 64)`
- canonical output is `(B, 4, 64, 64)`
- channel `2` remains mask logits; the model does not apply sigmoid internally
- if `mamba-ssm` is not installed, `mamba_backend: auto` uses the fallback gated SSM for smoke/debug runs
- set `neural_operator_type: none` or reduce `neural_operator_depth` if memory is tight

## Rebuilding The Sliding-Window Patch Cache
Train, validation, and test now all use sliding-window patchification with `patch_size=64` and `stride=60`.

Recommended workflow:

```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split all
python scripts/inspect_patch_cache.py --config configs/default.yaml
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/sanity_check_project.py --config configs/default.yaml
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Notes:
- all three splits now use deterministic sliding-window patch refs
- border patches are included so the full domain is covered
- if cache validation fails, rerun `python scripts/precompute_patch_cache.py --config configs/default.yaml --split all`
