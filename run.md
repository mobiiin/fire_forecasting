# Rebuilt Dataset Pipeline

The staged dataset pipeline preserves the configured manual train/validation/test fire holdout and saves engineered full-frame tensors in channel-first `(C,H,W)` format under scratch. It does not construct targets or fixed X/y training samples.

```bash
python scripts/build_engineered_frame_dataset.py --config configs/default.yaml
python scripts/visualize_engineered_frames.py --config configs/default.yaml --split train

Viewer controls: Left/Right browse timestamps, Up/Down browse channels while keeping the timestamp, and `n`/`p` browse fires. Use `--channel_group core` or `--channel N` to start from a specific channel.
python scripts/build_patch_index.py --config configs/default.yaml
python scripts/visualize_patch_index.py --config configs/default.yaml --split train
```

Stages: build full-frame engineered data, visualize channels, build spatial patch metadata, visualize patch boxes, then later add target construction, temporal sample indices, train-only normalization, and training. Patch indices define crops only; they do not save patch tensors. No `targets/` or fixed training-cache outputs are created here.

# Run Guide

Project base directory: `/home/mhabibp/fire_forecasting`

Use a config path for every experiment. The default is still `configs/default.yaml`, and examples live under `configs/experiments/`.

```bash
CONFIG=configs/experiments/convlstm_debug_fixed_subset.yaml
```

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

## Manual Fire Start Trimming
After regenerating `fire_dataset_index.json`, choose the first useful frame for each fire manually. The raw CAWFE files are not changed, and the trimmed index stores compact `temporal_trim` start/end indices instead of a full list of frame paths.

```bash
python scripts/manual_trim_fire_datasets.py \
  --input_index fire_dataset_index.json \
  --output_index fire_dataset_index_trimmed.json
```

To review only one fire:

```bash
python scripts/manual_trim_fire_datasets.py \
  --input_index fire_dataset_index.json \
  --only_fire MCKINNEY \
  --output_index fire_dataset_index_trimmed.json
```

Apply existing choices without interaction:

```bash
python scripts/manual_trim_fire_datasets.py \
  --apply_only \
  --input_index fire_dataset_index.json \
  --trim_config configs/manual_fire_trim.json \
  --output_index fire_dataset_index_trimmed.json
```

Inspect the compact trimmed index:

```bash
python scripts/inspect_multi_fire_dataset.py \
  --index fire_dataset_index_trimmed.json \
  --config configs/default.yaml
```

To train from the trimmed sequence window, set `fire_dataset_index_json: ../fire_dataset_index_trimmed.json` in `configs/default.yaml`, bump `cache.cache_version`, then rebuild cache and normalization:

```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split all
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
```

Old patch cache shards, normalization stats, and checkpoints are incompatible after changing trim decisions.

## Parallel Experiment Launches
Each experiment config should set a unique `experiment.name`, `paths.patch_cache_root`, and `paths.artifacts_root`. Normalization files live directly inside `paths.patch_cache_root`, so separate experiments still avoid cache/stat collisions without creating a separate normalization folder.

```bash
python scripts/sanity_check_project.py --config "$CONFIG"
python scripts/precompute_patch_cache.py --config "$CONFIG" --split all
python scripts/compute_normalization.py --config "$CONFIG" --from_cache
python scripts/train_forecasting_model.py --config "$CONFIG"
```

`compute_normalization.py` saves unique timestamped JSON/NPZ files and updates latest aliases by default:

```text
train_normalization_stats_<config_name>_<YYYYMMDD_HHMMSS>.json
train_normalization_stats_<config_name>_<YYYYMMDD_HHMMSS>.npz
normalization_stats.json
normalization_stats.npz
```

The current config keeps the stable aliases named `normalization_stats.json` and `normalization_stats.npz` directly in `paths.patch_cache_root`, beside `cache_manifest.json` and the split shard folders.

Use `--output_dir` or `--config_name` for one-off overrides. Use `--latest_as_copy` if you want latest aliases copied instead of symlinked, and `--no_latest_alias` only when preserving the current latest aliases intentionally.

On Palmetto, the generic Slurm wrappers take the config path as their first argument:

```bash
sbatch scripts/slurm_precompute_cache_with_config.sh "$CONFIG" all
sbatch scripts/slurm_train_with_config_a10080.sh "$CONFIG"
```

The patch-cache precompute step writes `.precompute_lock` inside the configured cache directory while it is running. If a previous job died and left that lock behind, inspect it first, then rerun with `--ignore_stale_lock` only when you are sure no other job owns that cache.

If you are not switching to the trimmed-index cache workflow yet, run the intermediate validation step:

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

This should:

- load only the training fires
- save timestamped normalization JSON/NPZ files and update latest aliases
- write a JSON sidecar with train-only provenance metadata
- save the resolved split file to `artifacts/splits/manual_fire_split_resolved.json`

Optional inspection steps before training:

```bash
python scripts/inspect_normalization_pipeline.py --config configs/default.yaml --split all
python scripts/inspect_dataset.py --config configs/default.yaml
python scripts/inspect_target_channels.py --config configs/default.yaml
python scripts/inspect_energy_release.py --config configs/default.yaml --split train
```

Visualize cached patches directly from scratch:

```bash
python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split train

python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split val \
  --sample_index 100

python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split test \
  --mode save \
  --random \
  --num_save_samples 10
```

Viewer controls: `n`/`p` next/previous sample, up/down input frame, `g`/`G` channel group, `v` view, `w` save current figure, `q` quit.

Use these checks to confirm:

- train, validation, and test fire names and sample counts look right
- normalization stats are train-only and applied exactly once
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
  --model_architecture convlstm_unet \
  --num_batches 50

python scripts/train_forecasting_model.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet
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

Check validation mode:

```bash
python scripts/sanity_check_project.py \
  --config configs/default.yaml
```

Expected outputs:

- source-of-truth run folder under `artifacts/runs/<architecture>/<run_name>/`
- best/latest checkpoints under `artifacts/runs/<architecture>/<run_name>/checkpoints/`
- CSV logs under `artifacts/runs/<architecture>/<run_name>/logs/`
- curve images under `artifacts/runs/<architecture>/<run_name>/figures/`
- compatibility checkpoint copies under `artifacts/checkpoints/<architecture>/`
- training and validation loss printed each epoch

If validation metrics are oscillating, check whether validation is using `fixed_subset_every_epoch` or `full_every_epoch`. For stable curves, use one of those modes for the whole run and do not mix capped validation with periodic full validation.

Early stopping is enabled by default for shared training runs:

```yaml
training:
  max_epochs: 50
  early_stopping:
    enabled: true
    monitor: val_loss
    mode: min
    patience: 8
    min_delta: 0.001
    start_epoch: 5
```

Override it from any shared training script:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml \
  --early_stopping_patience 12
```

Use `--disable_early_stopping` for a full `max_epochs` run. Check `metadata/run_summary.json` for `stopped_early`, `stop_reason`, and the `early_stopping` state.

Test the trained model:

```bash
python scripts/test_model.py --config configs/default.yaml
```

This evaluates the model on the held-out test fires from your manual split.

## Checking Training Outputs
After one or more local or Slurm training jobs finish, list the saved runs:

```bash
python scripts/list_training_runs.py \
  --root artifacts/runs
```

Verify expected files and checkpoint metadata:

```bash
python scripts/verify_training_outputs.py \
  --all \
  --root artifacts/runs
```

## Quantitative Testing After Training
After the training run folders are verified, generate paper-ready test metrics:

```bash
python scripts/list_training_runs.py \
  --root artifacts/runs

python scripts/verify_training_outputs.py \
  --all \
  --root artifacts/runs

python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture all \
  --paper_energy_metric log
```

New checkpoints include input-normalization metadata. Evaluation applies the same dataset/device normalization path used in training and checks that metadata when present. For intentional compatibility/debug runs against older checkpoints, add `--allow_normalization_mismatch`.

The paper table uses log-space energy by default: `Energy Log MAE` and `Active Energy Log MAE` are computed directly on output channel 3, which is `log1p(energy_release_MW)`. MW-space energy errors are saved as secondary diagnostics in the report/long metrics, but they are not used for the default Skill score.

For ConvLSTM only with auto-selected best checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture convlstm_unet \
  --paper_energy_metric log
```

For ConvLSTM with an explicit checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --paper_energy_metric log
```

For baselines only:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture baseline \
  --paper_energy_metric log
```

To run both quantitative metrics and qualitative figures:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode all \
  --split test \
  --model_architecture all \
  --num_samples 10 \
  --paper_energy_metric log
```

Then inspect:

```text
artifacts/results/<eval_name>/evaluation_report.md
```

JSON, CSV, and LaTeX sidecars are opt-in for quantitative runs. Add `--include_json_outputs`, `--include_csv_outputs`, or `--write_latex` when those files are needed.

## Qualitative Testing After Training
Use qualitative mode to inspect a fixed random set of samples. The same sample indices are reused across every evaluated model, so `--model_architecture all` produces aligned comparison figures.

For ConvLSTM with an explicit checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode qualitative \
  --split test \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --num_samples 10 \
  --qualitative_seed 42
```

For all available learned models plus baselines:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode qualitative \
  --split test \
  --model_architecture all \
  --num_samples 10 \
  --qualitative_seed 42
```

Then inspect:

```text
artifacts/results/qualitative/<eval_name>/images/sample_000.png
artifacts/results/qualitative/<eval_name>/qualitative_report.md
artifacts/results/qualitative/<eval_name>/qualitative_report.json
artifacts/results/qualitative/<eval_name>/selected_samples.json
artifacts/results/qualitative/<eval_name>/selected_models.json
```

Default qualitative output is one summary image per selected sample. Use `--qualitative_output_format pdf`, `--qualitative_dpi`, or `--save_individual_model_panels` only when you need different figure output. This mode is one-shot and does not perform recursive rollout.

## Debug A Suspicious Model Checkpoint
If a trained model looks worse than persistence or its qualitative outputs look wrong, inspect a few batches directly:

```bash
python scripts/list_training_runs.py --root artifacts/runs

python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --split test \
  --num_batches 2 \
  --num_samples_to_plot 4
```

To see how much normalization changes prediction scale on the same batches:

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --split test \
  --num_batches 2 \
  --compare_without_normalization
```

Then inspect:

```text
artifacts/debug_predictions/convlstm_unet/<timestamp>/diagnostics_summary.txt
artifacts/debug_predictions/convlstm_unet/<timestamp>/channel_stats.csv
artifacts/debug_predictions/convlstm_unet/<timestamp>/metrics_summary.json
artifacts/debug_predictions/convlstm_unet/<timestamp>/figures/
```

## ConvLSTM Mask-Gated Regression
The default ConvLSTM config now supports mask-gated regression to suppress diffuse background predictions. The model returns the same four channels, but channels `0`, `1`, and `3` are gated by predicted fire probability while channel `2` remains mask logits.

```yaml
convlstm_unet:
  use_mask_gated_regression: true
  regression_activation: relu
  mask_gate_mode: soft
  detach_mask_gate: false
  mask_gate_min: 0.0
  output_bias_init:
    enabled: true
    regression_bias: -2.0
    mask_bias: -2.0
    energy_bias: -2.0

training:
  loss:
    background_suppression:
      enabled: true
      architectures: [convlstm_unet]
      weight: 0.05
```

To ablate the old ConvLSTM behavior, set `convlstm_unet.use_mask_gated_regression: false` and `training.loss.background_suppression.enabled: false`, then retrain:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml
```

Run ConvLSTM diagnostic evaluation without retraining:

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --split test \
  --num_batches 20 \
  --run_background_diagnostics \
  --run_mask_gating_diagnostics \
  --run_oracle_gating_diagnostics
```

Compare best and latest checkpoints on the same selected batches:

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --split test \
  --num_batches 20 \
  --compare_checkpoints \
    artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
    artifacts/runs/convlstm_unet/<run_name>/checkpoints/latest_model.pt \
  --checkpoint_labels best latest
```

Diagnostic outputs include `background_diagnostics.csv`, `mask_gating_diagnostics.csv`, `oracle_gating_diagnostics.csv`, `output_activation_inspection.json`, and, for checkpoint comparison, `checkpoint_comparison.csv`.

Regenerate curves for a run if needed:

```bash
python scripts/plot_training_curves.py \
  --run_dir artifacts/runs/convlstm_unet/<run_name>
```

The best checkpoint for a run is:

```text
artifacts/runs/<architecture>/<run_name>/checkpoints/best_model.pt
```

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
- canonical ST-Mamba-Lite patch input is `(B, 5, 129, 64, 64)`
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
- canonical WeatherFormer-lite patch input is `(B, 5, 129, 64, 64)`
- canonical output is `(B, 4, 64, 64)`
- it uses factorized temporal attention and local spatial window attention
- the shifted-window path currently uses cyclic shifts without a masking scheme
- channel `2` remains mask logits; the model does not apply sigmoid internally

## CAWFE-Latte v1 End-to-End
CAWFE-Latte v1 is the first trainable end-to-end baseline for the fresh architecture. It uses four encoders, fire-query modality fusion, a small temporal CNN backbone, a shallow same-resolution decoder, and four heads.

Outputs are surface consumed fuel, canopy consumed fuel, fire mask logits, and log1p energy release. The v1 loss uses surface Huber weight 1, canopy Huber weight 1, mask BCE+Dice weight 5, energy-log Huber weight 1, and one auxiliary fire-support mask loss after local fused features with weight 0.2. There are no per-encoder auxiliary losses and no heavy backbone, neural operator, Mamba, or large transformer yet.

Smoke-test v1 with:

```bash
python scripts/smoke_test_cawfe_latte.py --config configs/default.yaml
```

Train with the example experiment config:

```bash
python scripts/train_forecasting_model.py \
  --config configs/experiments/cawfe_latte_v1.yaml
```

Evaluate with:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/experiments/cawfe_latte_v1.yaml \
  --mode quantitative \
  --split test \
  --model_architecture cawfe_latte
```

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


## Rebuilt target/sample pipeline

After full-frame engineering and patch-index creation, build full-frame targets and metadata-only temporal samples:

```bash
python scripts/estimate_fire_mask_thresholds.py --config configs/default.yaml --percentile 1.0 --update_config --derived_config_path configs/derived/default_with_fire_mask_thresholds.yaml
python scripts/build_target_dataset.py --config configs/derived/default_with_fire_mask_thresholds.yaml
python scripts/visualize_targets.py --config configs/default.yaml --split train
python scripts/build_temporal_sample_index.py --config configs/default.yaml --pattern all
python scripts/visualize_processed_samples.py --config configs/default.yaml --pattern consecutive5_h10 --split train
python scripts/compute_processed_dataset_normalization.py --config configs/default.yaml --pattern consecutive5_h10
python scripts/inspect_processed_dataset.py --config configs/default.yaml
```

The processed ConvLSTM ablations use dynamic full-frame loading rather than fixed patch shards:

```bash
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_consecutive5_h10.yaml
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_single1_h10.yaml
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_sparse5_h10.yaml
```

Targets are full-frame files, sample indices contain only references, and normalization is fitted on train samples only.


## Automated Data Preparation Pipeline

Run the rebuilt data stages in order with one command:

```bash
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml
```

Optional quicklooks, percentile, and pattern selection:

```bash
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --make-quicklooks
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --percentile 5.0
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --pattern sparse5_h10
```

The runner builds engineered frames, estimates train-only fire-mask thresholds, constructs targets using the frozen derived config, builds patch and temporal indices, computes train-only normalization, inspects the dataset, and optionally saves visualizations. It never trains models by default. Logs are written under `artifacts/logs/data_preparation/`; after success, the script prints the exact next training commands.
