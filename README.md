# ConvLSTM U-Net Wildfire Forecasting

## Overview
This project trains a ConvLSTM U-Net on one `.npy` tensor per timestamp to forecast future wildfire state from historical simulation frames.

The current default setup is still the multitask forecast introduced earlier:
- output channel `0`: surface consumed fuel
- output channel `1`: canopy consumed fuel
- output channel `2`: active fire / perimeter mask logits
- output channel `3`: `log1p` energy release total MW

## Conda Environment Setup
Create and activate the environment before running any scripts:

```bash
conda create -n fire_forecasting python=3.10 -y
conda activate fire_forecasting
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format
Each timestamp file is expected to have shape:

```text
(144, 144, 86)
```

Default channel layout:
- `0-79`: atmospheric variables
- `80-85`: flux + fuel variables

Configured via:

```yaml
channel_layout:
  atmospheric_channels: [0, 79]
  flux_channels: [80, 81, 82, 83]
  fuel_channels: [84, 85]
  surface_fuel_channel: 84
  canopy_fuel_channel: 85
  flux_mask_channel: 80
```

## Dataset Splitting Policy
## Training On Multiple Simulation Datasets
Use `data_dirs` to provide multiple dataset folders. Each folder is treated as an independent time series, so input sequences never cross folder boundaries.

In `multi_dataset_chronological` mode:
- each dataset is split chronologically into train/validation/test independently
- train splits are concatenated across datasets
- validation splits are concatenated across datasets
- test splits are concatenated across datasets
- no random frame splitting is used

Normalization rules:
- normalization stats are computed only from the combined training splits
- validation and test splits are not used for normalization
- sequences, previous-frame deltas, and initial-fuel references all stay local to the source dataset

This improves generalization by exposing the model to more fire regimes while still preserving chronological structure inside each simulation.

Example config:

```yaml
data_dirs:
  - ../keepz_08
  - ../keepz_08_tuo
split_mode: multi_dataset_chronological
train_fraction: 0.7
val_fraction: 0.15
test_fraction: 0.15
```

If the goal is strict cross-dataset generalization, add a future split mode that holds out whole datasets. The current `multi_dataset_chronological` mode uses all configured datasets in train/val/test splits chronologically.

## Inputs And Targets
Base input channels: `86`

With the default engineered features enabled, the dataset appends:
- `4` flux delta channels
- `2` fuel delta channels
- `2` step consumed fuel channels
- `2` cumulative consumed fuel channels
- `8` horizontal wind speed channels
- `1` low-level mean wind speed channel
- `8` updraft channels
- `16` wind direction unit-vector channels

Total input channels:

```text
86 + 10 + 33 = 129
```

Default model config:

```yaml
model:
  input_channels: 129
  output_channels: 4
```

## Energy Release Map Target
CAWFE provides sensible and latent heat fluxes in `W/m^2`. This project derives an additional downstream target called the energy release map by multiplying the summed fluxes by grid-cell area from `dx * dy`.

Formula:

```text
P_energy_MW(x,y,t) =
dx * dy *
(Q_sens_surface + Q_sens_canopy + Q_lat_surface + Q_lat_canopy)
/ 1e6
```

This is intentionally called the energy release map, not FRP, because the CAWFE inputs here are sensible and latent heat fluxes rather than radiative power. The model predicts `log1p(P_energy_MW)` as an additional continuous output target because the distribution is sparse and heavy-tailed. Evaluation reports both log-space error and physical MW error.

Per-cell area now comes from each fire dataset's `.geom` file rather than a single scalar `dx * dy`. The recommended discovery/index workflow is:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
```

The resulting `fire_dataset_index.json` is written to the project root by default and is used to pair processed `.npy` folders with the matching geometry source. If energy release is enabled, old 3-channel checkpoints are incompatible with the 4-channel model head and must not be reused without retraining.

## Manual Fire Splits
For cross-fire generalization experiments, the project now supports assigning whole fires to train, validation, and test with `split_mode: manual_fire_holdout` in [configs/default.yaml]. In this mode, `manual_fire_split.train_fires`, `val_fires`, and `test_fires` must match keys in `fire_dataset_index.json`, and validation/test fires are completely held out from training and normalization.

Recommended workflow:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
python scripts/precompute_patch_cache.py --config configs/default.yaml --split train
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/precompute_patch_cache.py --config configs/default.yaml --split val
python scripts/precompute_patch_cache.py --config configs/default.yaml --split test
python scripts/inspect_patch_cache.py --config configs/default.yaml --split all
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Choose training fires to cover a range of fire sizes, durations, energy-release regimes, and spatial extents. Hold out validation/test fires as unseen events when you want a scientifically cleaner cross-fire generalization result.

## Fast Patch Cache Training
The default config can train from precomputed patch shards under `/scratch/mhabibp/fire_forecasting_patch_cache/`. This avoids reopening full `.npy` frames and rebuilding full-domain engineered features, multitask targets, and energy-release maps inside every training batch.

Use `scripts/precompute_patch_cache.py` to save `X` patches shaped `(N,T,C,H,W)` and `y` patches shaped `(N,4,H,W)`. Inputs are stored unnormalized by default, so compute normalization from the train cache before training:

```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split train
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/precompute_patch_cache.py --config configs/default.yaml --split val
python scripts/precompute_patch_cache.py --config configs/default.yaml --split test
python scripts/inspect_patch_cache.py --config configs/default.yaml --split all
python scripts/train_forecasting_model.py --config configs/default.yaml
```

If `/scratch` purges the cache, training raises a clear error and asks you to rerun precompute unless `cache.allow_dynamic_fallback: true` is set.

## Training Performance Optimization
All model-specific training scripts now route through the shared optimized training path in `src/training/train.py`. The shared path supports GPU-aware batch sizing, BF16/FP16 AMP, TF32, GPU-side input normalization, shard-local cache batching, Slurm-aware DataLoader worker caps, timing CSV logs, optional CUDA prefetching, and optional `torch.compile`.

Useful commands:

```bash
python scripts/diagnose_training_pipeline.py \
  --config configs/default.yaml \
  --model_architecture cawfe_latte \
  --num_batches 50

python scripts/benchmark_patch_cache_io.py \
  --config configs/default.yaml \
  --split train \
  --num_batches 100

python scripts/train_forecasting_model.py \
  --config configs/default.yaml
```

On Palmetto, request enough CPUs for the DataLoader, keep the patch cache on `/scratch/mhabibp/`, and prefer BF16 on A100/H100 GPUs. If GPU utilization is low and timing logs show high `data_wait`, tune workers or cache locality first. If VRAM is underused and `data_wait` is low, increase batch size or enable `training.performance.auto_batch_size`.

## Training Run Outputs
Every training entry point now creates a unique run directory:

```text
artifacts/runs/<architecture>/<run_name>/
```

The run directory is the source of truth for checkpoints, logs, configs, metadata, and training curves. For example, the best ConvLSTM checkpoint for a run is:

```text
artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt
```

Each run stores `checkpoints/best_model.pt`, `checkpoints/latest_model.pt`, `logs/training_log.csv`, `logs/validation_log.csv`, `logs/timing_log.csv`, `figures/loss_curves.png`, `figures/metric_curves.png`, `configs/resolved_config.yaml`, and `metadata/run_summary.json`. Compatibility checkpoint copies are still written under `artifacts/checkpoints/<architecture>/`, but those files are only conveniences for older evaluation commands and can be replaced by the next run of the same architecture.

Useful commands:

```bash
python scripts/list_training_runs.py --root artifacts/runs

python scripts/verify_training_outputs.py --all --root artifacts/runs

python scripts/plot_training_curves.py \
  --run_dir artifacts/runs/cawfe_latte/<run_name>
```

You can also name a run explicitly:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml \
  --run_name cawfe_latte_main_seed42
```

The loss and metric curves are debugging plots for overfitting, instability, and training-speed inspection; they are not paper figures.

## Evaluating Trained Models
Use the paper-results evaluator after training jobs have produced run directories under `artifacts/runs/<architecture>/<run_name>/`. The script searches those run folders, selects the best checkpoint by validation metric, evaluates the held-out split, and writes paper-ready quantitative metrics.

Evaluate all trained models on the test split:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test
```

Evaluate only CAWFE-Latte:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture cawfe_latte
```

Primary outputs are written to:

```text
artifacts/results/quantitative/<eval_run_name>/paper_metrics.csv
artifacts/results/quantitative/<eval_run_name>/paper_table.tex
artifacts/results/quantitative/<eval_run_name>/per_fire_metrics.csv
```

Qualitative prediction and rollout mode is reserved for a later implementation; for now, use `--mode quantitative`.

## Atmospheric Engineered Features
From the raw atmospheric `U`, `V`, and `W` channels, the dataset can append three atmospheric feature groups for every input timestep.

1. Horizontal wind speed for each retained z-level:

```text
sqrt(U^2 + V^2)
```

2. Low-level mean wind speed using `low_level_indices`, default `[0, 1, 2]`:

```text
sqrt(mean(U_low)^2 + mean(V_low)^2)
```

3. Updraft for each retained z-level:

```text
max(W, 0)
```

4. Wind direction unit-vector features for each retained z-level:

```text
wind_speed = sqrt(U^2 + V^2 + eps)
wind_dir_cos = U / wind_speed
wind_dir_sin = V / wind_speed
```

These encode the direction the wind is blowing toward.

Why these features help:
- horizontal wind speed directly represents wind magnitude
- low-level wind is strongly related to near-surface fire spread
- updraft captures plume and convection strength
- wind-direction unit vectors preserve directional information without angle wraparound

Why not raw angles:
- raw `atan2`-style angle channels wrap at `360/0` degrees, which creates an artificial discontinuity for ML
- normalized `cos/sin` direction components are smoother and easier for the model to learn from
- meteorological wind direction is usually the direction wind comes from, but these features use the toward-direction vector because it is more directly aligned with transport and spread

Default channel-count example:
- raw channels: `86`
- fuel/flux engineered channels: `10`
- atmospheric engineered channels:
  `8` horizontal wind speed
  `1` low-level mean wind speed
  `8` updraft
  `16` wind direction unit-vector channels
  total atmospheric engineered = `33`
- total model input channels: `129`

## Leakage Rule
- input features may use time `t` and earlier only
- labels may use `t_future = t + prediction_horizon`
- engineered deltas and consumed-fuel input features must never use `t_future` or later frames

Why fuel/flux history is allowed as input:
- past fuel/flux are known historical state
- future fuel/perimeter are prediction targets

## Earthformer-lite Transformer Baseline
This project now includes `earthformer_lite`, an in-project simplified Earthformer-inspired model for comparison. It is not the full official Earthformer implementation.

The design uses axial space-time attention inspired by cuboid attention:
- temporal attention
- height attention
- width attention

Canonical patch-cache input/output:
- input: `(B, 6, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

The model predicts:
- surface consumed fuel
- canopy consumed fuel
- mask logits
- `log1p` energy release

Commands:

Smoke test:
```bash
python scripts/smoke_test_earthformer_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_earthformer_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_earthformer_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/earthformer_lite/best_model.pt --split test
```

Compare baselines and model:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/earthformer_lite/best_model.pt --model_architecture earthformer_lite
```

## ST-Mamba-Lite Architecture
This project also includes `st_mamba_lite`, a CAWFE-tailored spatiotemporal Mamba model for dense wildfire forecasting. It is inspired by MetMamba's route-based 3D scanning ideas and ST-Mamba's ST-Mixer / ST-SSM design pattern, but it is not an official reproduction of either paper.

Canonical patch-cache input/output:
- input: `(B, 6, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Output channels:
- surface consumed fuel
- canopy consumed fuel
- fire/perimeter mask logits
- `log1p` energy release map

The model uses:
- a per-timestep CNN stem
- route-based spatial-temporal Mamba scans over `(T, H, W)`
- local depthwise 3D convolution mixing
- temporal readout to a final 2D map
- a U-Net-style decoder for dense output reconstruction

Commands:

Smoke test:
```bash
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_st_mamba_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --split test
```

Compare with baselines:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --model_architecture st_mamba_lite
```

For real training, install `mamba-ssm` and set `st_mamba_lite.mamba_backend: mamba_ssm`. If `mamba_backend: auto` and `mamba-ssm` is unavailable, the code falls back to a gated sequence block meant for smoke testing and debugging only. Do not treat fallback-backend results as a publishable Mamba comparison.

OOM troubleshooting:
- reduce `batch_size`
- reduce `earthformer_lite.embed_dim`
- reduce `earthformer_lite.depths`
- reduce `earthformer_lite.num_heads`
- increase `training.gradient_accumulation_steps`
- keep `training.mixed_precision: true`

## WeatherFormer-lite Architecture
This project also includes `weatherformer_lite`, an in-project WeatherFormer-inspired factorized transformer tailored to CAWFE wildfire forecasting. It is not the official WeatherFormer implementation.

Canonical patch-cache input/output:
- input: `(B, 6, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Output channels:
- surface consumed fuel
- canopy consumed fuel
- fire/perimeter mask logits
- `log1p` energy release map

The model uses:
- learnable channel scaling and feature gating over the CAWFE input channels
- a per-timestep CNN stem
- temporal attention at each spatial cell
- local spatial window attention with optional cyclic shifted windows
- temporal readout to final 2D maps
- a U-Net-style decoder for dense prediction

Commands:

Smoke test:
```bash
python scripts/smoke_test_weatherformer_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_weatherformer_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_weatherformer_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/weatherformer_lite/best_model.pt --split test
```

Compare with baselines:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/weatherformer_lite/best_model.pt --model_architecture weatherformer_lite
```

Notes:
- this is a CAWFE-specific WeatherFormer-lite adaptation, not an official reproduction
- the shifted-window path currently uses cyclic shifts without a masking scheme
- channel `2` remains logits; the model does not apply sigmoid internally

## Sliding-Window Patch Cache
The project now supports deterministic sliding-window patchification for train, validation, and test. The current default is:
- `patch_size = 64`
- `stride = 60`

This gives nearly non-overlapping patches with a 4-pixel overlap while still covering the full fire domain, including border-aligned patches. Compared with the earlier sampled-train / sliding-eval behavior, it keeps training and evaluation patchification consistent.

Commands:

Rebuild the sliding-window patch cache:
```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split all
```

Inspect the cache:
```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
```

Compute normalization from the rebuilt cache:
```bash
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
```

Train:
```bash
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Notes:
- train, validation, and test now all use sliding-window patchification
- the train cache is much larger than the older sampled-train cache, so one epoch can be substantially longer
- if needed, reduce epochs or set `training.max_train_batches_per_epoch`
- after changing patch stride or patch mode, rebuild the scratch cache or bump `cache.cache_version`
- old sampled or stride-32 caches are incompatible with the current config

## ST-Mamba-Lite Baseline
This project is also being extended with `st_mamba_lite`, a CAWFE-tailored spatial-temporal Mamba baseline for dense wildfire forecasting. It should not be described as an official reproduction of any prior Mamba paper.

Planned/expected patch-cache input/output:
- input: `(B, 6, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Planned prediction targets:
- surface consumed fuel
- canopy consumed fuel
- mask logits
- `log1p` energy release

Expected command syntax for the model-specific scripts:

Smoke test:
```bash
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_st_mamba_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --split test
```

Expected comparison command once the architecture scripts are present:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/checkpoints/st_mamba_lite/best_model.pt --model_architecture st_mamba_lite
```

## Multitask Labels
For sample start index `i`:
- input frames: `i ... i + input_sequence_length - 1`
- current time: `t = i + input_sequence_length - 1`
- future time: `t_future = t + prediction_horizon`

Label definitions:

```text
surface consumed fuel = current surface fuel - future surface fuel
canopy consumed fuel = current canopy fuel - future canopy fuel
```

Mask target modes:

`active_flux`:

```text
mask = future flux channel > flux_fire_threshold
```

`burned_fuel`:

```text
mask = max(initial fuel - future surface/canopy fuel) > consumed_fuel_threshold
```

## Reconstructing Future Fuel Beds
The regression heads predict consumed fuel, not future fuel directly.

Reconstruction: 

```text
predicted future surface fuel = current surface fuel - predicted surface consumed fuel
predicted future canopy fuel = current canopy fuel - predicted canopy consumed fuel
```

These reconstructed maps are clamped to `>= 0`.

## Multi-Step Autoregressive Rollout
One-step forecasts can look good because adjacent timestamps are often similar. Multi-step rollout is the harder test: start from a true input window, predict one step ahead, write the predicted surface and canopy fuel back into the next raw frame, rebuild engineered features from that updated raw window, and repeat.

The rollout path currently supports only `window_mode: static`, which keeps the model input length equal to `input_sequence_length`. After every step, the oldest frame is dropped and the newly constructed next frame is appended.

Exogenous handling modes:
- `teacher_forced`: use the true future raw frame for non-fuel variables, but replace the surface and canopy fuel channels with predicted fuel
- `constant`: copy the previous rollout raw frame for non-fuel variables and replace the surface and canopy fuel channels with predicted fuel

The rollout script now randomly selects split samples by seed and generates one GIF per sample. Each GIF compares ground-truth future progression against the model's autoregressive rollout with consistent color scaling across steps. Metrics are still saved, but the primary output is the visual rollout sequence under `outputs/rollouts/<split>/<dataset_name>/`.

## Commands
All commands below assume the Conda environment above is already active.

### Core Python Scripts
- Inspect dataset split and file counts:
  `python scripts/inspect_dataset.py --config configs/default.yaml`
- Precompute train patch shards on scratch:
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split train`
- Compute normalization from the precomputed train patch cache:
  `python scripts/compute_normalization.py --config configs/default.yaml --from_cache`
- Precompute validation and test patch shards:
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split val`
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split test`
- Inspect the patch cache and save random previews:
  `python scripts/inspect_patch_cache.py --config configs/default.yaml --split all`
- Run the main project sanity check:
  `python scripts/sanity_check_project.py --config configs/default.yaml`
- Run lightweight smoke checks:
  `python scripts/smoke_checks.py --config configs/default.yaml`
- Train the model:
  `python scripts/train_forecasting_model.py --config configs/default.yaml`
- Evaluate the saved checkpoint on the configured test split:
  `python scripts/test_model.py --config configs/default.yaml --checkpoint-kind best`
- Probe model support for native and alternate spatial sizes without loading dataset files:
  `python scripts/test_spatial_size_compatibility.py --config configs/default.yaml`
- Diagnose why multitask mask output channel 2 generalizes poorly on external test:
  `python scripts/diagnose_mask_generalization.py --config configs/default.yaml --max_samples 200`
- Compare raw active-flux and burned-fuel mask definitions across splits:
  `python scripts/compare_mask_definitions.py --config configs/default.yaml --max_samples 200`
- Evaluate masks derived from predicted consumed fuel:
  `python scripts/evaluate_consumed_fuel_derived_mask.py --config configs/default.yaml --max_samples 200`
- Visualize validation predictions:
  `python scripts/visualize_predictions.py --config configs/default.yaml --split val --num_samples 10`
- Visualize test predictions:
  `python scripts/visualize_predictions.py --config configs/default.yaml --split test --num_samples 10`
- Visualize model vs persistence comparisons:

  `python scripts/visualize_model_vs_persistence.py --config configs/default.yaml --num_samples 20 --output_dir outputs/model_vs_persistence`
- Reconstruct future surface/canopy fuel beds from multitask predictions:

  `python scripts/reconstruct_fuel_bed_from_predictions.py --config configs/default.yaml --num_samples 10`
- Inspect engineered features for one sample:

  `python scripts/inspect_engineered_features.py --config configs/default.yaml --sample_index 0`
- Inspect atmospheric engineered features for one sample:

  `python scripts/inspect_atmospheric_features.py --config configs/default.yaml --sample_index 0`
- Legacy cache for engineered per-timestep tensors only:

  `python scripts/cache_engineered_dataset.py --config configs/default.yaml --output_dir ../keepz_05_engineered`

### Inspection And Analysis Scripts
- Inspect selected raw channel maps at selected timesteps:
  `python scripts/inspect_target_channels.py --config configs/default.yaml --channels 80 81 84 85 --timesteps 0 50 100`
- Launch the interactive raw input viewer:

  `python scripts/visualize_input_dataset.py --data-dir`
- Evaluate the persistence baseline on the configured test dataset:

  `python scripts/evaluate_persistence_baseline.py --config configs/default.yaml --num-visualizations 5`
- Compare persistence across candidate target channels:

  `python scripts/evaluate_persistence_all_candidate_targets.py --config configs/default.yaml --channels 50 51 52 53 54 55`
- Run multitask autoregressive rollout GIF generation with teacher-forced future exogenous variables:

  `python scripts/rollout_predictions.py --config configs/default.yaml --split test --num_samples 5 --rollout_steps 20`
- Run multitask autoregressive rollout GIF generation with constant exogenous variables:

  `python scripts/rollout_predictions.py --config configs/default.yaml --split test --num_samples 5 --rollout_steps 20 --exogenous_mode constant`

### Shell Wrappers
- Full pipeline wrapper:
  `bash scripts/run_full_pipeline.sh configs/default.yaml`
- Quick smoke-test wrapper:
  `bash scripts/run_quick_smoke_test.sh configs/default.yaml`
- Training-only wrapper:
  `bash scripts/run_training_only.sh configs/default.yaml`
- Visualization-only wrapper:
  `bash scripts/run_visualization_only.sh configs/default.yaml 10`

Use `--help` on any Python script for the full CLI:

```bash
python scripts/visualize_predictions.py --help
```

## CAWFE-Latte-Lite
`cawfe_latte_lite` is the custom CAWFE-specific architecture for this project. It separately encodes vertical atmospheric structure and fire/fuel state variables, applies fire-front attention, and uses a hybrid Transformer + Mamba backbone to predict the four dense multitask maps.

Detailed architecture notes are in [cawfe_latte.md](cawfe_latte.md).

```bash
python scripts/smoke_test_cawfe_latte_lite.py --config configs/default.yaml
python scripts/train_cawfe_latte_lite.py --config configs/default.yaml
python scripts/test_cawfe_latte_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte_lite/best_model.pt --split test
```

## Full CAWFE-Latte
`cawfe_latte` extends CAWFE-Latte-Lite with wind-guided directional feature modulation and an AFNO-style neural-operator bottleneck. This is the main custom paper architecture. Detailed documentation is in [cawfe_latte.md](cawfe_latte.md).

```bash
python scripts/smoke_test_cawfe_latte.py --config configs/default.yaml
python scripts/train_cawfe_latte.py --config configs/default.yaml
python scripts/test_cawfe_latte.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test
python scripts/visualize_cawfe_latte_aux.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test --num_samples 5
```

## CAWFE-Latte Hyperparameter Tuning
Only the main `cawfe_latte` paper model is tuned. ConvLSTM U-Net, Earthformer-lite, WeatherFormer-lite, CAWFE-ST-Mamba, and CAWFE-Latte-Lite use their fixed configs.

The tuner runs short validation-only CAWFE-Latte trials through the shared training pipeline, then writes:

- `artifacts/hparam/cawfe_latte/tuning_trials.csv`
- `artifacts/hparam/cawfe_latte/tuning_trials.jsonl`
- `artifacts/hparam/cawfe_latte/best_params.json`
- `artifacts/hparam/cawfe_latte/best_config.yaml`
- `artifacts/hparam/cawfe_latte/tuning_summary.txt`

```bash
# Tune on one A100 80GB GPU.
sbatch slurm/slurm_tune_cawfe_latte_a10080.sh

# Final full training from the tuned config.
sbatch slurm/slurm_train_cawfe_latte_tuned_a10080.sh

# Tuned ablations.
sbatch slurm/slurm_ablate_cawfe_latte_a10080.sh

# Or submit tune -> train -> ablate with Slurm dependencies.
bash scripts/submit_cawfe_latte_pipeline.sh
```

Local/debug commands:

```bash
python scripts/tune_cawfe_latte.py --config configs/default.yaml --num_trials 12 --output_dir artifacts/hparam/cawfe_latte
python scripts/train_cawfe_latte.py --config artifacts/hparam/cawfe_latte/best_config.yaml
python scripts/ablate_cawfe_latte.py --base_config artifacts/hparam/cawfe_latte/best_config.yaml --output_dir configs/ablations/cawfe_latte_tuned
```
