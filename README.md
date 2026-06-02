# ConvLSTM U-Net Wildfire Forecasting

## Overview
This project trains a ConvLSTM U-Net on one `.npy` tensor per timestamp to forecast future wildfire state from historical simulation frames.

The current default setup is still the multitask forecast introduced earlier:
- output channel `0`: surface consumed fuel
- output channel `1`: canopy consumed fuel
- output channel `2`: active fire / perimeter mask logits

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
  output_channels: 3
```

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

## Commands
All commands below assume the Conda environment above is already active.

### Core Python Scripts
- Inspect dataset split and file counts:
  `python scripts/inspect_dataset.py --config configs/default.yaml`
- Compute normalization from the main training split only:
  `python scripts/compute_normalization.py --config configs/default.yaml`
- Run the main project sanity check:
  `python scripts/sanity_check_project.py --config configs/default.yaml`
- Run lightweight smoke checks:
  `python scripts/smoke_checks.py --config configs/default.yaml`
- Train the model:
  `python scripts/train_convlstm_unet.py --config configs/default.yaml`
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
- Cache engineered per-timestep tensors to disk:
  `python scripts/cache_engineered_dataset.py --config configs/default.yaml --output_dir ../keepz_05_engineered`

### Inspection And Analysis Scripts
- Inspect selected raw channel maps at selected timesteps:
  `python scripts/inspect_target_channels.py --config configs/default.yaml --channels 80 81 84 85 --timesteps 0 50 100`
- Launch the interactive raw input viewer:
  `python scripts/visualize_input_dataset.py --config configs/default.yaml --file-index 0 --channel-start -6 --window-size 6`
- Evaluate the persistence baseline on the configured test dataset:
  `python scripts/evaluate_persistence_baseline.py --config configs/default.yaml --num-visualizations 5`
- Compare persistence across candidate target channels:
  `python scripts/evaluate_persistence_all_candidate_targets.py --config configs/default.yaml --channels 50 51 52 53 54 55`
- Run autoregressive rollout visualizations from the configured test dataset:
  `python scripts/rollout_predictions.py --config configs/default.yaml --start_index 0 --rollout_steps 30 --rollout_mode constant_exogenous`

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
