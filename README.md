# ConvLSTM U-Net Wildfire Forecasting

## Overview
This project trains a ConvLSTM U-Net on one `.npy` tensor per timestamp to forecast future wildfire state from historical simulation frames.

The current default setup is a 3-output multitask forecast:
- output channel `0`: surface consumed fuel
- output channel `1`: canopy consumed fuel
- output channel `2`: active fire / perimeter mask logits

No sigmoid is applied inside the model. The mask head is trained with logits.

## Conda Environment Setup
Create and activate the environment before running any scripts:

```bash
conda create -n fire_forecasting python=3.10 -y
conda activate fire_forecasting
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Dataset Shape
Each timestamp file is expected to have shape:

```text
(144, 144, 86)
```

Channel layout:
- `0-79`: atmospheric variables
- `80-85`: flux + fuel variables

Default configurable layout:

```yaml
channel_layout:
  atmospheric_channels: [0, 79]
  flux_channels: [80, 81, 82, 83]
  fuel_channels: [84, 85]
  surface_fuel_channel: 84
  canopy_fuel_channel: 85
  flux_mask_channel: 80
```

## Inputs
Base input channels: `86`

With the default engineered features enabled, the dataset appends:
- `4` flux delta channels
- `2` fuel delta channels
- `2` step consumed fuel channels
- `2` cumulative consumed fuel channels

Total input channels:

```text
86 + 10 = 96
```

So the default model config uses:

```yaml
model:
  input_channels: 96
  output_channels: 3
```

## Leakage Rule
- Input features may use time `t` and earlier only.
- Labels may use `t_future = t + prediction_horizon`.
- Engineered deltas and consumed-fuel input features must never use `t_future` or any later frame.

Why fuel/flux history is allowed as input:
- past fuel/flux are known historical state
- future fuel/perimeter are prediction targets

## Multitask Labels
For sample start index `i`:
- input frames: `i ... i + input_sequence_length - 1`
- current time: `t = i + input_sequence_length - 1`
- future time: `t_future = t + prediction_horizon`

### Channel 0
Surface consumed fuel:

```text
surface consumed fuel = current surface fuel - future surface fuel
```

### Channel 1
Canopy consumed fuel:

```text
canopy consumed fuel = current canopy fuel - future canopy fuel
```

### Channel 2
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
The multitask regression heads predict consumed fuel, not future fuel directly.

Reconstruction:

```text
predicted future surface fuel = current surface fuel - predicted surface consumed fuel
predicted future canopy fuel = current canopy fuel - predicted canopy consumed fuel
```

These reconstructed maps are clamped to `>= 0`.

## Normalization
Input normalization is computed after engineering, not on the raw 86-channel tensors.

That means the normalization archive must match the final model input channel count, which is `96` in the default setup.

Default target normalization for multitask is off:

```yaml
target_normalization:
  enabled: false
```

The binary mask target is never normalized.

## Visualization
Default multitask prediction figures are saved to:

```text
outputs/visualizations_multitask/
```

Each figure includes:
1. Current surface fuel
2. True future surface fuel
3. Predicted future surface fuel
4. Surface fuel prediction error
5. Current canopy fuel
6. True future canopy fuel
7. Predicted future canopy fuel
8. Canopy fuel prediction error
9. True mask
10. Predicted mask probability
11. Predicted binary mask
12. Predicted / true perimeter contour overlay

Reconstructed fuel-bed figures are saved to:

```text
outputs/reconstructed_fuel_beds/
```

Engineered-feature inspection figures are saved to:

```text
outputs/engineered_feature_inspection/
```

## Commands
Compute normalization:

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

Inspect engineered features:

```bash
python scripts/inspect_engineered_features.py --config configs/default.yaml --sample_index 0
```

Sanity check:

```bash
python scripts/sanity_check_project.py --config configs/default.yaml
```

Train:

```bash
python scripts/train_convlstm_unet.py --config configs/default.yaml
```

Test:

```bash
python scripts/test_model.py --config configs/default.yaml
```

Visualize predictions:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --num_samples 10
```

Reconstruct future fuel beds:

```bash
python scripts/reconstruct_fuel_bed_from_predictions.py --config configs/default.yaml --num_samples 10
```

Optional engineered-tensor caching:

```bash
python scripts/cache_engineered_dataset.py --config configs/default.yaml --output_dir ../keepz_05_engineered
```
